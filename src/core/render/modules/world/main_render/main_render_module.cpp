#include "core/render/modules/world/main_render/main_render_module.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/frame_generation.hpp"
#include "core/render/modules/world/main_render/submodules/world_prepare.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"
#include "core/util/parallel.hpp"
#include "core/vulkan/result.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;

// One pool per frame slot: acquireContext waits this slot's fence before reuse.
// These are inclusive GPU intervals, not sums of independently scheduled work.
struct MainRenderTimingCapture {
    enum Stage : uint32_t { WorldPrepare, FrameGraph, RayDispatch, Count };
    std::shared_ptr<vk::Device> device;
    VkQueryPool pool = VK_NULL_HANDLE;
    VkCommandBuffer frameCommand = VK_NULL_HANDLE;
    std::array<bool, Count> written{};
    float period = 0;
    uint64_t mask = UINT64_MAX;
    uint32_t slot = 0;
    uint32_t previousWidth = 0, previousHeight = 0;
    uint32_t traceWidth = 0, traceHeight = 0;
    std::chrono::steady_clock::time_point lastReport{};

    explicit MainRenderTimingCapture(const std::shared_ptr<FrameworkContext>& context)
        : device(context->device), slot(context->frameIndex) {
        period = context->physicalDevice->properties().limits.timestampPeriod;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(context->physicalDevice->vkPhysicalDevice(), &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(context->physicalDevice->vkPhysicalDevice(), &count, queues.data());
        const uint32_t bits = queues.at(context->physicalDevice->mainQueueIndex()).timestampValidBits;
        if (!bits || period <= 0) return;
        mask = bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1;
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = Count * 2;
        if (vkCreateQueryPool(device->vkDevice(), &info, nullptr, &pool) != VK_SUCCESS) pool = VK_NULL_HANDLE;
        if (pool != VK_NULL_HANDLE && slot == 0)
            std::fprintf(stderr, "[RADIANCE-E18] detailed_gpu_timing=enabled preserve_quality=1\n");
    }
    ~MainRenderTimingCapture() {
        if (pool != VK_NULL_HANDLE) vkDestroyQueryPool(device->vkDevice(), pool, nullptr);
    }
    void beginFrame(VkCommandBuffer command, uint32_t width, uint32_t height) {
        if (pool == VK_NULL_HANDLE) return;
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport >= std::chrono::seconds(5)) {
            static const char* names[] = {"world_prepare", "frame_graph", "ray_dispatch_first"};
            bool reported = false;
            for (uint32_t i = 0; i < Count; ++i) {
                if (!written[i]) continue;
                uint64_t values[2]{};
                if (vkGetQueryPoolResults(device->vkDevice(), pool, i * 2, 2, sizeof(values), values,
                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) continue;
                std::fprintf(stderr,
                    "[RADIANCE-E18-PERF] slot=%u render=%ux%u trace=%ux%u stage=%s gpu_interval_ms=%.3f\n",
                    slot, previousWidth, previousHeight, traceWidth, traceHeight, names[i],
                    double((values[1] - values[0]) & mask) * period / 1e6);
                reported = true;
            }
            if (reported) lastReport = now;
        }
        written.fill(false);
        previousWidth = width; previousHeight = height;
        traceWidth = traceHeight = 0;
        frameCommand = command;
        vkCmdResetQueryPool(command, pool, 0, Count * 2);
    }
    bool beginStage(VkCommandBuffer command, Stage stage) {
        // Async branch command buffers have their own scheduling; do not write
        // this frame's primary query pool from those branches or repeat a slot.
        if (pool == VK_NULL_HANDLE || command != frameCommand || written[stage]) return false;
        written[stage] = true;
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, stage * 2);
        return true;
    }
    void endStage(VkCommandBuffer command, Stage stage) {
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, stage * 2 + 1);
    }
};

static constexpr std::array<std::string_view, 22> defaultRenderLayerOrder = {
    "solid",
    "cutout_mipped",
    "cutout",
    "entity_solid",
    "entity_solid_z_offset_forward",
    "entity_cutout",
    "entity_cutout_no_cull",
    "entity_cutout_no_cull_z_offset",
    "entity_smooth_cutout",
    "entity_no_outline",
    "armor_cutout_no_cull",
    "entity_decal",
    "entity_alpha",
    "eyes",
    "translucent",
    "translucent_moving_block",
    "tripwire",
    "item_entity_translucent_cull",
    "entity_translucent",
    "entity_translucent_emissive",
    "armor_translucent",
    "default",
};

struct RenderIndirectMetadataHeader {
    uint32_t drawCount = 0;
    uint32_t metadataOffsetBytes = 40;
    uint32_t metadataStrideBytes = 48;
    uint32_t commandStrideBytes = static_cast<uint32_t>(sizeof(VkDrawIndirectCommand));
    uint32_t chunkInstanceIndexBase = 0;
    uint32_t chunkGeometryAddressIndexBase = 0;
    uint32_t entityInstanceCount = 0;
    uint32_t entityGeometryCount = 0;
    uint32_t chunkInstanceCount = 0;
    uint32_t chunkGeometrySlotCount = 0;
};
static_assert(sizeof(RenderIndirectMetadataHeader) == 40);

struct RenderIndirectDrawMetadata {
    uint32_t objectType = 0;
    uint32_t objectIndex = 0;
    uint32_t geometryIndex = 0;
    uint32_t instanceIndex = 0;
    uint32_t geometryAddressIndex = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t layerOrder = std::numeric_limits<uint32_t>::max();
    uint32_t renderLayerHashLow = 0;
    uint32_t renderLayerHashHigh = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
};
static_assert(sizeof(RenderIndirectDrawMetadata) == 48);

struct RenderIndirectDrawBatch {
    uint32_t sourceOffset = 0;
    uint32_t sourceCount = 0;
};
static_assert(sizeof(RenderIndirectDrawBatch) == 8);

struct RenderPipelineBatchKey {
    const vk::GraphicsPipeline *pipeline = nullptr;
    const vk::DeviceLocalBuffer *pbrBuffer = nullptr;
    const vk::DeviceLocalBuffer *indexBuffer = nullptr;

    bool operator==(const RenderPipelineBatchKey &rhs) const {
        return pipeline == rhs.pipeline && pbrBuffer == rhs.pbrBuffer && indexBuffer == rhs.indexBuffer;
    }
};

struct RenderPipelineBatchKeyHash {
    size_t operator()(const RenderPipelineBatchKey &key) const {
        size_t seed = 0;
        auto combine = [&](const void *ptr) {
            seed ^= std::hash<const void *>{}(ptr) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(key.pipeline);
        combine(key.pbrBuffer);
        combine(key.indexBuffer);
        return seed;
    }
};

struct RenderIndexedDrawCommand {
    uint32_t indexCount = 0;
    uint32_t instanceCount = 0;
    uint32_t firstIndex = 0;
    int32_t vertexOffset = 0;
    uint32_t firstInstance = 0;
};

static constexpr uint32_t renderIndirectMetadataDescriptorSet = 6;
static constexpr uint32_t renderIndirectMetadataDescriptorBinding = 0;
static constexpr uint32_t renderIndirectCommandsDescriptorBinding = 1;
static constexpr uint32_t renderIndirectWorkgroupSize = 64;
static constexpr uint32_t directionalShadowDepthDescriptorBinding = 16;
static constexpr uint32_t directionalShadowConfigDescriptorBinding = 24;
static constexpr uint32_t directionalShadowCommandsDescriptorBinding = 25;
static constexpr uint32_t directionalShadowMetadataDescriptorBinding = 26;
static constexpr uint32_t directionalShadowIndirectWorkgroupSize = 64;
static constexpr uint32_t segmentationMaskDescriptorSet = 3;
static constexpr uint32_t segmentationMaskDescriptorBinding = 23;
static constexpr VkFormat segmentationMaskFormat = VK_FORMAT_R8_UINT;
static constexpr VkImageUsageFlags segmentationMaskUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
static constexpr VkPipelineStageFlags2 segmentationMaskShaderStages = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
static constexpr VkAccessFlags2 segmentationMaskShaderAccess =
    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
static constexpr float directionalShadowMinDistanceBlocks = 32.0f;
static constexpr float directionalShadowMaxDistanceBlocks = 128.0f;
static constexpr float directionalShadowDepthBias = 0.00020f;
static constexpr VkFormat directionalShadowDepthFormat = VK_FORMAT_D32_SFLOAT;
static constexpr uint32_t renderIndirectDrawFlagEntity = 1u << 0u;
static constexpr uint32_t renderIndirectDrawFlagHand = 1u << 1u;
static constexpr uint32_t rayTracingMaskHand = 8u;

static std::filesystem::path worldPrepareCompiledShaderPath(std::string_view fileName) {
    return Renderer::folderPath / "shaders/world/ray_tracing/world_prepare" / std::string(fileName);
}

static uint64_t stableFnv1a64(std::string_view text) {
    uint64_t hash = 14695981039346656037ULL;
    for (char ch : text) {
        hash ^= static_cast<uint8_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static bool nameLooksRasterPrimary(std::string_view text) {
    std::string lower(text);
    lower = ShaderPackLoader::toLower(std::move(lower));
    return lower.find("raster_primary") != std::string::npos || lower.find("primary_surface") != std::string::npos;
}

static bool resourceListLooksRasterPrimary(const ShaderPackLoader::ResourceList &resources) {
    auto resourceUsesRasterPrimary = [](const std::string &name) { return nameLooksRasterPrimary(name); };
    if (std::any_of(resources.images.begin(), resources.images.end(), resourceUsesRasterPrimary)) { return true; }
    if (std::any_of(resources.buffers.begin(), resources.buffers.end(), resourceUsesRasterPrimary)) { return true; }
    for (const auto &access : resources.imageAccesses) {
        if (nameLooksRasterPrimary(access.name)) { return true; }
    }
    for (const auto &access : resources.bufferAccesses) {
        if (nameLooksRasterPrimary(access.name)) { return true; }
    }
    return false;
}

static bool renderPassLooksRasterPrimary(const ShaderPackLoader::RenderPassConfig &pass) {
    return nameLooksRasterPrimary(pass.name) || nameLooksRasterPrimary(pass.content) ||
           resourceListLooksRasterPrimary(pass.inputs) || resourceListLooksRasterPrimary(pass.outputs) ||
           std::any_of(pass.clearOutputs.begin(), pass.clearOutputs.end(),
                       [](const auto &entry) { return nameLooksRasterPrimary(entry.first); });
}

static bool renderPassIsHandOnly(const ShaderPackLoader::RenderPassConfig &pass) {
    return ShaderPackLoader::toLower(pass.content) == "hand";
}

static constexpr uint64_t renderLayerHashClouds = 0x7322de50865bfb97ULL;
static constexpr uint64_t renderLayerHashEyes = 0xef907f6092ccf1abULL;
static constexpr uint64_t renderLayerHashEntityTranslucentEmissive = 0x39f9af49cac7eb40ULL;
static constexpr uint64_t renderLayerHashEntityNoOutline = 0x45c4d550abd9dd51ULL;
static constexpr uint64_t renderLayerHashSolid = 0x3e71ea6112bd4c94ULL;
static constexpr uint64_t renderLayerHashEntitySolid = 0x3397aea4d7cd0a2aULL;
static constexpr uint64_t renderLayerHashEntitySolidZOffsetForward = 0x04fee3ad7c449b69ULL;

static uint64_t renderLayerHashFromMetadata(const RenderIndirectDrawMetadata &metadata) {
    return static_cast<uint64_t>(metadata.renderLayerHashLow) |
           (static_cast<uint64_t>(metadata.renderLayerHashHigh) << 32u);
}

static bool renderLayerDoesNotCastDirectionalShadow(const RenderIndirectDrawMetadata &metadata) {
    const uint64_t renderLayerHash = renderLayerHashFromMetadata(metadata);
    return renderLayerHash == renderLayerHashClouds || renderLayerHash == renderLayerHashEyes ||
           renderLayerHash == renderLayerHashEntityTranslucentEmissive ||
           renderLayerHash == renderLayerHashEntityNoOutline;
}

static bool isSolidFastDirectionalShadowLayer(const RenderIndirectDrawMetadata &metadata) {
    const uint64_t renderLayerHash = renderLayerHashFromMetadata(metadata);
    return renderLayerHash == renderLayerHashSolid || renderLayerHash == renderLayerHashEntitySolid ||
           renderLayerHash == renderLayerHashEntitySolidZOffsetForward;
}

static bool canCastDirectionalShadow(const RenderIndirectDrawMetadata &metadata) {
    return metadata.indexCount > 0 && (metadata.flags & renderIndirectDrawFlagHand) == 0u &&
           !renderLayerDoesNotCastDirectionalShadow(metadata);
}

static std::unordered_map<std::string, std::string>
mergeShaderDefinitions(const std::unordered_map<std::string, std::string> &base,
                       const std::unordered_map<std::string, std::string> &overrides) {
    std::unordered_map<std::string, std::string> merged = base;
    for (const auto &[key, value] : overrides) { merged[key] = value; }
    return merged;
}

static void appendShaderDefinitionsKey(std::string &key,
                                       const std::unordered_map<std::string, std::string> &definitions) {
    if (definitions.empty()) { return; }

    std::vector<std::pair<std::string, std::string>> sortedDefinitions(definitions.begin(), definitions.end());
    std::sort(sortedDefinitions.begin(), sortedDefinitions.end());
    for (const auto &[name, value] : sortedDefinitions) { key += "|def:" + name + "=" + value; }
}

static VkCullModeFlags parseCullMode(const std::string &value, const std::string &context) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower == "none") { return VK_CULL_MODE_NONE; }
    if (lower == "front") { return VK_CULL_MODE_FRONT_BIT; }
    if (lower == "back") { return VK_CULL_MODE_BACK_BIT; }
    if (lower == "front_and_back" || lower == "all") { return VK_CULL_MODE_FRONT_AND_BACK; }
    throw std::runtime_error("unsupported " + context + " cull mode: " + value);
}

static VkFrontFace parseFrontFace(const std::string &value, const std::string &context) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower == "counter_clockwise" || lower == "ccw") { return VK_FRONT_FACE_COUNTER_CLOCKWISE; }
    if (lower == "clockwise" || lower == "cw") { return VK_FRONT_FACE_CLOCKWISE; }
    throw std::runtime_error("unsupported " + context + " front_face: " + value);
}

static VkColorComponentFlags parseColorWriteMask(const std::string &value, const std::string &context) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower.empty() || lower == "none" || lower == "off" || lower == "0") { return 0; }
    if (lower == "all" || lower == "rgba") {
        return VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
               VK_COLOR_COMPONENT_A_BIT;
    }

    VkColorComponentFlags mask = 0;
    for (char ch : lower) {
        if (ch == 'r') {
            mask |= VK_COLOR_COMPONENT_R_BIT;
        } else if (ch == 'g') {
            mask |= VK_COLOR_COMPONENT_G_BIT;
        } else if (ch == 'b') {
            mask |= VK_COLOR_COMPONENT_B_BIT;
        } else if (ch == 'a') {
            mask |= VK_COLOR_COMPONENT_A_BIT;
        } else {
            throw std::runtime_error("unsupported " + context + " color_write_mask component: " + value);
        }
    }
    return mask;
}

std::string MainRenderModule::buildMainRenderDebugLabel(std::string_view passKind, std::string_view passName) {
    std::string label = "main_render/";
    label += passKind;
    label += "/";
    label += passName;
    return label;
}

VkCompareOp MainRenderModule::parseDepthCompare(const std::string &value) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower == "never") { return VK_COMPARE_OP_NEVER; }
    if (lower == "less") { return VK_COMPARE_OP_LESS; }
    if (lower == "equal") { return VK_COMPARE_OP_EQUAL; }
    if (lower == "less_or_equal" || lower == "lequal") { return VK_COMPARE_OP_LESS_OR_EQUAL; }
    if (lower == "greater") { return VK_COMPARE_OP_GREATER; }
    if (lower == "not_equal") { return VK_COMPARE_OP_NOT_EQUAL; }
    if (lower == "greater_or_equal" || lower == "gequal") { return VK_COMPARE_OP_GREATER_OR_EQUAL; }
    if (lower == "always") { return VK_COMPARE_OP_ALWAYS; }
    throw std::runtime_error("unsupported main_render depth_compare: " + value);
}

std::filesystem::path MainRenderModule::builtInShaderPackPath() {
    return Renderer::folderPath / "shaders/world/ray_tracing/advanced.zip";
}

std::shared_ptr<vk::Shader>
MainRenderModule::createShader(std::shared_ptr<vk::Device> device,
                               const std::filesystem::path &path,
                               VkShaderStageFlagBits stage,
                               const std::unordered_map<std::string, std::string> &definitions,
                               const std::vector<std::string> &includeDirectories,
                               const std::string &injectedSource) {
    return vk::Shader::create(device, path.string(), stage, definitions, includeDirectories, injectedSource);
}

struct MainRenderModule::MainRenderExecutionRuntime final : ShaderPack::ExecutionRuntime {
    std::shared_ptr<MainRenderModule> mainRenderModule;
    std::shared_ptr<MainRenderModuleContext> mainRenderContext;
    MainRenderFrameGraph *frameGraph = nullptr;
};

struct MainRenderModule::MainRenderFrameGraph final {
    enum class NodeKind {
        Pass,
        SharcUpdate,
        BufferUpdate,
    };

    enum class PhaseKind {
        Graphics,
        Compute,
        RayTracing,
        Transfer,
        Mixed,
    };

    struct Node {
        NodeKind kind = NodeKind::Pass;
        PassVariant passVariant;
        std::shared_ptr<RayTracingPass> sharcUpdatePass;
        std::string passName;
        ShaderPackLoader::ExecutionCommand::BufferUpdate bufferUpdate;
        ExecutionVariables executionVariables;
        std::vector<ExpressionEvaluator::Variable> expressionVariables;
    };

    struct Phase {
        PhaseKind kind = PhaseKind::Mixed;
        size_t firstNode = 0;
        size_t nodeCount = 0;
    };

    std::vector<Node> nodes;
    std::vector<Phase> phases;

    void clear() {
        nodes.clear();
        phases.clear();
    }

    void enqueuePass(PassVariant passVariant,
                     const ExecutionVariables &executionVariables,
                     const std::vector<ExpressionEvaluator::Variable> &expressionVariables) {
        nodes.push_back({
            .kind = NodeKind::Pass,
            .passVariant = std::move(passVariant),
            .executionVariables = executionVariables,
            .expressionVariables = expressionVariables,
        });
    }

    void enqueueSharcUpdate(std::string passName,
                            std::shared_ptr<RayTracingPass> sharcUpdatePass,
                            const ExecutionVariables &executionVariables,
                            const std::vector<ExpressionEvaluator::Variable> &expressionVariables) {
        nodes.push_back({
            .kind = NodeKind::SharcUpdate,
            .sharcUpdatePass = std::move(sharcUpdatePass),
            .passName = std::move(passName),
            .executionVariables = executionVariables,
            .expressionVariables = expressionVariables,
        });
    }

    void enqueueBufferUpdate(ShaderPackLoader::ExecutionCommand::BufferUpdate update) {
        nodes.push_back({
            .kind = NodeKind::BufferUpdate,
            .bufferUpdate = std::move(update),
        });
    }

    static PhaseKind phaseKindForPass(const PassVariant &passVariant) {
        return std::visit(
            [](const auto &compiledPass) {
                using PassType = std::decay_t<decltype(compiledPass)>;
                if constexpr (std::is_same_v<PassType, std::shared_ptr<FullScreenPass>>) {
                    return PhaseKind::Graphics;
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                    return PhaseKind::Graphics;
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                    return PhaseKind::RayTracing;
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                    return PhaseKind::Compute;
                }
                return PhaseKind::Mixed;
            },
            passVariant);
    }

    static PhaseKind phaseKindForNode(const Node &node) {
        switch (node.kind) {
            case NodeKind::Pass: return phaseKindForPass(node.passVariant);
            case NodeKind::SharcUpdate: return PhaseKind::Mixed;
            case NodeKind::BufferUpdate: return PhaseKind::Transfer;
        }
        return PhaseKind::Mixed;
    }

    void buildPhases() {
        phases.clear();
        for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
            const PhaseKind nodePhaseKind = phaseKindForNode(nodes[nodeIndex]);
            if (!phases.empty() && phases.back().kind == nodePhaseKind) {
                phases.back().nodeCount++;
                continue;
            }
            phases.push_back({
                .kind = nodePhaseKind,
                .firstNode = nodeIndex,
                .nodeCount = 1,
            });
        }
    }

    void record(MainRenderModule &module, MainRenderModuleContext &context);
};

struct MainRenderModule::MainRenderPassExecutionOp final : ShaderPack::ExecutionOp {
    explicit MainRenderPassExecutionOp(PassVariant mainRenderPassVariant)
        : mainRenderPassVariant(std::move(mainRenderPassVariant)) {}

    PassVariant mainRenderPassVariant;

    void operator()(ShaderPack::ExecutionRuntime &executionRuntime) const override {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.executePass");
        auto &runtime = dynamic_cast<MainRenderExecutionRuntime &>(executionRuntime);
        auto mainRenderModule = runtime.mainRenderModule;
        auto mainRenderContext = runtime.mainRenderContext;
        if (mainRenderModule == nullptr || mainRenderContext == nullptr) { return; }
        if (runtime.frameGraph == nullptr) { throw std::runtime_error("main_render frame graph is not active"); }
        runtime.frameGraph->enqueuePass(mainRenderPassVariant, runtime.executionVariables,
                                        runtime.expressionVariables());
    }
};

struct MainRenderModule::SharcUpdateExecutionOp final : ShaderPack::ExecutionOp {
    SharcUpdateExecutionOp(std::string passName, std::shared_ptr<RayTracingPass> sharcUpdatePass)
        : passName(std::move(passName)), sharcUpdatePass(std::move(sharcUpdatePass)) {}

    std::string passName;
    std::shared_ptr<RayTracingPass> sharcUpdatePass;

    void operator()(ShaderPack::ExecutionRuntime &executionRuntime) const override {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.executeSharcUpdate");
        auto &runtime = dynamic_cast<MainRenderExecutionRuntime &>(executionRuntime);
        auto mainRenderModule = runtime.mainRenderModule;
        auto mainRenderContext = runtime.mainRenderContext;
        if (mainRenderModule == nullptr || mainRenderContext == nullptr) { return; }
        if (runtime.frameGraph == nullptr) { throw std::runtime_error("main_render frame graph is not active"); }
        runtime.frameGraph->enqueueSharcUpdate(passName, sharcUpdatePass, runtime.executionVariables,
                                               runtime.expressionVariables());
    }
};

struct MainRenderModule::MainRenderBufferUpdateExecutionOp final : ShaderPack::ExecutionOp {
    explicit MainRenderBufferUpdateExecutionOp(ShaderPackLoader::ExecutionCommand::BufferUpdate update)
        : update(std::move(update)) {}

    ShaderPackLoader::ExecutionCommand::BufferUpdate update;

    void operator()(ShaderPack::ExecutionRuntime &executionRuntime) const override {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.executeBufferUpdate");
        auto &runtime = dynamic_cast<MainRenderExecutionRuntime &>(executionRuntime);
        auto mainRenderModule = runtime.mainRenderModule;
        auto mainRenderContext = runtime.mainRenderContext;
        if (mainRenderModule == nullptr || mainRenderContext == nullptr) { return; }
        if (runtime.frameGraph == nullptr) { throw std::runtime_error("main_render frame graph is not active"); }
        runtime.frameGraph->enqueueBufferUpdate(update);
    }
};

static constexpr VkAccessFlags2 mainRenderWriteAccessMask =
    VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

static constexpr VkAccessFlags2 mainRenderReadAccessMask =
    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT |
    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;

static bool mainRenderAccessWrites(VkAccessFlags2 accessMask) {
    return (accessMask & mainRenderWriteAccessMask) != 0;
}

static bool mainRenderAccessReads(VkAccessFlags2 accessMask) {
    return (accessMask & mainRenderReadAccessMask) != 0;
}

static VkPipelineStageFlags2 mainRenderShaderStageMaskForPass(VkPipelineStageFlags2 passStageMask, bool imageResource) {
    constexpr VkPipelineStageFlags2 shaderStageMask =
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    VkPipelineStageFlags2 passShaderStages = passStageMask & shaderStageMask;
    const bool graphicsPass = (passStageMask & (VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)) != 0;
    if (imageResource && graphicsPass) {
        passShaderStages &= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    }
    return passShaderStages == 0 ? passStageMask : passShaderStages;
}

static VkPipelineStageFlags2
mainRenderStageMaskForAccess(VkPipelineStageFlags2 passStageMask, VkAccessFlags2 accessMask, bool imageResource) {
    VkPipelineStageFlags2 stageMask = 0;
    constexpr VkAccessFlags2 shaderAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

    if ((accessMask & shaderAccessMask) != 0) {
        stageMask |= mainRenderShaderStageMaskForPass(passStageMask, imageResource);
    }
    if ((accessMask & (VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT)) != 0) {
        stageMask |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    }
    if ((accessMask & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT) != 0) {
        stageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    }
    if ((accessMask & (VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT)) != 0) {
        stageMask |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    }
    if ((accessMask & (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)) != 0) {
        stageMask |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    if ((accessMask &
         (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
        stageMask |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    }
    return stageMask == 0 ? passStageMask : stageMask;
}

void MainRenderModule::addFrameGraphImageBarrier(const ResolvedImageAccess &imageAccess,
                                                 std::vector<vk::CommandBuffer::ImageMemoryBarrier> &imageBarriers,
                                                 MainRenderModuleContext &context,
                                                 VkPipelineStageFlags2 passStageMask,
                                                 uint32_t frameIndex,
                                                 uint32_t queueIndex) {
    std::shared_ptr<vk::DeviceLocalImage> image = resolveImageResource(imageAccess.resource, frameIndex);
    if (image == nullptr) { return; }

    auto &trackedState = context.trackedImageStates[image.get()];
    const bool hasTrackedState = trackedState.initialized;
    const VkImageLayout oldLayout = hasTrackedState ? trackedState.layout : image->imageLayout();
    const VkPipelineStageFlags2 srcStageMask =
        hasTrackedState ? trackedState.stageMask :
                          (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                                                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    const VkAccessFlags2 trackedAccessMask =
        hasTrackedState ?
            trackedState.accessMask :
            (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    const VkPipelineStageFlags2 dstStageMask =
        mainRenderStageMaskForAccess(passStageMask, imageAccess.accessMask, true);

    const bool layoutChanged = oldLayout != imageAccess.layout;
    const bool sourceMayWrite =
        hasTrackedState ? mainRenderAccessWrites(trackedAccessMask) : oldLayout != VK_IMAGE_LAYOUT_UNDEFINED;
    const bool sourceMayRead = hasTrackedState && mainRenderAccessReads(trackedAccessMask);
    const bool needsSourceVisibility = sourceMayWrite &&
        (imageAccess.reads || imageAccess.writes || imageAccess.needsPreviousContent || layoutChanged);
    const bool needsExecutionOrdering = imageAccess.writes && (sourceMayWrite || sourceMayRead);
    const bool expandsReadScope = hasTrackedState && imageAccess.reads && !sourceMayWrite &&
        (dstStageMask & ~trackedState.stageMask) != 0;
    if (layoutChanged || needsSourceVisibility || needsExecutionOrdering || expandsReadScope) {
        imageBarriers.push_back({
            .srcStageMask = expandsReadScope ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT :
                (srcStageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : srcStageMask),
            .srcAccessMask = expandsReadScope ? VK_ACCESS_2_MEMORY_WRITE_BIT :
                (needsSourceVisibility ? trackedAccessMask : 0),
            .dstStageMask = dstStageMask,
            .dstAccessMask = (needsSourceVisibility || layoutChanged || expandsReadScope) ? imageAccess.accessMask : 0,
            .oldLayout = oldLayout,
            .newLayout = imageAccess.layout,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = image,
            .subresourceRange = image->fullSubresourceRange(),
        });
    }

    trackedState.layout = imageAccess.layout;
    trackedState.stageMask =
        !imageAccess.writes && !sourceMayWrite && !layoutChanged ? trackedState.stageMask | dstStageMask : dstStageMask;
    trackedState.accessMask = !imageAccess.writes && !sourceMayWrite && !layoutChanged ?
                                  trackedState.accessMask | imageAccess.accessMask :
                                  imageAccess.accessMask;
    trackedState.initialized = true;
    image->imageLayout() = imageAccess.layout;
}

void MainRenderModule::addFrameGraphBufferBarrier(const ResolvedBufferAccess &bufferAccess,
                                                  std::vector<vk::CommandBuffer::BufferMemoryBarrier> &bufferBarriers,
                                                  MainRenderModuleContext &context,
                                                  VkPipelineStageFlags2 passStageMask,
                                                  uint32_t frameIndex,
                                                  uint32_t queueIndex,
                                                  VkDeviceSize offset,
                                                  VkDeviceSize size) {
    if (bufferAccess.runtimeBuffer == nullptr) { return; }
    std::shared_ptr<vk::DeviceLocalBuffer> buffer = findRuntimeVKBuffer(*bufferAccess.runtimeBuffer, frameIndex);
    if (buffer == nullptr) { return; }

    auto &trackedState = context.trackedBufferStates[buffer.get()];
    const bool hasTrackedState = trackedState.initialized;
    const VkPipelineStageFlags2 dstStageMask =
        mainRenderStageMaskForAccess(passStageMask, bufferAccess.accessMask, false);

    const bool sourceMayWrite = hasTrackedState && mainRenderAccessWrites(trackedState.accessMask);
    const bool sourceMayRead = hasTrackedState && mainRenderAccessReads(trackedState.accessMask);
    const bool needsSourceVisibility = sourceMayWrite &&
        (bufferAccess.reads || bufferAccess.writes || bufferAccess.needsPreviousContent);
    const bool needsExecutionOrdering = bufferAccess.writes && (sourceMayWrite || sourceMayRead);
    const bool expandsReadScope = hasTrackedState && bufferAccess.reads && !sourceMayWrite &&
        (dstStageMask & ~trackedState.stageMask) != 0;
    if (needsSourceVisibility || needsExecutionOrdering || expandsReadScope) {
        bufferBarriers.push_back({
            .srcStageMask = expandsReadScope ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT :
                (trackedState.stageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : trackedState.stageMask),
            .srcAccessMask = expandsReadScope ? VK_ACCESS_2_MEMORY_WRITE_BIT :
                (needsSourceVisibility ? trackedState.accessMask : 0),
            .dstStageMask = dstStageMask,
            .dstAccessMask = (needsSourceVisibility || expandsReadScope) ? bufferAccess.accessMask : 0,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .buffer = buffer,
            .offset = offset,
            .size = size,
        });
    }

    trackedState.stageMask =
        !bufferAccess.writes && !sourceMayWrite ? trackedState.stageMask | dstStageMask : dstStageMask;
    trackedState.accessMask = !bufferAccess.writes && !sourceMayWrite ?
                                  trackedState.accessMask | bufferAccess.accessMask :
                                  bufferAccess.accessMask;
    trackedState.initialized = true;
}

void MainRenderModule::addFrameGraphResourceBarriers(
    const ResolvedPassResources &resources,
    std::vector<vk::CommandBuffer::BufferMemoryBarrier> &bufferBarriers,
    std::vector<vk::CommandBuffer::ImageMemoryBarrier> &imageBarriers,
    MainRenderModuleContext &context,
    VkPipelineStageFlags2 passStageMask,
    uint32_t frameIndex,
    uint32_t queueIndex) {
    for (const ResolvedImageAccess &imageAccess : resources.imageAccesses) {
        addFrameGraphImageBarrier(imageAccess, imageBarriers, context, passStageMask, frameIndex, queueIndex);
    }
    if (resources.fullScreenTarget.has_value()) {
        addFrameGraphImageBarrier(*resources.fullScreenTarget, imageBarriers, context, passStageMask, frameIndex,
                                  queueIndex);
    }
    for (const ResolvedImageAccess &target : resources.colorAttachmentTargets) {
        addFrameGraphImageBarrier(target, imageBarriers, context, passStageMask, frameIndex, queueIndex);
    }
    for (const ResolvedBufferAccess &bufferAccess : resources.bufferAccesses) {
        addFrameGraphBufferBarrier(bufferAccess, bufferBarriers, context, passStageMask, frameIndex, queueIndex);
    }
}

void MainRenderModule::addRenderDepthBarrier(std::vector<vk::CommandBuffer::ImageMemoryBarrier> &imageBarriers,
                                             MainRenderModuleContext &context,
                                             uint32_t frameIndex,
                                             uint32_t queueIndex,
                                             bool depthLoad) {
    ResolvedImageAccess depthAccess{
        .resource =
            ResolvedImageResource{
                .targetImages = &rasterDepthImages_,
                .runtimeTexture = nullptr,
                .isImportedRuntimeTexture = false,
            },
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .accessMask = depthLoad ? (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) :
                                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .reads = depthLoad,
        .writes = true,
        .needsPreviousContent = depthLoad,
    };
    addFrameGraphImageBarrier(depthAccess, imageBarriers, context,
                              VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                              frameIndex, queueIndex);
}

void MainRenderModule::uploadPassExecutionConstants(const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                                                    const ExecutionVariables &executionVariables,
                                                    MainRenderModuleContext &context,
                                                    VkShaderStageFlags shaderStages) {
    if (executionBuffer == nullptr) { return; }

    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    shaderPack_->uploadExecutionBuffer(ShaderPackLoader::Stage::RayTracing, executionBuffer, executionVariables,
                                       frameworkContext->worldCommandBuffer, context.mainRenderDescriptorTable,
                                       shaderPack_->executionSet(5u), framework->physicalDevice()->mainQueueIndex(),
                                       shaderStages);
}

void MainRenderModule::recordBufferUpdate(const ShaderPackLoader::ExecutionCommand::BufferUpdate &update,
                                          MainRenderModuleContext &context) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.recordBufferUpdate");
    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    const uint32_t frameIndex = frameworkContext->frameIndex;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();

    auto runtimeBuffer = findRuntimeBuffer(update.bufferName);
    if (!runtimeBuffer.has_value()) {
        throw std::runtime_error("main_render buffer_update references unknown runtime buffer: " + update.bufferName);
    }
    auto buffer = findRuntimeVKBuffer(runtimeBuffer->get(), frameIndex);
    if (buffer == nullptr) {
        throw std::runtime_error("main_render buffer_update has no allocated runtime buffer: " + update.bufferName);
    }
    const VkDeviceSize byteOffset = update.offset;
    const VkDeviceSize byteSize = update.uint32Values.size() * sizeof(uint32_t);
    if ((byteOffset & 3u) != 0u) {
        throw std::runtime_error("main_render buffer_update offset must be 4-byte aligned: " + update.bufferName);
    }
    if (byteSize == 0 || byteSize > 65536u || byteOffset + byteSize > buffer->size()) {
        throw std::runtime_error("main_render buffer_update range is invalid for runtime buffer: " + update.bufferName);
    }

    ResolvedBufferAccess transferWriteAccess{
        .runtimeBuffer = &runtimeBuffer->get(),
        .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .reads = false,
        .writes = true,
        .needsPreviousContent = false,
    };
    std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers;
    addFrameGraphBufferBarrier(transferWriteAccess, bufferBarriers, context, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               frameIndex, queueIndex, byteOffset, byteSize);
    if (!bufferBarriers.empty()) { worldCommandBuffer->barriersBufferImage(bufferBarriers, {}); }

    std::string debugLabelName = buildMainRenderDebugLabel("dynamic/buffer_update", update.bufferName);
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.30f, 0.75f, 0.95f);
    vkCmdUpdateBuffer(worldCommandBuffer->vkCommandBuffer(), buffer->vkBuffer(), byteOffset, byteSize,
                      update.uint32Values.data());

    auto &trackedState = context.trackedBufferStates[buffer.get()];
    trackedState.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    trackedState.accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    trackedState.initialized = true;
}

void MainRenderModule::clearSegmentationMask(MainRenderModuleContext &context) {
    if (context.segmentationMaskImage == nullptr) { return; }

    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();
    auto image = context.segmentationMaskImage;

    auto &trackedState = context.trackedImageStates[image.get()];
    const bool hasTrackedState = trackedState.initialized;
    const VkImageLayout oldLayout = hasTrackedState ? trackedState.layout : image->imageLayout();
    const VkPipelineStageFlags2 srcStageMask =
        hasTrackedState ? trackedState.stageMask :
                          (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                                                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    const VkAccessFlags2 srcAccessMask =
        hasTrackedState ?
            trackedState.accessMask :
            (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = srcStageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : srcStageMask,
                .srcAccessMask = srcAccessMask,
                .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .oldLayout = oldLayout,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = queueIndex,
                .dstQueueFamilyIndex = queueIndex,
                .image = image,
                .subresourceRange = image->fullSubresourceRange(),
            }});
    image->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkClearColorValue clearColor{};
    clearColor.uint32[0] = 1u;
    const VkImageSubresourceRange range = image->fullSubresourceRange();
    vkCmdClearColorImage(worldCommandBuffer->vkCommandBuffer(), image->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &clearColor, 1, &range);

    worldCommandBuffer->barriersBufferImage({}, {{
                                                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                    .dstStageMask = segmentationMaskShaderStages,
                                                    .dstAccessMask = segmentationMaskShaderAccess,
                                                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                    .srcQueueFamilyIndex = queueIndex,
                                                    .dstQueueFamilyIndex = queueIndex,
                                                    .image = image,
                                                    .subresourceRange = image->fullSubresourceRange(),
                                                }});

    trackedState.layout = VK_IMAGE_LAYOUT_GENERAL;
    trackedState.stageMask = segmentationMaskShaderStages;
    trackedState.accessMask = segmentationMaskShaderAccess;
    trackedState.initialized = true;
    image->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
}

void MainRenderModule::prepareSegmentationMaskForToneMapping(MainRenderModuleContext &context) {
    if (context.segmentationMaskImage == nullptr) { return; }

    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();
    auto image = context.segmentationMaskImage;

    auto &trackedState = context.trackedImageStates[image.get()];
    const bool hasTrackedState = trackedState.initialized;
    const VkImageLayout oldLayout = hasTrackedState ? trackedState.layout : image->imageLayout();
    const VkPipelineStageFlags2 srcStageMask =
        hasTrackedState ? trackedState.stageMask :
                          (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                                                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    const VkAccessFlags2 srcAccessMask =
        hasTrackedState ?
            trackedState.accessMask :
            (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = srcStageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : srcStageMask,
                .srcAccessMask = srcAccessMask,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout = oldLayout,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = queueIndex,
                .dstQueueFamilyIndex = queueIndex,
                .image = image,
                .subresourceRange = image->fullSubresourceRange(),
            }});

    trackedState.layout = VK_IMAGE_LAYOUT_GENERAL;
    trackedState.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    trackedState.accessMask = VK_ACCESS_2_SHADER_READ_BIT;
    trackedState.initialized = true;
    image->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
}

void MainRenderModule::MainRenderFrameGraph::record(MainRenderModule &module, MainRenderModuleContext &context) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.FrameGraph.record");
    buildPhases();

    auto frameworkContext = context.frameworkContext.lock();
    if (frameworkContext == nullptr) { throw std::runtime_error("main_render frame graph has no framework context"); }
    auto framework = frameworkContext->framework.lock();
    if (framework == nullptr) { throw std::runtime_error("main_render frame graph has no framework"); }
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    if (worldCommandBuffer == nullptr) { throw std::runtime_error("main_render frame graph has no command buffer"); }
    const uint32_t frameIndex = frameworkContext->frameIndex;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();

    constexpr VkShaderStageFlags executionShaderStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT |
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    auto submitGraphBarriers = [&](const ResolvedPassResources &resources, VkPipelineStageFlags2 passStageMask,
                                   bool usesRasterDepth, bool depthLoad) {
        std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers;
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;
        module.addFrameGraphResourceBarriers(resources, bufferBarriers, imageBarriers, context, passStageMask,
                                             frameIndex, queueIndex);
        if (usesRasterDepth) {
            module.addRenderDepthBarrier(imageBarriers, context, frameIndex, queueIndex, depthLoad);
        }
        if (!bufferBarriers.empty() || !imageBarriers.empty()) {
            worldCommandBuffer->barriersBufferImage(bufferBarriers, imageBarriers);
        }
    };

    auto recordPassNode = [&](const Node &node) {
        std::visit(
            [&](const auto &compiledPass) {
                using PassType = std::decay_t<decltype(compiledPass)>;
                if (compiledPass == nullptr) { return; }

                if constexpr (std::is_same_v<PassType, std::shared_ptr<FullScreenPass>>) {
                    std::string debugLabelName =
                        MainRenderModule::buildMainRenderDebugLabel("dynamic/fullscreen", compiledPass->config.name);
                    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.95f,
                                                                   0.90f, 0.20f);
                    submitGraphBarriers(module.resolvedResourcesForPass(*compiledPass),
                                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, false, false);
                    module.uploadPassExecutionConstants(compiledPass->executionBuffer, node.executionVariables, context,
                                                        executionShaderStages);
                    module.renderFullScreenPass(*compiledPass, context);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                    if (frameIndex >= compiledPass->framebuffers.size() ||
                        compiledPass->framebuffers[frameIndex] == nullptr) {
                        return;
                    }
                    std::string debugLabelName =
                        MainRenderModule::buildMainRenderDebugLabel("dynamic/raster", compiledPass->config.name);
                    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.95f,
                                                                   0.90f, 0.20f);
                    submitGraphBarriers(
                        module.resolvedResourcesForPass(*compiledPass),
                        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        true, compiledPass->config.depthLoad);
                    module.uploadPassExecutionConstants(compiledPass->executionBuffer, node.executionVariables, context,
                                                        executionShaderStages);
                    module.renderRenderPass(*compiledPass, context);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                    {
                        std::string prepareLabelName =
                            MainRenderModule::buildMainRenderDebugLabel("rt_prepare", compiledPass->config.name);
                        vk::CommandBuffer::ScopedDebugLabel prepareDebugLabel(
                            *worldCommandBuffer, prepareLabelName.c_str(), 0.85f, 0.55f, 0.20f);
                        if (!context.ensureRayTracingWorldPrepareResources()) { return; }
                    }
                    std::string debugLabelName =
                        MainRenderModule::buildMainRenderDebugLabel("dynamic/ray_tracing", compiledPass->config.name);
                    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.95f,
                                                                   0.90f, 0.20f);
                    submitGraphBarriers(module.resolvedResourcesForPass(*compiledPass),
                                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, false, false);
                    module.uploadPassExecutionConstants(compiledPass->executionBuffer, node.executionVariables, context,
                                                        executionShaderStages);
                    module.renderRayTracingPass(*compiledPass, context, node.expressionVariables);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                    std::string debugLabelName =
                        MainRenderModule::buildMainRenderDebugLabel("dynamic/compute", compiledPass->config.name);
                    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.95f,
                                                                   0.90f, 0.20f);
                    submitGraphBarriers(module.resolvedResourcesForPass(*compiledPass),
                                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, false, false);
                    module.uploadPassExecutionConstants(compiledPass->executionBuffer, node.executionVariables, context,
                                                        executionShaderStages);
                    module.renderComputePass(*compiledPass, context, node.expressionVariables);
                }
            },
            node.passVariant);
    };

    auto recordSharcUpdateNode = [&](const Node &node) {
        if (!module.hasSharcRuntime_ || !module.isSharcEnabled_ || node.sharcUpdatePass == nullptr) { return; }
        {
            std::string prepareLabelName = MainRenderModule::buildMainRenderDebugLabel("rt_prepare", node.passName);
            vk::CommandBuffer::ScopedDebugLabel prepareDebugLabel(*worldCommandBuffer, prepareLabelName.c_str(), 0.85f,
                                                                  0.55f, 0.20f);
            if (!context.ensureRayTracingWorldPrepareResources()) { return; }
        }

        std::string debugLabelName = MainRenderModule::buildMainRenderDebugLabel("dynamic/sharc_update", node.passName);
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 1.0f, 0.35f, 0.15f);
        submitGraphBarriers(module.resolvedResourcesForPass(*node.sharcUpdatePass),
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            false, false);
        module.uploadPassExecutionConstants(node.sharcUpdatePass->executionBuffer, node.executionVariables, context,
                                            executionShaderStages);
        module.renderSharcUpdateAndResolve(*node.sharcUpdatePass, context, node.expressionVariables);
    };

    auto recordNode = [&](const Node &node) {
        switch (node.kind) {
            case NodeKind::Pass: recordPassNode(node); break;
            case NodeKind::SharcUpdate: recordSharcUpdateNode(node); break;
            case NodeKind::BufferUpdate: module.recordBufferUpdate(node.bufferUpdate, context); break;
        }
    };

    struct Access {
        bool writes = false;
        std::optional<VkImageLayout> layout;
    };
    using AccessMap = std::unordered_map<const void *, Access>;
    struct PendingBranch {
        size_t submissionIndex;
        AccessMap accesses;
        decltype(context.trackedImageStates) imageStates;
        decltype(context.trackedBufferStates) bufferStates;
        bool writesSharc = false;
        bool joined = false;
    };
    std::vector<PendingBranch> pendingBranches;
    const void *sharcResource = &module.sharcResolvedBuffer_;

    auto asyncGroup = [&](const Node &node) -> std::string {
        if (!framework->device()->hasIndependentSecondaryQueue()) { return {}; }
        if (node.kind == NodeKind::SharcUpdate) {
            return node.sharcUpdatePass != nullptr ? node.sharcUpdatePass->config.asyncGroup : std::string{};
        }
        if (node.kind != NodeKind::Pass) { return {}; }
        return std::visit(
            [](const auto &pass) -> std::string {
                using PassType = std::decay_t<decltype(pass)>;
                if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>> ||
                              std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                    return pass != nullptr ? pass->config.asyncGroup : std::string{};
                }
                return {};
            },
            node.passVariant);
    };

    auto collectAccesses = [&](const Node &node, AccessMap &accesses, ResolvedPassResources &firstUses) {
        auto addImage = [&](const ResolvedImageAccess &access) {
            auto image = module.resolveImageResource(access.resource, frameIndex);
            if (image == nullptr) { return; }
            auto [iter, inserted] = accesses.try_emplace(image.get(), Access{access.writes, access.layout});
            if (inserted) { firstUses.imageAccesses.push_back(access); }
            iter->second.writes |= access.writes || iter->second.layout != access.layout;
        };
        auto addBuffer = [&](const ResolvedBufferAccess &access) {
            auto buffer = module.findRuntimeVKBuffer(*access.runtimeBuffer, frameIndex);
            if (buffer == nullptr) { return; }
            auto [iter, inserted] = accesses.try_emplace(buffer.get(), Access{access.writes, std::nullopt});
            if (inserted) { firstUses.bufferAccesses.push_back(access); }
            iter->second.writes |= access.writes;
        };
        auto addResources = [&](const ResolvedPassResources &resources) {
            for (const auto &access : resources.imageAccesses) { addImage(access); }
            if (resources.fullScreenTarget) { addImage(*resources.fullScreenTarget); }
            for (const auto &access : resources.colorAttachmentTargets) { addImage(access); }
            for (const auto &access : resources.bufferAccesses) { addBuffer(access); }
        };
        if (node.kind == NodeKind::BufferUpdate) {
            auto buffer = module.findRuntimeBuffer(node.bufferUpdate.bufferName);
            if (buffer) {
                addBuffer(
                    {.runtimeBuffer = &buffer->get(), .accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .writes = true});
            }
        } else if (node.kind == NodeKind::SharcUpdate) {
            if (node.sharcUpdatePass != nullptr && module.hasSharcRuntime_ && module.isSharcEnabled_) {
                addResources(module.resolvedResourcesForPass(*node.sharcUpdatePass));
                accesses[sharcResource].writes = true;
                accesses[node.sharcUpdatePass.get()].writes = true;
            }
        } else {
            std::visit(
                [&](const auto &pass) {
                    if (pass == nullptr) { return; }
                    addResources(module.resolvedResourcesForPass(*pass));
                    accesses[pass.get()].writes = true;
                    using PassType = std::decay_t<decltype(pass)>;
                    if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>> ||
                                  std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                        if (module.hasSharcRuntime_ && pass->config.querySharc) { accesses.try_emplace(sharcResource); }
                    } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                        accesses[module.rasterDepthImages_[frameIndex].get()].writes = true;
                    }
                },
                node.passVariant);
        }
    };

    auto joinBranch = [&](PendingBranch &branch) {
        if (branch.joined) { return; }
        frameworkContext->joinWorldAsyncBranch(branch.submissionIndex);
        for (const auto &[resource, access] : branch.accesses) {
            if (auto iter = branch.imageStates.find(resource); iter != branch.imageStates.end()) {
                auto &state = context.trackedImageStates[resource];
                if (access.writes) {
                    state = iter->second;
                } else {
                    state.stageMask |= iter->second.stageMask;
                    state.accessMask |= iter->second.accessMask;
                }
            }
            if (auto iter = branch.bufferStates.find(resource); iter != branch.bufferStates.end()) {
                auto &state = context.trackedBufferStates[resource];
                if (access.writes) {
                    state = iter->second;
                } else {
                    state.stageMask |= iter->second.stageMask;
                    state.accessMask |= iter->second.accessMask;
                }
            }
        }
        if (branch.writesSharc) { context.sharcQueryReady = true; }
        branch.joined = true;
    };

    for (size_t firstNode = 0; firstNode < nodes.size();) {
        const std::string group = asyncGroup(nodes[firstNode]);
        size_t endNode = firstNode + 1;
        if (!group.empty()) {
            while (endNode < nodes.size() && asyncGroup(nodes[endNode]) == group) { ++endNode; }
        }
        AccessMap accesses;
        ResolvedPassResources firstUses;
        for (size_t i = firstNode; i < endNode; ++i) { collectAccesses(nodes[i], accesses, firstUses); }
        for (auto &branch : pendingBranches) {
            if (branch.joined) { continue; }
            for (const auto &[resource, access] : accesses) {
                auto iter = branch.accesses.find(resource);
                if (iter != branch.accesses.end() &&
                    (access.writes || iter->second.writes || access.layout != iter->second.layout)) {
                    joinBranch(branch);
                    break;
                }
            }
        }
        if (group.empty()) {
            recordNode(nodes[firstNode]);
            firstNode = endNode;
            continue;
        }

        if (!context.ensureRayTracingWorldPrepareResources()) {
            firstNode = endNode;
            continue;
        }
        submitGraphBarriers(firstUses,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            false, false);
        const size_t branchSlot = pendingBranches.size();
        if (context.asyncBranchResources.size() <= branchSlot) {
            if (context.asyncCommandPool == nullptr) {
                context.asyncCommandPool = vk::CommandPool::create(framework->physicalDevice(), framework->device(),
                                                                   framework->device()->secondaryQueueFamilyIndex());
            }
            context.asyncBranchResources.push_back({
                .commandBuffer = vk::CommandBuffer::create(framework->device(), context.asyncCommandPool),
                .readySemaphore = vk::Semaphore::create(framework->device()),
                .completeSemaphore = vk::Semaphore::create(framework->device()),
            });
        }
        const auto &resources = context.asyncBranchResources[branchSlot];
        const size_t submissionIndex = frameworkContext->scheduleWorldAsyncBranch(
            resources.commandBuffer, resources.readySemaphore, resources.completeSemaphore);
        const auto mainCommands = worldCommandBuffer;
        auto mainImageStates = context.trackedImageStates;
        auto mainBufferStates = context.trackedBufferStates;
        const bool mainSharcReady = context.sharcQueryReady;
        for (auto &[resource, state] : context.trackedImageStates) {
            state.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            state.accessMask = 0;
        }
        for (auto &[resource, state] : context.trackedBufferStates) {
            state.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            state.accessMask = 0;
        }
        worldCommandBuffer = resources.commandBuffer;
        frameworkContext->worldCommandBuffer = worldCommandBuffer;
        worldCommandBuffer->begin();
        try {
            for (size_t i = firstNode; i < endNode; ++i) { recordNode(nodes[i]); }
            worldCommandBuffer->end();
        } catch (...) {
            frameworkContext->worldCommandBuffer = mainCommands;
            context.trackedImageStates = std::move(mainImageStates);
            context.trackedBufferStates = std::move(mainBufferStates);
            context.sharcQueryReady = mainSharcReady;
            throw;
        }
        const auto sharcAccess = accesses.find(sharcResource);
        const bool writesSharc = sharcAccess != accesses.end() && sharcAccess->second.writes;
        pendingBranches.push_back({
            .submissionIndex = submissionIndex,
            .accesses = std::move(accesses),
            .imageStates = std::move(context.trackedImageStates),
            .bufferStates = std::move(context.trackedBufferStates),
            .writesSharc = writesSharc,
        });
        worldCommandBuffer = mainCommands;
        frameworkContext->worldCommandBuffer = mainCommands;
        context.trackedImageStates = std::move(mainImageStates);
        context.trackedBufferStates = std::move(mainBufferStates);
        context.sharcQueryReady = writesSharc ? false : mainSharcReady;
        firstNode = endNode;
    }
    for (auto &branch : pendingBranches) { joinBranch(branch); }
}

MainRenderModule::MainRenderModule() {}

void MainRenderModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrNoisyOutputImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    firstHitDepthImages_.resize(size);
    indirectSpecularHitDistanceImages_.resize(size);
    indirectDiffuseHitDistanceImages_.resize(size);
    indirectDiffuseRadianceImages_.resize(size);
    indirectDiffuseDirectionImages_.resize(size);
    indirectSpecularRadianceImages_.resize(size);
    indirectSpecularDirectionImages_.resize(size);
    firstHitClearImages_.resize(size);
    firstHitRefractionImages_.resize(size);
    firstHitBaseEmissionImages_.resize(size);
    fogImages_.resize(size);
    directDiffuseRadianceImages_.resize(size);
    directSpecularRadianceImages_.resize(size);
    directHitDistanceImages_.resize(size);
    directLightGuideNormalRoughnessImages_.resize(size);
    directLightGuideMotionDepthImages_.resize(size);
    segmentationMaskImages_.resize(size);
    rasterDepthImages_.resize(size);

    worldPrepare_ = WorldPrepare::create(framework, shared_from_this());
}

bool MainRenderModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    return true;
}

bool MainRenderModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.size() != outputImageNum || formats.size() != outputImageNum) { return false; }
    uint32_t width = 0;
    uint32_t height = 0;
    bool set = false;
    for (auto &image : images) {
        if (image != nullptr) {
            if (!set) {
                width = image->width();
                height = image->height();
                set = true;
            } else if (image->width() != width || image->height() != height) {
                return false;
            }
        }
    }

    if (!set) { return false; }

    auto framework = framework_.lock();
    for (int i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width, height, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    hdrNoisyOutputImages_[frameIndex] = images[RADIANCE_OUTPUT];
    diffuseAlbedoImages_[frameIndex] = images[DIFFUSE_ALBEDO_METALLIC_OUTPUT];
    specularAlbedoImages_[frameIndex] = images[SPECULAR_ALBEDO_OUTPUT];
    normalRoughnessImages_[frameIndex] = images[NORMAL_ROUGHNESS_OUTPUT];
    motionVectorImages_[frameIndex] = images[MOTION_VECTOR_OUTPUT];
    linearDepthImages_[frameIndex] = images[LINEAR_DEPTH_OUTPUT];
    firstHitDepthImages_[frameIndex] = images[FIRST_HIT_DEPTH_OUTPUT];
    indirectSpecularHitDistanceImages_[frameIndex] = images[INDIRECT_SPECULAR_HIT_DISTANCE_OUTPUT];
    indirectDiffuseHitDistanceImages_[frameIndex] = images[INDIRECT_DIFFUSE_HIT_DISTANCE_OUTPUT];
    indirectDiffuseRadianceImages_[frameIndex] = images[INDIRECT_DIFFUSE_RADIANCE_OUTPUT];
    indirectDiffuseDirectionImages_[frameIndex] = images[INDIRECT_DIFFUSE_DIRECTION_OUTPUT];
    indirectSpecularRadianceImages_[frameIndex] = images[INDIRECT_SPECULAR_RADIANCE_OUTPUT];
    indirectSpecularDirectionImages_[frameIndex] = images[INDIRECT_SPECULAR_DIRECTION_OUTPUT];
    firstHitClearImages_[frameIndex] = images[FIRST_HIT_CLEAR_OUTPUT];
    firstHitRefractionImages_[frameIndex] = images[FIRST_HIT_REFRACTION_OUTPUT];
    firstHitBaseEmissionImages_[frameIndex] = images[FIRST_HIT_BASE_EMISSION_OUTPUT];
    fogImages_[frameIndex] = images[FOG_OUTPUT];
    directDiffuseRadianceImages_[frameIndex] = images[DIRECT_DIFFUSE_RADIANCE_OUTPUT];
    directSpecularRadianceImages_[frameIndex] = images[DIRECT_SPECULAR_RADIANCE_OUTPUT];
    directHitDistanceImages_[frameIndex] = images[DIRECT_HIT_DISTANCE_OUTPUT];
    directLightGuideNormalRoughnessImages_[frameIndex] = images[DIRECT_LIGHT_GUIDE_NORMAL_ROUGHNESS_OUTPUT];
    directLightGuideMotionDepthImages_[frameIndex] = images[DIRECT_LIGHT_GUIDE_MOTION_DEPTH_OUTPUT];

    return true;
}

const std::unordered_map<std::string, std::string> &MainRenderModule::captureExecutionVariables() const {
    return globalVariables_;
}

const std::unordered_map<std::string, std::string> &MainRenderModule::captureStaticAttributes() const {
    return staticAttributes_;
}

bool MainRenderModule::hasSegmentationMask() const {
    return shaderPack_ != nullptr && shaderPack_->hasSegmentationMask();
}

std::shared_ptr<vk::DeviceLocalImage> MainRenderModule::segmentationMaskImage(uint32_t frameIndex) const {
    if (frameIndex >= segmentationMaskImages_.size()) { return nullptr; }
    return segmentationMaskImages_[frameIndex];
}

std::string MainRenderModule::getAttributes(const std::vector<std::string> &attributes, const std::string &language) {
    std::string shaderPackPath;
    for (size_t i = 0; i + 1 < attributes.size(); i += 2) {
        if (attributes[i] == "render_pipeline.module.main_render.attribute.shader_pack_path") {
            shaderPackPath = attributes[i + 1];
            break;
        }
    }
    ShaderPackLoader::LoadResult loadResult =
        ShaderPackLoader::load(shaderPackPath.empty() ? std::filesystem::path{} : std::filesystem::path(shaderPackPath),
                               builtInShaderPackPath(), language);

    json result = json::object();
    result["success"] = loadResult.success;
    result["attributes"] = json::array();
    result["translations"] = json::object();
    if (!loadResult.success) {
        result["error"] = loadResult.error;
        return result.dump();
    }

    for (const auto &attribute : loadResult.shaderPack.attributes) {
        json attributeJson = {
            {"name", attribute.name},
            {"type", attribute.type},
            {"default_value", attribute.defaultValue},
        };
        if (!attribute.description.empty()) { attributeJson["description"] = attribute.description; }
        if (!attribute.group.empty()) { attributeJson["group"] = attribute.group; }
        if (!attribute.performanceImpact.empty()) { attributeJson["performance_impact"] = attribute.performanceImpact; }
        if (!attribute.visibleWhen.empty()) {
            attributeJson["visible_when"] = json::array();
            for (const auto &condition : attribute.visibleWhen) {
                json conditionJson = {
                    {"attribute", condition.attribute},
                };
                if (!condition.module.empty()) { conditionJson["module"] = condition.module; }
                if (!condition.values.empty()) { conditionJson["values"] = condition.values; }
                attributeJson["visible_when"].push_back(std::move(conditionJson));
            }
        }
        if (!attribute.linkedValues.empty()) {
            attributeJson["linked_values"] = json::object();
            for (const auto &[value, targetValues] : attribute.linkedValues) {
                attributeJson["linked_values"][value] = json::object();
                for (const auto &[name, linkedValue] : targetValues) {
                    attributeJson["linked_values"][value][name] = linkedValue;
                }
            }
        }
        if (attribute.preset.has_value()) {
            attributeJson["preset"] = {
                {"custom_value", attribute.preset->customValue},
                {"profiles", json::object()},
            };
            for (const auto &[profile, targetValues] : attribute.preset->profiles) {
                attributeJson["preset"]["profiles"][profile] = json::object();
                for (const auto &[name, value] : targetValues) {
                    attributeJson["preset"]["profiles"][profile][name] = value;
                }
            }
        }
        result["attributes"].push_back(std::move(attributeJson));
    }
    for (const auto &[key, value] : loadResult.shaderPack.translations) { result["translations"][key] = value; }
    return result.dump();
}

void MainRenderModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    staticAttributes_.clear();
    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];
        staticAttributes_[key] = value;
    }

    isJitterEnabled_ = true;
    isSharcEnabled_ = true;
    sharcDebugMode_ = 0;
    shaderPackPath_.clear();
    sharcSceneScale_ = defaultSharcSceneScale;
    sharcAccumulationFrameNum_ = defaultSharcAccumulationFrameNum;
    sharcStaleFrameNumMax_ = defaultSharcStaleFrameNumMax;
    sharcUpdateDownsampleFactor_ = defaultSharcUpdateDownsampleFactor;
    sharcCapacityLog2_ = defaultSharcCapacityLog2;
    sharcCapacity_ = defaultSharcCapacity;
    sharcResolveFrameStride_ = defaultSharcResolveFrameStride;
    currentSharcResolveEntryCount_ = sharcCapacity_;

    for (const auto &[key, value] : staticAttributes_) {
        if (key == "render_pipeline.module.main_render.attribute.use_jitter") {
            isJitterEnabled_ = value == "render_pipeline.true";
        } else if (key == "render_pipeline.module.main_render.attribute.use_sharc") {
            isSharcEnabled_ = value == "render_pipeline.true";
        } else if (key == "render_pipeline.module.main_render.attribute.sharc_resolve_frame_stride") {
            try {
                sharcResolveFrameStride_ = std::clamp(static_cast<uint32_t>(std::stoul(value)), 1u, 16u);
            } catch (...) { sharcResolveFrameStride_ = defaultSharcResolveFrameStride; }
        } else if (key == "render_pipeline.module.main_render.attribute.shader_pack_path") {
            shaderPackPath_ = value;
        }
    }
    Renderer::instance().buffers()->setUseJitter(isJitterEnabled_);
}

void MainRenderModule::loadShaderPack() {
    auto worldPipeline = worldPipeline_.lock();
    shaderPack_ = worldPipeline != nullptr ? worldPipeline->shaderPack() : nullptr;
    if (shaderPack_ == nullptr) { throw std::runtime_error("main_render shared shader pack runtime is unavailable"); }
    hasSharcRuntime_ = isSharcEnabled_ && shaderPack_->hasSharcRuntime();

    directionalShadowRasterEnabled_ = false;
}

void MainRenderModule::initExecutionVariables() {
    executionVariableConfigs_.clear();
    globalVariables_.clear();

    if (shaderPack_ == nullptr) { return; }
    shaderPack_->copyStageExecutionState(ShaderPackLoader::Stage::RayTracing, executionVariableConfigs_,
                                         globalVariables_);
}

std::vector<ExpressionEvaluator::Variable> MainRenderModule::executionExpressionVariables() const {
    std::vector<ExpressionEvaluator::Variable> variables;
    variables.reserve(16);

    const uint32_t renderWidth =
        hdrNoisyOutputImages_.empty() || hdrNoisyOutputImages_[0] == nullptr ? 0u : hdrNoisyOutputImages_[0]->width();
    const uint32_t renderHeight =
        hdrNoisyOutputImages_.empty() || hdrNoisyOutputImages_[0] == nullptr ? 0u : hdrNoisyOutputImages_[0]->height();
    variables.push_back({.name = "RENDER_WIDTH", .value = static_cast<double>(renderWidth)});
    variables.push_back({.name = "RENDER_HEIGHT", .value = static_cast<double>(renderHeight)});

    auto world = Renderer::instance().world();
    auto chunks = world == nullptr ? nullptr : world->chunks();
    if (chunks != nullptr) {
        const glm::ivec4 chunkGridInfo = chunks->chunkGridInfo();
        variables.push_back({.name = "CHUNK_SIZE_X", .value = static_cast<double>(chunkGridInfo.x)});
        variables.push_back({.name = "CHUNK_SIZE_Y", .value = static_cast<double>(chunkGridInfo.y)});
        variables.push_back({.name = "CHUNK_SIZE_Z", .value = static_cast<double>(chunkGridInfo.z)});
        variables.push_back({.name = "CHUNK_BOTTOM_SECTION_COORD", .value = static_cast<double>(chunkGridInfo.w)});
        variables.push_back({.name = "CHUNK_NUM", .value = static_cast<double>(chunks->chunks().size())});
        variables.push_back(
            {.name = "CHUNK_RENDERABLE_NUM", .value = static_cast<double>(chunks->renderableChunks().size())});
        variables.push_back({.name = "CHUNK_GEOMETRY_SLOT_NUM",
                             .value = static_cast<double>(chunks->chunkGeometryMetadataCache().geometrySlotCount)});
        variables.push_back({.name = "CHUNK_GEOMETRY_NUM",
                             .value = static_cast<double>(chunks->chunkGeometryMetadataCache().actualGeometryCount)});
    }

    uint32_t entityNum = 0;
    uint32_t entityGeometryNum = 0;
    auto entities = world == nullptr ? nullptr : world->entities();
    auto entityBatch = entities == nullptr ? nullptr : entities->peekEntityBatch();
    if (entityBatch != nullptr) {
        entityNum = static_cast<uint32_t>(entityBatch->entities.size());
        for (const auto &entity : entityBatch->entities) {
            if (entity == nullptr) { continue; }
            entityGeometryNum += entity->geometryCount;
        }
    }
    variables.push_back({.name = "ENTITY_NUM", .value = static_cast<double>(entityNum)});
    variables.push_back({.name = "ENTITY_GEOMETRY_NUM", .value = static_cast<double>(entityGeometryNum)});
    const double chunkGeometrySlots =
        chunks == nullptr ? 0.0 : static_cast<double>(chunks->chunkGeometryMetadataCache().geometrySlotCount);
    variables.push_back(
        {.name = "RASTER_DRAW_NUM", .value = chunkGeometrySlots + static_cast<double>(entityGeometryNum)});
    return variables;
}

double
MainRenderModule::evaluateNumericExpression(const std::string &expression,
                                            const std::vector<ExpressionEvaluator::Variable> &expressionVariables) {
    return shaderPack_->evaluateNumericExpression(ShaderPackLoader::Stage::RayTracing, expression, expressionVariables,
                                                  true);
}

std::optional<std::reference_wrapper<ShaderPackLoader::VariableConfig>>
MainRenderModule::findExecutionVariableConfig(std::string_view name) {
    auto executionIter = executionVariableConfigs_.find(std::string(name));
    if (executionIter != executionVariableConfigs_.end()) { return std::ref(executionIter->second); }
    return std::nullopt;
}

std::unordered_map<std::string, std::string> MainRenderModule::shaderDefinitionsWithRuntimeExpressions(
    const std::unordered_map<std::string, std::string> &definitions) const {
    std::unordered_map<std::string, std::string> shaderDefinitions = definitions;
    bool hasRenderWidth = false;
    bool hasRenderHeight = false;

    auto formatDefinitionValue = [](double value) {
        if (!std::isfinite(value)) { throw std::runtime_error("main_render runtime shader definition is not finite"); }
        std::ostringstream stream;
        stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return stream.str();
    };

    for (const auto &variable : executionExpressionVariables()) {
        if (variable.name.empty()) { continue; }
        if (variable.name == "RENDER_WIDTH") {
            if (variable.value <= 0.0) {
                throw std::runtime_error("main_render RENDER_WIDTH shader definition is not positive");
            }
            hasRenderWidth = true;
        } else if (variable.name == "RENDER_HEIGHT") {
            if (variable.value <= 0.0) {
                throw std::runtime_error("main_render RENDER_HEIGHT shader definition is not positive");
            }
            hasRenderHeight = true;
        }
        shaderDefinitions[variable.name] = formatDefinitionValue(variable.value);
    }

    if (!hasRenderWidth || !hasRenderHeight) {
        throw std::runtime_error("main_render shader definitions missing RENDER_WIDTH or RENDER_HEIGHT");
    }
    return shaderDefinitions;
}

void MainRenderModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    VkShaderStageFlags runtimeTextureStageFlags =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    mainRenderDescriptorTables_.resize(size);

    for (uint32_t frameIndex = 0; frameIndex < size; frameIndex++) {
        vk::DescriptorTableBuilder builder;

        builder.definePushConstant({
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = static_cast<uint32_t>(sizeof(DirectionalShadowPushConstants)),
        });

        auto &set0 = builder.beginDescriptorLayoutSet();
        auto &set0Bindings = set0.beginDescriptorLayoutSetBinding();
        set0Bindings.defineDescriptorLayoutSetBinding({
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = framework->device()->textureDescriptorCapacity(),
            .stageFlags = runtimeTextureStageFlags,
        });
        set0Bindings.endDescriptorLayoutSetBinding();
        set0.endDescriptorLayoutSet();

        auto &set1 = builder.beginDescriptorLayoutSet();
        auto &set1Bindings = set1.beginDescriptorLayoutSetBinding();
        set1Bindings
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                              VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 5,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                              VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 6,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 7,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 8,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 9,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 10,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = directionalShadowDepthDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = directionalShadowConfigDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = directionalShadowCommandsDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = directionalShadowMetadataDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            });
        set1Bindings.endDescriptorLayoutSetBinding();
        set1.endDescriptorLayoutSet();

        auto &set2 = builder.beginDescriptorLayoutSet();
        auto &set2Bindings = set2.beginDescriptorLayoutSetBinding();
        set2Bindings
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = runtimeTextureStageFlags,
            });
        set2Bindings.endDescriptorLayoutSetBinding();
        set2.endDescriptorLayoutSet();

        auto &set3 = builder.beginDescriptorLayoutSet();
        auto &set3Bindings = set3.beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < segmentationMaskDescriptorBinding; binding++) {
            set3Bindings.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                              VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                              VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            });
        }
        set3Bindings.defineDescriptorLayoutSetBinding({
            .binding = segmentationMaskDescriptorBinding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                          VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        });
        set3Bindings.endDescriptorLayoutSetBinding();
        set3.endDescriptorLayoutSet();

        auto &set4 = builder.beginDescriptorLayoutSet();
        auto &set4Bindings = set4.beginDescriptorLayoutSetBinding();
        set4Bindings
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            });
        set4Bindings.endDescriptorLayoutSetBinding();
        set4.endDescriptorLayoutSet();

        shaderPack_->defineRuntimeResourceDescriptorSet(builder, runtimeTextureStageFlags, runtimeTextureStageFlags,
                                                        runtimeTextureStageFlags);
        shaderPack_->defineExecutionDescriptorSet(builder, ShaderPackLoader::Stage::RayTracing,
                                                  runtimeTextureStageFlags | VK_SHADER_STAGE_COMPUTE_BIT);

        auto &set6 = builder.beginDescriptorLayoutSet();
        auto &set6Bindings = set6.beginPushDescriptorLayoutSetBinding();
        set6Bindings
            .defineDescriptorLayoutSetBinding({
                .binding = renderIndirectMetadataDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = renderIndirectCommandsDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
        set6Bindings.endDescriptorLayoutSetBinding();
        set6.endDescriptorLayoutSet();

        mainRenderDescriptorTables_[frameIndex] = builder.build(framework->device());

        mainRenderDescriptorTables_[frameIndex]->bindImage(hdrNoisyOutputImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL,
                                                           3, 0);
        mainRenderDescriptorTables_[frameIndex]->bindImage(diffuseAlbedoImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                           1);
        mainRenderDescriptorTables_[frameIndex]->bindImage(specularAlbedoImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL,
                                                           3, 2);
        mainRenderDescriptorTables_[frameIndex]->bindImage(normalRoughnessImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL,
                                                           3, 3);
        mainRenderDescriptorTables_[frameIndex]->bindImage(motionVectorImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                           4);
        mainRenderDescriptorTables_[frameIndex]->bindImage(linearDepthImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                           5);
        mainRenderDescriptorTables_[frameIndex]->bindImage(firstHitDepthImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                           6);
        mainRenderDescriptorTables_[frameIndex]->bindImage(indirectSpecularHitDistanceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 7);
        mainRenderDescriptorTables_[frameIndex]->bindImage(indirectDiffuseHitDistanceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 8);
        mainRenderDescriptorTables_[frameIndex]->bindImage(indirectDiffuseRadianceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 9);
        mainRenderDescriptorTables_[frameIndex]->bindImage(indirectDiffuseDirectionImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 10);
        mainRenderDescriptorTables_[frameIndex]->bindImage(indirectSpecularRadianceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 11);
        mainRenderDescriptorTables_[frameIndex]->bindImage(indirectSpecularDirectionImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 12);
        mainRenderDescriptorTables_[frameIndex]->bindImage(firstHitClearImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                           13);
        mainRenderDescriptorTables_[frameIndex]->bindImage(firstHitRefractionImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 14);
        mainRenderDescriptorTables_[frameIndex]->bindImage(firstHitBaseEmissionImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 15);
        mainRenderDescriptorTables_[frameIndex]->bindImage(fogImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 16);
        mainRenderDescriptorTables_[frameIndex]->bindImage(directDiffuseRadianceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 17);
        mainRenderDescriptorTables_[frameIndex]->bindImage(directSpecularRadianceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 18);
        mainRenderDescriptorTables_[frameIndex]->bindImage(directHitDistanceImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 19);
        mainRenderDescriptorTables_[frameIndex]->bindImage(directLightGuideNormalRoughnessImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 21);
        mainRenderDescriptorTables_[frameIndex]->bindImage(directLightGuideMotionDepthImages_[frameIndex],
                                                           VK_IMAGE_LAYOUT_GENERAL, 3, 22);
    }
}

void MainRenderModule::initRuntimeTextures() {
    auto framework = framework_.lock();
    auto device = framework->device();

    fullScreenVertexShader_ =
        vk::Shader::create(device, (Renderer::folderPath / "shaders/full_screen_vert.spv").string());
    if (!hdrNoisyOutputImages_.empty() && hdrNoisyOutputImages_[0] != nullptr) {
        for (uint32_t frameIndex = 0; frameIndex < hdrNoisyOutputImages_.size(); frameIndex++) {
            if (hdrNoisyOutputImages_[frameIndex] == nullptr) { continue; }
            rasterDepthImages_[frameIndex] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, hdrNoisyOutputImages_[frameIndex]->width(),
                hdrNoisyOutputImages_[frameIndex]->height(), 1, VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        }
    }
    if (hasSegmentationMask() && !hdrNoisyOutputImages_.empty() && hdrNoisyOutputImages_[0] != nullptr) {
        for (uint32_t frameIndex = 0; frameIndex < hdrNoisyOutputImages_.size(); frameIndex++) {
            if (hdrNoisyOutputImages_[frameIndex] == nullptr) { continue; }
            segmentationMaskImages_[frameIndex] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, hdrNoisyOutputImages_[frameIndex]->width(),
                hdrNoisyOutputImages_[frameIndex]->height(), 1, segmentationMaskFormat, segmentationMaskUsage);
            mainRenderDescriptorTables_[frameIndex]->bindImage(segmentationMaskImages_[frameIndex],
                                                               VK_IMAGE_LAYOUT_GENERAL, segmentationMaskDescriptorSet,
                                                               segmentationMaskDescriptorBinding);
        }
    }
    if (shaderPack_ != nullptr && !hdrNoisyOutputImages_.empty() && hdrNoisyOutputImages_[0] != nullptr) {
        shaderPack_->setRuntimeResourceExpressionVariables(executionExpressionVariables());
        shaderPack_->ensureRuntimeResources(hdrNoisyOutputImages_[0]->width(), hdrNoisyOutputImages_[0]->height());
    }
    initDirectionalShadowResources();
}

void MainRenderModule::initDirectionalShadowResources() {
    if (!directionalShadowRasterEnabled_) {
        directionalShadowRenderPass_ = nullptr;
        directionalShadowDepthSampler_ = nullptr;
        directionalShadowDepthImages_.clear();
        directionalShadowFramebuffers_.clear();
        directionalShadowDataBuffers_.clear();
        directionalShadowMetadataBuffers_.clear();
        directionalShadowCommandBuffers_.clear();
        return;
    }

    auto framework = framework_.lock();
    auto device = framework->device();
    const uint32_t frameCount = framework->swapchain()->imageCount();

    directionalShadowDepthSampler_ =
        vk::Sampler::create(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, true, VK_COMPARE_OP_LESS_OR_EQUAL);

    directionalShadowRenderPass_ = vk::RenderPassBuilder{}
                                       .beginAttachmentDescription()
                                       .defineAttachmentDescription({
                                           .format = directionalShadowDepthFormat,
                                           .samples = VK_SAMPLE_COUNT_1_BIT,
                                           .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                           .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                           .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                           .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                           .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       })
                                       .endAttachmentDescription()
                                       .beginAttachmentReference()
                                       .defineAttachmentReference({
                                           .attachment = 0,
                                           .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       })
                                       .endAttachmentReference()
                                       .beginSubpassDescription()
                                       .defineSubpassDescription({
                                           .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                           .depthStencilAttachmentIndex = 0,
                                       })
                                       .endSubpassDescription()
                                       .build(device);

    directionalShadowDepthImages_.resize(frameCount);
    directionalShadowFramebuffers_.resize(frameCount);
    directionalShadowDataBuffers_.resize(frameCount);
    directionalShadowMetadataBuffers_.resize(frameCount);
    directionalShadowCommandBuffers_.resize(frameCount);

    for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        directionalShadowDataBuffers_[frameIndex] = vk::HostVisibleBuffer::create(
            framework->vma(), device, sizeof(DirectionalShadowData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        mainRenderDescriptorTables_[frameIndex]->bindBuffer(directionalShadowDataBuffers_[frameIndex], 1,
                                                            directionalShadowConfigDescriptorBinding);

        DirectionalShadowData initialData{};
        initialData.lightViewProj = glm::mat4(1.0f);
        initialData.lightDirectionAndStrength = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        initialData.shadowParams =
            glm::vec4(1.0f / static_cast<float>(directionalShadowResolution), directionalShadowDepthBias, 1.0f, 0.0f);
        initialData.shadowDistanceParams =
            glm::vec4(directionalShadowMaxDistanceBlocks, directionalShadowMaxDistanceBlocks * 0.08f, 0.0f, 0.0f);
        directionalShadowDataBuffers_[frameIndex]->uploadToBuffer(&initialData);

        directionalShadowDepthImages_[frameIndex] = vk::DeviceLocalImage::create(
            device, framework->vma(), false, directionalShadowResolution, directionalShadowResolution, 1,
            directionalShadowDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        directionalShadowFramebuffers_[frameIndex] = vk::FramebufferBuilder{}
                                                         .beginAttachment()
                                                         .defineAttachment(directionalShadowDepthImages_[frameIndex])
                                                         .endAttachment()
                                                         .build(device, directionalShadowRenderPass_);

        mainRenderDescriptorTables_[frameIndex]->bindSamplerImage(
            directionalShadowDepthSampler_, directionalShadowDepthImages_[frameIndex],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, directionalShadowDepthDescriptorBinding, 0);
    }
}

void MainRenderModule::initRuntimeBuffers() {
    if (shaderPack_ != nullptr) {
        shaderPack_->setRuntimeResourceExpressionVariables(executionExpressionVariables());
        shaderPack_->refreshRuntimeBuffers();
    }
}

void MainRenderModule::refreshRuntimeBuffers(
    const std::vector<ExpressionEvaluator::Variable> &runtimeExpressionVariables) {
    if (shaderPack_ == nullptr) { return; }
    shaderPack_->setRuntimeResourceExpressionVariables(runtimeExpressionVariables);
    if (!shaderPack_->refreshRuntimeBuffers()) { return; }
    for (uint32_t i = 0; i < mainRenderDescriptorTables_.size(); i++) {
        shaderPack_->bindRuntimeResources(mainRenderDescriptorTables_[i], 5, i);
    }
}

void MainRenderModule::loadRuntimeResources() {
    auto framework = framework_.lock();
    for (uint32_t frameIndex = 0; frameIndex < framework->swapchain()->imageCount(); frameIndex++) {
        shaderPack_->bindRuntimeResources(mainRenderDescriptorTables_[frameIndex], 5, frameIndex);
    }
}

std::vector<std::shared_ptr<vk::Framebuffer>>
MainRenderModule::buildFramebuffers(std::shared_ptr<vk::DeviceLocalImage> image,
                                    std::shared_ptr<vk::RenderPass> renderPass) {
    auto framework = framework_.lock();
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers;
    if (image->layer() == 1) {
        framebuffers.push_back(vk::FramebufferBuilder{}.beginAttachment().defineAttachment(image).endAttachment().build(
            framework->device(), renderPass));
        return framebuffers;
    }

    framebuffers.resize(image->layer());
    for (uint32_t layerIndex = 0; layerIndex < image->layer(); layerIndex++) {
        framebuffers[layerIndex] = vk::FramebufferBuilder{}
                                       .beginAttachment()
                                       .defineAttachment(image, layerIndex + 1)
                                       .endAttachment()
                                       .build(framework->device(), renderPass);
    }
    return framebuffers;
}

std::shared_ptr<vk::DeviceLocalImage> MainRenderModule::findRuntimeVKTexture(ShaderPack::RuntimeTexture &runtimeTexture,
                                                                             uint32_t frameIndex) {
    return shaderPack_->findRuntimeVKTexture(runtimeTexture, frameIndex);
}

std::shared_ptr<vk::DeviceLocalBuffer> MainRenderModule::findRuntimeVKBuffer(ShaderPack::RuntimeBuffer &runtimeBuffer,
                                                                             uint32_t frameIndex) {
    return shaderPack_->findRuntimeVKBuffer(runtimeBuffer, frameIndex);
}

const std::vector<std::shared_ptr<vk::DeviceLocalImage>> *MainRenderModule::findTargetImages(std::string_view target) {
    if (target == TARGET_RADIANCE) return &hdrNoisyOutputImages_;
    if (target == TARGET_DIFFUSE_ALBEDO_METALLIC) return &diffuseAlbedoImages_;
    if (target == TARGET_SPECULAR_ALBEDO) return &specularAlbedoImages_;
    if (target == TARGET_NORMAL_ROUGHNESS) return &normalRoughnessImages_;
    if (target == TARGET_MOTION_VECTOR) return &motionVectorImages_;
    if (target == TARGET_LINEAR_DEPTH) return &linearDepthImages_;
    if (target == TARGET_FIRST_HIT_DEPTH) return &firstHitDepthImages_;
    if (target == TARGET_INDIRECT_SPECULAR_HIT_DISTANCE) return &indirectSpecularHitDistanceImages_;
    if (target == TARGET_INDIRECT_DIFFUSE_HIT_DISTANCE) return &indirectDiffuseHitDistanceImages_;
    if (target == TARGET_INDIRECT_DIFFUSE_RADIANCE) return &indirectDiffuseRadianceImages_;
    if (target == TARGET_INDIRECT_DIFFUSE_DIRECTION) return &indirectDiffuseDirectionImages_;
    if (target == TARGET_INDIRECT_SPECULAR_RADIANCE) return &indirectSpecularRadianceImages_;
    if (target == TARGET_INDIRECT_SPECULAR_DIRECTION) return &indirectSpecularDirectionImages_;
    if (target == TARGET_FIRST_HIT_CLEAR) return &firstHitClearImages_;
    if (target == TARGET_FIRST_HIT_REFRACTION) return &firstHitRefractionImages_;
    if (target == TARGET_FIRST_HIT_BASE_EMISSION) return &firstHitBaseEmissionImages_;
    if (target == TARGET_FOG_IMAGE) return &fogImages_;
    if (target == TARGET_DIRECT_DIFFUSE_RADIANCE) return &directDiffuseRadianceImages_;
    if (target == TARGET_DIRECT_SPECULAR_RADIANCE) return &directSpecularRadianceImages_;
    if (target == TARGET_DIRECT_HIT_DISTANCE) return &directHitDistanceImages_;
    if (target == TARGET_DIRECT_LIGHT_GUIDE_NORMAL_ROUGHNESS) return &directLightGuideNormalRoughnessImages_;
    if (target == TARGET_DIRECT_LIGHT_GUIDE_MOTION_DEPTH) return &directLightGuideMotionDepthImages_;
    return nullptr;
}

std::shared_ptr<vk::DeviceLocalImage> MainRenderModule::findTargetImage(const std::string &target,
                                                                        uint32_t frameIndex) {
    if (const auto *targetImages = findTargetImages(target); targetImages != nullptr) {
        return (*targetImages)[frameIndex];
    }

    if (auto runtimeTexture = findRuntimeTexture(target); runtimeTexture.has_value()) {
        if (runtimeTexture->get().config.imported) { return nullptr; }
        return findRuntimeVKTexture(runtimeTexture->get(), frameIndex);
    }
    return nullptr;
}

std::shared_ptr<vk::DeviceLocalImage> MainRenderModule::resolveImageResource(const ResolvedImageResource &resource,
                                                                             uint32_t frameIndex) {
    if (resource.isImportedRuntimeTexture) { return resource.runtimeTexture->importedImage; }
    if (resource.targetImages != nullptr) { return (*resource.targetImages)[frameIndex]; }
    if (resource.runtimeTexture != nullptr) { return findRuntimeVKTexture(*resource.runtimeTexture, frameIndex); }
    return nullptr;
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeTexture>>
MainRenderModule::findRuntimeTexture(std::string_view name) {
    if (shaderPack_ == nullptr) { return std::nullopt; }
    return shaderPack_->findRuntimeTexture(name);
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeBuffer>>
MainRenderModule::findRuntimeBuffer(std::string_view name) {
    if (shaderPack_ == nullptr) { return std::nullopt; }
    return shaderPack_->findRuntimeBuffer(name);
}

MainRenderModule::ResolvedImageResource MainRenderModule::resolveImageResource(std::string_view name) {
    auto runtimeTexture = findRuntimeTexture(name);
    if (runtimeTexture.has_value() && runtimeTexture->get().config.imported) {
        return {
            .targetImages = nullptr,
            .runtimeTexture = &runtimeTexture->get(),
            .isImportedRuntimeTexture = true,
        };
    }

    if (const auto *targetImages = findTargetImages(name); targetImages != nullptr) {
        return {
            .targetImages = targetImages,
            .runtimeTexture = nullptr,
            .isImportedRuntimeTexture = false,
        };
    }

    if (runtimeTexture.has_value()) {
        return {
            .targetImages = nullptr,
            .runtimeTexture = &runtimeTexture->get(),
            .isImportedRuntimeTexture = false,
        };
    }

    throw std::runtime_error("unknown pass image resource: " + std::string(name));
}

ShaderPack::RuntimeBuffer *MainRenderModule::resolveRuntimeBuffer(std::string_view name) {
    auto runtimeBuffer = findRuntimeBuffer(name);
    if (!runtimeBuffer.has_value()) { throw std::runtime_error("unknown pass buffer resource: " + std::string(name)); }
    return &runtimeBuffer->get();
}

MainRenderModule::ResolvedPassResources
MainRenderModule::resolvePassResources(const ShaderPackLoader::ResourceList &inputs,
                                       const ShaderPackLoader::ResourceList &outputs,
                                       const std::optional<std::string> &indirectArgsBuffer,
                                       std::optional<std::string_view> fullScreenTarget,
                                       bool outputsAreColorAttachments) {
    ResolvedPassResources resources;
    resources.imageAccesses.reserve(inputs.imageAccesses.size() + outputs.imageAccesses.size());
    resources.bufferAccesses.reserve(inputs.bufferAccesses.size() + outputs.bufferAccesses.size());

    struct AccessTraits {
        VkAccessFlags2 accessMask = 0;
        bool reads = false;
        bool writes = false;
        bool needsPreviousContent = false;
    };

    auto shaderAccessTraits = [](ShaderPackLoader::ResourceAccessKind kind, bool bufferResource) {
        switch (kind) {
            case ShaderPackLoader::ResourceAccessKind::Read:
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_SHADER_READ_BIT,
                    .reads = true,
                    .writes = false,
                    .needsPreviousContent = true,
                };
            case ShaderPackLoader::ResourceAccessKind::Write:
            case ShaderPackLoader::ResourceAccessKind::Overwrite:
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                    .reads = false,
                    .writes = true,
                    .needsPreviousContent = false,
                };
            case ShaderPackLoader::ResourceAccessKind::ReadWrite:
            case ShaderPackLoader::ResourceAccessKind::AppendCounter:
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    .reads = true,
                    .writes = true,
                    .needsPreviousContent = true,
                };
            case ShaderPackLoader::ResourceAccessKind::Append:
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                    .reads = false,
                    .writes = true,
                    .needsPreviousContent = false,
                };
            case ShaderPackLoader::ResourceAccessKind::IndirectRead:
                if (!bufferResource) { throw std::runtime_error("indirect_read is only valid for buffer resources"); }
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                    .reads = true,
                    .writes = false,
                    .needsPreviousContent = true,
                };
        }
        throw std::runtime_error("unsupported shader resource access kind");
    };

    auto colorAttachmentAccessTraits = [](ShaderPackLoader::ResourceAccessKind kind) {
        switch (kind) {
            case ShaderPackLoader::ResourceAccessKind::Write:
            case ShaderPackLoader::ResourceAccessKind::Overwrite:
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .reads = false,
                    .writes = true,
                    .needsPreviousContent = false,
                };
            case ShaderPackLoader::ResourceAccessKind::Read:
            case ShaderPackLoader::ResourceAccessKind::ReadWrite:
            case ShaderPackLoader::ResourceAccessKind::Append:
            case ShaderPackLoader::ResourceAccessKind::AppendCounter:
                return AccessTraits{
                    .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .reads = true,
                    .writes = true,
                    .needsPreviousContent = true,
                };
            case ShaderPackLoader::ResourceAccessKind::IndirectRead:
                throw std::runtime_error("indirect_read is not valid for color attachment resources");
        }
        throw std::runtime_error("unsupported color attachment access kind");
    };

    auto sameImageResource = [](const ResolvedImageResource &lhs, const ResolvedImageResource &rhs) {
        return lhs.targetImages == rhs.targetImages && lhs.runtimeTexture == rhs.runtimeTexture &&
               lhs.isImportedRuntimeTexture == rhs.isImportedRuntimeTexture;
    };

    auto addImageAccess = [&](ResolvedImageResource resource, VkImageLayout layout, const AccessTraits &traits) {
        for (ResolvedImageAccess &imageAccess : resources.imageAccesses) {
            if (!sameImageResource(imageAccess.resource, resource)) { continue; }
            imageAccess.layout = layout;
            imageAccess.accessMask |= traits.accessMask;
            imageAccess.reads = imageAccess.reads || traits.reads;
            imageAccess.writes = imageAccess.writes || traits.writes;
            imageAccess.needsPreviousContent = imageAccess.needsPreviousContent || traits.needsPreviousContent;
            return;
        }
        resources.imageAccesses.push_back({
            .resource = resource,
            .layout = layout,
            .accessMask = traits.accessMask,
            .reads = traits.reads,
            .writes = traits.writes,
            .needsPreviousContent = traits.needsPreviousContent,
        });
    };

    auto addBufferAccess = [&](ShaderPack::RuntimeBuffer *runtimeBuffer, const AccessTraits &traits) {
        for (ResolvedBufferAccess &bufferAccess : resources.bufferAccesses) {
            if (bufferAccess.runtimeBuffer != runtimeBuffer) { continue; }
            bufferAccess.accessMask |= traits.accessMask;
            bufferAccess.reads = bufferAccess.reads || traits.reads;
            bufferAccess.writes = bufferAccess.writes || traits.writes;
            bufferAccess.needsPreviousContent = bufferAccess.needsPreviousContent || traits.needsPreviousContent;
            return;
        }
        resources.bufferAccesses.push_back({
            .runtimeBuffer = runtimeBuffer,
            .accessMask = traits.accessMask,
            .reads = traits.reads,
            .writes = traits.writes,
            .needsPreviousContent = traits.needsPreviousContent,
        });
    };

    for (const ShaderPackLoader::ResourceAccess &input : inputs.imageAccesses) {
        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
        auto runtimeTexture = findRuntimeTexture(input.name);
        if (runtimeTexture.has_value() && runtimeTexture->get().config.imported) {
            layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else if (runtimeTexture.has_value() && !runtimeTexture->get().config.storageBinding.has_value() &&
                   runtimeTexture->get().config.sampledBinding.has_value()) {
            layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        addImageAccess(resolveImageResource(input.name), layout, shaderAccessTraits(input.kind, false));
    }

    for (const ShaderPackLoader::ResourceAccess &input : inputs.bufferAccesses) {
        addBufferAccess(resolveRuntimeBuffer(input.name), shaderAccessTraits(input.kind, true));
    }

    if (!outputsAreColorAttachments) {
        for (const ShaderPackLoader::ResourceAccess &output : outputs.imageAccesses) {
            auto runtimeTexture = findRuntimeTexture(output.name);
            if (runtimeTexture.has_value() && runtimeTexture->get().config.imported) {
                throw std::runtime_error("pass output cannot be an imported texture: " + output.name);
            }
            addImageAccess(resolveImageResource(output.name), VK_IMAGE_LAYOUT_GENERAL,
                           shaderAccessTraits(output.kind, false));
        }
    }

    for (const ShaderPackLoader::ResourceAccess &output : outputs.bufferAccesses) {
        addBufferAccess(resolveRuntimeBuffer(output.name), shaderAccessTraits(output.kind, true));
    }

    if (indirectArgsBuffer.has_value()) {
        resources.indirectArgsBuffer = resolveRuntimeBuffer(*indirectArgsBuffer);
        addBufferAccess(resources.indirectArgsBuffer, AccessTraits{
                                                          .accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                                          .reads = true,
                                                          .writes = false,
                                                          .needsPreviousContent = true,
                                                      });
    }

    if (fullScreenTarget.has_value()) {
        auto runtimeTexture = findRuntimeTexture(*fullScreenTarget);
        if (runtimeTexture.has_value() && runtimeTexture->get().config.imported) {
            throw std::runtime_error("pass output cannot be an imported texture: " + std::string(*fullScreenTarget));
        }
        resources.fullScreenTarget = ResolvedImageAccess{
            .resource = resolveImageResource(*fullScreenTarget),
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .reads = false,
            .writes = true,
            .needsPreviousContent = false,
        };
    }

    if (outputsAreColorAttachments) {
        for (const ShaderPackLoader::ResourceAccess &output : outputs.imageAccesses) {
            if (fullScreenTarget.has_value() && output.name == *fullScreenTarget) { continue; }
            auto runtimeTexture = findRuntimeTexture(output.name);
            if (runtimeTexture.has_value() && runtimeTexture->get().config.imported) {
                throw std::runtime_error("pass output cannot be an imported texture: " + output.name);
            }
            const AccessTraits traits = colorAttachmentAccessTraits(output.kind);
            resources.colorAttachmentTargets.push_back({
                .resource = resolveImageResource(output.name),
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .accessMask = traits.accessMask,
                .reads = traits.reads,
                .writes = traits.writes,
                .needsPreviousContent = traits.needsPreviousContent,
            });
        }
    }

    return resources;
}

const MainRenderModule::ResolvedPassResources &
MainRenderModule::resolvedResourcesForPass(const FullScreenPass &pass) const {
    return fullScreenPassResources_.at(&pass);
}

const MainRenderModule::ResolvedPassResources &
MainRenderModule::resolvedResourcesForPass(const RenderPass &pass) const {
    return renderPassResources_.at(&pass);
}

const MainRenderModule::ResolvedPassResources &
MainRenderModule::resolvedResourcesForPass(const RayTracingPass &pass) const {
    return rayTracingPassResources_.at(&pass);
}

const MainRenderModule::ResolvedPassResources &
MainRenderModule::resolvedResourcesForPass(const ComputePass &pass) const {
    return computePassResources_.at(&pass);
}

void MainRenderModule::initSharc() {
    sharcConfigBuffers_.clear();
    sharcHashEntriesBuffer_ = nullptr;
    sharcLockBuffer_ = nullptr;
    sharcAccumulationBuffer_ = nullptr;
    sharcResolvedBuffer_ = nullptr;

    if (!hasSharcRuntime_) { return; }

    sharcFrameIndex_ = 0;
    sharcPrevCameraPos_ = glm::dvec3(0.0);
    isFirstSharcFrame_ = true;
    currentSharcResolveEntryCount_ = sharcCapacity_;

    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t size = framework->swapchain()->imageCount();

    sharcConfigBuffers_.resize(size);

    const VkBufferUsageFlags sharcStorageUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const size_t sharcEntryCount = static_cast<size_t>(sharcCapacity_);
    sharcHashEntriesBuffer_ = vk::DeviceLocalBuffer::create(vma, device, false, sharcEntryCount * sizeof(uint64_t),
                                                            sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);
    sharcLockBuffer_ = vk::DeviceLocalBuffer::create(vma, device, false, sharcEntryCount * sizeof(uint32_t),
                                                     sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);
    sharcAccumulationBuffer_ = vk::DeviceLocalBuffer::create(vma, device, false, sharcEntryCount * sizeof(uint32_t) * 4,
                                                             sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);
    sharcResolvedBuffer_ = vk::DeviceLocalBuffer::create(vma, device, false, sharcEntryCount * sizeof(uint32_t) * 4,
                                                         sharcStorageUsage, 0, VMA_MEMORY_USAGE_GPU_ONLY);

    auto clearCommandPool = vk::CommandPool::create(framework->physicalDevice(), device);
    auto clearCommandBuffer = vk::CommandBuffer::create(device, clearCommandPool);
    clearCommandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcHashEntriesBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcLockBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcAccumulationBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(clearCommandBuffer->vkCommandBuffer(), sharcResolvedBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
    clearCommandBuffer->end();
    clearCommandBuffer->submitMainQueueIndividual(device);
    {
        std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
        vk::requireSuccess(vkQueueWaitIdle(device->mainVkQueue()), "vkQueueWaitIdle(SHARC initialization)");
    }

    for (uint32_t i = 0; i < size; i++) {
        sharcConfigBuffers_[i] =
            vk::HostVisibleBuffer::create(vma, device, sizeof(SharcConfigData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mainRenderDescriptorTables_[i]->bindBuffer(sharcConfigBuffers_[i], 4, 0);
        mainRenderDescriptorTables_[i]->bindBuffer(sharcHashEntriesBuffer_, 4, 1);
        mainRenderDescriptorTables_[i]->bindBuffer(sharcAccumulationBuffer_, 4, 2);
        mainRenderDescriptorTables_[i]->bindBuffer(sharcResolvedBuffer_, 4, 3);
        mainRenderDescriptorTables_[i]->bindBuffer(sharcLockBuffer_, 4, 4);
    }
}

void MainRenderModule::updateSharcConfig(uint32_t frameIndex) {
    auto buffers = Renderer::instance().buffers();
    auto worldUbo = reinterpret_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
    if (worldUbo == nullptr) { return; }

    glm::dvec3 currentCameraPos(worldUbo->cameraPos.x, worldUbo->cameraPos.y, worldUbo->cameraPos.z);
    if (isFirstSharcFrame_) {
        sharcPrevCameraPos_ = currentCameraPos;
        isFirstSharcFrame_ = false;
    }

    auto splitAddress = [](VkDeviceAddress address) {
        return std::array<uint32_t, 2>{static_cast<uint32_t>(address & 0xFFFFFFFFu),
                                       static_cast<uint32_t>((address >> 32) & 0xFFFFFFFFu)};
    };

    SharcConfigData config{};
    const auto hashEntriesAddress = splitAddress(sharcHashEntriesBuffer_->bufferAddress());
    const auto lockAddress = splitAddress(sharcLockBuffer_->bufferAddress());
    const auto accumulationAddress = splitAddress(sharcAccumulationBuffer_->bufferAddress());
    const auto resolvedAddress = splitAddress(sharcResolvedBuffer_->bufferAddress());
    config.hashEntriesAddress[0] = hashEntriesAddress[0];
    config.hashEntriesAddress[1] = hashEntriesAddress[1];
    config.lockAddress[0] = lockAddress[0];
    config.lockAddress[1] = lockAddress[1];
    config.accumulationAddress[0] = accumulationAddress[0];
    config.accumulationAddress[1] = accumulationAddress[1];
    config.resolvedAddress[0] = resolvedAddress[0];
    config.resolvedAddress[1] = resolvedAddress[1];
    config.cameraPosition = glm::vec4(static_cast<float>(currentCameraPos.x), static_cast<float>(currentCameraPos.y),
                                      static_cast<float>(currentCameraPos.z), 0.0f);
    config.cameraPositionPrev =
        glm::vec4(static_cast<float>(sharcPrevCameraPos_.x), static_cast<float>(sharcPrevCameraPos_.y),
                  static_cast<float>(sharcPrevCameraPos_.z), 0.0f);
    config.sceneScale = sharcSceneScale_;
    config.radianceScale = 1000.0f;
    config.accumulationFrameNum = sharcAccumulationFrameNum_;
    config.staleFrameNumMax = sharcStaleFrameNumMax_;
    config.capacity = sharcCapacity_;
    config.frameIndex = sharcFrameIndex_;
    config.enableAntiFireflyFilter = 1;
    config.useLockBuffer = 0;
    config.debugMode = sharcDebugMode_;
    config.updateDownsampleFactor = sharcUpdateDownsampleFactor_;
    config.resolveFrameStride = std::max(1u, sharcResolveFrameStride_);
    const uint32_t resolveFrame = sharcFrameIndex_ % config.resolveFrameStride;
    const uint32_t resolveBaseEntry =
        static_cast<uint32_t>((static_cast<uint64_t>(sharcCapacity_) * resolveFrame) / config.resolveFrameStride);
    const uint32_t resolveEndEntry =
        static_cast<uint32_t>((static_cast<uint64_t>(sharcCapacity_) * (static_cast<uint64_t>(resolveFrame) + 1u)) /
                              config.resolveFrameStride);
    config.resolveBaseEntry = resolveBaseEntry;
    config.resolveEntryCount = resolveEndEntry - resolveBaseEntry;
    currentSharcResolveEntryCount_ = config.resolveEntryCount;

    sharcConfigBuffers_[frameIndex]->uploadToBuffer(&config);

    sharcPrevCameraPos_ = currentCameraPos;
    sharcFrameIndex_++;
}

MainRenderModule::DirectionalShadowConfigUpdate
MainRenderModule::updateDirectionalShadowConfig(MainRenderModuleContext &context, uint32_t frameIndex) {
    auto buffers = Renderer::instance().buffers();
    auto worldUbo = reinterpret_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
    auto skyUbo = reinterpret_cast<vk::Data::SkyUBO *>(buffers->skyUniformBuffer()->mappedPtr());
    if (worldUbo == nullptr || skyUbo == nullptr || frameIndex >= directionalShadowDataBuffers_.size() ||
        directionalShadowDataBuffers_[frameIndex] == nullptr) {
        return {};
    }

    auto hashConfigData = [](const DirectionalShadowData &data) {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&data);
        uint64_t seed = 14695981039346656037ULL;
        for (size_t i = 0; i < sizeof(DirectionalShadowData); i++) {
            seed ^= bytes[i];
            seed *= 1099511628211ULL;
        }
        return seed;
    };

    auto uploadConfigIfChanged = [&](const DirectionalShadowData &data) {
        const uint64_t dataSignature = hashConfigData(data);
        bool updated = true;
        if (void *mappedPtr = directionalShadowDataBuffers_[frameIndex]->mappedPtr(); mappedPtr != nullptr) {
            updated = std::memcmp(mappedPtr, &data, sizeof(DirectionalShadowData)) != 0;
        }
        if (updated) {
            directionalShadowDataBuffers_[frameIndex]->uploadToBuffer(const_cast<DirectionalShadowData *>(&data));
        }
        return DirectionalShadowConfigUpdate{
            .valid = true,
            .updated = updated,
            .signature = dataSignature,
        };
    };

    auto normalizeSafe = [](glm::vec3 value, glm::vec3 fallback) {
        const float len2 = glm::dot(value, value);
        if (!std::isfinite(len2) || len2 <= 1e-8f) { return fallback; }
        return value * glm::inversesqrt(len2);
    };

    auto applyCelestialSouthOffset = [&](glm::vec3 direction) {
        constexpr float southOffsetSinAngle = 0.17364817766693033f;
        constexpr float southOffsetCosAngle = 0.984807753012208f;
        glm::vec3 rotated(direction.x, direction.y * southOffsetCosAngle - direction.z * southOffsetSinAngle,
                          direction.y * southOffsetSinAngle + direction.z * southOffsetCosAngle);
        return normalizeSafe(rotated, glm::vec3(0.0f, 1.0f, 0.0f));
    };

    glm::vec3 sunDirection =
        applyCelestialSouthOffset(normalizeSafe(glm::vec3(skyUbo->sunDirection), glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 moonDirection = -sunDirection;
    const float sunScale = std::max(sunDirection.y, 0.0f);
    const float moonScale = std::max(moonDirection.y, 0.0f);
    const bool useSun = sunScale >= moonScale;
    glm::vec3 lightDirection = useSun ? sunDirection : moonDirection;
    const float lightScale = useSun ? sunScale : moonScale;
    const bool enabled = worldUbo->skyType == 1 && lightScale > 1e-4f;

    const float renderDistanceBlocks =
        worldUbo->renderDistanceBlocks == 0 ? 192.0f : static_cast<float>(worldUbo->renderDistanceBlocks);
    const float shadowDistance =
        std::clamp(renderDistanceBlocks, directionalShadowMinDistanceBlocks, directionalShadowMaxDistanceBlocks);
    const float shadowFadeDistance = std::max(shadowDistance * 0.08f, 8.0f);

    const int32_t configuredWorldBottom = worldUbo->worldBottomY;
    const int32_t configuredWorldTop = worldUbo->worldTopY;
    const float worldBottomY =
        configuredWorldTop > configuredWorldBottom ? static_cast<float>(configuredWorldBottom) : -64.0f;
    const float worldTopY =
        configuredWorldTop > configuredWorldBottom ? static_cast<float>(configuredWorldTop) : 320.0f;
    const float cameraWorldY = static_cast<float>(worldUbo->cameraPos.y);
    const float relativeWorldBottomY = worldBottomY - cameraWorldY;
    const float relativeWorldTopY = worldTopY - cameraWorldY;

    DirectionalShadowData shadowData{};
    shadowData.lightViewProj = glm::mat4(1.0f);
    shadowData.lightDirectionAndStrength = glm::vec4(lightDirection, lightScale);
    shadowData.shadowParams = glm::vec4(1.0f / static_cast<float>(directionalShadowResolution),
                                        directionalShadowDepthBias, 1.0f, enabled ? 1.0f : 0.0f);
    shadowData.shadowDistanceParams = glm::vec4(shadowDistance, shadowFadeDistance, 0.0f, 0.0f);

    if (!enabled) {
        (void)context;
        return uploadConfigIfChanged(shadowData);
    }

    const glm::mat4 invView = worldUbo->cameraEffectedViewMatInv;
    const glm::mat4 proj = worldUbo->cameraProjMat;
    const glm::vec3 cameraOrigin = glm::vec3(0.0f);
    const glm::vec3 cameraRight =
        normalizeSafe(glm::vec3(invView * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::vec3 cameraUp =
        normalizeSafe(glm::vec3(invView * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 cameraForward =
        normalizeSafe(glm::vec3(invView * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)), glm::vec3(0.0f, 0.0f, -1.0f));
    const float tanHalfFovX = 1.0f / std::max(std::abs(proj[0][0]), 1e-4f);
    const float tanHalfFovY = 1.0f / std::max(std::abs(proj[1][1]), 1e-4f);

    const glm::vec3 lightZ = normalizeSafe(-lightDirection, glm::vec3(0.0f, -1.0f, 0.0f));
    const glm::vec3 lightUpCandidate =
        std::abs(lightZ.y) > 0.95f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 lightX = normalizeSafe(glm::cross(lightUpCandidate, lightZ), glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::vec3 lightY = normalizeSafe(glm::cross(lightZ, lightX), glm::vec3(0.0f, 1.0f, 0.0f));

    auto buildFrustumCorners = [&](float nearDistance, float farDistance) {
        std::array<glm::vec3, 8> corners{};
        uint32_t cursor = 0;
        for (float distance : {nearDistance, farDistance}) {
            const float halfWidth = distance * tanHalfFovX;
            const float halfHeight = distance * tanHalfFovY;
            const glm::vec3 center = cameraOrigin + cameraForward * distance;
            for (float ySign : {-1.0f, 1.0f}) {
                for (float xSign : {-1.0f, 1.0f}) {
                    corners[cursor++] = center + cameraRight * (xSign * halfWidth) + cameraUp * (ySign * halfHeight);
                }
            }
        }
        return corners;
    };

    auto lightSpacePosition = [&](glm::vec3 value) {
        return glm::vec3(glm::dot(value, lightX), glm::dot(value, lightY), glm::dot(value, lightZ));
    };

    auto buildLightMatrix = [&](float shadowNear, float shadowFar) {
        std::array<glm::vec3, 8> corners = buildFrustumCorners(shadowNear, shadowFar);
        float minX = std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();
        for (const glm::vec3 &corner : corners) {
            const glm::vec3 cornerLight = lightSpacePosition(corner);
            minX = std::min(minX, cornerLight.x);
            maxX = std::max(maxX, cornerLight.x);
            minY = std::min(minY, cornerLight.y);
            maxY = std::max(maxY, cornerLight.y);
            minZ = std::min(minZ, cornerLight.z);
            maxZ = std::max(maxZ, cornerLight.z);
            glm::vec3 worldBottomCorner = corner;
            worldBottomCorner.y = relativeWorldBottomY;
            glm::vec3 worldTopCorner = corner;
            worldTopCorner.y = relativeWorldTopY;
            const float bottomZ = lightSpacePosition(worldBottomCorner).z;
            const float topZ = lightSpacePosition(worldTopCorner).z;
            minZ = std::min(minZ, std::min(bottomZ, topZ));
            maxZ = std::max(maxZ, std::max(bottomZ, topZ));
        }
        const float casterMargin = std::max(24.0f, shadowFar * 0.20f);
        minZ -= casterMargin;
        maxZ += casterMargin;

        const float xyMargin = std::max(2.0f, shadowFar * 0.025f);
        minX -= xyMargin;
        maxX += xyMargin;
        minY -= xyMargin;
        maxY += xyMargin;
        const float rangeXUnsnapped = std::max(maxX - minX, 1.0f);
        const float rangeYUnsnapped = std::max(maxY - minY, 1.0f);
        const float shadowResolution = static_cast<float>(directionalShadowResolution);
        const float texelSizeX = rangeXUnsnapped / shadowResolution;
        const float texelSizeY = rangeYUnsnapped / shadowResolution;
        minX = std::floor(minX / texelSizeX) * texelSizeX;
        maxX = minX + std::ceil(rangeXUnsnapped / texelSizeX) * texelSizeX;
        minY = std::floor(minY / texelSizeY) * texelSizeY;
        maxY = minY + std::ceil(rangeYUnsnapped / texelSizeY) * texelSizeY;
        const float rangeX = std::max(maxX - minX, 1e-3f);
        const float rangeY = std::max(maxY - minY, 1e-3f);
        const float rangeZ = std::max(maxZ - minZ, 1e-3f);

        glm::mat4 matrix(0.0f);
        matrix[0][0] = 2.0f * lightX.x / rangeX;
        matrix[1][0] = 2.0f * lightX.y / rangeX;
        matrix[2][0] = 2.0f * lightX.z / rangeX;
        matrix[3][0] = -(maxX + minX) / rangeX;

        matrix[0][1] = 2.0f * lightY.x / rangeY;
        matrix[1][1] = 2.0f * lightY.y / rangeY;
        matrix[2][1] = 2.0f * lightY.z / rangeY;
        matrix[3][1] = -(maxY + minY) / rangeY;

        matrix[0][2] = lightZ.x / rangeZ;
        matrix[1][2] = lightZ.y / rangeZ;
        matrix[2][2] = lightZ.z / rangeZ;
        matrix[3][2] = -minZ / rangeZ;
        matrix[3][3] = 1.0f;

        shadowData.lightViewProj = matrix;
    };

    buildLightMatrix(0.05f, shadowDistance);

    (void)context;
    return uploadConfigIfChanged(shadowData);
}

void MainRenderModule::renderDirectionalShadowMaps(MainRenderModuleContext &context,
                                                   const std::shared_ptr<vk::DeviceLocalBuffer> &indirectCommandBuffer,
                                                   uint32_t solidDrawCount,
                                                   uint32_t alphaOpaqueDrawCount) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.renderDirectionalShadowMaps");
    const uint32_t drawCount = solidDrawCount + alphaOpaqueDrawCount;
    MCVR_PROFILE_METRIC("shadow.draw_count", drawCount);
    MCVR_PROFILE_METRIC("shadow.solid_draw_count", solidDrawCount);
    MCVR_PROFILE_METRIC("shadow.alpha_opaque_draw_count", alphaOpaqueDrawCount);
    MCVR_PROFILE_METRIC("shadow.resolution", directionalShadowResolution);
    if (!directionalShadowRasterEnabled_ || drawCount == 0 || indirectCommandBuffer == nullptr ||
        directionalShadowRenderPass_ == nullptr || directionalShadowSolidPipeline_ == nullptr ||
        directionalShadowOpaquePipeline_ == nullptr) {
        return;
    }

    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    const uint32_t frameIndex = frameworkContext->frameIndex;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();
    if (frameIndex >= directionalShadowDepthImages_.size() || frameIndex >= directionalShadowFramebuffers_.size()) {
        return;
    }
    auto depthImage = directionalShadowDepthImages_[frameIndex];
    auto framebuffer = directionalShadowFramebuffers_[frameIndex];
    if (depthImage == nullptr || framebuffer == nullptr || frameIndex >= directionalShadowDataBuffers_.size() ||
        directionalShadowDataBuffers_[frameIndex] == nullptr) {
        return;
    }

    auto imageBarrierToAttachment = [&](const std::shared_ptr<vk::DeviceLocalImage> &image, VkImageLayout newLayout,
                                        VkAccessFlags2 dstAccessMask, VkPipelineStageFlags2 dstStageMask) {
        if (image == nullptr) { return; }
        const VkImageLayout oldLayout = image->imageLayout();
        if (oldLayout == newLayout) { return; }
        const bool uninitialized = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED;
        worldCommandBuffer->barriersBufferImage(
            {},
            {
                {
                    .srcStageMask =
                        uninitialized ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    .srcAccessMask = uninitialized ? 0 : (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT),
                    .dstStageMask = dstStageMask,
                    .dstAccessMask = dstAccessMask,
                    .oldLayout = oldLayout,
                    .newLayout = newLayout,
                    .srcQueueFamilyIndex = queueIndex,
                    .dstQueueFamilyIndex = queueIndex,
                    .image = image,
                    .subresourceRange = image->fullSubresourceRange(),
                },
            });
        image->imageLayout() = newLayout;
    };

    imageBarrierToAttachment(
        depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT);

    VkClearValue depthClear{};
    depthClear.depthStencil = {.depth = 1.0f, .stencil = 0};

    worldCommandBuffer->beginRenderPass({
        .renderPass = directionalShadowRenderPass_,
        .framebuffer = framebuffer,
        .renderAreaExtent = {.width = directionalShadowResolution, .height = directionalShadowResolution},
        .clearValues = {depthClear},
    });
    {
        std::string debugLabelName = MainRenderModule::buildMainRenderDebugLabel("shadow", "directional");
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.75f, 0.35f,
                                                       0.20f);
        worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);

        DirectionalShadowData *shadowData =
            reinterpret_cast<DirectionalShadowData *>(directionalShadowDataBuffers_[frameIndex]->mappedPtr());
        DirectionalShadowPushConstants pushConstants{};
        pushConstants.lightViewProj = shadowData->lightViewProj;
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), context.mainRenderDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DirectionalShadowPushConstants), &pushConstants);

        vkCmdBindPipeline(worldCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          directionalShadowSolidPipeline_->vkPipeline());
        if (solidDrawCount > 0) {
            MCVR_PROFILE_SCOPE("native.MainRenderModule.renderDirectionalShadowMaps.solid");
            vkCmdDrawIndirect(worldCommandBuffer->vkCommandBuffer(), indirectCommandBuffer->vkBuffer(), 0,
                              solidDrawCount, sizeof(VkDrawIndirectCommand));
        }

        vkCmdBindPipeline(worldCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          directionalShadowOpaquePipeline_->vkPipeline());
        if (alphaOpaqueDrawCount > 0) {
            MCVR_PROFILE_SCOPE("native.MainRenderModule.renderDirectionalShadowMaps.alpha_opaque");
            const VkDeviceSize alphaOpaqueCommandOffset =
                static_cast<VkDeviceSize>(solidDrawCount) * sizeof(VkDrawIndirectCommand);
            vkCmdDrawIndirect(worldCommandBuffer->vkCommandBuffer(), indirectCommandBuffer->vkBuffer(),
                              alphaOpaqueCommandOffset, alphaOpaqueDrawCount, sizeof(VkDrawIndirectCommand));
        }

        worldCommandBuffer->endRenderPass();
    }
    depthImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    worldCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask =
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .srcQueueFamilyIndex = queueIndex,
                    .dstQueueFamilyIndex = queueIndex,
                    .image = depthImage,
                    .subresourceRange = depthImage->fullSubresourceRange(),
                },
            });
    depthImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void MainRenderModule::initFullScreenPassTargets(FullScreenPass &pass,
                                                 std::shared_ptr<vk::Device> device,
                                                 uint32_t frameCount) {
    std::shared_ptr<vk::DeviceLocalImage> targetImage = findTargetImage(pass.config.target, 0);
    pass.renderPass = vk::RenderPassBuilder{}
                          .beginAttachmentDescription()
                          .defineAttachmentDescription({
                              .format = targetImage->vkFormat(),
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                              .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                              .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                              .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                              .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          })
                          .endAttachmentDescription()
                          .beginAttachmentReference()
                          .defineAttachmentReference({
                              .attachment = 0,
                              .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          })
                          .endAttachmentReference()
                          .beginSubpassDescription()
                          .defineSubpassDescription({
                              .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                              .colorAttachmentIndices = {0},
                          })
                          .endSubpassDescription()
                          .build(device);

    pass.framebuffers.resize(frameCount);
    for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        pass.framebuffers[frameIndex] =
            buildFramebuffers(findTargetImage(pass.config.target, frameIndex), pass.renderPass);
    }
}

std::vector<ShaderPack::ShaderCreateInfo>
MainRenderModule::collectFullScreenPassShaderRequests(const FullScreenPass &pass,
                                                      const std::unordered_map<std::string, std::string> &definitions) {
    std::shared_ptr<vk::DeviceLocalImage> targetImage = findTargetImage(pass.config.target, 0);
    const bool implicitCubeFace =
        targetImage->layer() == 6 && !pass.config.face.has_value() && !pass.config.layer.has_value();
    const uint32_t pipelineCount = implicitCubeFace ? 6u : 1u;
    const uint32_t executionSet = shaderPack_->executionSet(5u);
    const std::unordered_map<std::string, std::string> baseDefinitions =
        shaderDefinitionsWithRuntimeExpressions(definitions);

    std::vector<ShaderPack::ShaderCreateInfo> requests(pipelineCount);
    for (uint32_t pipelineIndex = 0; pipelineIndex < pipelineCount; pipelineIndex++) {
        std::unordered_map<std::string, std::string> shaderDefinitions = baseDefinitions;
        if (targetImage->layer() == 6) {
            uint32_t faceIndex = pass.config.face.value_or(pipelineIndex);
            shaderDefinitions["FACE"] = std::to_string(faceIndex);
        }
        requests[pipelineIndex] = {
            .path = pass.config.fragmentShaderPath,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .definitions = std::move(shaderDefinitions),
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        };
    }
    return requests;
}

void MainRenderModule::buildFullScreenPassPipelines(FullScreenPass &pass,
                                                    std::shared_ptr<vk::Device> device,
                                                    const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                                    size_t &shaderOffset) {
    std::shared_ptr<vk::DeviceLocalImage> targetImage = findTargetImage(pass.config.target, 0);
    const bool implicitCubeFace =
        targetImage->layer() == 6 && !pass.config.face.has_value() && !pass.config.layer.has_value();
    const uint32_t pipelineCount = implicitCubeFace ? 6u : 1u;

    pass.fragmentShaders.assign(compiledShaders.begin() + static_cast<ptrdiff_t>(shaderOffset),
                                compiledShaders.begin() + static_cast<ptrdiff_t>(shaderOffset + pipelineCount));
    shaderOffset += pipelineCount;
    pass.pipelines.resize(pipelineCount);

    for (uint32_t pipelineIndex = 0; pipelineIndex < pipelineCount; pipelineIndex++) {
        pass.pipelines[pipelineIndex] =
            vk::GraphicsPipelineBuilder{}
                .defineRenderPass(pass.renderPass, 0)
                .beginShaderStage()
                .defineShaderStage(fullScreenVertexShader_, VK_SHADER_STAGE_VERTEX_BIT)
                .defineShaderStage(pass.fragmentShaders[pipelineIndex], VK_SHADER_STAGE_FRAGMENT_BIT)
                .endShaderStage()
                .defineVertexInputState<void>()
                .defineViewportScissorState({
                    .viewport =
                        {
                            .x = 0.0f,
                            .y = 0.0f,
                            .width = static_cast<float>(targetImage->width()),
                            .height = static_cast<float>(targetImage->height()),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f,
                        },
                    .scissor =
                        {
                            .offset = {.x = 0, .y = 0},
                            .extent = {.width = targetImage->width(), .height = targetImage->height()},
                        },
                })
                .beginColorBlendAttachmentState()
                .defineDefaultColorBlendAttachmentState()
                .endColorBlendAttachmentState()
                .definePipelineLayout(mainRenderDescriptorTables_[0])
                .build(device);
    }
}

void MainRenderModule::initRenderPassTargets(RenderPass &pass,
                                             std::shared_ptr<vk::Device> device,
                                             uint32_t frameCount) {
    if (pass.config.outputs.images.empty()) {
        throw std::runtime_error("main_render render pass has no outputs: " + pass.config.name);
    }
    if (hdrNoisyOutputImages_.empty() || hdrNoisyOutputImages_[0] == nullptr) {
        throw std::runtime_error("main_render render pass missing reference output image: " + pass.config.name);
    }

    vk::RenderPassBuilder renderPassBuilder;
    auto &attachmentBuilder = renderPassBuilder.beginAttachmentDescription();
    for (const std::string &target : pass.config.outputs.images) {
        auto targetImage = findTargetImage(target, 0);
        if (targetImage == nullptr) { throw std::runtime_error("invalid main_render render pass target: " + target); }
        const bool clearTarget = pass.config.clearOutputs.find(target) != pass.config.clearOutputs.end();
        attachmentBuilder.defineAttachmentDescription({
            .format = targetImage->vkFormat(),
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = clearTarget ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        });
    }
    const uint32_t depthAttachmentIndex = static_cast<uint32_t>(pass.config.outputs.images.size());
    attachmentBuilder.defineAttachmentDescription({
        .format = rasterDepthImages_[0]->vkFormat(),
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = pass.config.depthLoad ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    });
    attachmentBuilder.endAttachmentDescription();

    auto &referenceBuilder = renderPassBuilder.beginAttachmentReference();
    for (uint32_t attachmentIndex = 0; attachmentIndex < pass.config.outputs.images.size(); attachmentIndex++) {
        referenceBuilder.defineAttachmentReference({
            .attachment = attachmentIndex,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        });
    }
    referenceBuilder
        .defineAttachmentReference({
            .attachment = depthAttachmentIndex,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        })
        .endAttachmentReference();

    std::vector<uint32_t> colorAttachmentIndices;
    colorAttachmentIndices.reserve(pass.config.outputs.images.size());
    for (uint32_t attachmentIndex = 0; attachmentIndex < pass.config.outputs.images.size(); attachmentIndex++) {
        colorAttachmentIndices.push_back(attachmentIndex);
    }

    pass.renderPass = renderPassBuilder.beginSubpassDescription()
                          .defineSubpassDescription({
                              .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                              .colorAttachmentIndices = std::move(colorAttachmentIndices),
                              .depthStencilAttachmentIndex = depthAttachmentIndex,
                          })
                          .endSubpassDescription()
                          .build(device);

    pass.framebuffers.resize(frameCount);
    for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        vk::FramebufferBuilder framebufferBuilder;
        auto &attachments = framebufferBuilder.beginAttachment();
        for (const std::string &target : pass.config.outputs.images) {
            attachments.defineAttachment(findTargetImage(target, frameIndex));
        }
        pass.framebuffers[frameIndex] =
            attachments.defineAttachment(rasterDepthImages_[frameIndex]).endAttachment().build(device, pass.renderPass);
    }
}

std::vector<ShaderPack::ShaderCreateInfo>
MainRenderModule::collectRenderPassShaderRequests(const RenderPass &pass,
                                                  const std::unordered_map<std::string, std::string> &definitions) {
    const uint32_t executionSet = shaderPack_->executionSet(5u);
    std::vector<ShaderPack::ShaderCreateInfo> requests;
    requests.reserve(pass.config.pipelineConfigs.size() * 2);
    for (const auto &[variantName, pipelineConfig] : pass.config.pipelineConfigs) {
        (void)variantName;
        const std::unordered_map<std::string, std::string> shaderDefinitions =
            shaderDefinitionsWithRuntimeExpressions(mergeShaderDefinitions(definitions, pipelineConfig.definitions));
        requests.push_back({
            .path = pipelineConfig.shader.vertexShaderPath,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .definitions = shaderDefinitions,
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        });
        requests.push_back({
            .path = pipelineConfig.shader.fragmentShaderPath,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .definitions = shaderDefinitions,
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        });
    }
    return requests;
}

void MainRenderModule::buildRenderPassPipelines(RenderPass &pass,
                                                std::shared_ptr<vk::Device> device,
                                                const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                                size_t &shaderOffset) {
    if (pass.config.indirectDraw.has_value()) {
        if (!device->hasMultiDrawIndirect()) {
            throw std::runtime_error("main_render render pass indirect_draw requires multiDrawIndirect: " +
                                     pass.config.name);
        }
        if (!device->hasDrawIndirectFirstInstance()) {
            throw std::runtime_error("main_render render pass indirect_draw requires drawIndirectFirstInstance: " +
                                     pass.config.name);
        }
    }

    for (const auto &[variantName, pipelineConfig] : pass.config.pipelineConfigs) {
        auto &variant = pass.shaderVariants[variantName];
        variant.config = pipelineConfig;
        variant.vertexShader = compiledShaders[shaderOffset++];
        variant.fragmentShader = compiledShaders[shaderOffset++];
    }

    auto colorBlendAttachment = [](bool enableBlend, VkColorComponentFlags colorWriteMask) {
        VkPipelineColorBlendAttachmentState state{
            .blendEnable = enableBlend ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = colorWriteMask,
        };
        if (!enableBlend) {
            state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        }
        return state;
    };

    auto noBlendAttachment = [](VkColorComponentFlags colorWriteMask) {
        return VkPipelineColorBlendAttachmentState{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = colorWriteMask,
        };
    };

    auto isScalarDepthTarget = [](std::string_view target) {
        return target == MainRenderModule::TARGET_LINEAR_DEPTH || target == MainRenderModule::TARGET_FIRST_HIT_DEPTH ||
               target == MainRenderModule::TARGET_INDIRECT_SPECULAR_HIT_DISTANCE ||
               target == MainRenderModule::TARGET_INDIRECT_DIFFUSE_HIT_DISTANCE ||
               target == MainRenderModule::TARGET_DIRECT_HIT_DISTANCE;
    };

    VkPipelineVertexInputStateCreateInfo emptyVertexInputState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };

    auto &pbrVertexLayoutInfo = vk::Vertex::vertexLayoutInfo<vk::VertexFormat::EntityPBRVertex>();
    VkPipelineVertexInputStateCreateInfo vertexInputState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &pbrVertexLayoutInfo.bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(pbrVertexLayoutInfo.attributeDescriptions.size()),
        .pVertexAttributeDescriptions = pbrVertexLayoutInfo.attributeDescriptions.data(),
    };
    const VkPipelineVertexInputStateCreateInfo &pipelineVertexInputState =
        (pass.config.indirectDraw.has_value() || pass.rendersHandOnly) ? emptyVertexInputState : vertexInputState;

    std::unordered_map<std::string, std::shared_ptr<vk::GraphicsPipeline>> pipelineCache;
    for (auto &[variantName, variant] : pass.shaderVariants) {
        auto pipelineConfigIter = pass.config.pipelineConfigs.find(variantName);
        if (pipelineConfigIter == pass.config.pipelineConfigs.end()) {
            throw std::runtime_error("main_render render pass pipeline config is missing: " + pass.config.name + "." +
                                     variantName);
        }
        const ShaderPackLoader::RenderPipelineConfig &pipelineConfig = pipelineConfigIter->second;
        const bool depthTest = pipelineConfig.depthTest;
        const bool depthWrite = pipelineConfig.depthWrite;
        const VkCompareOp depthCompare = parseDepthCompare(pipelineConfig.depthCompare);
        const bool colorBlend = pipelineConfig.colorBlend;
        const VkCullModeFlags cullMode = parseCullMode(pipelineConfig.cull, pass.config.name + "." + variantName);
        const VkFrontFace frontFace = parseFrontFace(pipelineConfig.frontFace, pass.config.name + "." + variantName);
        if (!pipelineConfig.colorWriteMasks.empty() &&
            pipelineConfig.colorWriteMasks.size() != pass.config.outputs.images.size()) {
            throw std::runtime_error("main_render render pass pipeline color_write_mask count mismatch: " +
                                     pass.config.name + "." + variantName);
        }
        std::string colorWriteMaskKey;
        for (const std::string &mask : pipelineConfig.colorWriteMasks) { colorWriteMaskKey += "|mask=" + mask; }
        std::string pipelineKey =
            pipelineConfig.shader.vertexShaderPath.string() + "|" + pipelineConfig.shader.fragmentShaderPath.string() +
            "|depth_test=" + (depthTest ? "1" : "0") + "|depth_write=" + (depthWrite ? "1" : "0") +
            "|depth_compare=" + pipelineConfig.depthCompare + "|color_blend=" + (colorBlend ? "1" : "0") +
            colorWriteMaskKey + "|cull=" + pipelineConfig.cull + "|front_face=" + pipelineConfig.frontFace;
        appendShaderDefinitionsKey(pipelineKey, pipelineConfig.definitions);
        auto cachedPipelineIter = pipelineCache.find(pipelineKey);
        if (cachedPipelineIter != pipelineCache.end()) {
            variant.pipeline = cachedPipelineIter->second;
            continue;
        }

        vk::GraphicsPipelineBuilder pipelineBuilder;
        auto &colorBlendBuilder = pipelineBuilder.defineRenderPass(pass.renderPass, 0)
                                      .beginShaderStage()
                                      .defineShaderStage(variant.vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
                                      .defineShaderStage(variant.fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
                                      .endShaderStage()
                                      .defineVertexInputState(pipelineVertexInputState)
                                      .defineViewportScissorState({
                                          .viewport = {.x = 0.0f,
                                                       .y = 0.0f,
                                                       .width = static_cast<float>(hdrNoisyOutputImages_[0]->width()),
                                                       .height = static_cast<float>(hdrNoisyOutputImages_[0]->height()),
                                                       .minDepth = 0.0f,
                                                       .maxDepth = 1.0f},
                                          .scissor = {.offset = {.x = 0, .y = 0},
                                                      .extent = {.width = hdrNoisyOutputImages_[0]->width(),
                                                                 .height = hdrNoisyOutputImages_[0]->height()}},
                                      })
                                      .defineDepthStencilState({
                                          .depthTestEnable = depthTest ? VK_TRUE : VK_FALSE,
                                          .depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE,
                                          .depthCompareOp = depthCompare,
                                          .depthBoundsTestEnable = VK_FALSE,
                                          .stencilTestEnable = VK_FALSE,
                                          .minDepthBounds = 0.0f,
                                          .maxDepthBounds = 1.0f,
                                      })
                                      .defineRasterizationState(VkPipelineRasterizationStateCreateInfo{
                                          .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                          .depthClampEnable = VK_FALSE,
                                          .rasterizerDiscardEnable = VK_FALSE,
                                          .polygonMode = VK_POLYGON_MODE_FILL,
                                          .cullMode = cullMode,
                                          .frontFace = frontFace,
                                          .depthBiasEnable = VK_FALSE,
                                          .depthBiasConstantFactor = 0.0f,
                                          .depthBiasClamp = 0.0f,
                                          .depthBiasSlopeFactor = 0.0f,
                                          .lineWidth = 1.0f,
                                      })
                                      .beginColorBlendAttachmentState();
        for (size_t attachmentIndex = 0; attachmentIndex < pass.config.outputs.images.size(); attachmentIndex++) {
            const std::string &target = pass.config.outputs.images[attachmentIndex];
            VkColorComponentFlags colorWriteMask = isScalarDepthTarget(target) ?
                                                       VK_COLOR_COMPONENT_R_BIT :
                                                       (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
            if (!pipelineConfig.colorWriteMasks.empty()) {
                colorWriteMask = parseColorWriteMask(pipelineConfig.colorWriteMasks[attachmentIndex],
                                                     pass.config.name + "." + variantName + ".color_write_mask[" +
                                                         std::to_string(attachmentIndex) + "]");
            }
            colorBlendBuilder.defineColorBlendAttachmentState(isScalarDepthTarget(target) ?
                                                                  noBlendAttachment(colorWriteMask) :
                                                                  colorBlendAttachment(colorBlend, colorWriteMask));
        }
        variant.pipeline = colorBlendBuilder.endColorBlendAttachmentState()
                               .definePipelineLayout(mainRenderDescriptorTables_[0])
                               .build(device);
        pipelineCache[pipelineKey] = variant.pipeline;
    }

    if (pass.config.indirectDraw.has_value()) {
        pass.indirectCommandComputeShader =
            vk::Shader::create(device, worldPrepareCompiledShaderPath("primary_surface_indirect_comp.spv").string());
        pass.indirectCommandComputePipeline = vk::ComputePipelineBuilder{}
                                                  .defineShader(pass.indirectCommandComputeShader)
                                                  .definePipelineLayout(mainRenderDescriptorTables_[0])
                                                  .build(device);
    } else {
        pass.indirectCommandComputeShader = nullptr;
        pass.indirectCommandComputePipeline = nullptr;
    }
}

void MainRenderModule::initDirectionalShadowPipelines() {
    if (!directionalShadowRasterEnabled_) {
        directionalShadowSolidVertexShader_ = nullptr;
        directionalShadowSolidFragmentShader_ = nullptr;
        directionalShadowVertexShader_ = nullptr;
        directionalShadowOpaqueFragmentShader_ = nullptr;
        directionalShadowIndirectComputeShader_ = nullptr;
        directionalShadowSolidPipeline_ = nullptr;
        directionalShadowOpaquePipeline_ = nullptr;
        directionalShadowIndirectComputePipeline_ = nullptr;
        return;
    }
    if (directionalShadowRenderPass_ == nullptr || mainRenderDescriptorTables_.empty() ||
        mainRenderDescriptorTables_[0] == nullptr) {
        return;
    }

    auto framework = framework_.lock();
    auto device = framework->device();

    directionalShadowSolidVertexShader_ =
        vk::Shader::create(device, worldPrepareCompiledShaderPath("directional_shadow_solid_vert.spv").string());
    directionalShadowSolidFragmentShader_ =
        vk::Shader::create(device, worldPrepareCompiledShaderPath("directional_shadow_solid_frag.spv").string());
    directionalShadowVertexShader_ =
        vk::Shader::create(device, worldPrepareCompiledShaderPath("directional_shadow_vert.spv").string());
    directionalShadowOpaqueFragmentShader_ =
        vk::Shader::create(device, worldPrepareCompiledShaderPath("directional_shadow_opaque_frag.spv").string());
    directionalShadowIndirectComputeShader_ =
        vk::Shader::create(device, worldPrepareCompiledShaderPath("directional_shadow_visibility_comp.spv").string());

    auto buildPipeline = [&](std::shared_ptr<vk::Shader> vertexShader, std::shared_ptr<vk::Shader> fragmentShader,
                             VkBool32 depthWrite) {
        return vk::GraphicsPipelineBuilder{}
            .defineRenderPass(directionalShadowRenderPass_, 0)
            .beginShaderStage()
            .defineShaderStage(vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
            .defineShaderStage(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
            .endShaderStage()
            .defineVertexInputState<void>()
            .defineViewportScissorState({
                .viewport = {.x = 0.0f,
                             .y = 0.0f,
                             .width = static_cast<float>(directionalShadowResolution),
                             .height = static_cast<float>(directionalShadowResolution),
                             .minDepth = 0.0f,
                             .maxDepth = 1.0f},
                .scissor = {.offset = {.x = 0, .y = 0},
                            .extent = {.width = directionalShadowResolution, .height = directionalShadowResolution}},
            })
            .defineDepthStencilState({
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = depthWrite,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .minDepthBounds = 0.0f,
                .maxDepthBounds = 1.0f,
            })
            .defineRasterizationState(VkPipelineRasterizationStateCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .depthBiasConstantFactor = 0.0f,
                .depthBiasClamp = 0.0f,
                .depthBiasSlopeFactor = 0.0f,
                .lineWidth = 1.0f,
            })
            .definePipelineLayout(mainRenderDescriptorTables_[0])
            .build(device);
    };

    directionalShadowSolidPipeline_ =
        buildPipeline(directionalShadowSolidVertexShader_, directionalShadowSolidFragmentShader_, VK_TRUE);
    directionalShadowOpaquePipeline_ =
        buildPipeline(directionalShadowVertexShader_, directionalShadowOpaqueFragmentShader_, VK_TRUE);
    directionalShadowIndirectComputePipeline_ = vk::ComputePipelineBuilder{}
                                                    .defineShader(directionalShadowIndirectComputeShader_)
                                                    .definePipelineLayout(mainRenderDescriptorTables_[0])
                                                    .build(device);
}

void MainRenderModule::addRenderIndirectResources(RenderPass &pass, ResolvedPassResources &resources) {
    (void)pass;
    (void)resources;
}

void MainRenderModule::renderHandOnlyPass(RenderPass &pass, MainRenderModuleContext &context) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.renderHandOnlyPass");
    if (!pass.rendersHandOnly) { throw std::runtime_error("renderHandOnlyPass called for a non-HAND render pass"); }
    if (pass.config.indirectDraw.has_value() || pass.indirectCommandComputeShader != nullptr ||
        pass.indirectCommandComputePipeline != nullptr || !pass.indirectCommandBuffers.empty()) {
        throw std::runtime_error("HAND render pass cannot use indirect raster resources: " + pass.config.name);
    }
    if (pass.config.depthLoad) {
        throw std::runtime_error("HAND render pass must clear its private depth attachment: " + pass.config.name);
    }

    auto frameworkContext = context.frameworkContext.lock();
    if (frameworkContext == nullptr) { throw std::runtime_error("HAND render pass has no framework context"); }
    auto framework = frameworkContext->framework.lock();
    if (framework == nullptr) { throw std::runtime_error("HAND render pass has no framework"); }
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    if (worldCommandBuffer == nullptr) { throw std::runtime_error("HAND render pass has no command buffer"); }
    const uint32_t frameIndex = frameworkContext->frameIndex;
    if (frameIndex >= pass.framebuffers.size()) {
        throw std::runtime_error("HAND render pass framebuffer index is out of range: " + pass.config.name);
    }
    if (pass.framebuffers[frameIndex] == nullptr || pass.renderPass == nullptr) {
        throw std::runtime_error("HAND render pass framebuffer is unavailable: " + pass.config.name);
    }
    if (context.mainRenderDescriptorTable == nullptr) {
        throw std::runtime_error("HAND render pass descriptor table is unavailable: " + pass.config.name);
    }
    if (frameIndex >= rasterDepthImages_.size() || rasterDepthImages_[frameIndex] == nullptr) {
        throw std::runtime_error("HAND render pass depth attachment is unavailable: " + pass.config.name);
    }
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();
    const ResolvedPassResources &resources = resolvedResourcesForPass(pass);

    if (pass.config.outputs.images.empty() ||
        resources.colorAttachmentTargets.size() != pass.config.outputs.images.size()) {
        throw std::runtime_error("HAND render pass color attachment list is invalid: " + pass.config.name);
    }
    for (const std::string &outputName : pass.config.outputs.images) {
        if (pass.config.clearOutputs.find(outputName) == pass.config.clearOutputs.end()) {
            throw std::runtime_error("HAND render pass output is not cleared: " + outputName);
        }
    }

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> colorImages;
    colorImages.reserve(resources.colorAttachmentTargets.size());
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    for (const ResolvedImageAccess &target : resources.colorAttachmentTargets) {
        auto image = resolveImageResource(target.resource, frameIndex);
        if (image == nullptr || image->width() == 0 || image->height() == 0) {
            throw std::runtime_error("HAND render pass color attachment is unavailable: " + pass.config.name);
        }
        if (colorImages.empty()) {
            renderWidth = image->width();
            renderHeight = image->height();
        } else if (image->width() != renderWidth || image->height() != renderHeight) {
            throw std::runtime_error("HAND render pass color attachment dimensions differ: " + pass.config.name);
        }
        colorImages.push_back(std::move(image));
    }
    if (rasterDepthImages_[frameIndex]->width() != renderWidth ||
        rasterDepthImages_[frameIndex]->height() != renderHeight) {
        throw std::runtime_error("HAND render pass depth attachment dimensions differ: " + pass.config.name);
    }

    auto entityBatch =
        context.worldPrepareContext == nullptr ? nullptr : context.worldPrepareContext->rasterEntityBatch;
    const uint32_t entityInstanceCount = entityBatch == nullptr ? 0u : [&]() {
        if (entityBatch->entities.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("HAND entity instance count exceeds uint32 range");
        }
        return static_cast<uint32_t>(entityBatch->entities.size());
    }();

    std::vector<uint32_t> entityGeometryBaseIndices(entityInstanceCount, 0u);
    uint32_t entityGeometryCount = 0u;
    if (entityBatch != nullptr) {
        for (uint32_t entityIndex = 0; entityIndex < entityInstanceCount; ++entityIndex) {
            entityGeometryBaseIndices[entityIndex] = entityGeometryCount;
            const auto &entity = entityBatch->entities[entityIndex];
            if (entity == nullptr) { continue; }
            if (entity->geometryCount > std::numeric_limits<uint32_t>::max() - entityGeometryCount) {
                throw std::runtime_error("HAND entity geometry count exceeds uint32 range");
            }
            entityGeometryCount += entity->geometryCount;
        }
    }

    auto renderLayerNameAt = [](const std::shared_ptr<std::vector<std::string>> &names,
                                uint32_t geometryIndex) -> std::string_view {
        if (names == nullptr || geometryIndex >= names->size()) { return {}; }
        return (*names)[geometryIndex];
    };

    auto bufferElementOffset = [](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                  const std::shared_ptr<std::vector<VkDeviceAddress>> &addresses,
                                  uint32_t geometryIndex, VkDeviceSize elementSize, uint32_t &elementOffset) {
        if (buffer == nullptr || addresses == nullptr || geometryIndex >= addresses->size()) { return false; }
        const VkDeviceAddress address = (*addresses)[geometryIndex];
        const VkDeviceAddress bufferAddress = buffer->bufferAddress();
        if (address < bufferAddress) { return false; }
        const VkDeviceAddress byteOffset = address - bufferAddress;
        if (byteOffset % elementSize != 0) { return false; }
        const uint64_t offset = byteOffset / elementSize;
        if (offset > std::numeric_limits<uint32_t>::max()) { return false; }
        elementOffset = static_cast<uint32_t>(offset);
        return true;
    };

    struct HandDraw {
        vk::GraphicsPipeline *pipeline = nullptr;
        RenderIndirectDrawMetadata metadata{};
        uint32_t indexCount = 0;
    };

    std::vector<HandDraw> draws;
    if (entityBatch != nullptr) {
        for (uint32_t entityIndex = 0; entityIndex < entityInstanceCount; ++entityIndex) {
            const auto &entity = entityBatch->entities[entityIndex];
            if (entity == nullptr || (static_cast<uint32_t>(entity->rayTracingFlag) & rayTracingMaskHand) == 0u) {
                continue;
            }
            if (entity->geometryCount == 0u) { continue; }
            if (entity->vertexCounts == nullptr || entity->indexCounts == nullptr ||
                entity->indexBufferAddresses == nullptr || entity->pbrBufferAddresses == nullptr ||
                entity->indexBuffer == nullptr || entity->pbrBuffer == nullptr) {
                throw std::runtime_error("HAND entity is missing geometry resources");
            }
            if (entity->vertexCounts->size() < entity->geometryCount ||
                entity->indexCounts->size() < entity->geometryCount ||
                entity->indexBufferAddresses->size() < entity->geometryCount ||
                entity->pbrBufferAddresses->size() < entity->geometryCount) {
                throw std::runtime_error("HAND entity geometry metadata is truncated");
            }

            for (uint32_t geometryIndex = 0; geometryIndex < entity->geometryCount; ++geometryIndex) {
                const uint32_t vertexCount = (*entity->vertexCounts)[geometryIndex];
                const uint32_t indexCount = (*entity->indexCounts)[geometryIndex];
                const uint32_t rasterIndexCount = indexCount - indexCount % 3u;
                if (vertexCount == 0u || rasterIndexCount == 0u) { continue; }

                uint32_t vertexOffset = 0u;
                uint32_t firstIndex = 0u;
                if (!bufferElementOffset(entity->pbrBuffer, entity->pbrBufferAddresses, geometryIndex,
                                         sizeof(vk::VertexFormat::EntityPBRVertex), vertexOffset) ||
                    !bufferElementOffset(entity->indexBuffer, entity->indexBufferAddresses, geometryIndex,
                                         sizeof(uint32_t), firstIndex)) {
                    throw std::runtime_error("HAND entity geometry address is invalid");
                }
                if (vertexOffset > std::numeric_limits<uint32_t>::max() - vertexCount ||
                    firstIndex > std::numeric_limits<uint32_t>::max() - indexCount ||
                    static_cast<uint64_t>(vertexOffset) + vertexCount >
                        entity->pbrBuffer->size() / sizeof(vk::VertexFormat::EntityPBRVertex) ||
                    static_cast<uint64_t>(firstIndex) + indexCount > entity->indexBuffer->size() / sizeof(uint32_t)) {
                    throw std::runtime_error("HAND entity geometry range is invalid");
                }

                const std::string_view layerName = renderLayerNameAt(entity->geometryGroupNames, geometryIndex);
                const RenderPass::ShaderVariant *variant = nullptr;
                if (auto iter = pass.shaderVariants.find(std::string(layerName)); iter != pass.shaderVariants.end()) {
                    variant = &iter->second;
                } else if (auto iter = pass.shaderVariants.find("default"); iter != pass.shaderVariants.end()) {
                    variant = &iter->second;
                } else if (auto iter = pass.shaderVariants.find("hand"); iter != pass.shaderVariants.end()) {
                    variant = &iter->second;
                }
                if (variant == nullptr || variant->pipeline == nullptr) {
                    throw std::runtime_error("HAND render pass has no pipeline for layer: " + std::string(layerName));
                }

                if (entityGeometryBaseIndices[entityIndex] > std::numeric_limits<uint32_t>::max() - geometryIndex) {
                    throw std::runtime_error("HAND entity geometry address index exceeds uint32 range");
                }
                const uint32_t geometryAddressIndex = entityGeometryBaseIndices[entityIndex] + geometryIndex;
                const uint64_t renderLayerHash = stableFnv1a64(layerName);
                if (draws.size() >= std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("HAND raster draw count exceeds uint32 range");
                }
                draws.push_back({
                    .pipeline = variant->pipeline.get(),
                    .metadata =
                        RenderIndirectDrawMetadata{
                            .objectType = 1u,
                            .objectIndex = entityIndex,
                            .geometryIndex = geometryIndex,
                            .instanceIndex = entityIndex,
                            .geometryAddressIndex = geometryAddressIndex,
                            .vertexCount = vertexCount,
                            .indexCount = rasterIndexCount,
                            .layerOrder = std::numeric_limits<uint32_t>::max(),
                            .renderLayerHashLow = static_cast<uint32_t>(renderLayerHash & 0xffffffffu),
                            .renderLayerHashHigh = static_cast<uint32_t>(renderLayerHash >> 32u),
                            .flags = renderIndirectDrawFlagEntity | renderIndirectDrawFlagHand,
                            .reserved = 0u,
                        },
                    .indexCount = rasterIndexCount,
                });
            }
        }
    }

    for (const HandDraw &draw : draws) {
        if (draw.metadata.objectType != 1u || (draw.metadata.flags & renderIndirectDrawFlagHand) == 0u) {
            throw std::runtime_error("hand raster pass produced a non-HAND draw");
        }
    }
    MCVR_PROFILE_METRIC("raster.hand_only", 1);
    MCVR_PROFILE_METRIC("raster.hand_draw_count", draws.size());

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers;
    auto addVertexIndexBarrier = [&](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer) {
        if (buffer == nullptr) { return; }
        auto trackedStateIter = context.trackedBufferStates.find(buffer.get());
        if (trackedStateIter == context.trackedBufferStates.end()) { return; }
        auto &trackedState = trackedStateIter->second;
        constexpr VkAccessFlags2 writeAccessMask =
            VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        constexpr VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        constexpr VkAccessFlags2 dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        if (trackedState.initialized && (trackedState.accessMask & writeAccessMask) == 0 &&
            (trackedState.stageMask & dstStageMask) == dstStageMask &&
            (trackedState.accessMask & dstAccessMask) == dstAccessMask) {
            return;
        }
        if (trackedState.initialized && (trackedState.accessMask & writeAccessMask) != 0) {
            bufferBarriers.push_back({
                .srcStageMask =
                    trackedState.stageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : trackedState.stageMask,
                .srcAccessMask = trackedState.accessMask,
                .dstStageMask = dstStageMask,
                .dstAccessMask = dstAccessMask,
                .srcQueueFamilyIndex = queueIndex,
                .dstQueueFamilyIndex = queueIndex,
                .buffer = buffer,
            });
        }
        trackedState.stageMask = dstStageMask;
        trackedState.accessMask = dstAccessMask;
        trackedState.initialized = true;
    };
    if (entityBatch != nullptr) {
        for (const auto &entity : entityBatch->entities) {
            if (entity == nullptr || (static_cast<uint32_t>(entity->rayTracingFlag) & rayTracingMaskHand) == 0u) {
                continue;
            }
            addVertexIndexBarrier(entity->pbrBuffer);
            addVertexIndexBarrier(entity->indexBuffer);
        }
    }
    if (!bufferBarriers.empty()) { worldCommandBuffer->barriersBufferImage(bufferBarriers, {}); }

    std::shared_ptr<vk::DeviceLocalBuffer> metadataBuffer;
    if (!draws.empty()) {
        const uint32_t swapchainImageCount = framework->swapchain()->imageCount();
        const uint64_t metadataBytes = sizeof(RenderIndirectMetadataHeader) +
                                       static_cast<uint64_t>(draws.size()) * sizeof(RenderIndirectDrawMetadata);
        if (frameIndex >= swapchainImageCount) {
            throw std::runtime_error("HAND render pass frame index exceeds swapchain image count");
        }
        if (pass.handMetadataBuffers.size() < swapchainImageCount) {
            pass.handMetadataBuffers.resize(swapchainImageCount);
        }
        auto &slot = pass.handMetadataBuffers[frameIndex];
        if (slot == nullptr || slot->size() < metadataBytes) {
            framework->frameResourceRetainer().retain(slot);
            slot = vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), true, metadataBytes,
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        metadataBuffer = slot;
        RenderIndirectMetadataHeader header{
            .drawCount = static_cast<uint32_t>(draws.size()),
            .metadataOffsetBytes = static_cast<uint32_t>(sizeof(RenderIndirectMetadataHeader)),
            .metadataStrideBytes = static_cast<uint32_t>(sizeof(RenderIndirectDrawMetadata)),
            .commandStrideBytes = static_cast<uint32_t>(sizeof(VkDrawIndirectCommand)),
            .chunkInstanceIndexBase = entityInstanceCount,
            .chunkGeometryAddressIndexBase = entityGeometryCount,
            .entityInstanceCount = entityInstanceCount,
            .entityGeometryCount = entityGeometryCount,
            .chunkInstanceCount = 0u,
            .chunkGeometrySlotCount = 0u,
        };
        metadataBuffer->uploadToStagingBuffer(&header, sizeof(header), 0);
        metadataBuffer->uploadToBuffer(worldCommandBuffer, sizeof(header), 0, 0);

        std::vector<RenderIndirectDrawMetadata> metadata;
        metadata.reserve(draws.size());
        for (const HandDraw &draw : draws) { metadata.push_back(draw.metadata); }
        const size_t metadataPayloadBytes = metadata.size() * sizeof(RenderIndirectDrawMetadata);
        metadataBuffer->uploadToStagingBuffer(metadata.data(), metadataPayloadBytes, sizeof(header));
        metadataBuffer->uploadToBuffer(worldCommandBuffer, metadataPayloadBytes, sizeof(header), sizeof(header));
        worldCommandBuffer->barriersBufferImage({{
                                                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                    .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                    .srcQueueFamilyIndex = queueIndex,
                                                    .dstQueueFamilyIndex = queueIndex,
                                                    .buffer = metadataBuffer,
                                                    .offset = 0,
                                                    .size = metadataBytes,
                                                }},
                                                {});
        auto &trackedState = context.trackedBufferStates[metadataBuffer.get()];
        trackedState.stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        trackedState.accessMask = VK_ACCESS_2_SHADER_READ_BIT;
        trackedState.initialized = true;
    }

    VkClearValue depthClear{};
    depthClear.depthStencil = {.depth = 1.0f, .stencil = 0};
    std::vector<VkClearValue> clearValues(pass.config.outputs.images.size() + 1);
    for (size_t attachmentIndex = 0; attachmentIndex < pass.config.outputs.images.size(); ++attachmentIndex) {
        auto clearIter = pass.config.clearOutputs.find(pass.config.outputs.images[attachmentIndex]);
        if (clearIter == pass.config.clearOutputs.end()) { continue; }
        for (size_t componentIndex = 0; componentIndex < clearIter->second.size(); ++componentIndex) {
            clearValues[attachmentIndex].color.uint32[componentIndex] = clearIter->second[componentIndex];
        }
    }
    clearValues.back() = depthClear;

    worldCommandBuffer->beginRenderPass({
        .renderPass = pass.renderPass,
        .framebuffer = pass.framebuffers[frameIndex],
        .renderAreaExtent = {renderWidth, renderHeight},
        .clearValues = clearValues,
    });
    worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);
    if (metadataBuffer != nullptr) {
        worldCommandBuffer->pushDescriptorBuffers(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                  renderIndirectMetadataDescriptorSet,
                                                  {{.binding = renderIndirectMetadataDescriptorBinding,
                                                    .buffer = metadataBuffer->vkBuffer(),
                                                    .size = metadataBuffer->size(),
                                                    .owner = metadataBuffer}});
    }

    vk::GraphicsPipeline *boundPipeline = nullptr;
    for (uint32_t drawIndex = 0; drawIndex < draws.size(); ++drawIndex) {
        const HandDraw &draw = draws[drawIndex];
        if (draw.pipeline != boundPipeline) {
            vkCmdBindPipeline(worldCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              draw.pipeline->vkPipeline());
            boundPipeline = draw.pipeline;
        }
        worldCommandBuffer->draw(draw.indexCount, 1u, 0u, drawIndex);
    }
    worldCommandBuffer->endRenderPass();
    for (const ResolvedImageAccess &target : resources.colorAttachmentTargets) {
        if (auto image = resolveImageResource(target.resource, frameIndex); image != nullptr) {
            image->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }
    rasterDepthImages_[frameIndex]->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
}

void MainRenderModule::renderRenderPass(RenderPass &pass, MainRenderModuleContext &context) {
    MCVR_PROFILE_SCOPE(std::string("native.MainRenderModule.renderRenderPass.") + pass.config.name);
    const std::string renderPassProfilePrefix =
        std::string("native.MainRenderModule.renderRenderPass.") + pass.config.name + ".";
    auto frameworkContext = context.frameworkContext.lock();
    if (frameworkContext == nullptr) { throw std::runtime_error("render pass has no framework context"); }
    auto framework = frameworkContext->framework.lock();
    if (framework == nullptr) { throw std::runtime_error("render pass has no framework"); }
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    if (worldCommandBuffer == nullptr) { throw std::runtime_error("render pass has no command buffer"); }
    const uint32_t frameIndex = frameworkContext->frameIndex;
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();
    const ResolvedPassResources &resources = resolvedResourcesForPass(pass);
    if (pass.rendersHandOnly) {
        renderHandOnlyPass(pass, context);
        return;
    }
    throw std::runtime_error("generic ray-tracing raster passes are disabled in the HAND-only copy: " +
                             pass.config.name);

    const bool shaderPullingIndirect = pass.config.indirectDraw.has_value();
    MCVR_PROFILE_METRIC("raster.indirect", shaderPullingIndirect ? 1 : 0);
    MCVR_PROFILE_METRIC("raster.primary", pass.rendersPrimaryRaster ? 1 : 0);

    if (frameIndex >= pass.framebuffers.size() || pass.framebuffers[frameIndex] == nullptr) { return; }
    if (rasterDepthImages_.size() <= frameIndex || rasterDepthImages_[frameIndex] == nullptr) {
        throw std::runtime_error("render pass depth attachment is unavailable: " + pass.config.name);
    }
    if (context.worldPrepareContext == nullptr) {
        throw std::runtime_error("generic render pass has no world prepare context: " + pass.config.name);
    }

    const auto &renderableChunks = context.worldPrepareContext->rasterRenderableChunks;
    const auto &chunkRenderRecords = context.worldPrepareContext->rasterChunkRenderRecords;
    const auto &chunkDrawDirtySlots = context.worldPrepareContext->rasterChunkDrawDirtySlots;
    const uint64_t chunkRenderVersion = context.worldPrepareContext->rasterChunkRenderVersion;
    const uint64_t chunkDrawDirtyStartVersion = context.worldPrepareContext->rasterChunkDrawDirtyStartVersion;
    const uint32_t chunkInstanceCount = static_cast<uint32_t>(chunkRenderRecords.size());
    const uint32_t chunkGeometrySlotCount = context.worldPrepareContext->rasterChunkGeometrySlotCount;
    auto entityBatch = context.worldPrepareContext->rasterEntityBatch;
    const uint32_t vistaInstanceCount = context.worldPrepareContext->vistaInstanceCount;
    const uint32_t entityInstanceCount =
        (entityBatch == nullptr ? 0u : static_cast<uint32_t>(entityBatch->entities.size())) + vistaInstanceCount;

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers;
    std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;

    context.trackedBufferStates.reserve(context.trackedBufferStates.size() + entityInstanceCount * 3 + 8);

    auto addVertexIndexBarrier = [&](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                     VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask) {
        if (buffer == nullptr) { return; }
        auto trackedStateIter = context.trackedBufferStates.find(buffer.get());
        if (trackedStateIter == context.trackedBufferStates.end()) { return; }
        auto &trackedState = trackedStateIter->second;
        constexpr VkAccessFlags2 writeAccessMask =
            VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        if (trackedState.initialized && (trackedState.accessMask & writeAccessMask) == 0 &&
            (trackedState.stageMask & dstStageMask) == dstStageMask &&
            (trackedState.accessMask & dstAccessMask) == dstAccessMask) {
            return;
        }

        if (trackedState.initialized && (trackedState.accessMask & writeAccessMask) != 0) {
            bufferBarriers.push_back({
                .srcStageMask =
                    trackedState.stageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : trackedState.stageMask,
                .srcAccessMask = trackedState.accessMask,
                .dstStageMask = dstStageMask,
                .dstAccessMask = dstAccessMask,
                .srcQueueFamilyIndex = queueIndex,
                .dstQueueFamilyIndex = queueIndex,
                .buffer = buffer,
            });
        }
        trackedState.stageMask = dstStageMask;
        trackedState.accessMask = dstAccessMask;
        trackedState.initialized = true;
    };

    std::vector<uint32_t> entityGeometryBaseIndices(entityInstanceCount, 0);
    uint32_t entityGeometryCount = 0;
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "geometry.inputs");
        if (entityBatch != nullptr) {
            for (uint32_t entityIndex = 0; entityIndex < entityBatch->entities.size(); entityIndex++) {
                entityGeometryBaseIndices[entityIndex] = entityGeometryCount;
                const auto &entity = entityBatch->entities[entityIndex];
                if (entity != nullptr) { entityGeometryCount += entity->geometryCount; }
            }
        }
    }
    entityGeometryCount += vistaInstanceCount;
    const VkPipelineStageFlags2 geometryBufferStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    const VkAccessFlags2 geometryBufferAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    const VkPipelineStageFlags2 indexBufferStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    const VkAccessFlags2 indexBufferAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "barriers.geometry");
        MCVR_PROFILE_METRIC("raster.tracked_buffer_count", context.trackedBufferStates.size());
        if (!context.trackedBufferStates.empty()) {
            if (entityBatch != nullptr) {
                for (const auto &entity : entityBatch->entities) {
                    if (entity == nullptr) { continue; }
                    addVertexIndexBarrier(entity->pbrBuffer, geometryBufferStageMask, geometryBufferAccessMask);
                    addVertexIndexBarrier(entity->indexBuffer, indexBufferStageMask, indexBufferAccessMask);
                }
            }
            for (const auto &chunk : renderableChunks) {
                if (chunk == nullptr) { continue; }
                addVertexIndexBarrier(chunk->pbrBuffer, geometryBufferStageMask, geometryBufferAccessMask);
                addVertexIndexBarrier(chunk->indexBuffer, indexBufferStageMask, indexBufferAccessMask);
            }
        }

        if (!bufferBarriers.empty() || !imageBarriers.empty()) {
            worldCommandBuffer->barriersBufferImage(bufferBarriers, imageBarriers);
        }
    }

    auto renderLayerNameAt = [](const std::shared_ptr<std::vector<std::string>> &names,
                                uint32_t geometryIndex) -> std::string_view {
        if (names == nullptr || geometryIndex >= names->size()) { return {}; }
        return (*names)[geometryIndex];
    };

    struct RenderLayerInfo {
        const RenderPass::ShaderVariant *variant = nullptr;
        uint32_t layerOrder = std::numeric_limits<uint32_t>::max();
        uint64_t renderLayerHash = 0;
        bool batchSortAllowed = false;
        bool drawable = false;
    };

    struct RenderGeometryCandidate {
        bool entity = false;
        uint32_t objectIndex = 0;
        uint32_t geometryIndex = 0;
        uint32_t layerInfoIndex = 0;
    };

    const bool useConfiguredLayerOrder = !pass.config.layerOrder.empty();
    const size_t layerOrderSize =
        useConfiguredLayerOrder ? pass.config.layerOrder.size() : defaultRenderLayerOrder.size();
    std::unordered_map<std::string, uint32_t> layerOrderIndices;
    const bool enableBatchSort =
        pass.config.indirectDraw.has_value() && !pass.config.indirectDraw->batchSortLayers.empty();
    std::unordered_set<std::string> batchSortLayerNames;
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "layers.build");
        layerOrderIndices.reserve(layerOrderSize);
        for (uint32_t i = 0; i < layerOrderSize; i++) {
            std::string_view layerName =
                useConfiguredLayerOrder ? std::string_view(pass.config.layerOrder[i]) : defaultRenderLayerOrder[i];
            layerOrderIndices.try_emplace(std::string(layerName), i);
        }

        if (enableBatchSort) {
            batchSortLayerNames.reserve(pass.config.indirectDraw->batchSortLayers.size());
            for (const std::string &layerName : pass.config.indirectDraw->batchSortLayers) {
                batchSortLayerNames.insert(layerName);
            }
        }
    }

    auto layerOrderFor = [&](const std::string &renderLayerName) {
        auto iter = layerOrderIndices.find(renderLayerName);
        return iter == layerOrderIndices.end() ? std::numeric_limits<uint32_t>::max() : iter->second;
    };

    std::vector<RenderGeometryCandidate> entityGeometryCandidates;
    entityGeometryCandidates.reserve(entityGeometryCount);

    std::unordered_map<std::string, uint32_t> renderLayerInfoIndices;
    renderLayerInfoIndices.reserve(64);
    std::vector<RenderLayerInfo> renderLayerInfos;
    renderLayerInfos.reserve(64);
    auto renderLayerInfoIndexFor = [&](std::string_view renderLayerName) -> uint32_t {
        std::string layerName(renderLayerName);
        auto existingIter = renderLayerInfoIndices.find(layerName);
        if (existingIter != renderLayerInfoIndices.end()) { return existingIter->second; }

        const uint32_t layerInfoIndex = static_cast<uint32_t>(renderLayerInfos.size());
        auto [insertIter, inserted] = renderLayerInfoIndices.emplace(std::move(layerName), layerInfoIndex);
        (void)inserted;
        const std::string &cachedLayerName = insertIter->first;

        auto exactShaderIter = pass.shaderVariants.find(cachedLayerName);
        const bool exactShader = exactShaderIter != pass.shaderVariants.end();
        auto defaultShaderIter = pass.shaderVariants.find("default");
        const bool defaultShader = !exactShader && defaultShaderIter != pass.shaderVariants.end();
        const RenderPass::ShaderVariant *variant = nullptr;
        if (exactShader) {
            variant = &exactShaderIter->second;
        } else if (defaultShader) {
            variant = &defaultShaderIter->second;
        }

        renderLayerInfos.push_back({
            .variant = variant,
            .layerOrder = layerOrderFor(cachedLayerName),
            .renderLayerHash = stableFnv1a64(cachedLayerName),
            .batchSortAllowed =
                enableBatchSort && batchSortLayerNames.find(cachedLayerName) != batchSortLayerNames.end(),
            .drawable = variant != nullptr,
        });
        return layerInfoIndex;
    };

    auto appendEntityGeometryCandidate = [&](uint32_t objectIndex, uint32_t geometryIndex,
                                             std::string_view renderLayerName) {
        const uint32_t layerInfoIndex = renderLayerInfoIndexFor(renderLayerName);
        RenderLayerInfo &layerInfo = renderLayerInfos[layerInfoIndex];
        if (!layerInfo.drawable) { return; }
        entityGeometryCandidates.push_back({
            .entity = true,
            .objectIndex = objectIndex,
            .geometryIndex = geometryIndex,
            .layerInfoIndex = layerInfoIndex,
        });
    };

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "entities.collectCandidates");
        if (entityBatch != nullptr) {
            for (uint32_t entityIndex = 0; entityIndex < entityBatch->entities.size(); entityIndex++) {
                const auto &entity = entityBatch->entities[entityIndex];
                if (entity == nullptr || entity->geometryGroupNames == nullptr) { continue; }
                for (uint32_t geometryIndex = 0; geometryIndex < entity->geometryCount; geometryIndex++) {
                    appendEntityGeometryCandidate(entityIndex, geometryIndex,
                                                  renderLayerNameAt(entity->geometryGroupNames, geometryIndex));
                }
            }
        }
    }
    auto bufferElementOffset = [](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                  const std::shared_ptr<std::vector<VkDeviceAddress>> &addresses,
                                  uint32_t geometryIndex, VkDeviceSize elementSize, uint32_t &elementOffset) {
        if (buffer == nullptr || addresses == nullptr || geometryIndex >= addresses->size()) { return false; }
        const VkDeviceAddress address = (*addresses)[geometryIndex];
        const VkDeviceAddress bufferAddress = buffer->bufferAddress();
        if (address < bufferAddress) { return false; }
        const VkDeviceAddress byteOffset = address - bufferAddress;
        if (byteOffset % elementSize != 0) { return false; }
        const uint64_t offset = byteOffset / elementSize;
        if (offset > std::numeric_limits<uint32_t>::max()) { return false; }
        elementOffset = static_cast<uint32_t>(offset);
        return true;
    };

    struct PreparedRenderDraw {
        vk::GraphicsPipeline *pipeline = nullptr;
        vk::DeviceLocalBuffer *pbrBuffer = nullptr;
        vk::DeviceLocalBuffer *indexBuffer = nullptr;
        RenderIndexedDrawCommand indexedCommand{};
        RenderIndirectDrawMetadata metadata{};
        uint32_t pipelineBatchKeyIndex = std::numeric_limits<uint32_t>::max();
        bool batchSortAllowed = false;
    };

    auto prepareRenderDraw =
        [&](const RenderGeometryCandidate &geometryCandidate) -> std::optional<PreparedRenderDraw> {
        const RenderLayerInfo &layerInfo = renderLayerInfos[geometryCandidate.layerInfoIndex];
        const RenderPass::ShaderVariant *variant = layerInfo.variant;
        if (variant == nullptr || variant->pipeline == nullptr) { return std::nullopt; }

        vk::DeviceLocalBuffer *pbrBuffer = nullptr;
        vk::DeviceLocalBuffer *indexBuffer = nullptr;
        const std::vector<uint32_t> *vertexCounts = nullptr;
        const std::vector<uint32_t> *indexCounts = nullptr;
        uint32_t firstIndex = 0;
        uint32_t vertexOffset = 0;
        uint32_t instanceIndex = 0;
        uint32_t firstInstance = 0;
        uint32_t geometryAddressIndex = geometryCandidate.geometryIndex;
        uint32_t drawFlags = 0;
        size_t vertexStride = sizeof(vk::VertexFormat::EntityPBRVertex);

        if (geometryCandidate.entity) {
            if (entityBatch == nullptr || geometryCandidate.objectIndex >= entityBatch->entities.size()) {
                return std::nullopt;
            }
            const auto &entity = entityBatch->entities[geometryCandidate.objectIndex];
            if (entity == nullptr || entity->vertexCounts == nullptr || entity->indexCounts == nullptr) {
                return std::nullopt;
            }
            drawFlags |= renderIndirectDrawFlagEntity;
            if ((static_cast<uint32_t>(entity->rayTracingFlag) & rayTracingMaskHand) != 0u) {
                drawFlags |= renderIndirectDrawFlagHand;
            }
            if (geometryCandidate.geometryIndex >= entity->vertexCounts->size() ||
                geometryCandidate.geometryIndex >= entity->indexCounts->size()) {
                return std::nullopt;
            }
            pbrBuffer = entity->pbrBuffer.get();
            indexBuffer = entity->indexBuffer.get();
            vertexCounts = entity->vertexCounts.get();
            indexCounts = entity->indexCounts.get();
            geometryAddressIndex =
                entityGeometryBaseIndices[geometryCandidate.objectIndex] + geometryCandidate.geometryIndex;
            if (!bufferElementOffset(entity->indexBuffer, entity->indexBufferAddresses, geometryCandidate.geometryIndex,
                                     sizeof(uint32_t), firstIndex)) {
                return std::nullopt;
            }
            if (!bufferElementOffset(entity->pbrBuffer, entity->pbrBufferAddresses, geometryCandidate.geometryIndex,
                                     sizeof(vk::VertexFormat::EntityPBRVertex), vertexOffset)) {
                return std::nullopt;
            }
            instanceIndex = geometryCandidate.objectIndex;
            firstInstance = geometryCandidate.objectIndex;
        } else {
            if (geometryCandidate.objectIndex >= renderableChunks.size() ||
                geometryCandidate.objectIndex >= chunkInstanceCount ||
                geometryCandidate.objectIndex >= chunkRenderRecords.size()) {
                return std::nullopt;
            }
            const auto &chunk = renderableChunks[geometryCandidate.objectIndex];
            if (chunk == nullptr || chunk->vertexCounts == nullptr || chunk->indexCounts == nullptr) {
                return std::nullopt;
            }
            const ChunkRenderRecord &renderRecord = chunkRenderRecords[geometryCandidate.objectIndex];
            const uint32_t renderRecordGeometryCount =
                static_cast<uint32_t>(std::max(renderRecord.positionAndGeometryCount.w, 0));
            if (geometryCandidate.geometryIndex >= chunk->vertexCounts->size() ||
                geometryCandidate.geometryIndex >= chunk->indexCounts->size() ||
                geometryCandidate.geometryIndex >= renderRecordGeometryCount) {
                return std::nullopt;
            }
            const uint64_t localGeometryAddressIndex =
                static_cast<uint64_t>(renderRecord.offsetsMaskFlags.x) + geometryCandidate.geometryIndex;
            if (localGeometryAddressIndex >= chunkGeometrySlotCount ||
                entityInstanceCount > std::numeric_limits<uint32_t>::max() - geometryCandidate.objectIndex) {
                return std::nullopt;
            }
            if (!bufferElementOffset(chunk->indexBuffer, chunk->indexBufferAddresses, geometryCandidate.geometryIndex,
                                     sizeof(uint32_t), firstIndex)) {
                return std::nullopt;
            }
            if (!bufferElementOffset(chunk->pbrBuffer, chunk->pbrBufferAddresses, geometryCandidate.geometryIndex,
                                     sizeof(vk::VertexFormat::ChunkPBRVertex), vertexOffset)) {
                return std::nullopt;
            }
            vertexStride = sizeof(vk::VertexFormat::ChunkPBRVertex);
            pbrBuffer = chunk->pbrBuffer.get();
            indexBuffer = chunk->indexBuffer.get();
            vertexCounts = chunk->vertexCounts.get();
            indexCounts = chunk->indexCounts.get();
            instanceIndex = geometryCandidate.objectIndex;
            firstInstance = entityInstanceCount + geometryCandidate.objectIndex;
            geometryAddressIndex = static_cast<uint32_t>(localGeometryAddressIndex);
        }

        if (pbrBuffer == nullptr || indexBuffer == nullptr) { return std::nullopt; }
        const uint32_t vertexCount = (*vertexCounts)[geometryCandidate.geometryIndex];
        const uint32_t indexCount = (*indexCounts)[geometryCandidate.geometryIndex];
        const uint32_t rasterIndexCount = indexCount - indexCount % 3u;
        if (rasterIndexCount == 0 || vertexCount == 0) { return std::nullopt; }
        if (vertexOffset > std::numeric_limits<uint32_t>::max() - vertexCount) { return std::nullopt; }
        if (firstIndex > std::numeric_limits<uint32_t>::max() - indexCount) { return std::nullopt; }
        if (static_cast<uint64_t>(firstIndex) + indexCount > indexBuffer->size() / sizeof(uint32_t)) {
            return std::nullopt;
        }
        if (static_cast<uint64_t>(vertexOffset) + vertexCount > pbrBuffer->size() / vertexStride) {
            return std::nullopt;
        }
        if (vertexOffset > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) { return std::nullopt; }

        PreparedRenderDraw preparedDraw;
        preparedDraw.pipeline = variant->pipeline.get();
        preparedDraw.pbrBuffer = pbrBuffer;
        preparedDraw.indexBuffer = indexBuffer;
        preparedDraw.indexedCommand = RenderIndexedDrawCommand{
            .indexCount = rasterIndexCount,
            .instanceCount = 1,
            .firstIndex = firstIndex,
            .vertexOffset = static_cast<int32_t>(vertexOffset),
            .firstInstance = firstInstance,
        };
        preparedDraw.metadata = RenderIndirectDrawMetadata{
            .objectType = geometryCandidate.entity ? 1u : 0u,
            .objectIndex = geometryCandidate.objectIndex,
            .geometryIndex = geometryCandidate.geometryIndex,
            .instanceIndex = instanceIndex,
            .geometryAddressIndex = geometryAddressIndex,
            .vertexCount = vertexCount,
            .indexCount = indexCount,
            .layerOrder = layerInfo.layerOrder,
            .renderLayerHashLow = static_cast<uint32_t>(layerInfo.renderLayerHash & 0xffffffffu),
            .renderLayerHashHigh = static_cast<uint32_t>(layerInfo.renderLayerHash >> 32u),
            .flags = drawFlags,
            .reserved = 0u,
        };
        preparedDraw.batchSortAllowed = layerInfo.batchSortAllowed;
        return preparedDraw;
    };

    struct ChunkDrawCacheKey {
        const Chunk1 *chunk = nullptr;
        uint32_t renderSlot = 0;
        int32_t chunkX = 0;
        int32_t chunkY = 0;
        int32_t chunkZ = 0;
        int64_t latestVersion = 0;
        int64_t blasVersion = 0;
        uint32_t geometryCount = 0;
        uint32_t renderRecordGeometryOffset = 0;
        uint32_t renderRecordGeometryCount = 0;
        const void *indexBufferAddresses = nullptr;
        const void *pbrBufferAddresses = nullptr;
        const void *indexBuffer = nullptr;
        const void *pbrBuffer = nullptr;
        const void *geometryGroupNames = nullptr;
        const void *vertexCounts = nullptr;
        const void *indexCounts = nullptr;
        bool shaderPullingIndirect = false;
        uint64_t passSignature = 0;

        bool operator==(const ChunkDrawCacheKey &rhs) const {
            return chunk == rhs.chunk && renderSlot == rhs.renderSlot && chunkX == rhs.chunkX && chunkY == rhs.chunkY &&
                   chunkZ == rhs.chunkZ && latestVersion == rhs.latestVersion && blasVersion == rhs.blasVersion &&
                   geometryCount == rhs.geometryCount && renderRecordGeometryOffset == rhs.renderRecordGeometryOffset &&
                   renderRecordGeometryCount == rhs.renderRecordGeometryCount &&
                   indexBufferAddresses == rhs.indexBufferAddresses && pbrBufferAddresses == rhs.pbrBufferAddresses &&
                   indexBuffer == rhs.indexBuffer && pbrBuffer == rhs.pbrBuffer &&
                   geometryGroupNames == rhs.geometryGroupNames && vertexCounts == rhs.vertexCounts &&
                   indexCounts == rhs.indexCounts && shaderPullingIndirect == rhs.shaderPullingIndirect &&
                   passSignature == rhs.passSignature;
        }
    };

    struct CachedChunkDrawEntry {
        bool valid = false;
        ChunkDrawCacheKey key{};
        uint32_t geometryOffset = 0;
        uint32_t geometryCount = 0;
        uint32_t entityInstanceCount = std::numeric_limits<uint32_t>::max();
        uint32_t entityGeometryCount = std::numeric_limits<uint32_t>::max();
        std::vector<PreparedRenderDraw> draws;
        std::vector<uint32_t> drawIndexByGeometryIndex;
        std::vector<uint32_t> sortedDrawIndices;
        std::vector<uint32_t> shadowSolidMetadataIndices;
        std::vector<uint32_t> shadowAlphaOpaqueMetadataIndices;
    };

    struct FrameIndirectUploadState {
        const vk::DeviceLocalBuffer *metadataBuffer = nullptr;
        uint64_t metadataBytes = 0;
        uint64_t metadataVersion = 0;
        uint64_t headerVersion = 0;
        uint64_t chunkMetadataVersion = 0;
        uint64_t entityMetadataHash = 0;
        uint32_t chunkMetadataCount = 0;
        uint32_t chunkLeadingMetadataCount = 0;
        uint32_t entityMetadataCount = 0;
        uint32_t entityLeadingMetadataCount = 0;
        bool metadataValid = false;
    };

    struct MetadataIndexRange {
        uint32_t offset = 0;
        uint32_t count = 0;
    };

    struct ChunkMetadataDirtyUpdate {
        uint64_t version = 0;
        std::vector<MetadataIndexRange> primaryRanges;
        std::vector<MetadataIndexRange> shadowSolidRanges;
        std::vector<MetadataIndexRange> shadowAlphaOpaqueRanges;
    };

    struct RenderPassDrawMetadataCache {
        uint64_t passSignature = 0;
        uint64_t chunkRenderVersion = std::numeric_limits<uint64_t>::max();
        uint32_t chunkEntityInstanceCount = std::numeric_limits<uint32_t>::max();
        uint32_t chunkEntityGeometryCount = std::numeric_limits<uint32_t>::max();
        std::vector<CachedChunkDrawEntry> chunkEntries;
        std::unordered_map<RenderPipelineBatchKey, uint32_t, RenderPipelineBatchKeyHash> pipelineBatchKeyIndices;
        std::vector<RenderPipelineBatchKey> pipelineBatchKeys;
        std::vector<size_t> layerBucketCounts;
        std::vector<size_t> layerBucketOffsets;
        std::vector<size_t> layerBucketCursors;
        std::vector<PreparedRenderDraw> layerSortedDraws;
        std::vector<uint32_t> batchGroupIndices;
        std::vector<uint32_t> batchTouchedKeys;
        std::vector<uint32_t> batchCandidateGroupIndices;
        std::vector<size_t> batchGroupCounts;
        std::vector<size_t> batchGroupOffsets;
        std::vector<size_t> batchGroupCursors;
        std::vector<PreparedRenderDraw> batchGroupedDraws;
        std::vector<PreparedRenderDraw> preparedDraws;
        std::vector<PreparedRenderDraw> entityPreparedDraws;
        std::vector<RenderIndirectDrawBatch> indirectDrawBatches;
        std::vector<RenderIndirectDrawBatch> chunkIndirectDrawBatches;
        bool chunkIndirectDrawBatchesDirty = true;
        std::vector<PreparedRenderDraw> chunkPreparedDraws;
        bool chunkPreparedDrawsDirty = true;
        std::vector<PreparedRenderDraw> sortedChunkPreparedDraws;
        bool sortedChunkPreparedDrawsDirty = true;
        uint64_t chunkMetadataVersion = 0;
        std::vector<RenderIndirectDrawMetadata> chunkPrimaryMetadata;
        std::vector<RenderIndirectDrawMetadata> chunkShadowSolidMetadata;
        std::vector<RenderIndirectDrawMetadata> chunkShadowAlphaOpaqueMetadata;
        bool chunkMetadataDirty = true;
        std::vector<ChunkMetadataDirtyUpdate> chunkMetadataDirtyUpdates;
        std::vector<RenderIndirectDrawMetadata> entityPrimaryMetadata;
        std::vector<RenderIndirectDrawMetadata> entityShadowSolidMetadata;
        std::vector<RenderIndirectDrawMetadata> entityShadowAlphaOpaqueMetadata;
        std::vector<uint8_t> indirectMetadataUpload;
        std::vector<uint8_t> shadowMetadataUpload;
        std::vector<FrameIndirectUploadState> frameUploadStates;
        std::vector<FrameIndirectUploadState> shadowFrameUploadStates;
    };

    auto hashCombine64 = [](uint64_t &seed, uint64_t value) {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6u) + (seed >> 2u);
    };
    auto pointerHash64 = [](const void *ptr) -> uint64_t {
        return static_cast<uint64_t>(std::hash<const void *>{}(ptr));
    };

    uint64_t passDrawCacheSignature = 0;
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "cache.signature");
        passDrawCacheSignature = stableFnv1a64(pass.config.name);
        hashCombine64(passDrawCacheSignature, shaderPullingIndirect ? 1u : 0u);
        hashCombine64(passDrawCacheSignature, enableBatchSort ? 1u : 0u);
        hashCombine64(passDrawCacheSignature, sizeof(RenderIndirectDrawMetadata));
        hashCombine64(passDrawCacheSignature, sizeof(VkDrawIndirectCommand));
        hashCombine64(passDrawCacheSignature, layerOrderSize);
        for (uint32_t i = 0; i < layerOrderSize; i++) {
            std::string_view layerName =
                useConfiguredLayerOrder ? std::string_view(pass.config.layerOrder[i]) : defaultRenderLayerOrder[i];
            hashCombine64(passDrawCacheSignature, i);
            hashCombine64(passDrawCacheSignature, stableFnv1a64(layerName));
        }
        hashCombine64(passDrawCacheSignature, pass.config.indirectDraw.has_value() ? renderIndirectWorkgroupSize : 0u);
        if (pass.config.indirectDraw.has_value()) {
            for (const std::string &layerName : pass.config.indirectDraw->batchSortLayers) {
                hashCombine64(passDrawCacheSignature, stableFnv1a64(layerName));
            }
        }
        uint64_t shaderVariantsSignature = 0;
        for (const auto &[layerName, variant] : pass.shaderVariants) {
            uint64_t variantSignature = stableFnv1a64(layerName);
            hashCombine64(variantSignature, pointerHash64(variant.pipeline.get()));
            hashCombine64(variantSignature, pointerHash64(variant.vertexShader.get()));
            hashCombine64(variantSignature, pointerHash64(variant.fragmentShader.get()));
            shaderVariantsSignature ^= variantSignature;
        }
        hashCombine64(passDrawCacheSignature, shaderVariantsSignature);
    }

    static std::unordered_map<const RenderPass *, RenderPassDrawMetadataCache> renderPassDrawMetadataCaches;
    RenderPassDrawMetadataCache &drawMetadataCache = renderPassDrawMetadataCaches[&pass];
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "cache.state");
        if (drawMetadataCache.passSignature != passDrawCacheSignature) {
            drawMetadataCache.passSignature = passDrawCacheSignature;
            drawMetadataCache.chunkRenderVersion = std::numeric_limits<uint64_t>::max();
            drawMetadataCache.chunkEntityInstanceCount = std::numeric_limits<uint32_t>::max();
            drawMetadataCache.chunkEntityGeometryCount = std::numeric_limits<uint32_t>::max();
            drawMetadataCache.chunkEntries.clear();
            drawMetadataCache.pipelineBatchKeyIndices.clear();
            drawMetadataCache.pipelineBatchKeys.clear();
            drawMetadataCache.layerBucketCounts.clear();
            drawMetadataCache.layerBucketOffsets.clear();
            drawMetadataCache.layerBucketCursors.clear();
            drawMetadataCache.layerSortedDraws.clear();
            drawMetadataCache.batchGroupIndices.clear();
            drawMetadataCache.batchTouchedKeys.clear();
            drawMetadataCache.batchCandidateGroupIndices.clear();
            drawMetadataCache.batchGroupCounts.clear();
            drawMetadataCache.batchGroupOffsets.clear();
            drawMetadataCache.batchGroupCursors.clear();
            drawMetadataCache.batchGroupedDraws.clear();
            drawMetadataCache.preparedDraws.clear();
            drawMetadataCache.entityPreparedDraws.clear();
            drawMetadataCache.indirectDrawBatches.clear();
            drawMetadataCache.chunkIndirectDrawBatches.clear();
            drawMetadataCache.chunkIndirectDrawBatchesDirty = true;
            drawMetadataCache.chunkPreparedDraws.clear();
            drawMetadataCache.chunkPreparedDrawsDirty = true;
            drawMetadataCache.sortedChunkPreparedDraws.clear();
            drawMetadataCache.sortedChunkPreparedDrawsDirty = true;
            drawMetadataCache.chunkMetadataVersion = 0;
            drawMetadataCache.chunkPrimaryMetadata.clear();
            drawMetadataCache.chunkShadowSolidMetadata.clear();
            drawMetadataCache.chunkShadowAlphaOpaqueMetadata.clear();
            drawMetadataCache.chunkMetadataDirty = true;
            drawMetadataCache.chunkMetadataDirtyUpdates.clear();
            drawMetadataCache.entityPrimaryMetadata.clear();
            drawMetadataCache.entityShadowSolidMetadata.clear();
            drawMetadataCache.entityShadowAlphaOpaqueMetadata.clear();
            drawMetadataCache.indirectMetadataUpload.clear();
            drawMetadataCache.shadowMetadataUpload.clear();
            drawMetadataCache.frameUploadStates.clear();
            drawMetadataCache.shadowFrameUploadStates.clear();
        }
        if (drawMetadataCache.chunkEntries.size() != renderableChunks.size()) {
            drawMetadataCache.chunkRenderVersion = std::numeric_limits<uint64_t>::max();
            drawMetadataCache.chunkPreparedDrawsDirty = true;
            drawMetadataCache.chunkIndirectDrawBatchesDirty = true;
            drawMetadataCache.chunkMetadataDirty = true;
        }
        drawMetadataCache.chunkEntries.resize(renderableChunks.size());
    }

    auto &pipelineBatchKeyIndices = drawMetadataCache.pipelineBatchKeyIndices;
    auto &pipelineBatchKeys = drawMetadataCache.pipelineBatchKeys;
    if (enableBatchSort) {
        const size_t pipelineBatchKeyCapacity = std::max<size_t>(pipelineBatchKeys.size(), renderLayerInfos.size() + 8);
        pipelineBatchKeyIndices.reserve(pipelineBatchKeyCapacity);
        pipelineBatchKeys.reserve(pipelineBatchKeyCapacity);
    }
    auto pipelineBatchKeyIndexFor = [&](const PreparedRenderDraw &preparedDraw) {
        RenderPipelineBatchKey pipelineBatchKey{
            .pipeline = preparedDraw.pipeline,
            .pbrBuffer = shaderPullingIndirect ? nullptr : preparedDraw.pbrBuffer,
            .indexBuffer = shaderPullingIndirect ? nullptr : preparedDraw.indexBuffer,
        };
        auto [pipelineBatchKeyIter, inserted] =
            pipelineBatchKeyIndices.emplace(pipelineBatchKey, static_cast<uint32_t>(pipelineBatchKeys.size()));
        if (inserted) { pipelineBatchKeys.push_back(pipelineBatchKey); }
        return pipelineBatchKeyIter->second;
    };

    auto makeChunkDrawCacheKey =
        [&](uint32_t chunkIndex, const std::shared_ptr<ChunkRenderData> &chunk) -> std::optional<ChunkDrawCacheKey> {
        if (chunk == nullptr || chunk->geometryGroupNames == nullptr || chunk->vertexCounts == nullptr ||
            chunk->indexCounts == nullptr || chunkIndex >= chunkRenderRecords.size()) {
            return std::nullopt;
        }

        const ChunkRenderRecord &renderRecord = chunkRenderRecords[chunkIndex];
        const uint32_t renderRecordGeometryOffset = renderRecord.offsetsMaskFlags.x;
        const uint32_t renderRecordGeometryCount =
            static_cast<uint32_t>(std::max(renderRecord.positionAndGeometryCount.w, 0));
        if (static_cast<uint64_t>(renderRecordGeometryOffset) + renderRecordGeometryCount > chunkGeometrySlotCount) {
            return std::nullopt;
        }

        return ChunkDrawCacheKey{
            .chunk = chunk->sourceChunk,
            .renderSlot = chunkIndex,
            .chunkX = chunk->x,
            .chunkY = chunk->y,
            .chunkZ = chunk->z,
            .latestVersion = chunk->latestVersion,
            .blasVersion = chunk->blasVersion,
            .geometryCount = chunk->geometryCount,
            .renderRecordGeometryOffset = renderRecordGeometryOffset,
            .renderRecordGeometryCount = renderRecordGeometryCount,
            .indexBufferAddresses = chunk->indexBufferAddresses.get(),
            .pbrBufferAddresses = chunk->pbrBufferAddresses.get(),
            .indexBuffer = chunk->indexBuffer.get(),
            .pbrBuffer = chunk->pbrBuffer.get(),
            .geometryGroupNames = chunk->geometryGroupNames.get(),
            .vertexCounts = chunk->vertexCounts.get(),
            .indexCounts = chunk->indexCounts.get(),
            .shaderPullingIndirect = shaderPullingIndirect,
            .passSignature = passDrawCacheSignature,
        };
    };

    auto updateChunkDrawCacheEntryFrameOffsets = [&](CachedChunkDrawEntry &entry) {
        if (!entry.valid) { return; }
        if (entry.key.renderSlot > std::numeric_limits<uint32_t>::max() - entityInstanceCount) {
            entry.valid = false;
            return;
        }
        const uint32_t instanceIndex = entityInstanceCount + entry.key.renderSlot;
        for (PreparedRenderDraw &draw : entry.draws) { draw.indexedCommand.firstInstance = instanceIndex; }
        entry.entityInstanceCount = entityInstanceCount;
        entry.entityGeometryCount = entityGeometryCount;
    };

    auto rebuildChunkDrawCacheEntry = [&](CachedChunkDrawEntry &entry, uint32_t chunkIndex,
                                          const std::shared_ptr<ChunkRenderData> &chunk, const ChunkDrawCacheKey &key) {
        entry.valid = false;
        entry.key = key;
        entry.geometryOffset = key.renderRecordGeometryOffset;
        entry.geometryCount = std::min(key.geometryCount, key.renderRecordGeometryCount);
        entry.entityInstanceCount = std::numeric_limits<uint32_t>::max();
        entry.entityGeometryCount = std::numeric_limits<uint32_t>::max();
        entry.draws.clear();
        entry.draws.reserve(chunk->geometryCount);
        constexpr uint32_t invalidDrawIndex = std::numeric_limits<uint32_t>::max();
        entry.drawIndexByGeometryIndex.assign(chunk->geometryCount, invalidDrawIndex);
        entry.sortedDrawIndices.clear();
        entry.shadowSolidMetadataIndices.clear();
        entry.shadowAlphaOpaqueMetadataIndices.clear();

        const uint32_t drawableGeometryCount = std::min(chunk->geometryCount, key.renderRecordGeometryCount);
        for (uint32_t geometryIndex = 0; geometryIndex < drawableGeometryCount; geometryIndex++) {
            const uint32_t layerInfoIndex =
                renderLayerInfoIndexFor(renderLayerNameAt(chunk->geometryGroupNames, geometryIndex));
            const RenderLayerInfo &layerInfo = renderLayerInfos[layerInfoIndex];
            if (!layerInfo.drawable) { continue; }

            RenderGeometryCandidate geometryCandidate{
                .entity = false,
                .objectIndex = chunkIndex,
                .geometryIndex = geometryIndex,
                .layerInfoIndex = layerInfoIndex,
            };
            std::optional<PreparedRenderDraw> preparedDraw = prepareRenderDraw(geometryCandidate);
            if (!preparedDraw.has_value()) { continue; }

            preparedDraw->indexedCommand.firstInstance = 0;
            preparedDraw->metadata.objectIndex = chunkIndex;
            preparedDraw->metadata.instanceIndex = chunkIndex;
            preparedDraw->metadata.geometryAddressIndex = key.renderRecordGeometryOffset + geometryIndex;
            if (enableBatchSort) { preparedDraw->pipelineBatchKeyIndex = pipelineBatchKeyIndexFor(*preparedDraw); }
            entry.drawIndexByGeometryIndex[geometryIndex] = static_cast<uint32_t>(entry.draws.size());
            entry.draws.push_back(std::move(*preparedDraw));
        }
        entry.valid = true;
        updateChunkDrawCacheEntryFrameOffsets(entry);
    };

    auto addMetadataDirtyRange = [](std::vector<MetadataIndexRange> &ranges, uint32_t index) {
        if (index == std::numeric_limits<uint32_t>::max()) { return; }
        const uint64_t indexEnd = static_cast<uint64_t>(index) + 1u;
        for (MetadataIndexRange &range : ranges) {
            const uint64_t rangeBegin = range.offset;
            const uint64_t rangeEnd = static_cast<uint64_t>(range.offset) + range.count;
            if (indexEnd < rangeBegin || static_cast<uint64_t>(index) > rangeEnd) { continue; }
            const uint32_t newBegin = std::min(range.offset, index);
            const uint64_t newEnd = std::max(rangeEnd, indexEnd);
            range.offset = newBegin;
            range.count = static_cast<uint32_t>(newEnd - newBegin);
            return;
        }
        ranges.push_back({.offset = index, .count = 1});
    };

    auto normalizeMetadataDirtyRanges = [](std::vector<MetadataIndexRange> &ranges) {
        if (ranges.size() < 2) { return; }
        std::sort(ranges.begin(), ranges.end(),
                  [](const MetadataIndexRange &lhs, const MetadataIndexRange &rhs) { return lhs.offset < rhs.offset; });
        size_t writeIndex = 0;
        for (size_t readIndex = 1; readIndex < ranges.size(); readIndex++) {
            MetadataIndexRange &dst = ranges[writeIndex];
            const MetadataIndexRange &src = ranges[readIndex];
            const uint64_t dstEnd = static_cast<uint64_t>(dst.offset) + dst.count;
            const uint64_t srcEnd = static_cast<uint64_t>(src.offset) + src.count;
            if (src.offset <= dstEnd) {
                dst.count = static_cast<uint32_t>(std::max(dstEnd, srcEnd) - dst.offset);
            } else {
                writeIndex++;
                if (writeIndex != readIndex) { ranges[writeIndex] = src; }
            }
        }
        ranges.resize(writeIndex + 1);
    };

    auto bumpChunkMetadataVersion = [&]() {
        drawMetadataCache.chunkMetadataVersion++;
        if (drawMetadataCache.chunkMetadataVersion == 0) { drawMetadataCache.chunkMetadataVersion = 1; }
    };

    auto appendChunkMetadataDirtyUpdate = [&](ChunkMetadataDirtyUpdate &&dirtyUpdate) {
        normalizeMetadataDirtyRanges(dirtyUpdate.primaryRanges);
        normalizeMetadataDirtyRanges(dirtyUpdate.shadowSolidRanges);
        normalizeMetadataDirtyRanges(dirtyUpdate.shadowAlphaOpaqueRanges);
        if (dirtyUpdate.primaryRanges.empty() && dirtyUpdate.shadowSolidRanges.empty() &&
            dirtyUpdate.shadowAlphaOpaqueRanges.empty()) {
            return;
        }
        dirtyUpdate.version = drawMetadataCache.chunkMetadataVersion;
        drawMetadataCache.chunkMetadataDirtyUpdates.push_back(std::move(dirtyUpdate));
        constexpr size_t maxRetainedDirtyUpdates = 16;
        if (drawMetadataCache.chunkMetadataDirtyUpdates.size() > maxRetainedDirtyUpdates) {
            drawMetadataCache.chunkMetadataDirtyUpdates.erase(drawMetadataCache.chunkMetadataDirtyUpdates.begin(),
                                                              drawMetadataCache.chunkMetadataDirtyUpdates.end() -
                                                                  maxRetainedDirtyUpdates);
        }
    };

    auto drawKeepsSortedSlot = [&](const PreparedRenderDraw &oldDraw, const PreparedRenderDraw &newDraw) {
        const RenderIndirectDrawMetadata &oldMetadata = oldDraw.metadata;
        const RenderIndirectDrawMetadata &newMetadata = newDraw.metadata;
        return oldDraw.pipeline == newDraw.pipeline && oldDraw.pipelineBatchKeyIndex == newDraw.pipelineBatchKeyIndex &&
               oldDraw.batchSortAllowed == newDraw.batchSortAllowed &&
               oldMetadata.objectType == newMetadata.objectType && oldMetadata.objectIndex == newMetadata.objectIndex &&
               oldMetadata.geometryIndex == newMetadata.geometryIndex &&
               oldMetadata.layerOrder == newMetadata.layerOrder &&
               oldMetadata.renderLayerHashLow == newMetadata.renderLayerHashLow &&
               oldMetadata.renderLayerHashHigh == newMetadata.renderLayerHashHigh &&
               oldMetadata.flags == newMetadata.flags &&
               canCastDirectionalShadow(oldMetadata) == canCastDirectionalShadow(newMetadata) &&
               isSolidFastDirectionalShadowLayer(oldMetadata) == isSolidFastDirectionalShadowLayer(newMetadata);
    };

    auto tryReplaceChunkDrawCacheEntryInPlace = [&](CachedChunkDrawEntry &entry, CachedChunkDrawEntry &&newEntry) {
        constexpr uint32_t invalidDrawIndex = std::numeric_limits<uint32_t>::max();
        if (!shaderPullingIndirect || !entry.valid || !newEntry.valid ||
            drawMetadataCache.sortedChunkPreparedDrawsDirty || drawMetadataCache.chunkMetadataDirty ||
            entry.draws.size() != newEntry.draws.size() || entry.sortedDrawIndices.size() != entry.draws.size() ||
            entry.shadowSolidMetadataIndices.size() != entry.draws.size() ||
            entry.shadowAlphaOpaqueMetadataIndices.size() != entry.draws.size() ||
            drawMetadataCache.chunkPrimaryMetadata.size() != drawMetadataCache.sortedChunkPreparedDraws.size()) {
            return false;
        }

        for (size_t drawIndex = 0; drawIndex < entry.draws.size(); drawIndex++) {
            const PreparedRenderDraw &oldDraw = entry.draws[drawIndex];
            const PreparedRenderDraw &newDraw = newEntry.draws[drawIndex];
            const uint32_t sortedDrawIndex = entry.sortedDrawIndices[drawIndex];
            if (sortedDrawIndex == invalidDrawIndex ||
                sortedDrawIndex >= drawMetadataCache.sortedChunkPreparedDraws.size() ||
                sortedDrawIndex >= drawMetadataCache.chunkPrimaryMetadata.size() ||
                !drawKeepsSortedSlot(oldDraw, newDraw)) {
                return false;
            }

            const PreparedRenderDraw &sortedDraw = drawMetadataCache.sortedChunkPreparedDraws[sortedDrawIndex];
            if (sortedDraw.metadata.objectType != 0u ||
                sortedDraw.metadata.objectIndex != oldDraw.metadata.objectIndex ||
                sortedDraw.metadata.geometryIndex != oldDraw.metadata.geometryIndex) {
                return false;
            }

            const bool castsShadow = pass.rendersPrimaryRaster && canCastDirectionalShadow(oldDraw.metadata);
            const bool solidShadow = castsShadow && isSolidFastDirectionalShadowLayer(oldDraw.metadata);
            const bool alphaOpaqueShadow = castsShadow && !solidShadow;
            const uint32_t solidMetadataIndex = entry.shadowSolidMetadataIndices[drawIndex];
            const uint32_t alphaOpaqueMetadataIndex = entry.shadowAlphaOpaqueMetadataIndices[drawIndex];
            if (solidShadow) {
                if (solidMetadataIndex == invalidDrawIndex ||
                    solidMetadataIndex >= drawMetadataCache.chunkShadowSolidMetadata.size()) {
                    return false;
                }
            } else if (solidMetadataIndex != invalidDrawIndex) {
                return false;
            }
            if (alphaOpaqueShadow) {
                if (alphaOpaqueMetadataIndex == invalidDrawIndex ||
                    alphaOpaqueMetadataIndex >= drawMetadataCache.chunkShadowAlphaOpaqueMetadata.size()) {
                    return false;
                }
            } else if (alphaOpaqueMetadataIndex != invalidDrawIndex) {
                return false;
            }
        }

        newEntry.sortedDrawIndices = entry.sortedDrawIndices;
        newEntry.shadowSolidMetadataIndices = entry.shadowSolidMetadataIndices;
        newEntry.shadowAlphaOpaqueMetadataIndices = entry.shadowAlphaOpaqueMetadataIndices;

        ChunkMetadataDirtyUpdate dirtyUpdate;
        for (size_t drawIndex = 0; drawIndex < newEntry.draws.size(); drawIndex++) {
            const PreparedRenderDraw &newDraw = newEntry.draws[drawIndex];
            const uint32_t sortedDrawIndex = newEntry.sortedDrawIndices[drawIndex];
            drawMetadataCache.sortedChunkPreparedDraws[sortedDrawIndex] = newDraw;
            drawMetadataCache.chunkPrimaryMetadata[sortedDrawIndex] = newDraw.metadata;
            addMetadataDirtyRange(dirtyUpdate.primaryRanges, sortedDrawIndex);

            const uint32_t solidMetadataIndex = newEntry.shadowSolidMetadataIndices[drawIndex];
            if (solidMetadataIndex != invalidDrawIndex) {
                drawMetadataCache.chunkShadowSolidMetadata[solidMetadataIndex] = newDraw.metadata;
                addMetadataDirtyRange(dirtyUpdate.shadowSolidRanges, solidMetadataIndex);
            }
            const uint32_t alphaOpaqueMetadataIndex = newEntry.shadowAlphaOpaqueMetadataIndices[drawIndex];
            if (alphaOpaqueMetadataIndex != invalidDrawIndex) {
                drawMetadataCache.chunkShadowAlphaOpaqueMetadata[alphaOpaqueMetadataIndex] = newDraw.metadata;
                addMetadataDirtyRange(dirtyUpdate.shadowAlphaOpaqueRanges, alphaOpaqueMetadataIndex);
            }
        }

        entry = std::move(newEntry);
        bumpChunkMetadataVersion();
        appendChunkMetadataDirtyUpdate(std::move(dirtyUpdate));
        return true;
    };

    auto &preparedDraws = drawMetadataCache.preparedDraws;
    preparedDraws.clear();
    preparedDraws.reserve(std::max(renderableChunks.size() * 2, drawMetadataCache.chunkPreparedDraws.size()) +
                          entityGeometryCount);
    const std::vector<PreparedRenderDraw> *drawsForRender = &preparedDraws;
    auto &entityPreparedDraws = drawMetadataCache.entityPreparedDraws;
    entityPreparedDraws.clear();
    entityPreparedDraws.reserve(entityGeometryCount);
    auto appendEntityPreparedDraw = [&](PreparedRenderDraw preparedDraw) {
        if (enableBatchSort && preparedDraw.pipelineBatchKeyIndex == std::numeric_limits<uint32_t>::max()) {
            preparedDraw.pipelineBatchKeyIndex = pipelineBatchKeyIndexFor(preparedDraw);
        }
        entityPreparedDraws.push_back(std::move(preparedDraw));
    };

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "chunks.prepareDraws");
        bool chunkPreparedDrawsDirty = drawMetadataCache.chunkPreparedDrawsDirty;
        const bool chunkEntityOffsetsAffectCachedDraws = !shaderPullingIndirect;
        const bool chunkRenderCacheDirty = drawMetadataCache.chunkRenderVersion != chunkRenderVersion ||
                                           drawMetadataCache.chunkEntries.size() != renderableChunks.size();
        if (chunkRenderCacheDirty) {
            const bool chunkEntityOffsetsDirty = drawMetadataCache.chunkEntityInstanceCount != entityInstanceCount ||
                                                 drawMetadataCache.chunkEntityGeometryCount != entityGeometryCount;
            const bool canUseDirtySlots = drawMetadataCache.chunkEntries.size() == renderableChunks.size() &&
                                          drawMetadataCache.chunkRenderVersion == chunkDrawDirtyStartVersion &&
                                          (!chunkEntityOffsetsDirty || !chunkEntityOffsetsAffectCachedDraws) &&
                                          !chunkDrawDirtySlots.empty();
            uint32_t checkedChunkCacheEntries = 0;
            uint32_t rebuiltChunkCacheEntries = 0;
            uint32_t invalidatedChunkCacheEntries = 0;
            uint32_t offsetUpdatedChunkCacheEntries = 0;
            uint32_t incrementallyReplacedChunkCacheEntries = 0;
            auto processChunkDrawCacheSlot = [&](uint32_t chunkIndex) {
                checkedChunkCacheEntries++;
                const auto &chunk = renderableChunks[chunkIndex];
                std::optional<ChunkDrawCacheKey> key = makeChunkDrawCacheKey(chunkIndex, chunk);
                if (!key.has_value()) {
                    CachedChunkDrawEntry &entry = drawMetadataCache.chunkEntries[chunkIndex];
                    if (entry.valid) {
                        entry.valid = false;
                        chunkPreparedDrawsDirty = true;
                        invalidatedChunkCacheEntries++;
                    }
                    return;
                }

                CachedChunkDrawEntry &entry = drawMetadataCache.chunkEntries[chunkIndex];
                if (!entry.valid || !(entry.key == *key)) {
                    if (entry.valid) {
                        CachedChunkDrawEntry newEntry;
                        rebuildChunkDrawCacheEntry(newEntry, chunkIndex, chunk, *key);
                        if (tryReplaceChunkDrawCacheEntryInPlace(entry, std::move(newEntry))) {
                            incrementallyReplacedChunkCacheEntries++;
                        } else {
                            rebuildChunkDrawCacheEntry(entry, chunkIndex, chunk, *key);
                            chunkPreparedDrawsDirty = true;
                        }
                    } else {
                        rebuildChunkDrawCacheEntry(entry, chunkIndex, chunk, *key);
                        chunkPreparedDrawsDirty = true;
                    }
                    rebuiltChunkCacheEntries++;
                } else if (entry.entityInstanceCount != entityInstanceCount ||
                           entry.entityGeometryCount != entityGeometryCount) {
                    updateChunkDrawCacheEntryFrameOffsets(entry);
                    if (!entry.valid || chunkEntityOffsetsAffectCachedDraws) { chunkPreparedDrawsDirty = true; }
                    offsetUpdatedChunkCacheEntries++;
                }
            };

            if (canUseDirtySlots) {
                for (uint32_t chunkIndex : chunkDrawDirtySlots) {
                    if (chunkIndex >= renderableChunks.size()) {
                        chunkPreparedDrawsDirty = true;
                        continue;
                    }
                    processChunkDrawCacheSlot(chunkIndex);
                }
            } else {
                for (uint32_t chunkIndex = 0; chunkIndex < renderableChunks.size(); chunkIndex++) {
                    processChunkDrawCacheSlot(chunkIndex);
                }
            }
            MCVR_PROFILE_METRIC("raster.chunk_cache_dirty_slot_path", canUseDirtySlots ? 1 : 0);
            MCVR_PROFILE_METRIC("raster.chunk_cache_dirty_slot_count", chunkDrawDirtySlots.size());
            MCVR_PROFILE_METRIC("raster.chunk_cache_checked_count", checkedChunkCacheEntries);
            MCVR_PROFILE_METRIC("raster.chunk_cache_rebuilt_count", rebuiltChunkCacheEntries);
            MCVR_PROFILE_METRIC("raster.chunk_cache_invalidated_count", invalidatedChunkCacheEntries);
            MCVR_PROFILE_METRIC("raster.chunk_cache_offset_updated_count", offsetUpdatedChunkCacheEntries);
            MCVR_PROFILE_METRIC("raster.chunk_cache_incremental_replace_count", incrementallyReplacedChunkCacheEntries);
            drawMetadataCache.chunkRenderVersion = chunkRenderVersion;
            drawMetadataCache.chunkEntityInstanceCount = entityInstanceCount;
            drawMetadataCache.chunkEntityGeometryCount = entityGeometryCount;
        } else if (drawMetadataCache.chunkEntityInstanceCount != entityInstanceCount ||
                   drawMetadataCache.chunkEntityGeometryCount != entityGeometryCount) {
            uint32_t offsetUpdatedChunkCacheEntries = 0;
            if (chunkEntityOffsetsAffectCachedDraws) {
                for (CachedChunkDrawEntry &entry : drawMetadataCache.chunkEntries) {
                    const uint32_t previousEntityInstanceCount = entry.entityInstanceCount;
                    const uint32_t previousEntityGeometryCount = entry.entityGeometryCount;
                    updateChunkDrawCacheEntryFrameOffsets(entry);
                    if (!entry.valid) {
                        chunkPreparedDrawsDirty = true;
                    } else if (previousEntityInstanceCount != entry.entityInstanceCount ||
                               previousEntityGeometryCount != entry.entityGeometryCount) {
                        chunkPreparedDrawsDirty = true;
                        offsetUpdatedChunkCacheEntries++;
                    }
                }
            }
            MCVR_PROFILE_METRIC("raster.chunk_cache_offset_update_only", 1);
            MCVR_PROFILE_METRIC("raster.chunk_cache_offset_updated_count", offsetUpdatedChunkCacheEntries);
            drawMetadataCache.chunkEntityInstanceCount = entityInstanceCount;
            drawMetadataCache.chunkEntityGeometryCount = entityGeometryCount;
        }
        auto &chunkPreparedDraws = drawMetadataCache.chunkPreparedDraws;
        if (chunkPreparedDrawsDirty) {
            size_t chunkPreparedDrawCount = 0;
            for (const CachedChunkDrawEntry &entry : drawMetadataCache.chunkEntries) {
                if (entry.valid) { chunkPreparedDrawCount += entry.draws.size(); }
            }
            chunkPreparedDraws.clear();
            chunkPreparedDraws.reserve(chunkPreparedDrawCount);
            for (const CachedChunkDrawEntry &entry : drawMetadataCache.chunkEntries) {
                if (!entry.valid || entry.draws.empty()) { continue; }
                chunkPreparedDraws.insert(chunkPreparedDraws.end(), entry.draws.begin(), entry.draws.end());
            }
            drawMetadataCache.chunkPreparedDrawsDirty = false;
            drawMetadataCache.sortedChunkPreparedDrawsDirty = true;
            drawMetadataCache.chunkIndirectDrawBatchesDirty = true;
            drawMetadataCache.chunkMetadataDirty = true;
        }
    }

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "entities.prepareDraws");
        for (const RenderGeometryCandidate &geometryCandidate : entityGeometryCandidates) {
            std::optional<PreparedRenderDraw> preparedDraw = prepareRenderDraw(geometryCandidate);
            if (!preparedDraw.has_value()) { continue; }
            appendEntityPreparedDraw(std::move(*preparedDraw));
        }
    }

    auto sortPreparedDrawList = [&](std::vector<PreparedRenderDraw> &drawList) {
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "sort.layer");
            if (drawList.empty()) { return; }
            const size_t unorderedLayerBucket = layerOrderSize;
            auto &layerBucketCounts = drawMetadataCache.layerBucketCounts;
            auto &layerBucketOffsets = drawMetadataCache.layerBucketOffsets;
            auto &layerBucketCursors = drawMetadataCache.layerBucketCursors;
            auto &sortedPreparedDraws = drawMetadataCache.layerSortedDraws;
            layerBucketCounts.assign(layerOrderSize + 1, 0);
            for (const PreparedRenderDraw &preparedDraw : drawList) {
                const uint32_t layerOrder = preparedDraw.metadata.layerOrder;
                const size_t bucketIndex =
                    layerOrder == std::numeric_limits<uint32_t>::max() || layerOrder >= layerOrderSize ?
                        unorderedLayerBucket :
                        static_cast<size_t>(layerOrder);
                layerBucketCounts[bucketIndex]++;
            }

            layerBucketOffsets.resize(layerBucketCounts.size());
            size_t layerBucketOffset = 0;
            for (size_t bucketIndex = 0; bucketIndex < layerBucketCounts.size(); bucketIndex++) {
                layerBucketOffsets[bucketIndex] = layerBucketOffset;
                layerBucketOffset += layerBucketCounts[bucketIndex];
            }
            layerBucketCursors = layerBucketOffsets;
            sortedPreparedDraws.resize(drawList.size());
            for (PreparedRenderDraw &preparedDraw : drawList) {
                const uint32_t layerOrder = preparedDraw.metadata.layerOrder;
                const size_t bucketIndex =
                    layerOrder == std::numeric_limits<uint32_t>::max() || layerOrder >= layerOrderSize ?
                        unorderedLayerBucket :
                        static_cast<size_t>(layerOrder);
                sortedPreparedDraws[layerBucketCursors[bucketIndex]++] = std::move(preparedDraw);
            }
            drawList.swap(sortedPreparedDraws);
        }

        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "sort.batch");
            if (!enableBatchSort) { return; }
            constexpr uint32_t invalidBatchGroupIndex = std::numeric_limits<uint32_t>::max();
            auto &pipelineBatchGroupIndices = drawMetadataCache.batchGroupIndices;
            auto &touchedBatchKeys = drawMetadataCache.batchTouchedKeys;
            auto &candidateBatchGroupIndices = drawMetadataCache.batchCandidateGroupIndices;
            auto &groupCounts = drawMetadataCache.batchGroupCounts;
            auto &groupOffsets = drawMetadataCache.batchGroupOffsets;
            auto &groupCursors = drawMetadataCache.batchGroupCursors;
            auto &batchGroupedDraws = drawMetadataCache.batchGroupedDraws;
            pipelineBatchGroupIndices.assign(pipelineBatchKeys.size(), invalidBatchGroupIndex);
            size_t sortRangeBegin = 0;
            while (sortRangeBegin < drawList.size()) {
                if (!drawList[sortRangeBegin].batchSortAllowed) {
                    sortRangeBegin++;
                    continue;
                }

                size_t sortRangeEnd = sortRangeBegin + 1;
                while (sortRangeEnd < drawList.size() && drawList[sortRangeEnd].batchSortAllowed) { sortRangeEnd++; }

                const size_t sortRangeCount = sortRangeEnd - sortRangeBegin;
                if (sortRangeCount > 1) {
                    touchedBatchKeys.clear();
                    candidateBatchGroupIndices.resize(sortRangeCount);
                    groupCounts.clear();

                    for (size_t localDrawIndex = 0; localDrawIndex < sortRangeCount; localDrawIndex++) {
                        const PreparedRenderDraw &draw = drawList[sortRangeBegin + localDrawIndex];
                        uint32_t groupIndex = pipelineBatchGroupIndices[draw.pipelineBatchKeyIndex];
                        if (groupIndex == invalidBatchGroupIndex) {
                            groupIndex = static_cast<uint32_t>(groupCounts.size());
                            pipelineBatchGroupIndices[draw.pipelineBatchKeyIndex] = groupIndex;
                            touchedBatchKeys.push_back(draw.pipelineBatchKeyIndex);
                            groupCounts.push_back(0);
                        }
                        candidateBatchGroupIndices[localDrawIndex] = groupIndex;
                        groupCounts[groupIndex]++;
                    }

                    groupOffsets.resize(groupCounts.size());
                    size_t groupOffset = 0;
                    for (size_t groupIndex = 0; groupIndex < groupCounts.size(); groupIndex++) {
                        groupOffsets[groupIndex] = groupOffset;
                        groupOffset += groupCounts[groupIndex];
                    }

                    groupCursors = groupOffsets;
                    batchGroupedDraws.resize(sortRangeCount);
                    for (size_t localDrawIndex = 0; localDrawIndex < sortRangeCount; localDrawIndex++) {
                        const uint32_t groupIndex = candidateBatchGroupIndices[localDrawIndex];
                        batchGroupedDraws[groupCursors[groupIndex]++] =
                            std::move(drawList[sortRangeBegin + localDrawIndex]);
                    }
                    for (size_t localDrawIndex = 0; localDrawIndex < sortRangeCount; localDrawIndex++) {
                        drawList[sortRangeBegin + localDrawIndex] = std::move(batchGroupedDraws[localDrawIndex]);
                    }
                    for (uint32_t pipelineBatchKeyIndex : touchedBatchKeys) {
                        pipelineBatchGroupIndices[pipelineBatchKeyIndex] = invalidBatchGroupIndex;
                    }
                }
                sortRangeBegin = sortRangeEnd;
            }
        }
    };

    auto &sortedChunkPreparedDraws = drawMetadataCache.sortedChunkPreparedDraws;
    if (drawMetadataCache.sortedChunkPreparedDrawsDirty) {
        sortedChunkPreparedDraws = drawMetadataCache.chunkPreparedDraws;
        sortPreparedDrawList(sortedChunkPreparedDraws);
        constexpr uint32_t invalidDrawIndex = std::numeric_limits<uint32_t>::max();
        for (CachedChunkDrawEntry &entry : drawMetadataCache.chunkEntries) {
            entry.sortedDrawIndices.assign(entry.draws.size(), invalidDrawIndex);
            entry.shadowSolidMetadataIndices.assign(entry.draws.size(), invalidDrawIndex);
            entry.shadowAlphaOpaqueMetadataIndices.assign(entry.draws.size(), invalidDrawIndex);
        }
        for (uint32_t sortedDrawIndex = 0; sortedDrawIndex < sortedChunkPreparedDraws.size(); sortedDrawIndex++) {
            const PreparedRenderDraw &draw = sortedChunkPreparedDraws[sortedDrawIndex];
            const RenderIndirectDrawMetadata &metadata = draw.metadata;
            if (metadata.objectType != 0u || metadata.objectIndex >= drawMetadataCache.chunkEntries.size()) {
                continue;
            }
            CachedChunkDrawEntry &entry = drawMetadataCache.chunkEntries[metadata.objectIndex];
            if (metadata.geometryIndex >= entry.drawIndexByGeometryIndex.size()) { continue; }
            const uint32_t localDrawIndex = entry.drawIndexByGeometryIndex[metadata.geometryIndex];
            if (localDrawIndex == invalidDrawIndex || localDrawIndex >= entry.sortedDrawIndices.size()) { continue; }
            entry.sortedDrawIndices[localDrawIndex] = sortedDrawIndex;
        }
        drawMetadataCache.sortedChunkPreparedDrawsDirty = false;
        drawMetadataCache.chunkIndirectDrawBatchesDirty = true;
        drawMetadataCache.chunkMetadataDirty = true;
    }
    MCVR_PROFILE_METRIC("raster.chunk_draw_count", sortedChunkPreparedDraws.size());
    MCVR_PROFILE_METRIC("raster.entity_draw_count", entityPreparedDraws.size());

    if (entityPreparedDraws.empty()) {
        drawsForRender = &sortedChunkPreparedDraws;
    } else {
        sortPreparedDrawList(entityPreparedDraws);
        preparedDraws.reserve(sortedChunkPreparedDraws.size() + entityPreparedDraws.size());
        preparedDraws.insert(preparedDraws.end(), sortedChunkPreparedDraws.begin(), sortedChunkPreparedDraws.end());
        preparedDraws.insert(preparedDraws.end(), entityPreparedDraws.begin(), entityPreparedDraws.end());
        drawsForRender = &preparedDraws;
    }
    const std::vector<PreparedRenderDraw> &draws = *drawsForRender;

    auto beginGraphicsRenderPass = [&]() {
        VkClearValue depthClear{};
        depthClear.depthStencil = {.depth = 1.0f, .stencil = 0};
        std::vector<VkClearValue> clearValues(pass.config.outputs.images.size() + 1);
        for (size_t attachmentIndex = 0; attachmentIndex < pass.config.outputs.images.size(); ++attachmentIndex) {
            const auto clearIter = pass.config.clearOutputs.find(pass.config.outputs.images[attachmentIndex]);
            if (clearIter == pass.config.clearOutputs.end()) { continue; }

            for (size_t componentIndex = 0; componentIndex < clearIter->second.size(); ++componentIndex) {
                clearValues[attachmentIndex].color.uint32[componentIndex] = clearIter->second[componentIndex];
            }
        }
        clearValues.back() = depthClear;

        worldCommandBuffer->beginRenderPass({
            .renderPass = pass.renderPass,
            .framebuffer = pass.framebuffers[frameIndex],
            .renderAreaExtent = {hdrNoisyOutputImages_[frameIndex]->width(),
                                 hdrNoisyOutputImages_[frameIndex]->height()},
            .clearValues = clearValues,
        });

        worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);
    };

    auto finishGraphicsRenderPass = [&]() {
        worldCommandBuffer->endRenderPass();
        for (const ResolvedImageAccess &target : resources.colorAttachmentTargets) {
            if (auto image = resolveImageResource(target.resource, frameIndex); image != nullptr) {
                image->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
        }
        rasterDepthImages_[frameIndex]->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    };

    if (!pass.config.indirectDraw.has_value()) {
        const uint32_t swapchainImageCount = framework->swapchain()->imageCount();
        const uint64_t directMetadataBytes = sizeof(RenderIndirectMetadataHeader) +
                                             static_cast<uint64_t>(draws.size()) * sizeof(RenderIndirectDrawMetadata);
        std::shared_ptr<vk::DeviceLocalBuffer> directMetadataBuffer;

        if (!draws.empty()) {
            if (pass.indirectMetadataBuffers.size() < swapchainImageCount) {
                pass.indirectMetadataBuffers.resize(swapchainImageCount);
            }

            auto &metadataBuffer = pass.indirectMetadataBuffers[frameIndex];
            if (metadataBuffer == nullptr || metadataBuffer->size() < directMetadataBytes) {
                framework->frameResourceRetainer().retain(metadataBuffer);
                metadataBuffer =
                    vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), true, directMetadataBytes,
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY);
            }
            directMetadataBuffer = metadataBuffer;

            RenderIndirectMetadataHeader metadataHeader{
                .drawCount = static_cast<uint32_t>(draws.size()),
                .metadataOffsetBytes = static_cast<uint32_t>(sizeof(RenderIndirectMetadataHeader)),
                .metadataStrideBytes = static_cast<uint32_t>(sizeof(RenderIndirectDrawMetadata)),
                .commandStrideBytes = static_cast<uint32_t>(sizeof(VkDrawIndirectCommand)),
                .chunkInstanceIndexBase = entityInstanceCount,
                .chunkGeometryAddressIndexBase = entityGeometryCount,
                .entityInstanceCount = entityInstanceCount,
                .entityGeometryCount = entityGeometryCount,
                .chunkInstanceCount = chunkInstanceCount,
                .chunkGeometrySlotCount = chunkGeometrySlotCount,
            };
            directMetadataBuffer->uploadToStagingBuffer(&metadataHeader, sizeof(metadataHeader), 0);
            directMetadataBuffer->uploadToBuffer(worldCommandBuffer, sizeof(metadataHeader), 0, 0);

            std::vector<RenderIndirectDrawMetadata> directMetadata;
            directMetadata.reserve(draws.size());
            for (const PreparedRenderDraw &draw : draws) { directMetadata.push_back(draw.metadata); }
            const size_t metadataBytes = directMetadata.size() * sizeof(RenderIndirectDrawMetadata);
            if (metadataBytes > 0) {
                const size_t metadataOffset = sizeof(RenderIndirectMetadataHeader);
                directMetadataBuffer->uploadToStagingBuffer(directMetadata.data(), metadataBytes, metadataOffset);
                directMetadataBuffer->uploadToBuffer(worldCommandBuffer, metadataBytes, metadataOffset, metadataOffset);
            }

            worldCommandBuffer->barriersBufferImage({{
                                                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                        .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                                        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                        .srcQueueFamilyIndex = queueIndex,
                                                        .dstQueueFamilyIndex = queueIndex,
                                                        .buffer = directMetadataBuffer,
                                                        .offset = 0,
                                                        .size = directMetadataBytes,
                                                    }},
                                                    {});

            auto &trackedState = context.trackedBufferStates[directMetadataBuffer.get()];
            trackedState.stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            trackedState.accessMask = VK_ACCESS_2_SHADER_READ_BIT;
            trackedState.initialized = true;
        }

        beginGraphicsRenderPass();
        vk::GraphicsPipeline *boundPipeline = nullptr;
        if (directMetadataBuffer != nullptr) {
            worldCommandBuffer->pushDescriptorBuffers(
                context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS, renderIndirectMetadataDescriptorSet,
                {{.binding = renderIndirectMetadataDescriptorBinding,
                  .buffer = directMetadataBuffer->vkBuffer(),
                  .size = directMetadataBuffer->size(),
                  .owner = directMetadataBuffer}});
        }
        for (uint32_t drawIndex = 0; drawIndex < draws.size(); drawIndex++) {
            const PreparedRenderDraw &draw = draws[drawIndex];
            if (draw.pipeline != boundPipeline) {
                vkCmdBindPipeline(worldCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  draw.pipeline->vkPipeline());
                boundPipeline = draw.pipeline;
            }
            worldCommandBuffer->draw(draw.indexedCommand.indexCount, draw.indexedCommand.instanceCount, 0, drawIndex);
        }
        finishGraphicsRenderPass();
        return;
    }

    const uint32_t swapchainImageCount = framework->swapchain()->imageCount();
    auto &indirectDrawBatches = drawMetadataCache.indirectDrawBatches;
    auto &chunkIndirectDrawBatches = drawMetadataCache.chunkIndirectDrawBatches;
    const std::vector<RenderIndirectDrawBatch> *indirectDrawBatchesForRender = nullptr;
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "indirect.buildBatches");
        const bool chunkBatchCacheHit = !drawMetadataCache.chunkIndirectDrawBatchesDirty;
        size_t chunkBatchScanCount = 0;
        size_t entityBatchScanCount = 0;
        auto drawsCanShareIndirectBatch = [&](const PreparedRenderDraw &lhs, const PreparedRenderDraw &rhs) {
            const bool hasSamePipeline = lhs.pipeline == rhs.pipeline;
            const bool hasSameGeometryBuffers = lhs.pbrBuffer == rhs.pbrBuffer && lhs.indexBuffer == rhs.indexBuffer;
            return hasSamePipeline && (shaderPullingIndirect || hasSameGeometryBuffers);
        };
        auto appendIndirectDrawBatchesForRange = [&](std::vector<RenderIndirectDrawBatch> &batchList,
                                                     const std::vector<PreparedRenderDraw> &sourceDraws,
                                                     size_t rangeBegin, size_t rangeEnd) {
            size_t scannedDrawCount = 0;
            size_t groupStart = rangeBegin;
            while (groupStart < rangeEnd) {
                const PreparedRenderDraw &groupDraw = sourceDraws[groupStart];
                size_t groupEnd = groupStart + 1;
                while (groupEnd < rangeEnd && drawsCanShareIndirectBatch(groupDraw, sourceDraws[groupEnd])) {
                    groupEnd++;
                }

                const uint32_t sourceOffset = static_cast<uint32_t>(groupStart);
                const uint32_t sourceCount = static_cast<uint32_t>(groupEnd - groupStart);
                if (!batchList.empty()) {
                    RenderIndirectDrawBatch &previousBatch = batchList.back();
                    const size_t previousBatchEnd =
                        static_cast<size_t>(previousBatch.sourceOffset) + previousBatch.sourceCount;
                    if (previousBatchEnd == groupStart &&
                        drawsCanShareIndirectBatch(sourceDraws[previousBatch.sourceOffset], groupDraw)) {
                        previousBatch.sourceCount += sourceCount;
                    } else {
                        batchList.push_back({.sourceOffset = sourceOffset, .sourceCount = sourceCount});
                    }
                } else {
                    batchList.push_back({.sourceOffset = sourceOffset, .sourceCount = sourceCount});
                }
                scannedDrawCount += groupEnd - groupStart;
                groupStart = groupEnd;
            }
            return scannedDrawCount;
        };

        if (drawMetadataCache.chunkIndirectDrawBatchesDirty) {
            chunkIndirectDrawBatches.clear();
            chunkIndirectDrawBatches.reserve(sortedChunkPreparedDraws.size());
            chunkBatchScanCount = appendIndirectDrawBatchesForRange(chunkIndirectDrawBatches, sortedChunkPreparedDraws,
                                                                    0, sortedChunkPreparedDraws.size());
            drawMetadataCache.chunkIndirectDrawBatchesDirty = false;
        }

        if (entityPreparedDraws.empty()) {
            indirectDrawBatchesForRender = &chunkIndirectDrawBatches;
        } else {
            indirectDrawBatches.clear();
            indirectDrawBatches.reserve(chunkIndirectDrawBatches.size() + entityPreparedDraws.size());
            indirectDrawBatches.insert(indirectDrawBatches.end(), chunkIndirectDrawBatches.begin(),
                                       chunkIndirectDrawBatches.end());
            entityBatchScanCount = appendIndirectDrawBatchesForRange(indirectDrawBatches, draws,
                                                                     sortedChunkPreparedDraws.size(), draws.size());
            indirectDrawBatchesForRender = &indirectDrawBatches;
        }
        MCVR_PROFILE_METRIC("raster.indirect_chunk_batch_cache_hit", chunkBatchCacheHit ? 1 : 0);
        MCVR_PROFILE_METRIC("raster.indirect_chunk_batch_scan_count", chunkBatchScanCount);
        MCVR_PROFILE_METRIC("raster.indirect_entity_batch_scan_count", entityBatchScanCount);
        MCVR_PROFILE_METRIC("raster.indirect_chunk_batch_count", chunkIndirectDrawBatches.size());
    }
    const std::vector<RenderIndirectDrawBatch> &indirectDrawBatchesForRecord = *indirectDrawBatchesForRender;
    MCVR_PROFILE_METRIC("raster.prepared_draw_count", draws.size());
    MCVR_PROFILE_METRIC("raster.indirect_batch_count", indirectDrawBatchesForRecord.size());

    const uint64_t indirectMetadataBytes =
        sizeof(RenderIndirectMetadataHeader) + static_cast<uint64_t>(draws.size()) * sizeof(RenderIndirectDrawMetadata);
    const uint64_t indirectCommandBytes = static_cast<uint64_t>(draws.size()) * sizeof(VkDrawIndirectCommand);
    MCVR_PROFILE_METRIC("raster.indirect_metadata_bytes", indirectMetadataBytes);
    MCVR_PROFILE_METRIC("raster.indirect_command_bytes", indirectCommandBytes);

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "buffers.primary.resize");
        if (pass.indirectMetadataBuffers.size() < swapchainImageCount) {
            pass.indirectMetadataBuffers.resize(swapchainImageCount);
        }
        if (pass.indirectCommandBuffers.size() < swapchainImageCount) {
            pass.indirectCommandBuffers.resize(swapchainImageCount);
        }
    }

    std::shared_ptr<vk::DeviceLocalBuffer> &indirectMetadataBuffer = pass.indirectMetadataBuffers[frameIndex];
    std::shared_ptr<vk::DeviceLocalBuffer> indirectCommandBuffer;
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "buffers.primary.ensure");
        if (indirectMetadataBuffer == nullptr || indirectMetadataBuffer->size() < indirectMetadataBytes) {
            framework->frameResourceRetainer().retain(indirectMetadataBuffer);
            indirectMetadataBuffer =
                vk::DeviceLocalBuffer::create(framework->vma(), framework->device(), true, indirectMetadataBytes,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY);
        }

        if (indirectCommandBytes > 0) {
            indirectCommandBuffer = pass.indirectCommandBuffers[frameIndex];
            if (indirectCommandBuffer == nullptr || indirectCommandBuffer->size() < indirectCommandBytes) {
                framework->frameResourceRetainer().retain(indirectCommandBuffer);
                indirectCommandBuffer = vk::DeviceLocalBuffer::create(
                    framework->vma(), framework->device(), false, indirectCommandBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 0,
                    VMA_MEMORY_USAGE_GPU_ONLY);
                pass.indirectCommandBuffers[frameIndex] = indirectCommandBuffer;
            }
        }
    }

    if (drawMetadataCache.chunkMetadataDirty) {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.chunk.cachePack");
        auto &chunkPrimaryMetadata = drawMetadataCache.chunkPrimaryMetadata;
        auto &chunkShadowSolidMetadata = drawMetadataCache.chunkShadowSolidMetadata;
        auto &chunkShadowAlphaOpaqueMetadata = drawMetadataCache.chunkShadowAlphaOpaqueMetadata;
        chunkPrimaryMetadata.clear();
        chunkPrimaryMetadata.reserve(sortedChunkPreparedDraws.size());
        chunkShadowSolidMetadata.clear();
        chunkShadowAlphaOpaqueMetadata.clear();
        chunkShadowSolidMetadata.reserve(sortedChunkPreparedDraws.size());
        chunkShadowAlphaOpaqueMetadata.reserve(sortedChunkPreparedDraws.size());

        constexpr uint32_t invalidDrawIndex = std::numeric_limits<uint32_t>::max();
        for (CachedChunkDrawEntry &entry : drawMetadataCache.chunkEntries) {
            entry.shadowSolidMetadataIndices.assign(entry.draws.size(), invalidDrawIndex);
            entry.shadowAlphaOpaqueMetadataIndices.assign(entry.draws.size(), invalidDrawIndex);
        }
        for (PreparedRenderDraw &draw : sortedChunkPreparedDraws) {
            const RenderIndirectDrawMetadata &metadata = draw.metadata;
            chunkPrimaryMetadata.push_back(metadata);
            if (!pass.rendersPrimaryRaster || !canCastDirectionalShadow(metadata)) { continue; }
            CachedChunkDrawEntry *entry = nullptr;
            uint32_t localDrawIndex = invalidDrawIndex;
            if (metadata.objectType == 0u && metadata.objectIndex < drawMetadataCache.chunkEntries.size()) {
                entry = &drawMetadataCache.chunkEntries[metadata.objectIndex];
                if (metadata.geometryIndex < entry->drawIndexByGeometryIndex.size()) {
                    localDrawIndex = entry->drawIndexByGeometryIndex[metadata.geometryIndex];
                }
            }
            if (isSolidFastDirectionalShadowLayer(metadata)) {
                if (entry != nullptr && localDrawIndex != invalidDrawIndex &&
                    localDrawIndex < entry->shadowSolidMetadataIndices.size()) {
                    entry->shadowSolidMetadataIndices[localDrawIndex] =
                        static_cast<uint32_t>(chunkShadowSolidMetadata.size());
                }
                chunkShadowSolidMetadata.push_back(metadata);
            } else {
                if (entry != nullptr && localDrawIndex != invalidDrawIndex &&
                    localDrawIndex < entry->shadowAlphaOpaqueMetadataIndices.size()) {
                    entry->shadowAlphaOpaqueMetadataIndices[localDrawIndex] =
                        static_cast<uint32_t>(chunkShadowAlphaOpaqueMetadata.size());
                }
                chunkShadowAlphaOpaqueMetadata.push_back(metadata);
            }
        }

        drawMetadataCache.chunkMetadataDirty = false;
        bumpChunkMetadataVersion();
        drawMetadataCache.chunkMetadataDirtyUpdates.clear();
    }

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.entity.build");
        auto &entityPrimaryMetadata = drawMetadataCache.entityPrimaryMetadata;
        auto &entityShadowSolidMetadata = drawMetadataCache.entityShadowSolidMetadata;
        auto &entityShadowAlphaOpaqueMetadata = drawMetadataCache.entityShadowAlphaOpaqueMetadata;
        entityPrimaryMetadata.clear();
        entityPrimaryMetadata.reserve(entityPreparedDraws.size());
        entityShadowSolidMetadata.clear();
        entityShadowAlphaOpaqueMetadata.clear();
        entityShadowSolidMetadata.reserve(entityPreparedDraws.size());
        entityShadowAlphaOpaqueMetadata.reserve(entityPreparedDraws.size());

        for (const PreparedRenderDraw &draw : entityPreparedDraws) {
            const RenderIndirectDrawMetadata &metadata = draw.metadata;
            entityPrimaryMetadata.push_back(metadata);
            if (!pass.rendersPrimaryRaster || !canCastDirectionalShadow(metadata)) { continue; }
            if (isSolidFastDirectionalShadowLayer(metadata)) {
                entityShadowSolidMetadata.push_back(metadata);
            } else {
                entityShadowAlphaOpaqueMetadata.push_back(metadata);
            }
        }
    }

    const uint32_t directionalShadowSolidDrawCount =
        static_cast<uint32_t>(drawMetadataCache.chunkShadowSolidMetadata.size()) +
        static_cast<uint32_t>(drawMetadataCache.entityShadowSolidMetadata.size());
    const uint32_t directionalShadowAlphaOpaqueDrawCount =
        static_cast<uint32_t>(drawMetadataCache.chunkShadowAlphaOpaqueMetadata.size()) +
        static_cast<uint32_t>(drawMetadataCache.entityShadowAlphaOpaqueMetadata.size());
    const uint32_t directionalShadowDrawCount = directionalShadowSolidDrawCount + directionalShadowAlphaOpaqueDrawCount;
    const uint64_t directionalShadowMetadataBytes =
        sizeof(RenderIndirectMetadataHeader) +
        static_cast<uint64_t>(directionalShadowDrawCount) * sizeof(RenderIndirectDrawMetadata);
    const uint64_t directionalShadowCommandBytes =
        static_cast<uint64_t>(directionalShadowDrawCount) * sizeof(VkDrawIndirectCommand);
    MCVR_PROFILE_METRIC("shadow.candidate_draw_count", directionalShadowDrawCount);
    MCVR_PROFILE_METRIC("shadow.solid_candidate_draw_count", directionalShadowSolidDrawCount);
    MCVR_PROFILE_METRIC("shadow.alpha_opaque_candidate_draw_count", directionalShadowAlphaOpaqueDrawCount);
    MCVR_PROFILE_METRIC("shadow.filtered_draw_count",
                        draws.size() >= directionalShadowDrawCount ? draws.size() - directionalShadowDrawCount : 0);
    MCVR_PROFILE_METRIC("shadow.indirect_metadata_bytes", directionalShadowMetadataBytes);
    MCVR_PROFILE_METRIC("shadow.indirect_command_bytes", directionalShadowCommandBytes);

    RenderIndirectMetadataHeader metadataHeader{
        .drawCount = static_cast<uint32_t>(draws.size()),
        .metadataOffsetBytes = static_cast<uint32_t>(sizeof(RenderIndirectMetadataHeader)),
        .metadataStrideBytes = static_cast<uint32_t>(sizeof(RenderIndirectDrawMetadata)),
        .commandStrideBytes = static_cast<uint32_t>(sizeof(VkDrawIndirectCommand)),
        .chunkInstanceIndexBase = entityInstanceCount,
        .chunkGeometryAddressIndexBase = entityGeometryCount,
        .entityInstanceCount = entityInstanceCount,
        .entityGeometryCount = entityGeometryCount,
        .chunkInstanceCount = chunkInstanceCount,
        .chunkGeometrySlotCount = chunkGeometrySlotCount,
    };
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.primary.frameState");
        if (drawMetadataCache.frameUploadStates.size() < swapchainImageCount) {
            drawMetadataCache.frameUploadStates.resize(swapchainImageCount);
        }
    }
    FrameIndirectUploadState &frameUploadState = drawMetadataCache.frameUploadStates[frameIndex];

    uint64_t primaryMetadataVersion = drawMetadataCache.chunkMetadataVersion;
    hashCombine64(primaryMetadataVersion, indirectMetadataBytes);
    hashCombine64(primaryMetadataVersion, entityInstanceCount);
    hashCombine64(primaryMetadataVersion, entityGeometryCount);
    hashCombine64(primaryMetadataVersion, chunkInstanceCount);
    hashCombine64(primaryMetadataVersion, chunkGeometrySlotCount);
    hashCombine64(primaryMetadataVersion, drawMetadataCache.chunkPrimaryMetadata.size());
    hashCombine64(primaryMetadataVersion, drawMetadataCache.entityPrimaryMetadata.size());
    bool uploadIndirectMetadata = false;
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.primary.uploadCheck");
        if (frameUploadState.metadataBuffer != indirectMetadataBuffer.get()) {
            frameUploadState.metadataBuffer = indirectMetadataBuffer.get();
            frameUploadState.metadataValid = false;
        }
        frameUploadState.metadataBytes = indirectMetadataBytes;
        uploadIndirectMetadata = !frameUploadState.metadataValid ||
                                 frameUploadState.metadataVersion != primaryMetadataVersion ||
                                 !drawMetadataCache.entityPrimaryMetadata.empty();
    }
    auto &indirectMetadataUpload = drawMetadataCache.indirectMetadataUpload;
    const uint64_t primaryChunkMetadataVersion = drawMetadataCache.chunkMetadataVersion;
    const uint32_t primaryChunkMetadataCount = static_cast<uint32_t>(drawMetadataCache.chunkPrimaryMetadata.size());
    const uint32_t primaryEntityMetadataCount = static_cast<uint32_t>(drawMetadataCache.entityPrimaryMetadata.size());
    const size_t primaryEntityMetadataBytes =
        drawMetadataCache.entityPrimaryMetadata.size() * sizeof(RenderIndirectDrawMetadata);
    const size_t primaryEntityMetadataOffset =
        sizeof(RenderIndirectMetadataHeader) +
        drawMetadataCache.chunkPrimaryMetadata.size() * sizeof(RenderIndirectDrawMetadata);
    const bool primaryChunkMetadataChanged = frameUploadState.chunkMetadataVersion != primaryChunkMetadataVersion;
    const bool primaryChunkLayoutChanged = frameUploadState.chunkMetadataCount != primaryChunkMetadataCount;
    std::vector<MetadataIndexRange> primaryChunkUploadRanges;
    bool primaryChunkUploadRangesValid = false;
    if (uploadIndirectMetadata && frameUploadState.metadataValid && primaryChunkMetadataChanged &&
        !primaryChunkLayoutChanged && frameUploadState.chunkMetadataVersion < primaryChunkMetadataVersion) {
        primaryChunkUploadRangesValid = true;
        for (const ChunkMetadataDirtyUpdate &dirtyUpdate : drawMetadataCache.chunkMetadataDirtyUpdates) {
            if (dirtyUpdate.version <= frameUploadState.chunkMetadataVersion) { continue; }
            if (dirtyUpdate.version > primaryChunkMetadataVersion) { break; }
            for (const MetadataIndexRange &range : dirtyUpdate.primaryRanges) {
                for (uint32_t i = 0; i < range.count; i++) {
                    addMetadataDirtyRange(primaryChunkUploadRanges, range.offset + i);
                }
            }
        }
        if (primaryChunkUploadRanges.empty() || drawMetadataCache.chunkMetadataDirtyUpdates.empty() ||
            drawMetadataCache.chunkMetadataDirtyUpdates.front().version > frameUploadState.chunkMetadataVersion + 1) {
            primaryChunkUploadRangesValid = false;
            primaryChunkUploadRanges.clear();
        } else {
            normalizeMetadataDirtyRanges(primaryChunkUploadRanges);
        }
    }
    const bool uploadPrimaryMetadataFull =
        uploadIndirectMetadata && (!frameUploadState.metadataValid || primaryChunkLayoutChanged ||
                                   (primaryChunkMetadataChanged && !primaryChunkUploadRangesValid));
    const bool uploadPrimaryMetadataHeader =
        uploadIndirectMetadata && (!frameUploadState.metadataValid || primaryChunkMetadataChanged ||
                                   frameUploadState.chunkMetadataCount != primaryChunkMetadataCount ||
                                   frameUploadState.entityMetadataCount != primaryEntityMetadataCount ||
                                   frameUploadState.headerVersion != primaryMetadataVersion);
    const bool uploadPrimaryMetadataEntity = uploadIndirectMetadata && primaryEntityMetadataBytes > 0;
    const size_t primaryMetadataUploadBytes =
        uploadIndirectMetadata ?
            (uploadPrimaryMetadataFull ?
                 static_cast<size_t>(indirectMetadataBytes) :
                 ((uploadPrimaryMetadataHeader ? sizeof(RenderIndirectMetadataHeader) : 0u) +
                  (primaryChunkUploadRangesValid ?
                       std::accumulate(primaryChunkUploadRanges.begin(), primaryChunkUploadRanges.end(), size_t{0},
                                       [](size_t sum, const MetadataIndexRange &range) {
                                           return sum +
                                                  static_cast<size_t>(range.count) * sizeof(RenderIndirectDrawMetadata);
                                       }) :
                       0u) +
                  (uploadPrimaryMetadataEntity ? primaryEntityMetadataBytes : 0u))) :
            0u;
    MCVR_PROFILE_METRIC("raster.metadata_primary_upload_full", uploadPrimaryMetadataFull ? 1 : 0);
    MCVR_PROFILE_METRIC("raster.metadata_primary_upload_header", uploadPrimaryMetadataHeader ? 1 : 0);
    MCVR_PROFILE_METRIC("raster.metadata_primary_upload_entity", uploadPrimaryMetadataEntity ? 1 : 0);
    MCVR_PROFILE_METRIC("raster.metadata_primary_upload_chunk_range",
                        primaryChunkUploadRangesValid ? primaryChunkUploadRanges.size() : 0);
    MCVR_PROFILE_METRIC("raster.metadata_primary_upload_bytes", primaryMetadataUploadBytes);
    MCVR_PROFILE_METRIC("raster.metadata_primary_total_bytes", indirectMetadataBytes);
    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.primary.pack");
        if (uploadIndirectMetadata) {
            if (uploadPrimaryMetadataFull) {
                indirectMetadataUpload.resize(static_cast<size_t>(indirectMetadataBytes));
                std::memcpy(indirectMetadataUpload.data(), &metadataHeader, sizeof(metadataHeader));
                uint8_t *metadataWrite = indirectMetadataUpload.data() + sizeof(RenderIndirectMetadataHeader);
                const size_t chunkMetadataBytes =
                    drawMetadataCache.chunkPrimaryMetadata.size() * sizeof(RenderIndirectDrawMetadata);
                if (chunkMetadataBytes > 0) {
                    std::memcpy(metadataWrite, drawMetadataCache.chunkPrimaryMetadata.data(), chunkMetadataBytes);
                    metadataWrite += chunkMetadataBytes;
                }
                if (primaryEntityMetadataBytes > 0) {
                    std::memcpy(metadataWrite, drawMetadataCache.entityPrimaryMetadata.data(),
                                primaryEntityMetadataBytes);
                }
            }
        }
    }

    auto &indirectMetadataTrackedState = context.trackedBufferStates[indirectMetadataBuffer.get()];
    auto *indirectCommandTrackedState =
        indirectCommandBuffer == nullptr ? nullptr : &context.trackedBufferStates[indirectCommandBuffer.get()];

    auto barrierToTransferWrite = [&](auto &trackedState, const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                                      uint64_t byteCount) {
        if (byteCount == 0) { return; }
        if (trackedState.initialized) {
            worldCommandBuffer->barriersBufferImage(
                {
                    {
                        .srcStageMask =
                            trackedState.stageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : trackedState.stageMask,
                        .srcAccessMask = trackedState.accessMask,
                        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .srcQueueFamilyIndex = queueIndex,
                        .dstQueueFamilyIndex = queueIndex,
                        .buffer = buffer,
                        .offset = 0,
                        .size = byteCount,
                    },
                },
                {});
        }
        trackedState.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        trackedState.accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        trackedState.initialized = true;
    };

    auto barrierToRead = [&](auto &trackedState, const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                             uint64_t byteCount, VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask) {
        if (byteCount > 0 && trackedState.initialized &&
            (trackedState.accessMask &
             (VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT)) != 0) {
            worldCommandBuffer->barriersBufferImage(
                {
                    {
                        .srcStageMask =
                            trackedState.stageMask == 0 ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : trackedState.stageMask,
                        .srcAccessMask = trackedState.accessMask,
                        .dstStageMask = dstStageMask,
                        .dstAccessMask = dstAccessMask,
                        .srcQueueFamilyIndex = queueIndex,
                        .dstQueueFamilyIndex = queueIndex,
                        .buffer = buffer,
                        .offset = 0,
                        .size = byteCount,
                    },
                },
                {});
        }
        trackedState.stageMask = dstStageMask;
        trackedState.accessMask = dstAccessMask;
        trackedState.initialized = true;
    };

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.primary.upload");
        if (uploadIndirectMetadata) {
            barrierToTransferWrite(indirectMetadataTrackedState, indirectMetadataBuffer, indirectMetadataBytes);
            if (uploadPrimaryMetadataFull) {
                indirectMetadataBuffer->uploadToStagingBuffer(indirectMetadataUpload.data(),
                                                              indirectMetadataUpload.size(), 0);
                indirectMetadataBuffer->uploadToBuffer(worldCommandBuffer, indirectMetadataUpload.size(), 0, 0);
            } else {
                if (uploadPrimaryMetadataHeader) {
                    indirectMetadataBuffer->uploadToStagingBuffer(&metadataHeader, sizeof(metadataHeader), 0);
                    indirectMetadataBuffer->uploadToBuffer(worldCommandBuffer, sizeof(metadataHeader), 0, 0);
                }
                if (primaryChunkUploadRangesValid) {
                    for (const MetadataIndexRange &range : primaryChunkUploadRanges) {
                        const size_t rangeBytes = static_cast<size_t>(range.count) * sizeof(RenderIndirectDrawMetadata);
                        const size_t rangeOffset =
                            sizeof(RenderIndirectMetadataHeader) +
                            static_cast<size_t>(range.offset) * sizeof(RenderIndirectDrawMetadata);
                        indirectMetadataBuffer->uploadToStagingBuffer(
                            drawMetadataCache.chunkPrimaryMetadata.data() + range.offset, rangeBytes, rangeOffset);
                        indirectMetadataBuffer->uploadToBuffer(worldCommandBuffer, rangeBytes, rangeOffset,
                                                               rangeOffset);
                    }
                }
                if (uploadPrimaryMetadataEntity && !drawMetadataCache.entityPrimaryMetadata.empty()) {
                    indirectMetadataBuffer->uploadToStagingBuffer(drawMetadataCache.entityPrimaryMetadata.data(),
                                                                  primaryEntityMetadataBytes,
                                                                  primaryEntityMetadataOffset);
                    indirectMetadataBuffer->uploadToBuffer(worldCommandBuffer, primaryEntityMetadataBytes,
                                                           primaryEntityMetadataOffset, primaryEntityMetadataOffset);
                }
            }
        }
        barrierToRead(indirectMetadataTrackedState, indirectMetadataBuffer, indirectMetadataBytes,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                      VK_ACCESS_2_SHADER_READ_BIT);

        frameUploadState.metadataVersion = primaryMetadataVersion;
        frameUploadState.headerVersion = primaryMetadataVersion;
        frameUploadState.chunkMetadataVersion = primaryChunkMetadataVersion;
        frameUploadState.entityMetadataHash = 0;
        frameUploadState.chunkMetadataCount = primaryChunkMetadataCount;
        frameUploadState.chunkLeadingMetadataCount = primaryChunkMetadataCount;
        frameUploadState.entityMetadataCount = primaryEntityMetadataCount;
        frameUploadState.entityLeadingMetadataCount = 0;
        frameUploadState.metadataValid = true;
    }

    MCVR_PROFILE_METRIC("shadow.enabled_for_pass",
                        directionalShadowRasterEnabled_ && pass.rendersPrimaryRaster ? 1 : 0);
    if (directionalShadowRasterEnabled_ && pass.rendersPrimaryRaster && directionalShadowDrawCount > 0 &&
        directionalShadowCommandBytes > 0) {
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "buffers.shadow.resize");
            if (directionalShadowIndirectComputePipeline_ == nullptr) {
                throw std::runtime_error("main_render directional shadow indirect command pipeline is unavailable");
            }
            if (directionalShadowMetadataBuffers_.size() < swapchainImageCount) {
                directionalShadowMetadataBuffers_.resize(swapchainImageCount);
            }
            if (directionalShadowCommandBuffers_.size() < swapchainImageCount) {
                directionalShadowCommandBuffers_.resize(swapchainImageCount);
            }
        }
        std::shared_ptr<vk::DeviceLocalBuffer> &shadowMetadataBuffer = directionalShadowMetadataBuffers_[frameIndex];
        std::shared_ptr<vk::DeviceLocalBuffer> &shadowCommandBuffer = directionalShadowCommandBuffers_[frameIndex];
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "buffers.shadow.ensure");
            if (shadowMetadataBuffer == nullptr || shadowMetadataBuffer->size() < directionalShadowMetadataBytes) {
                framework->frameResourceRetainer().retain(shadowMetadataBuffer);
                shadowMetadataBuffer = vk::DeviceLocalBuffer::create(
                    framework->vma(), framework->device(), true, directionalShadowMetadataBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY);
            }
            context.mainRenderDescriptorTable->bindBuffer(shadowMetadataBuffer, 1,
                                                          directionalShadowMetadataDescriptorBinding);

            if (shadowCommandBuffer == nullptr || shadowCommandBuffer->size() < directionalShadowCommandBytes) {
                framework->frameResourceRetainer().retain(shadowCommandBuffer);
                shadowCommandBuffer = vk::DeviceLocalBuffer::create(
                    framework->vma(), framework->device(), false, directionalShadowCommandBytes,
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            }
            context.mainRenderDescriptorTable->bindBuffer(shadowCommandBuffer, 1,
                                                          directionalShadowCommandsDescriptorBinding);
        }

        RenderIndirectMetadataHeader shadowMetadataHeader{
            .drawCount = directionalShadowDrawCount,
            .metadataOffsetBytes = static_cast<uint32_t>(sizeof(RenderIndirectMetadataHeader)),
            .metadataStrideBytes = static_cast<uint32_t>(sizeof(RenderIndirectDrawMetadata)),
            .commandStrideBytes = static_cast<uint32_t>(sizeof(VkDrawIndirectCommand)),
            .chunkInstanceIndexBase = entityInstanceCount,
            .chunkGeometryAddressIndexBase = entityGeometryCount,
            .entityInstanceCount = entityInstanceCount,
            .entityGeometryCount = entityGeometryCount,
            .chunkInstanceCount = chunkInstanceCount,
            .chunkGeometrySlotCount = chunkGeometrySlotCount,
        };

        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.shadow.frameState");
            if (drawMetadataCache.shadowFrameUploadStates.size() < swapchainImageCount) {
                drawMetadataCache.shadowFrameUploadStates.resize(swapchainImageCount);
            }
        }
        FrameIndirectUploadState &shadowFrameUploadState = drawMetadataCache.shadowFrameUploadStates[frameIndex];
        uint64_t shadowMetadataVersion = drawMetadataCache.chunkMetadataVersion;
        hashCombine64(shadowMetadataVersion, directionalShadowMetadataBytes);
        hashCombine64(shadowMetadataVersion, entityInstanceCount);
        hashCombine64(shadowMetadataVersion, entityGeometryCount);
        hashCombine64(shadowMetadataVersion, chunkInstanceCount);
        hashCombine64(shadowMetadataVersion, chunkGeometrySlotCount);
        hashCombine64(shadowMetadataVersion, drawMetadataCache.chunkShadowSolidMetadata.size());
        hashCombine64(shadowMetadataVersion, drawMetadataCache.chunkShadowAlphaOpaqueMetadata.size());
        hashCombine64(shadowMetadataVersion, drawMetadataCache.entityShadowSolidMetadata.size());
        hashCombine64(shadowMetadataVersion, drawMetadataCache.entityShadowAlphaOpaqueMetadata.size());
        bool uploadShadowMetadata = false;
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.shadow.uploadCheck");
            if (shadowFrameUploadState.metadataBuffer != shadowMetadataBuffer.get()) {
                shadowFrameUploadState.metadataBuffer = shadowMetadataBuffer.get();
                shadowFrameUploadState.metadataValid = false;
            }
            shadowFrameUploadState.metadataBytes = directionalShadowMetadataBytes;
            uploadShadowMetadata = !shadowFrameUploadState.metadataValid ||
                                   shadowFrameUploadState.metadataVersion != shadowMetadataVersion ||
                                   !drawMetadataCache.entityShadowSolidMetadata.empty() ||
                                   !drawMetadataCache.entityShadowAlphaOpaqueMetadata.empty();
        }
        auto &shadowMetadataUpload = drawMetadataCache.shadowMetadataUpload;
        const uint64_t shadowChunkMetadataVersion = drawMetadataCache.chunkMetadataVersion;
        const uint32_t shadowChunkMetadataCount =
            static_cast<uint32_t>(drawMetadataCache.chunkShadowSolidMetadata.size() +
                                  drawMetadataCache.chunkShadowAlphaOpaqueMetadata.size());
        const uint32_t shadowEntityMetadataCount =
            static_cast<uint32_t>(drawMetadataCache.entityShadowSolidMetadata.size() +
                                  drawMetadataCache.entityShadowAlphaOpaqueMetadata.size());
        const uint32_t shadowEntitySolidMetadataCount =
            static_cast<uint32_t>(drawMetadataCache.entityShadowSolidMetadata.size());
        const uint32_t shadowChunkSolidMetadataCount =
            static_cast<uint32_t>(drawMetadataCache.chunkShadowSolidMetadata.size());
        const size_t shadowEntitySolidBytes =
            drawMetadataCache.entityShadowSolidMetadata.size() * sizeof(RenderIndirectDrawMetadata);
        const size_t shadowEntityAlphaOpaqueBytes =
            drawMetadataCache.entityShadowAlphaOpaqueMetadata.size() * sizeof(RenderIndirectDrawMetadata);
        const size_t shadowEntitySolidOffset =
            sizeof(RenderIndirectMetadataHeader) +
            drawMetadataCache.chunkShadowSolidMetadata.size() * sizeof(RenderIndirectDrawMetadata);
        const size_t shadowEntityAlphaOpaqueOffset =
            sizeof(RenderIndirectMetadataHeader) +
            (drawMetadataCache.chunkShadowSolidMetadata.size() + drawMetadataCache.entityShadowSolidMetadata.size() +
             drawMetadataCache.chunkShadowAlphaOpaqueMetadata.size()) *
                sizeof(RenderIndirectDrawMetadata);
        const bool shadowChunkMetadataChanged =
            shadowFrameUploadState.chunkMetadataVersion != shadowChunkMetadataVersion;
        const bool shadowChunkLayoutChanged =
            shadowFrameUploadState.chunkMetadataCount != shadowChunkMetadataCount ||
            shadowFrameUploadState.chunkLeadingMetadataCount != shadowChunkSolidMetadataCount ||
            shadowFrameUploadState.entityLeadingMetadataCount != shadowEntitySolidMetadataCount;
        std::vector<MetadataIndexRange> shadowSolidUploadRanges;
        std::vector<MetadataIndexRange> shadowAlphaOpaqueUploadRanges;
        bool shadowChunkUploadRangesValid = false;
        if (uploadShadowMetadata && shadowFrameUploadState.metadataValid && shadowChunkMetadataChanged &&
            !shadowChunkLayoutChanged && shadowFrameUploadState.chunkMetadataVersion < shadowChunkMetadataVersion) {
            shadowChunkUploadRangesValid = true;
            for (const ChunkMetadataDirtyUpdate &dirtyUpdate : drawMetadataCache.chunkMetadataDirtyUpdates) {
                if (dirtyUpdate.version <= shadowFrameUploadState.chunkMetadataVersion) { continue; }
                if (dirtyUpdate.version > shadowChunkMetadataVersion) { break; }
                for (const MetadataIndexRange &range : dirtyUpdate.shadowSolidRanges) {
                    for (uint32_t i = 0; i < range.count; i++) {
                        addMetadataDirtyRange(shadowSolidUploadRanges, range.offset + i);
                    }
                }
                for (const MetadataIndexRange &range : dirtyUpdate.shadowAlphaOpaqueRanges) {
                    for (uint32_t i = 0; i < range.count; i++) {
                        addMetadataDirtyRange(shadowAlphaOpaqueUploadRanges, range.offset + i);
                    }
                }
            }
            if ((shadowSolidUploadRanges.empty() && shadowAlphaOpaqueUploadRanges.empty()) ||
                drawMetadataCache.chunkMetadataDirtyUpdates.empty() ||
                drawMetadataCache.chunkMetadataDirtyUpdates.front().version >
                    shadowFrameUploadState.chunkMetadataVersion + 1) {
                shadowChunkUploadRangesValid = false;
                shadowSolidUploadRanges.clear();
                shadowAlphaOpaqueUploadRanges.clear();
            } else {
                normalizeMetadataDirtyRanges(shadowSolidUploadRanges);
                normalizeMetadataDirtyRanges(shadowAlphaOpaqueUploadRanges);
            }
        }
        const size_t shadowChunkRangeUploadBytes =
            shadowChunkUploadRangesValid ?
                (std::accumulate(shadowSolidUploadRanges.begin(), shadowSolidUploadRanges.end(), size_t{0},
                                 [](size_t sum, const MetadataIndexRange &range) {
                                     return sum + static_cast<size_t>(range.count) * sizeof(RenderIndirectDrawMetadata);
                                 }) +
                 std::accumulate(shadowAlphaOpaqueUploadRanges.begin(), shadowAlphaOpaqueUploadRanges.end(), size_t{0},
                                 [](size_t sum, const MetadataIndexRange &range) {
                                     return sum + static_cast<size_t>(range.count) * sizeof(RenderIndirectDrawMetadata);
                                 })) :
                0u;
        const bool uploadShadowMetadataFull =
            uploadShadowMetadata && (!shadowFrameUploadState.metadataValid || shadowChunkLayoutChanged ||
                                     (shadowChunkMetadataChanged && !shadowChunkUploadRangesValid));
        const bool uploadShadowMetadataHeader =
            uploadShadowMetadata && (!shadowFrameUploadState.metadataValid || shadowChunkMetadataChanged ||
                                     shadowFrameUploadState.chunkMetadataCount != shadowChunkMetadataCount ||
                                     shadowFrameUploadState.entityMetadataCount != shadowEntityMetadataCount ||
                                     shadowFrameUploadState.headerVersion != shadowMetadataVersion);
        const bool uploadShadowMetadataEntity =
            uploadShadowMetadata && (shadowEntitySolidBytes > 0 || shadowEntityAlphaOpaqueBytes > 0);
        const size_t shadowMetadataUploadBytes =
            uploadShadowMetadata ?
                (uploadShadowMetadataFull ?
                     static_cast<size_t>(directionalShadowMetadataBytes) :
                     ((uploadShadowMetadataHeader ? sizeof(RenderIndirectMetadataHeader) : 0u) +
                      shadowChunkRangeUploadBytes +
                      (uploadShadowMetadataEntity ? shadowEntitySolidBytes + shadowEntityAlphaOpaqueBytes : 0u))) :
                0u;
        MCVR_PROFILE_METRIC("shadow.metadata_upload_full", uploadShadowMetadataFull ? 1 : 0);
        MCVR_PROFILE_METRIC("shadow.metadata_upload_header", uploadShadowMetadataHeader ? 1 : 0);
        MCVR_PROFILE_METRIC("shadow.metadata_upload_entity", uploadShadowMetadataEntity ? 1 : 0);
        MCVR_PROFILE_METRIC(
            "shadow.metadata_upload_chunk_range",
            shadowChunkUploadRangesValid ? shadowSolidUploadRanges.size() + shadowAlphaOpaqueUploadRanges.size() : 0);
        MCVR_PROFILE_METRIC("shadow.metadata_upload_bytes", shadowMetadataUploadBytes);
        MCVR_PROFILE_METRIC("shadow.metadata_total_bytes", directionalShadowMetadataBytes);
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.shadow.pack");
            if (uploadShadowMetadata) {
                if (uploadShadowMetadataFull) {
                    shadowMetadataUpload.resize(static_cast<size_t>(directionalShadowMetadataBytes));
                    std::memcpy(shadowMetadataUpload.data(), &shadowMetadataHeader, sizeof(shadowMetadataHeader));
                    uint8_t *metadataWrite = shadowMetadataUpload.data() + sizeof(RenderIndirectMetadataHeader);

                    const size_t chunkSolidBytes =
                        drawMetadataCache.chunkShadowSolidMetadata.size() * sizeof(RenderIndirectDrawMetadata);
                    if (chunkSolidBytes > 0) {
                        std::memcpy(metadataWrite, drawMetadataCache.chunkShadowSolidMetadata.data(), chunkSolidBytes);
                        metadataWrite += chunkSolidBytes;
                    }
                    if (shadowEntitySolidBytes > 0) {
                        std::memcpy(metadataWrite, drawMetadataCache.entityShadowSolidMetadata.data(),
                                    shadowEntitySolidBytes);
                        metadataWrite += shadowEntitySolidBytes;
                    }
                    const size_t chunkAlphaOpaqueBytes =
                        drawMetadataCache.chunkShadowAlphaOpaqueMetadata.size() * sizeof(RenderIndirectDrawMetadata);
                    if (chunkAlphaOpaqueBytes > 0) {
                        std::memcpy(metadataWrite, drawMetadataCache.chunkShadowAlphaOpaqueMetadata.data(),
                                    chunkAlphaOpaqueBytes);
                        metadataWrite += chunkAlphaOpaqueBytes;
                    }
                    if (shadowEntityAlphaOpaqueBytes > 0) {
                        std::memcpy(metadataWrite, drawMetadataCache.entityShadowAlphaOpaqueMetadata.data(),
                                    shadowEntityAlphaOpaqueBytes);
                    }
                }
            }
        }

        auto &shadowMetadataTrackedState = context.trackedBufferStates[shadowMetadataBuffer.get()];
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "metadata.shadow.upload");
            if (uploadShadowMetadata) {
                barrierToTransferWrite(shadowMetadataTrackedState, shadowMetadataBuffer,
                                       directionalShadowMetadataBytes);
                if (uploadShadowMetadataFull) {
                    shadowMetadataBuffer->uploadToStagingBuffer(shadowMetadataUpload.data(),
                                                                shadowMetadataUpload.size(), 0);
                    shadowMetadataBuffer->uploadToBuffer(worldCommandBuffer, shadowMetadataUpload.size(), 0, 0);
                } else {
                    if (uploadShadowMetadataHeader) {
                        shadowMetadataBuffer->uploadToStagingBuffer(&shadowMetadataHeader, sizeof(shadowMetadataHeader),
                                                                    0);
                        shadowMetadataBuffer->uploadToBuffer(worldCommandBuffer, sizeof(shadowMetadataHeader), 0, 0);
                    }
                    if (shadowChunkUploadRangesValid) {
                        for (const MetadataIndexRange &range : shadowSolidUploadRanges) {
                            const size_t rangeBytes =
                                static_cast<size_t>(range.count) * sizeof(RenderIndirectDrawMetadata);
                            const size_t rangeOffset =
                                sizeof(RenderIndirectMetadataHeader) +
                                static_cast<size_t>(range.offset) * sizeof(RenderIndirectDrawMetadata);
                            shadowMetadataBuffer->uploadToStagingBuffer(
                                drawMetadataCache.chunkShadowSolidMetadata.data() + range.offset, rangeBytes,
                                rangeOffset);
                            shadowMetadataBuffer->uploadToBuffer(worldCommandBuffer, rangeBytes, rangeOffset,
                                                                 rangeOffset);
                        }
                        for (const MetadataIndexRange &range : shadowAlphaOpaqueUploadRanges) {
                            const size_t rangeBytes =
                                static_cast<size_t>(range.count) * sizeof(RenderIndirectDrawMetadata);
                            const size_t rangeOffset = sizeof(RenderIndirectMetadataHeader) +
                                                       (static_cast<size_t>(shadowChunkSolidMetadataCount) +
                                                        shadowEntitySolidMetadataCount + range.offset) *
                                                           sizeof(RenderIndirectDrawMetadata);
                            shadowMetadataBuffer->uploadToStagingBuffer(
                                drawMetadataCache.chunkShadowAlphaOpaqueMetadata.data() + range.offset, rangeBytes,
                                rangeOffset);
                            shadowMetadataBuffer->uploadToBuffer(worldCommandBuffer, rangeBytes, rangeOffset,
                                                                 rangeOffset);
                        }
                    }
                    if (uploadShadowMetadataEntity && shadowEntitySolidBytes > 0) {
                        shadowMetadataBuffer->uploadToStagingBuffer(drawMetadataCache.entityShadowSolidMetadata.data(),
                                                                    shadowEntitySolidBytes, shadowEntitySolidOffset);
                        shadowMetadataBuffer->uploadToBuffer(worldCommandBuffer, shadowEntitySolidBytes,
                                                             shadowEntitySolidOffset, shadowEntitySolidOffset);
                    }
                    if (uploadShadowMetadataEntity && shadowEntityAlphaOpaqueBytes > 0) {
                        shadowMetadataBuffer->uploadToStagingBuffer(
                            drawMetadataCache.entityShadowAlphaOpaqueMetadata.data(), shadowEntityAlphaOpaqueBytes,
                            shadowEntityAlphaOpaqueOffset);
                        shadowMetadataBuffer->uploadToBuffer(worldCommandBuffer, shadowEntityAlphaOpaqueBytes,
                                                             shadowEntityAlphaOpaqueOffset,
                                                             shadowEntityAlphaOpaqueOffset);
                    }
                }
            }
            barrierToRead(shadowMetadataTrackedState, shadowMetadataBuffer, directionalShadowMetadataBytes,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT);
            shadowFrameUploadState.metadataVersion = shadowMetadataVersion;
            shadowFrameUploadState.headerVersion = shadowMetadataVersion;
            shadowFrameUploadState.chunkMetadataVersion = shadowChunkMetadataVersion;
            shadowFrameUploadState.entityMetadataHash = 0;
            shadowFrameUploadState.chunkMetadataCount = shadowChunkMetadataCount;
            shadowFrameUploadState.chunkLeadingMetadataCount = shadowChunkSolidMetadataCount;
            shadowFrameUploadState.entityMetadataCount = shadowEntityMetadataCount;
            shadowFrameUploadState.entityLeadingMetadataCount = shadowEntitySolidMetadataCount;
            shadowFrameUploadState.metadataValid = true;
        }

        DirectionalShadowConfigUpdate shadowConfigUpdate{};
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "shadow.config");
            shadowConfigUpdate = updateDirectionalShadowConfig(context, frameIndex);
        }
        MCVR_PROFILE_METRIC("shadow.config_valid", shadowConfigUpdate.valid ? 1 : 0);
        MCVR_PROFILE_METRIC("shadow.config_updated", shadowConfigUpdate.updated ? 1 : 0);

        auto &shadowCommandTrackedState = context.trackedBufferStates[shadowCommandBuffer.get()];
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "shadow.indirectRecord");
            if (shadowCommandTrackedState.initialized) {
                worldCommandBuffer->barriersBufferImage(
                    {
                        {
                            .srcStageMask = shadowCommandTrackedState.stageMask == 0 ?
                                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                                shadowCommandTrackedState.stageMask,
                            .srcAccessMask = shadowCommandTrackedState.accessMask,
                            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                            .srcQueueFamilyIndex = queueIndex,
                            .dstQueueFamilyIndex = queueIndex,
                            .buffer = shadowCommandBuffer,
                            .offset = 0,
                            .size = directionalShadowCommandBytes,
                        },
                    },
                    {});
            }

            const uint32_t shadowIndirectDispatchGroups =
                (directionalShadowDrawCount + directionalShadowIndirectWorkgroupSize - 1) /
                directionalShadowIndirectWorkgroupSize;
            MCVR_PROFILE_METRIC("shadow.indirect_dispatch_groups", shadowIndirectDispatchGroups);
            {
                std::string debugLabelName =
                    MainRenderModule::buildMainRenderDebugLabel("shadow_indirect", pass.config.name);
                vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.85f,
                                                               0.30f, 0.15f);
                worldCommandBuffer
                    ->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
                    ->bindComputePipeline(directionalShadowIndirectComputePipeline_);
                vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), shadowIndirectDispatchGroups, 1, 1);
            }

            worldCommandBuffer->barriersBufferImage(
                {
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                        .srcQueueFamilyIndex = queueIndex,
                        .dstQueueFamilyIndex = queueIndex,
                        .buffer = shadowCommandBuffer,
                        .offset = 0,
                        .size = directionalShadowCommandBytes,
                    },
                },
                {});
            shadowCommandTrackedState.stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            shadowCommandTrackedState.accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            shadowCommandTrackedState.initialized = true;
        }
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "shadow.record");
            renderDirectionalShadowMaps(context, shadowCommandBuffer, directionalShadowSolidDrawCount,
                                        directionalShadowAlphaOpaqueDrawCount);
        }
    }

    if (!draws.empty()) {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "raster.indirectRecord");
        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "raster.indirectRecord.validate");
            if (pass.indirectCommandComputePipeline == nullptr) {
                throw std::runtime_error("main_render render pass indirect_draw command pipeline is unavailable: " +
                                         pass.config.name);
            }
            if (indirectCommandBuffer == nullptr) {
                throw std::runtime_error("main_render render pass indirect_draw internal buffers are unavailable: " +
                                         pass.config.name);
            }
        }

        const uint32_t indirectDispatchGroups =
            (static_cast<uint32_t>(draws.size()) + renderIndirectWorkgroupSize - 1) / renderIndirectWorkgroupSize;
        MCVR_PROFILE_METRIC("raster.indirect_dispatch_groups", indirectDispatchGroups);
        {
            std::string debugLabelName =
                MainRenderModule::buildMainRenderDebugLabel("raster_indirect", pass.config.name);
            vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.25f, 0.70f,
                                                           0.95f);
            worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE);
            worldCommandBuffer->pushDescriptorBuffers(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                      renderIndirectMetadataDescriptorSet,
                                                      {{.binding = renderIndirectMetadataDescriptorBinding,
                                                        .buffer = indirectMetadataBuffer->vkBuffer(),
                                                        .size = indirectMetadataBuffer->size(),
                                                        .owner = indirectMetadataBuffer},
                                                       {.binding = renderIndirectCommandsDescriptorBinding,
                                                        .buffer = indirectCommandBuffer->vkBuffer(),
                                                        .size = indirectCommandBuffer->size(),
                                                        .owner = indirectCommandBuffer}});
            worldCommandBuffer->bindComputePipeline(pass.indirectCommandComputePipeline);
            vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), indirectDispatchGroups, 1, 1);
        }

        {
            MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "raster.indirectRecord.barrier");
            worldCommandBuffer->barriersBufferImage(
                {
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                        .srcQueueFamilyIndex = queueIndex,
                        .dstQueueFamilyIndex = queueIndex,
                        .buffer = indirectCommandBuffer,
                        .offset = 0,
                        .size = indirectCommandBytes,
                    },
                },
                {});
            if (indirectCommandTrackedState != nullptr) {
                indirectCommandTrackedState->stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
                indirectCommandTrackedState->accessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
                indirectCommandTrackedState->initialized = true;
            }
        }
    }

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "raster.beginRenderPass");
        beginGraphicsRenderPass();
    }

    if (shaderPullingIndirect && indirectMetadataBuffer != nullptr) {
        worldCommandBuffer->pushDescriptorBuffers(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                  renderIndirectMetadataDescriptorSet,
                                                  {{.binding = renderIndirectMetadataDescriptorBinding,
                                                    .buffer = indirectMetadataBuffer->vkBuffer(),
                                                    .size = indirectMetadataBuffer->size(),
                                                    .owner = indirectMetadataBuffer}});
    }

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "raster.drawRecord");
        std::string debugLabelName = MainRenderModule::buildMainRenderDebugLabel("raster_draw", pass.config.name);
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.25f, 0.85f,
                                                       0.45f);
        vk::GraphicsPipeline *boundPipeline = nullptr;
        for (const RenderIndirectDrawBatch &drawBatch : indirectDrawBatchesForRecord) {
            const PreparedRenderDraw &groupDraw = draws[drawBatch.sourceOffset];
            if (groupDraw.pipeline != boundPipeline) {
                vkCmdBindPipeline(worldCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  groupDraw.pipeline->vkPipeline());
                boundPipeline = groupDraw.pipeline;
            }

            vkCmdDrawIndirect(worldCommandBuffer->vkCommandBuffer(), indirectCommandBuffer->vkBuffer(),
                              static_cast<VkDeviceSize>(drawBatch.sourceOffset) * sizeof(VkDrawIndirectCommand),
                              drawBatch.sourceCount, sizeof(VkDrawIndirectCommand));
        }
    }

    {
        MCVR_PROFILE_SCOPE(renderPassProfilePrefix + "raster.finishRenderPass");
        finishGraphicsRenderPass();
    }
}

std::vector<ShaderPack::ShaderCreateInfo>
MainRenderModule::collectRayTracingPassShaderRequests(RayTracingPass &pass,
                                                      const std::unordered_map<std::string, std::string> &definitions) {
    pass.missShaders.clear();
    pass.hitShaderGroups.clear();
    pass.hitGroupNameToIndex.clear();
    pass.shadowHitGroupIndex = 0;
    pass.fallbackHitGroupIndex = 0;
    pass.missGroupCount = 0;
    pass.hitGroupCount = 0;

    const uint32_t executionSet = shaderPack_->executionSet(5u);
    std::unordered_map<std::string, std::string> shaderDefinitions =
        shaderDefinitionsWithRuntimeExpressions(definitions);
    const auto framework = framework_.lock();
    const auto device = framework == nullptr ? nullptr : framework->device();
    shaderDefinitions["MCVR_SER_EXT"] = device != nullptr && device->hasShaderInvocationReorderEXT() ? "1" : "0";
    shaderDefinitions["MCVR_SER_NV"] = device != nullptr && device->hasShaderInvocationReorderNV() ? "1" : "0";

    std::vector<ShaderPack::ShaderCreateInfo> requests;

    for (size_t i = 0; i < pass.config.missShaders.size(); i++) {
        requests.push_back({
            .path = pass.config.missShaders[i].shaderPath,
            .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
            .definitions = shaderDefinitions,
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        });
    }

    std::vector<RayTracingPass::HitAssignment> hitAssignments;

    pass.hitShaderGroups.resize(pass.config.hitGroups.size());
    for (size_t groupIndex = 0; groupIndex < pass.config.hitGroups.size(); groupIndex++) {
        const auto &groupConfig = pass.config.hitGroups[groupIndex];
        const std::unordered_map<std::string, std::string> hitGroupDefinitions =
            shaderDefinitionsWithRuntimeExpressions(mergeShaderDefinitions(definitions, groupConfig.definitions));
        HitShaderGroup group;
        group.name = groupConfig.name;
        group.type = groupConfig.type;
        if (groupConfig.closestHit.has_value()) {
            requests.push_back({
                .path = *groupConfig.closestHit,
                .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                .definitions = hitGroupDefinitions,
                .executionStage = ShaderPackLoader::Stage::RayTracing,
                .executionSet = executionSet,
            });
            hitAssignments.push_back({groupIndex, 0});
        }
        if (groupConfig.anyHit.has_value()) {
            requests.push_back({
                .path = *groupConfig.anyHit,
                .stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                .definitions = hitGroupDefinitions,
                .executionStage = ShaderPackLoader::Stage::RayTracing,
                .executionSet = executionSet,
            });
            hitAssignments.push_back({groupIndex, 1});
        }
        if (groupConfig.intersection.has_value()) {
            requests.push_back({
                .path = *groupConfig.intersection,
                .stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                .definitions = hitGroupDefinitions,
                .executionStage = ShaderPackLoader::Stage::RayTracing,
                .executionSet = executionSet,
            });
            hitAssignments.push_back({groupIndex, 2});
        }

        pass.hitShaderGroups[groupIndex] = std::move(group);
        pass.hitGroupNameToIndex[pass.hitShaderGroups[groupIndex].name] = static_cast<uint32_t>(groupIndex);
        if (pass.hitShaderGroups[groupIndex].name == pass.config.defaultHitGroupName) {
            pass.fallbackHitGroupIndex = static_cast<uint32_t>(groupIndex);
        }
        if (pass.hitShaderGroups[groupIndex].name == "shadow") {
            pass.shadowHitGroupIndex = static_cast<uint32_t>(groupIndex);
        }
    }

    pass.missRequestCount_ = pass.config.missShaders.size();
    pass.hitRequestCount_ = hitAssignments.size();
    pass.hitAssignments_ = std::move(hitAssignments);

    std::unordered_map<std::string, std::string> queryDefinitions = shaderDefinitions;
    if (pass.querySharcEnabled) {
        queryDefinitions["USE_SHARC"] = "1";
        queryDefinitions["SHARC_QUERY"] = "1";
    }

    std::unordered_map<std::string, std::string> updateDefinitions = shaderDefinitions;
    if (pass.isSharcUpdatePass) {
        updateDefinitions["USE_SHARC"] = "1";
        updateDefinitions["SHARC_UPDATE"] = "1";
    }

    requests.push_back({
        .path = pass.config.rayGenShaderPath,
        .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        .definitions = queryDefinitions,
        .executionStage = ShaderPackLoader::Stage::RayTracing,
        .executionSet = executionSet,
    });
    pass.rayGenRequestCount_ = 1;
    if (pass.isSharcUpdatePass) {
        requests.push_back({
            .path = pass.config.rayGenShaderPath,
            .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
            .definitions = updateDefinitions,
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        });
        pass.rayGenRequestCount_ = 2;
    }

    pass.sharcRequestCount_ = 0;
    if (pass.isSharcUpdatePass) {
        if (!shaderPack_->shaderPack().sharc.has_value()) {
            throw std::runtime_error("missing sharc config for update pass: " + pass.config.name);
        }
        requests.push_back({
            .path = shaderPack_->shaderPack().sharc->resolveCompShaderPath,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .definitions = updateDefinitions,
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        });
        pass.sharcRequestCount_ = 1;
    }

    return requests;
}

void MainRenderModule::buildRayTracingPassPipelines(RayTracingPass &pass,
                                                    std::shared_ptr<vk::Device> device,
                                                    const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                                    size_t &shaderOffset) {
    auto framework = framework_.lock();
    uint32_t frameCount = framework->swapchain()->imageCount();

    pass.rayGenUpdateShader = nullptr;
    pass.rayGenQueryShader = nullptr;
    pass.updatePipeline = nullptr;
    pass.queryPipeline = nullptr;
    pass.updateSbts.clear();
    pass.querySbts.clear();
    pass.sharcResolveCompShader = nullptr;
    pass.sharcResolvePipeline = nullptr;

    for (size_t i = 0; i < pass.missRequestCount_; i++) {
        pass.missShaders.push_back({
            .name = pass.config.missShaders[i].name,
            .index = static_cast<uint32_t>(i),
            .shader = compiledShaders[shaderOffset + i],
        });
    }
    shaderOffset += pass.missRequestCount_;

    for (size_t i = 0; i < pass.hitRequestCount_; i++) {
        auto &group = pass.hitShaderGroups[pass.hitAssignments_[i].groupIndex];
        switch (pass.hitAssignments_[i].shaderType) {
            case 0: group.closestHitShader = compiledShaders[shaderOffset + i]; break;
            case 1: group.anyHitShader = compiledShaders[shaderOffset + i]; break;
            case 2: group.intersectionShader = compiledShaders[shaderOffset + i]; break;
        }
    }
    shaderOffset += pass.hitRequestCount_;
    if (pass.hitGroupNameToIndex.find("shadow") == pass.hitGroupNameToIndex.end()) {
        pass.shadowHitGroupIndex = pass.fallbackHitGroupIndex;
    }
    pass.hitGroupCount = static_cast<uint32_t>(pass.hitShaderGroups.size());
    pass.missGroupCount = static_cast<uint32_t>(pass.missShaders.size());

    auto buildRtPipeline = [&](const std::shared_ptr<vk::Shader> &rayGenShader) {
        vk::RayTracingPipelineBuilder builder;
        auto &stageBuilder = builder.beginShaderStage();

        uint32_t stageIndex = 0;
        stageBuilder.defineShaderStage(rayGenShader, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
        stageIndex++;

        std::vector<uint32_t> missStageIndices;
        for (const auto &missShader : pass.missShaders) {
            missStageIndices.push_back(stageIndex);
            stageBuilder.defineShaderStage(missShader.shader, VK_SHADER_STAGE_MISS_BIT_KHR);
            stageIndex++;
        }

        struct HitStageIndices {
            VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            uint32_t closestHit = VK_SHADER_UNUSED_KHR;
            uint32_t anyHit = VK_SHADER_UNUSED_KHR;
            uint32_t intersection = VK_SHADER_UNUSED_KHR;
        };
        std::vector<HitStageIndices> hitStageIndices;
        for (const auto &group : pass.hitShaderGroups) {
            HitStageIndices indices;
            indices.type = group.type;
            if (group.closestHitShader) {
                indices.closestHit = stageIndex;
                stageBuilder.defineShaderStage(group.closestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
                stageIndex++;
            }
            if (group.anyHitShader) {
                indices.anyHit = stageIndex;
                stageBuilder.defineShaderStage(group.anyHitShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
                stageIndex++;
            }
            if (group.intersectionShader) {
                indices.intersection = stageIndex;
                stageBuilder.defineShaderStage(group.intersectionShader, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
                stageIndex++;
            }
            hitStageIndices.push_back(indices);
        }
        stageBuilder.endShaderStage();

        auto &groupBuilder = builder.beginShaderGroup();
        groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR,
                                       VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
        for (uint32_t missStageIndex : missStageIndices) {
            groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, missStageIndex,
                                           VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
        }
        for (const auto &indices : hitStageIndices) {
            groupBuilder.defineShaderGroup(indices.type, VK_SHADER_UNUSED_KHR, indices.closestHit, indices.anyHit,
                                           indices.intersection);
        }
        groupBuilder.endShaderGroup();

        return builder.definePipelineLayout(mainRenderDescriptorTables_[0])
            .defineMaxRayRecursionDepth(pass.config.maxRayRecursionDepth)
            .build(device);
    };

    pass.rayGenQueryShader = compiledShaders[shaderOffset];
    if (pass.isSharcUpdatePass && pass.rayGenRequestCount_ > 1) {
        pass.rayGenUpdateShader = compiledShaders[shaderOffset + 1];
    } else {
        pass.rayGenUpdateShader = pass.rayGenQueryShader;
    }
    shaderOffset += pass.rayGenRequestCount_;

    pass.queryPipeline = buildRtPipeline(pass.rayGenQueryShader);
    pass.updatePipeline = pass.isSharcUpdatePass ? buildRtPipeline(pass.rayGenUpdateShader) : pass.queryPipeline;

    if (pass.isSharcUpdatePass && pass.sharcRequestCount_ > 0) {
        pass.sharcResolveCompShader = compiledShaders[shaderOffset];
        shaderOffset += pass.sharcRequestCount_;
        pass.sharcResolvePipeline = vk::ComputePipelineBuilder{}
                                        .defineShader(pass.sharcResolveCompShader)
                                        .definePipelineLayout(mainRenderDescriptorTables_[0])
                                        .build(device);
    }

    pass.querySbts.resize(frameCount);
    if (pass.isSharcUpdatePass) { pass.updateSbts.resize(frameCount); }
    for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        if (pass.isSharcUpdatePass) {
            pass.updateSbts[frameIndex] = vk::SBT::create(framework->physicalDevice(), device, framework->vma(),
                                                          pass.updatePipeline, pass.missGroupCount, pass.hitGroupCount);
        }
        pass.querySbts[frameIndex] = vk::SBT::create(framework->physicalDevice(), device, framework->vma(),
                                                     pass.queryPipeline, pass.missGroupCount, pass.hitGroupCount);
    }
}

std::vector<ShaderPack::ShaderCreateInfo>
MainRenderModule::collectComputePassShaderRequests(const ComputePass &pass,
                                                   const std::unordered_map<std::string, std::string> &definitions) {
    const uint32_t executionSet = shaderPack_->executionSet(5u);
    std::unordered_map<std::string, std::string> computeDefinitions =
        shaderDefinitionsWithRuntimeExpressions(definitions);
    if (hasSharcRuntime_ && pass.config.querySharc) {
        computeDefinitions["USE_SHARC"] = "1";
        computeDefinitions["SHARC_QUERY"] = "1";
    }
    return {{
        .path = pass.config.computeShaderPath,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .definitions = computeDefinitions,
        .executionStage = ShaderPackLoader::Stage::RayTracing,
        .executionSet = executionSet,
    }};
}

void MainRenderModule::buildComputePassPipelines(ComputePass &pass,
                                                 std::shared_ptr<vk::Device> device,
                                                 const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                                 size_t &shaderOffset) {
    pass.computeShader = compiledShaders[shaderOffset++];
    pass.pipeline = vk::ComputePipelineBuilder{}
                        .defineShader(pass.computeShader)
                        .definePipelineLayout(mainRenderDescriptorTables_[0])
                        .build(device);
}

size_t MainRenderModule::shaderRequestCountForPass(PassVariant &passVariant) {
    return std::visit(
        [&](auto &pass) -> size_t {
            using PassType = std::decay_t<decltype(pass)>;
            if constexpr (std::is_same_v<PassType, std::shared_ptr<FullScreenPass>>) {
                std::shared_ptr<vk::DeviceLocalImage> targetImage = findTargetImage(pass->config.target, 0);
                const bool implicitCubeFace =
                    targetImage->layer() == 6 && !pass->config.face.has_value() && !pass->config.layer.has_value();
                return implicitCubeFace ? 6u : 1u;
            } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                return pass->config.pipelineConfigs.size() * 2u;
            } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                return pass->missRequestCount_ + pass->hitRequestCount_ + pass->rayGenRequestCount_ +
                       pass->sharcRequestCount_;
            } else if constexpr (std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                return 1u;
            }
            return 0u;
        },
        passVariant);
}

void MainRenderModule::uploadStaticRayTracingPassSbts(std::shared_ptr<vk::Device> device) {
    auto framework = framework_.lock();
    if (framework == nullptr) { return; }

    bool hasStaticSbts = false;
    for (auto &passVariant : passes_) {
        bool passHasStaticSbts = std::visit(
            [](auto &pass) {
                using PassType = std::decay_t<decltype(pass)>;
                if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                    return !pass->querySbts.empty() || !pass->updateSbts.empty();
                }
                return false;
            },
            passVariant);
        if (passHasStaticSbts) {
            hasStaticSbts = true;
            break;
        }
    }
    if (!hasStaticSbts) { return; }

    auto commandPool = vk::CommandPool::create(framework->physicalDevice(), device);
    auto commandBuffer = vk::CommandBuffer::create(device, commandPool);
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    for (auto &passVariant : passes_) {
        std::visit(
            [&](auto &pass) {
                using PassType = std::decay_t<decltype(pass)>;
                if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                    for (auto &sbt : pass->updateSbts) {
                        if (sbt != nullptr) { sbt->uploadStaticSBT(commandBuffer); }
                    }
                    for (auto &sbt : pass->querySbts) {
                        if (sbt != nullptr) { sbt->uploadStaticSBT(commandBuffer); }
                    }
                }
            },
            passVariant);
    }
    commandBuffer->end()->submitMainQueueIndividual(device);
    {
        std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
        vk::requireSuccess(vkQueueWaitIdle(device->mainVkQueue()), "vkQueueWaitIdle(shader binding upload)");
    }
}

void MainRenderModule::renderFullScreenPass(const FullScreenPass &pass, MainRenderModuleContext &context) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.renderFullScreenPass");
    auto frameworkContext = context.frameworkContext.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    uint32_t frameIndex = frameworkContext->frameIndex;
    const ResolvedPassResources &resources = resolvedResourcesForPass(pass);

    std::shared_ptr<vk::DeviceLocalImage> targetImage =
        resolveImageResource(resources.fullScreenTarget->resource, frameIndex);
    if (targetImage == nullptr) { return; }
    int faceCount = static_cast<int>(pass.framebuffers[frameIndex].size());
    int faceStart = 0;
    int faceEnd = faceCount;
    if (pass.config.face.has_value()) {
        faceStart = static_cast<int>(*pass.config.face);
        faceEnd = faceStart + 1;
    } else if (pass.config.layer.has_value()) {
        faceStart = static_cast<int>(*pass.config.layer);
        faceEnd = faceStart + 1;
    }
    for (int faceIndex = faceStart; faceIndex < faceEnd; faceIndex++) {
        size_t pipelineIndex = pass.pipelines.size() == 1 ? 0 : static_cast<size_t>(faceIndex);
        std::string label = buildMainRenderDebugLabel("fullscreen", pass.config.name);
        if (faceCount > 1) { label += "/face_" + std::to_string(faceIndex); }
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, label.c_str(), 0.15f, 0.55f, 1.0f);
        worldCommandBuffer->beginRenderPass({
            .renderPass = pass.renderPass,
            .framebuffer = pass.framebuffers[frameIndex][faceIndex],
            .renderAreaExtent = {targetImage->width(), targetImage->height()},
            .clearValues = {},
        });
        worldCommandBuffer->bindGraphicsPipeline(pass.pipelines[pipelineIndex])
            ->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);
        worldCommandBuffer->draw(3, 1)->endRenderPass();
    }
}

void MainRenderModule::renderRayTracingPass(RayTracingPass &pass,
                                            MainRenderModuleContext &context,
                                            const std::vector<ExpressionEvaluator::Variable> &expressionVariables) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.renderRayTracingPass");
    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    uint32_t frameIndex = frameworkContext->frameIndex;
    const ResolvedPassResources &resources = resolvedResourcesForPass(pass);

    if (!context.ensureRayTracingWorldPrepareResources()) { return; }

    {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.setupRayTracingHitGroupSbt");
        context.worldPrepareContext->setupHitGroupSbt(pass.hitGroupNameToIndex, pass.fallbackHitGroupIndex,
                                                      pass.shadowHitGroupIndex, worldCommandBuffer, nullptr,
                                                      pass.querySbts[frameIndex]);
    }

    auto evaluateTraceDimension = [&](const std::string &expression, const std::string &axis) {
        double value = evaluateNumericExpression(expression, expressionVariables);
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error("invalid ray tracing dispatch " + axis + " for pass " + pass.config.name + ": " +
                                     expression);
        }
        return static_cast<uint32_t>(std::max(1.0, std::ceil(value)));
    };

    const uint32_t traceWidth = evaluateTraceDimension(pass.config.width, "width");
    const uint32_t traceHeight = evaluateTraceDimension(pass.config.height, "height");
    const uint32_t traceDepth = evaluateTraceDimension(pass.config.depth, "depth");
    std::string label = buildMainRenderDebugLabel("ray_tracing", pass.config.name);
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, label.c_str(), 0.85f, 0.35f, 1.0f);

    if (pass.querySharcEnabled) { prepareSharcResources(context, false); }

    auto command = worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable,
                                                           VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
    {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.bindRayTracingPipeline");
        command->bindRTPipeline(pass.queryPipeline);
    }
    MCVR_PROFILE_METRIC("ray_tracing.dispatch_width", traceWidth);
    MCVR_PROFILE_METRIC("ray_tracing.dispatch_height", traceHeight);
    MCVR_PROFILE_METRIC("ray_tracing.dispatch_depth", traceDepth);
    const bool captureDispatch = context.timingCapture &&
        context.timingCapture->beginStage(worldCommandBuffer->vkCommandBuffer(), MainRenderTimingCapture::RayDispatch);
    if (captureDispatch) {
        context.timingCapture->traceWidth = traceWidth;
        context.timingCapture->traceHeight = traceHeight;
    }
    if (pass.config.indirectArgsBuffer.has_value()) {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.traceRaysIndirectCommand");
        command->raytracingIndirect(pass.querySbts[frameIndex],
                                    findRuntimeVKBuffer(*resources.indirectArgsBuffer, frameIndex)->bufferAddress());
    } else {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.traceRaysCommand");
        command->raytracing(pass.querySbts[frameIndex], traceWidth, traceHeight, traceDepth);
    }
    if (captureDispatch) context.timingCapture->endStage(worldCommandBuffer->vkCommandBuffer(),
                                                       MainRenderTimingCapture::RayDispatch);
}

void MainRenderModule::prepareSharcResources(MainRenderModuleContext &context, bool writes) {
    if (!writes && context.sharcQueryReady) { return; }
    auto frameworkContext = context.frameworkContext.lock();
    const VkPipelineStageFlags2 shaderStages =
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    std::vector<vk::CommandBuffer::BufferMemoryBarrier> barriers;
    for (const auto &buffer :
         {sharcHashEntriesBuffer_, sharcLockBuffer_, sharcAccumulationBuffer_, sharcResolvedBuffer_}) {
        if (buffer == nullptr) { continue; }
        barriers.push_back({
            .srcStageMask = shaderStages | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
                             (writes ? VK_ACCESS_2_SHADER_READ_BIT : 0),
            .dstStageMask = writes ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR : shaderStages,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | (writes ? VK_ACCESS_2_SHADER_WRITE_BIT : 0),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
        });
    }
    if (!barriers.empty()) { frameworkContext->worldCommandBuffer->barriersBufferImage(barriers, {}); }
    context.sharcQueryReady = !writes;
}

void MainRenderModule::renderSharcUpdateAndResolve(
    RayTracingPass &pass,
    MainRenderModuleContext &context,
    const std::vector<ExpressionEvaluator::Variable> &expressionVariables) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.renderSharcUpdateAndResolve");
    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    uint32_t frameIndex = frameworkContext->frameIndex;

    if (!context.ensureRayTracingWorldPrepareResources()) { return; }
    if (!pass.isSharcUpdatePass || pass.updatePipeline == nullptr || pass.sharcResolvePipeline == nullptr ||
        pass.updateSbts.empty()) {
        throw std::runtime_error("invalid sharc update pass state: " + pass.config.name);
    }

    context.worldPrepareContext->setupHitGroupSbt(pass.hitGroupNameToIndex, pass.fallbackHitGroupIndex,
                                                  pass.shadowHitGroupIndex, worldCommandBuffer,
                                                  pass.updateSbts[frameIndex], nullptr);

    updateSharcConfig(frameIndex);
    context.mainRenderDescriptorTable->bindBuffer(sharcConfigBuffers_[frameIndex], 4, 0);

    auto evaluateTraceDimension = [&](const std::string &expression, const std::string &axis) {
        double value = evaluateNumericExpression(expression, expressionVariables);
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error("invalid ray tracing dispatch " + axis + " for pass " + pass.config.name + ": " +
                                     expression);
        }
        return static_cast<uint32_t>(std::max(1.0, std::ceil(value)));
    };

    const uint32_t traceWidth = evaluateTraceDimension(pass.config.width, "width");
    const uint32_t traceHeight = evaluateTraceDimension(pass.config.height, "height");
    const uint32_t updateDownsampleFactor = std::max(1u, sharcUpdateDownsampleFactor_);
    const uint32_t updateWidth = (traceWidth + updateDownsampleFactor - 1) / updateDownsampleFactor;
    const uint32_t updateHeight = (traceHeight + updateDownsampleFactor - 1) / updateDownsampleFactor;
    prepareSharcResources(context, true);
    {
        std::string label = buildMainRenderDebugLabel("sharc_update_trace", pass.config.name);
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, label.c_str(), 1.0f, 0.45f, 0.25f);
        worldCommandBuffer
            ->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
            ->bindRTPipeline(pass.updatePipeline)
            ->raytracing(pass.updateSbts[frameIndex], updateWidth, updateHeight, 1);
    }

    worldCommandBuffer->barriersMemory({{
        .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
    }});

    {
        std::string label = buildMainRenderDebugLabel("sharc_resolve", pass.config.name);
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, label.c_str(), 1.0f, 0.70f, 0.20f);
        worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(pass.sharcResolvePipeline);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(),
                      (currentSharcResolveEntryCount_ + sharcResolveWorkgroupSize - 1) / sharcResolveWorkgroupSize, 1,
                      1);
    }
}

void MainRenderModule::renderComputePass(const ComputePass &pass,
                                         MainRenderModuleContext &context,
                                         const std::vector<ExpressionEvaluator::Variable> &expressionVariables) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.renderComputePass");
    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    uint32_t frameIndex = frameworkContext->frameIndex;
    const ResolvedPassResources &resources = resolvedResourcesForPass(pass);

    if (hasSharcRuntime_ && pass.config.querySharc) { prepareSharcResources(context, false); }

    std::string label = buildMainRenderDebugLabel("compute", pass.config.name);
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, label.c_str(), 0.20f, 0.90f, 0.60f);
    worldCommandBuffer->bindDescriptorTable(context.mainRenderDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(pass.pipeline);
    if (pass.config.indirectArgsBuffer.has_value()) {
        vkCmdDispatchIndirect(worldCommandBuffer->vkCommandBuffer(),
                              findRuntimeVKBuffer(*resources.indirectArgsBuffer, frameIndex)->vkBuffer(), 0);
    } else {
        auto evaluateDispatchDimension = [&](const std::string &expression, const std::string &axis) {
            double value = evaluateNumericExpression(expression, expressionVariables);
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::runtime_error("invalid compute dispatch " + axis + " for pass " + pass.config.name + ": " +
                                         expression);
            }
            return static_cast<uint32_t>(std::max(1.0, std::ceil(value)));
        };

        const uint32_t gx = evaluateDispatchDimension(pass.config.gx, "gx");
        const uint32_t gy = evaluateDispatchDimension(pass.config.gy, "gy");
        const uint32_t gz = evaluateDispatchDimension(pass.config.gz, "gz");
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), gx, gy, gz);
    }
}

void MainRenderModule::initPipelines() {
    auto framework = framework_.lock();
    auto device = framework->device();
    uint32_t frameCount = framework->swapchain()->imageCount();
    const auto &shaderPack = shaderPack_->shaderPack();
    size_t executionBufferSize = shaderPack.rayTracingExecution.variables.size() * sizeof(float);
    const std::unordered_set<std::string> activePassNames =
        shaderPack_->activePassNames(ShaderPackLoader::Stage::RayTracing, true);

    passes_.clear();
    passNameToPass_.clear();
    fullScreenPassResources_.clear();
    renderPassResources_.clear();
    rayTracingPassResources_.clear();
    computePassResources_.clear();
    sharcUpdatePass_ = nullptr;
    hasRasterIndirectPass_ = false;

    for (const auto &passConfig : shaderPack.passes) {
        if (passConfig.stage != ShaderPackLoader::Stage::RayTracing) { continue; }
        const std::string *passName = nullptr;
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen: passName = &passConfig.fullScreen.name; break;
            case ShaderPackLoader::PassConfig::Type::Render: passName = &passConfig.render.name; break;
            case ShaderPackLoader::PassConfig::Type::RayTracing: passName = &passConfig.rayTracing.name; break;
            case ShaderPackLoader::PassConfig::Type::Compute: passName = &passConfig.compute.name; break;
        }
        if (passName != nullptr && activePassNames.find(*passName) == activePassNames.end()) { continue; }

        if (passConfig.type == ShaderPackLoader::PassConfig::Type::Render) {
            if (!renderPassIsHandOnly(passConfig.render)) { continue; }
            if (passConfig.render.indirectDraw.has_value()) {
                throw std::runtime_error("HAND raster pass cannot declare indirect_draw: " + passConfig.render.name);
            }
        }
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen: {
                std::shared_ptr<vk::DeviceLocalImage> targetImage = findTargetImage(passConfig.fullScreen.target, 0);
                if (targetImage == nullptr) {
                    throw std::runtime_error("Invalid full-screen pass target: " + passConfig.fullScreen.target);
                }
                if (passConfig.fullScreen.face.has_value() && targetImage->layer() != 6) {
                    throw std::runtime_error("Full-screen pass face requires cube target: " +
                                             passConfig.fullScreen.name);
                }
                if (passConfig.fullScreen.layer.has_value() && targetImage->layer() <= 1) {
                    throw std::runtime_error("Full-screen pass layer requires array target: " +
                                             passConfig.fullScreen.name);
                }
                if (passConfig.fullScreen.layer.has_value() && *passConfig.fullScreen.layer >= targetImage->layer()) {
                    throw std::runtime_error("Full-screen pass layer out of range: " + passConfig.fullScreen.name);
                }

                auto pass = std::make_shared<FullScreenPass>();
                pass->config = passConfig.fullScreen;
                pass->executionBuffer =
                    ShaderPack::createPassExecutionBuffer(device, framework->vma(), executionBufferSize);
                initFullScreenPassTargets(*pass, device, frameCount);
                fullScreenPassResources_.emplace(
                    pass.get(), resolvePassResources(pass->config.inputs, pass->config.outputs, std::nullopt,
                                                     std::optional<std::string_view>(pass->config.target), true));

                passes_.push_back(pass);
                passNameToPass_.emplace(pass->config.name, pass);
                break;
            }
            case ShaderPackLoader::PassConfig::Type::Render: {
                auto pass = std::make_shared<RenderPass>();
                pass->config = passConfig.render;
                hasRasterIndirectPass_ = false;
                pass->rendersHandOnly = renderPassIsHandOnly(pass->config);
                pass->rendersPrimaryRaster = !pass->rendersHandOnly && renderPassLooksRasterPrimary(pass->config);
                pass->executionBuffer =
                    ShaderPack::createPassExecutionBuffer(device, framework->vma(), executionBufferSize);
                initRenderPassTargets(*pass, device, frameCount);
                ResolvedPassResources resources =
                    resolvePassResources(pass->config.inputs, pass->config.outputs, std::nullopt, std::nullopt, true);
                for (size_t outputIndex = 0; outputIndex < pass->config.outputs.images.size() &&
                                             outputIndex < resources.colorAttachmentTargets.size();
                     ++outputIndex) {
                    if (pass->config.clearOutputs.find(pass->config.outputs.images[outputIndex]) ==
                        pass->config.clearOutputs.end()) {
                        continue;
                    }
                    ResolvedImageAccess &targetAccess = resources.colorAttachmentTargets[outputIndex];
                    targetAccess.accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    targetAccess.reads = false;
                    targetAccess.writes = true;
                    targetAccess.needsPreviousContent = false;
                }
                addRenderIndirectResources(*pass, resources);
                renderPassResources_.emplace(pass.get(), std::move(resources));

                passes_.push_back(pass);
                passNameToPass_.emplace(pass->config.name, pass);
                break;
            }
            case ShaderPackLoader::PassConfig::Type::RayTracing: {
                auto pass = std::make_shared<RayTracingPass>();
                pass->config = passConfig.rayTracing;
                pass->querySharcEnabled = hasSharcRuntime_ && passConfig.rayTracing.querySharc;
                pass->isSharcUpdatePass = false;
                pass->executionBuffer =
                    ShaderPack::createPassExecutionBuffer(device, framework->vma(), executionBufferSize);
                rayTracingPassResources_.emplace(pass.get(),
                                                 resolvePassResources(pass->config.inputs, pass->config.outputs,
                                                                      pass->config.indirectArgsBuffer, std::nullopt));

                passes_.push_back(pass);
                passNameToPass_.emplace(pass->config.name, pass);
                break;
            }
            case ShaderPackLoader::PassConfig::Type::Compute: {
                auto pass = std::make_shared<ComputePass>();
                pass->config = passConfig.compute;
                pass->executionBuffer =
                    ShaderPack::createPassExecutionBuffer(device, framework->vma(), executionBufferSize);
                computePassResources_.emplace(pass.get(),
                                              resolvePassResources(pass->config.inputs, pass->config.outputs,
                                                                   pass->config.indirectArgsBuffer, std::nullopt));

                passes_.push_back(pass);
                passNameToPass_.emplace(pass->config.name, pass);
                break;
            }
        }
    }

    if (hasSharcRuntime_) {
        if (!shaderPack_->shaderPack().sharc.has_value()) { throw std::runtime_error("missing sharc config"); }
        auto passIter = passNameToPass_.find(shaderPack_->shaderPack().sharc->updatePassName);
        if (passIter == passNameToPass_.end()) {
            throw std::runtime_error("unknown sharc update pass: " + shaderPack_->shaderPack().sharc->updatePassName);
        }
        auto updatePass = std::get_if<std::shared_ptr<RayTracingPass>>(&passIter->second);
        if (updatePass == nullptr || *updatePass == nullptr) {
            throw std::runtime_error("sharc update pass must be ray_tracing: " +
                                     shaderPack_->shaderPack().sharc->updatePassName);
        }
        sharcUpdatePass_ = *updatePass;
        sharcUpdatePass_->isSharcUpdatePass = true;
    }

    std::vector<ShaderPack::ShaderCreateInfo> allShaderRequests;
    for (auto &passVariant : passes_) {
        std::visit(
            [&](auto &pass) {
                using PassType = std::decay_t<decltype(pass)>;
                std::vector<ShaderPack::ShaderCreateInfo> requests;
                if constexpr (std::is_same_v<PassType, std::shared_ptr<FullScreenPass>>) {
                    requests = collectFullScreenPassShaderRequests(*pass, pass->config.definitions);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                    requests = collectRenderPassShaderRequests(*pass, pass->config.definitions);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                    requests = collectRayTracingPassShaderRequests(*pass, pass->config.definitions);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                    requests = collectComputePassShaderRequests(*pass, pass->config.definitions);
                }
                allShaderRequests.insert(allShaderRequests.end(), requests.begin(), requests.end());
            },
            passVariant);
    }

    std::vector<std::shared_ptr<vk::Shader>> allCompiledShaders = shaderPack_->createShaders(device, allShaderRequests);
    initDirectionalShadowPipelines();

    std::vector<size_t> shaderOffsets(passes_.size());
    size_t totalShaderCount = 0;
    for (size_t i = 0; i < passes_.size(); ++i) {
        shaderOffsets[i] = totalShaderCount;
        totalShaderCount += shaderRequestCountForPass(passes_[i]);
    }
    if (totalShaderCount != allCompiledShaders.size()) {
        throw std::runtime_error("main render shader offset count mismatch");
    }

    auto buildPassAtIndex = [&](size_t passIndex) {
        size_t shaderOffset = shaderOffsets[passIndex];
        std::visit(
            [&](auto &pass) {
                using PassType = std::decay_t<decltype(pass)>;
                if constexpr (std::is_same_v<PassType, std::shared_ptr<FullScreenPass>>) {
                    buildFullScreenPassPipelines(*pass, device, allCompiledShaders, shaderOffset);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                    buildRenderPassPipelines(*pass, device, allCompiledShaders, shaderOffset);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RayTracingPass>>) {
                    buildRayTracingPassPipelines(*pass, device, allCompiledShaders, shaderOffset);
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<ComputePass>>) {
                    buildComputePassPipelines(*pass, device, allCompiledShaders, shaderOffset);
                }
            },
            passes_[passIndex]);
    };

    mcvr::parallelFor(
        passes_.size(), [&](size_t passIndex) { buildPassAtIndex(passIndex); }, "MCVR_RT_PIPELINE_BUILD_THREADS");
    uploadStaticRayTracingPassSbts(device);
    compileExecutionProgram();
}

void MainRenderModule::initSBTs() {}

void MainRenderModule::compileExecutionProgram() {
    compiledExecutionProgram_ = nullptr;
    if (shaderPack_ == nullptr) { return; }

    auto passExecutionOpBuilder = [this](const std::string &passName) -> std::unique_ptr<ShaderPack::ExecutionOp> {
        auto passIter = passNameToPass_.find(passName);
        if (passIter == passNameToPass_.end()) { return std::make_unique<ShaderPack::SequentialOp>(); }
        return std::make_unique<MainRenderPassExecutionOp>(passIter->second);
    };

    auto sharcUpdateExecutionOpBuilder =
        [this](const std::string &passName) -> std::unique_ptr<ShaderPack::ExecutionOp> {
        if (!hasSharcRuntime_) { return std::make_unique<ShaderPack::SequentialOp>(); }
        if (!shaderPack_->shaderPack().sharc.has_value()) {
            throw std::runtime_error("sharc_update requires root.sharc");
        }
        if (passName != shaderPack_->shaderPack().sharc->updatePassName) {
            throw std::runtime_error("sharc_update pass mismatch: " + passName);
        }
        if (sharcUpdatePass_ == nullptr) { throw std::runtime_error("missing sharc update pass: " + passName); }
        return std::make_unique<SharcUpdateExecutionOp>(passName, sharcUpdatePass_);
    };
    auto bufferUpdateExecutionOpBuilder =
        [](const ShaderPackLoader::ExecutionCommand::BufferUpdate &update) -> std::unique_ptr<ShaderPack::ExecutionOp> {
        return std::make_unique<MainRenderBufferUpdateExecutionOp>(update);
    };

    compiledExecutionProgram_ =
        shaderPack_->compileExecutionProgram(ShaderPackLoader::Stage::RayTracing, true, passExecutionOpBuilder,
                                             sharcUpdateExecutionOpBuilder, bufferUpdateExecutionOpBuilder);
}

void MainRenderModule::initContexts() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    contexts_.resize(size);

    for (uint32_t i = 0; i < size; i++) {
        contexts_[i] =
            MainRenderModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
        worldPrepare_->contexts_[i]->mainRenderModuleContext =
            std::static_pointer_cast<MainRenderModuleContext>(contexts_[i]);
    }
}

void MainRenderModule::build() {
    auto framework = framework_.lock();
    (void)framework;

    loadShaderPack();
    initExecutionVariables();
    worldPrepare_->build();
    initDescriptorTables();
    initRuntimeTextures();
    initRuntimeBuffers();
    loadRuntimeResources();
    initSharc();
    initPipelines();
    initContexts();
}

std::vector<std::shared_ptr<WorldModuleContext>> &MainRenderModule::contexts() {
    return contexts_;
}

void MainRenderModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    for (uint32_t i = 0; i < size; i++) {
        if (mainRenderDescriptorTables_[i] != nullptr) {
            mainRenderDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                             0, 0, index);
        }
    }
}

void MainRenderModule::preClose() {
    contexts_.clear();
    passes_.clear();
    passNameToPass_.clear();
    sharcUpdatePass_ = nullptr;
    compiledExecutionProgram_ = nullptr;
    executionVariableConfigs_.clear();
    globalVariables_.clear();
    worldPrepare_ = nullptr;
    shaderPack_ = nullptr;
}

MainRenderModuleContext::MainRenderModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<MainRenderModule> mainRenderModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      mainRenderModule(mainRenderModule),
      mainRenderDescriptorTable(mainRenderModule->mainRenderDescriptorTables_[frameworkContext->frameIndex]),
      hdrNoisyOutputImage(mainRenderModule->hdrNoisyOutputImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(mainRenderModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(mainRenderModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(mainRenderModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(mainRenderModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(mainRenderModule->linearDepthImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(mainRenderModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      indirectSpecularHitDistanceImage(
          mainRenderModule->indirectSpecularHitDistanceImages_[frameworkContext->frameIndex]),
      indirectDiffuseHitDistanceImage(
          mainRenderModule->indirectDiffuseHitDistanceImages_[frameworkContext->frameIndex]),
      indirectDiffuseRadianceImage(mainRenderModule->indirectDiffuseRadianceImages_[frameworkContext->frameIndex]),
      indirectDiffuseDirectionImage(mainRenderModule->indirectDiffuseDirectionImages_[frameworkContext->frameIndex]),
      indirectSpecularRadianceImage(mainRenderModule->indirectSpecularRadianceImages_[frameworkContext->frameIndex]),
      indirectSpecularDirectionImage(mainRenderModule->indirectSpecularDirectionImages_[frameworkContext->frameIndex]),
      firstHitClearImage(mainRenderModule->firstHitClearImages_[frameworkContext->frameIndex]),
      firstHitRefractionImage(mainRenderModule->firstHitRefractionImages_[frameworkContext->frameIndex]),
      firstHitBaseEmissionImage(mainRenderModule->firstHitBaseEmissionImages_[frameworkContext->frameIndex]),
      fogImage(mainRenderModule->fogImages_[frameworkContext->frameIndex]),
      directDiffuseRadianceImage(mainRenderModule->directDiffuseRadianceImages_[frameworkContext->frameIndex]),
      directSpecularRadianceImage(mainRenderModule->directSpecularRadianceImages_[frameworkContext->frameIndex]),
      directHitDistanceImage(mainRenderModule->directHitDistanceImages_[frameworkContext->frameIndex]),
      directLightGuideNormalRoughnessImage(
          mainRenderModule->directLightGuideNormalRoughnessImages_[frameworkContext->frameIndex]),
      directLightGuideMotionDepthImage(
          mainRenderModule->directLightGuideMotionDepthImages_[frameworkContext->frameIndex]),
      segmentationMaskImage(mainRenderModule->segmentationMaskImages_[frameworkContext->frameIndex]),
      worldPrepareContext(mainRenderModule->worldPrepare_->contexts_[frameworkContext->frameIndex]) {}

bool MainRenderModuleContext::populateFrameGenerationInputs(mcvr::FrameGenerationInputs &inputs) {
    if (linearDepthImage == nullptr || motionVectorImage == nullptr) { return false; }

    inputs.linearDepth = linearDepthImage;
    inputs.rawMotionVector = motionVectorImage;
    inputs.renderWidth = linearDepthImage->width();
    inputs.renderHeight = linearDepthImage->height();
    return true;
}

void MainRenderModuleContext::bindWorldPrepareResources(bool includeAccelerationStructure) {
    MCVR_PROFILE_SCOPE("native.MainRenderModule.bindWorldPrepareResources");
    if (worldPrepareContext == nullptr || worldPrepareContext->tlasInstanceBuffer == nullptr) { return; }

    if (includeAccelerationStructure && worldPrepareContext->tlas != nullptr) {
        if (boundWorldPrepareTlas != worldPrepareContext->tlas) {
            MCVR_PROFILE_SCOPE("native.MainRenderModule.bindWorldPrepareTlas");
            mainRenderDescriptorTable->bindAS(worldPrepareContext->tlas, 1, 0);
            boundWorldPrepareTlas = worldPrepareContext->tlas;
#if defined(MCVR_ENABLE_PROFILER)
            ++worldPrepareTlasBindCount;
#endif
        } else {
#if defined(MCVR_ENABLE_PROFILER)
            ++worldPrepareTlasBindSkippedCount;
#endif
        }
    }
    if (!worldPrepareBuffersBound) {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.bindWorldPrepareBuffers");
        auto buffers = Renderer::instance().buffers();
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->blasOffsetsBuffer, 1, 1);
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->indexBufferAddr, 1, 2);
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->lastIndexBufferAddr, 1, 3);
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->pbrBufferAddr, 1, 4);
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->lastPbrBufferAddr, 1, 5);
        mainRenderDescriptorTable->bindBuffer(buffers->textureMappingBuffer(), 1, 7);
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->lastObjToWorldMat, 1, 8);
        mainRenderDescriptorTable->bindBuffer(worldPrepareContext->tlasInstanceBuffer, 1, 10);
        worldPrepareBuffersBound = true;
#if defined(MCVR_ENABLE_PROFILER)
        worldPrepareBufferBindCount += 8;
#endif
    } else {
#if defined(MCVR_ENABLE_PROFILER)
        worldPrepareBufferBindSkippedCount += 8;
#endif
    }
}

bool MainRenderModuleContext::ensureRayTracingWorldPrepareResources() {
    if (worldPrepareContext == nullptr || !worldPrepareContext->prepareAccelerationStructures()) { return false; }
    bindWorldPrepareResources(true);
    return worldPrepareContext->tlas != nullptr;
}

void MainRenderModuleContext::render() {
    MCVR_PROFILE_SCOPE("native.MainRenderModuleContext.render");
    worldPrepareBuffersBound = false;
    boundWorldPrepareTlas.reset();
#if defined(MCVR_ENABLE_PROFILER)
    worldPrepareBufferBindCount = 0;
    worldPrepareBufferBindSkippedCount = 0;
    worldPrepareTlasBindCount = 0;
    worldPrepareTlasBindSkippedCount = 0;
#endif
    auto module = mainRenderModule.lock();
    if (module == nullptr) { return; }

    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    (void)framework;

    auto buffers = Renderer::instance().buffers();
    auto chunks = Renderer::instance().world()->chunks();
    if (!timingCapture) {
        const char* capture = std::getenv("MCVR_PERFORMANCE_CAPTURE");
        if (capture && std::strcmp(capture, "1") == 0)
            timingCapture = std::make_shared<MainRenderTimingCapture>(context);
    }
    if (timingCapture) timingCapture->beginFrame(worldCommandBuffer->vkCommandBuffer(),
                                               hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height());
    if (!buffers->hasWorldHistory()) { module->initExecutionVariables(); }
    trackedImageStates.clear();
    trackedBufferStates.clear();
    sharcQueryReady = false;

    mainRenderDescriptorTable->bindBuffer(buffers->worldUniformBuffer(), 2, 0);
    mainRenderDescriptorTable->bindBuffer(buffers->lastWorldUniformBuffer(), 2, 1);
    mainRenderDescriptorTable->bindBuffer(buffers->skyUniformBuffer(), 2, 2);

    const auto accelerationStructureBuildMode = module->shaderPack_ == nullptr ?
                                                    ShaderPackLoader::AccelerationStructureBuildMode::Sync :
                                                    module->shaderPack_->accelerationStructureBuildMode();
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/world_prepare", 0.75f, 0.45f,
                                                       0.20f);
        const bool capture = timingCapture && timingCapture->beginStage(worldCommandBuffer->vkCommandBuffer(),
                                                                      MainRenderTimingCapture::WorldPrepare);
        worldPrepareContext->render(accelerationStructureBuildMode);
        if (capture) timingCapture->endStage(worldCommandBuffer->vkCommandBuffer(), MainRenderTimingCapture::WorldPrepare);
    }

    const std::vector<ExpressionEvaluator::Variable> runtimeExpressionVariables =
        module->executionExpressionVariables();
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/runtime_buffers", 0.70f, 0.50f,
                                                       0.25f);
        module->refreshRuntimeBuffers(runtimeExpressionVariables);
    }

    if (Renderer::options.collectChunkEmission && chunks != nullptr && chunks->chunkPackedData() != nullptr) {
        mainRenderDescriptorTable->bindBuffer(chunks->chunkPackedData(), 1, 9);
    }
    bindWorldPrepareResources(true);
    {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/clear_segmentation", 0.65f,
                                                       0.55f, 0.30f);
        module->clearSegmentationMask(*this);
    }

    MainRenderModule::ExecutionVariables variables;
    variables.reserve(module->globalVariables_.size());
    for (const auto &[name, value] : module->globalVariables_) {
        auto config = module->findExecutionVariableConfig(name);
        variables.emplace(name, MainRenderModule::ExecutionVariable{
                                    .name = name,
                                    .value = value,
                                    .type = config.has_value() ? config->get().type : "",
                                });
    }

    if (module->compiledExecutionProgram_ != nullptr) {
        MCVR_PROFILE_SCOPE("native.MainRenderModule.executeProgram");
        MainRenderModule::MainRenderFrameGraph frameGraph;
        MainRenderModule::MainRenderExecutionRuntime runtime;
        runtime.executionVariables = std::move(variables);
        runtime.runtimeExpressionVariables = runtimeExpressionVariables;
        runtime.loopLimit = MainRenderModule::executionLoopLimit;
        runtime.mainRenderModule = module;
        runtime.mainRenderContext = shared_from_this();
        runtime.frameGraph = &frameGraph;
        (*module->compiledExecutionProgram_)(runtime);
        variables = std::move(runtime.executionVariables);
        {
            vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/frame_graph", 0.60f, 0.60f,
                                                           0.35f);
            const bool capture = timingCapture && timingCapture->beginStage(worldCommandBuffer->vkCommandBuffer(),
                                                                          MainRenderTimingCapture::FrameGraph);
            frameGraph.record(*module, *this);
            if (capture) timingCapture->endStage(worldCommandBuffer->vkCommandBuffer(), MainRenderTimingCapture::FrameGraph);
        }
        {
            vk::CommandBuffer::ScopedDebugLabel debugLabel(
                *worldCommandBuffer, "main_render/prepare_segmentation_tone_mapping", 0.55f, 0.65f, 0.40f);
            module->prepareSegmentationMaskForToneMapping(*this);
        }
    } else if (!module->shaderPack_->execution(ShaderPackLoader::Stage::RayTracing).commands.empty()) {
        throw std::runtime_error("main_render execution program was not compiled");
    }

    if (accelerationStructureBuildMode == ShaderPackLoader::AccelerationStructureBuildMode::Deferred) {
        vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "main_render/flush_pending_blas", 0.50f,
                                                       0.70f, 0.45f);
        worldPrepareContext->flushPendingHighPriorityBLASBuilds();
    }

    for (auto &[name, value] : module->globalVariables_) {
        auto iter = variables.find(name);
        if (iter != variables.end()) { value = iter->second.value; }
    }
    MCVR_PROFILE_METRIC("world_descriptors.buffer_bind_count", worldPrepareBufferBindCount);
    MCVR_PROFILE_METRIC("world_descriptors.buffer_bind_skipped_count", worldPrepareBufferBindSkippedCount);
    MCVR_PROFILE_METRIC("world_descriptors.tlas_bind_count", worldPrepareTlasBindCount);
    MCVR_PROFILE_METRIC("world_descriptors.tlas_bind_skipped_count", worldPrepareTlasBindSkippedCount);
}
