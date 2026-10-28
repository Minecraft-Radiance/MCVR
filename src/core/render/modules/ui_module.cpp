#include "core/render/modules/ui_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"

#include <fstream>
#include <regex>
#include <stdexcept>

namespace {
bool isFlatUiProbeEnabled() {
#ifdef __APPLE__
    static const bool enabled = [] {
        const char *mode = std::getenv("MCVR_UI_PRESENT_PROBE");
        return mode != nullptr && std::strcmp(mode, "flat-ui") == 0;
    }();
    return enabled;
#else
    return false;
#endif
}
}

UIModule::UIModule() {}

UIModule::~UIModule() {
}

void UIModule::init(std::shared_ptr<Framework> framework) {
    vk::ResourceMemoryTagScope memoryTag("ui");
    framework_ = framework;

    initOverlayDescriptorTablesAndFrameSamplers();
    initOverlayDrawImages();
    initOverlayDrawRenderPass();
    initOverlayDrawFrameBuffers();

    initOverlayPostImages();
    initOverlayPostRenderPass();
    initOverlayPostFrameBuffers();
    initOverlayPostPipelineTypes();
    initOverlayPostPipelines();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_.resize(size);
    for (int i = 0; i < size; i++) {
        contexts_[i] = UIModuleContext::create(framework->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<UIModuleContext>> &UIModule::contexts() {
    return contexts_;
}

std::vector<std::shared_ptr<vk::DescriptorTable>> &UIModule::overlayDescriptorTables() {
    return overlayDescriptorTables_;
}

const std::vector<OverlayDynamicDrawShaderInfo> &UIModule::overlayDynamicDrawShaders() const {
    return overlayDynamicDrawShaders_;
}

uint32_t UIModule::registerOverlayDrawShader(const std::string &key,
                                             uint32_t vertexFormatType,
                                             uint32_t drawMode,
                                             uint32_t uniformSize,
                                             const std::string &vertexShaderPath,
                                             const std::string &fragmentShaderPath,
                                             const std::unordered_map<std::string, std::string> &definitions) {
    enum class OverlayAttributeNumericKind {
        FLOAT,
        SINT,
        UINT,
    };

    struct OverlayAttributeType {
        OverlayAttributeNumericKind numericKind;
    };

    auto overlayTopologyForDrawMode = [](uint32_t overlayDrawMode) -> VkPrimitiveTopology {
        switch (overlayDrawMode) {
            case 1: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
            case 7: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            default: throw std::runtime_error("Unsupported overlay draw mode");
        }
    };
    auto overlayVertexLayoutFor = [](uint32_t overlayVertexFormatType) -> vk::VertexLayoutInfo {
        switch (overlayVertexFormatType) {
            case 0: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexLightNormal>();
            case 1: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexOverlayLightNormal>();
            case 2: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColorLight>();
            case 3: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionOnly>();
            case 4: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColor>();
            case 5: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorNormal>();
            case 6: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorLight>();
            case 7: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTex>();
            case 8: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColor>();
            case 9: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexLight>();
            case 10: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexLightColor>();
            case 11: return vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColorNormal>();
            default: throw std::runtime_error("Unsupported overlay vertex format type");
        }
    };
    auto parseOverlayVertexInputs =
        [](const std::string &overlayVertexShaderPath) -> std::unordered_map<uint32_t, OverlayAttributeType> {
        std::ifstream sourceFile(overlayVertexShaderPath, std::ios::binary);
        if (!sourceFile.is_open()) { return {}; }

        std::string sourceText{std::istreambuf_iterator<char>(sourceFile), std::istreambuf_iterator<char>()};
        std::regex inputPattern(
            R"(layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*in\s+(float|int|uint|vec[234]|ivec[234]|uvec[234])\s+\w+\s*;)");
        std::unordered_map<uint32_t, OverlayAttributeType> result;

        for (std::sregex_iterator it(sourceText.begin(), sourceText.end(), inputPattern), end; it != end; ++it) {
            uint32_t location = static_cast<uint32_t>(std::stoul((*it)[1].str()));
            std::string type = (*it)[2].str();

            OverlayAttributeNumericKind numericKind = OverlayAttributeNumericKind::FLOAT;
            if (!type.empty() && type[0] == 'i') {
                numericKind = OverlayAttributeNumericKind::SINT;
            } else if (!type.empty() && type[0] == 'u') {
                numericKind = OverlayAttributeNumericKind::UINT;
            }

            result[location] = OverlayAttributeType{numericKind};
        }

        return result;
    };
    auto adaptOverlayAttributeFormat = [](VkFormat format, OverlayAttributeNumericKind numericKind) -> VkFormat {
        if (numericKind == OverlayAttributeNumericKind::FLOAT) {
            switch (format) {
                case VK_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SSCALED;
                case VK_FORMAT_R16G16_UINT: return VK_FORMAT_R16G16_USCALED;
                case VK_FORMAT_R16_SINT: return VK_FORMAT_R16_SSCALED;
                case VK_FORMAT_R16_UINT: return VK_FORMAT_R16_USCALED;
                case VK_FORMAT_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_SSCALED;
                case VK_FORMAT_R8G8B8A8_UINT: return VK_FORMAT_R8G8B8A8_USCALED;
                default: return format;
            }
        }
        return format;
    };
    auto makeOverlayVertexLayout = [&](uint32_t overlayVertexFormatType,
                                       const std::string &overlayVertexShaderPath) -> vk::VertexLayoutInfo {
        vk::VertexLayoutInfo layout = overlayVertexLayoutFor(overlayVertexFormatType);
        std::unordered_map<uint32_t, OverlayAttributeType> inputs = parseOverlayVertexInputs(overlayVertexShaderPath);

        for (VkVertexInputAttributeDescription &attribute : layout.attributeDescriptions) {
            auto it = inputs.find(attribute.location);
            if (it == inputs.end()) { continue; }
            attribute.format = adaptOverlayAttributeFormat(attribute.format, it->second.numericKind);
        }

        return layout;
    };

    auto existing = overlayDynamicDrawShaderIds_.find(key);
    if (existing != overlayDynamicDrawShaderIds_.end()) { return existing->second; }

    auto framework = framework_.lock();
    if (framework == nullptr) { throw std::runtime_error("Framework is not available"); }
    bool descriptorRangeChanged = Renderer::instance().buffers()->registerOverlayDrawUniformSize(uniformSize);
    if (descriptorRangeChanged) {
        for (uint32_t frameIndex = 0; frameIndex < overlayDescriptorTables_.size(); ++frameIndex) {
            refreshOverlayDescriptorTable(frameIndex);
        }
    }

    OverlayDynamicDrawShaderInfo info{};
    info.key = key;
    info.vertexFormatType = vertexFormatType;
    info.drawMode = drawMode;
    info.uniformSize = uniformSize;
    info.vertexShaderPath = vertexShaderPath;
    info.fragmentShaderPath = fragmentShaderPath;
    info.definitions = definitions;
    info.topology = overlayTopologyForDrawMode(drawMode);
    info.shaders.vertexShader =
        vk::Shader::create(framework->device(), vertexShaderPath, VK_SHADER_STAGE_VERTEX_BIT, definitions);
    std::string effectiveFragmentPath = fragmentShaderPath;
    if (isFlatUiProbeEnabled()) {
        // E07 changes the fragment stage only: no textures, discard or UBOs.
        std::filesystem::path flatPath = std::filesystem::path(fragmentShaderPath).parent_path() /
                                         "radiance-e07-flat-ui.frag";
        if (!std::filesystem::exists(flatPath)) {
            std::ofstream source(flatPath, std::ios::binary);
            source << "#version 460\nlayout(location = 0) out vec4 outColor;\n"
                      "void main() { outColor = vec4(0.0, 1.0, 0.0, 1.0); }\n";
            if (!source) { throw std::runtime_error("E07: cannot write flat UI fragment shader"); }
        }
        effectiveFragmentPath = flatPath.string();
        std::fprintf(stderr, "[RADIANCE-E07] flat fragment registered key=%s vertexFormat=%u uniformBytes=%u\n",
                     key.c_str(), vertexFormatType, uniformSize);
    }
    info.shaders.fragmentShader =
        vk::Shader::create(framework->device(), effectiveFragmentPath, VK_SHADER_STAGE_FRAGMENT_BIT, definitions);

    info.vertexLayout = makeOverlayVertexLayout(vertexFormatType, vertexShaderPath);
    if (framework->device()->hasDynamicColorBlendState()) {
        vk::DynamicGraphicsPipelineBuilder builder{1};
        builder.defineRenderPass(overlayDrawRenderPass_, 0)
            .beginShaderStage()
            .defineShaderStage(info.shaders.vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
            .defineShaderStage(info.shaders.fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
            .endShaderStage();
        builder.defineVertexInputState(info.vertexLayout);
        info.pipeline = builder.defineInputAssemblyState(info.topology)
                            .definePipelineLayout(overlayDescriptorTables_[0])
                            .build(framework->device());
    }

    uint32_t shaderId = overlayDynamicDrawShaders_.size();
    overlayDynamicDrawShaderIds_[key] = shaderId;
    overlayDynamicDrawShaders_.push_back(std::move(info));
    return shaderId;
}

const OverlayDynamicDrawShaderInfo &UIModule::overlayDrawShaderInfo(uint32_t shaderId) const {
    return overlayDynamicDrawShaders_.at(shaderId);
}

std::shared_ptr<vk::DynamicGraphicsPipeline> UIModule::overlayDrawPipeline(
    uint32_t shaderId, VkPipelineColorBlendAttachmentState colorState) {
    auto framework = framework_.lock();
    auto &info = overlayDynamicDrawShaders_.at(shaderId);
    if (framework->device()->hasDynamicColorBlendState()) { return info.pipeline; }

    // Explicit fields avoid padding-dependent keys and preserve separate alpha
    // blending. Cache ownership keeps variants alive while frames are in flight.
    std::array<uint32_t, 8> key{
        colorState.blendEnable, uint32_t(colorState.srcColorBlendFactor), uint32_t(colorState.dstColorBlendFactor),
        uint32_t(colorState.colorBlendOp), uint32_t(colorState.srcAlphaBlendFactor),
        uint32_t(colorState.dstAlphaBlendFactor), uint32_t(colorState.alphaBlendOp), colorState.colorWriteMask,
    };
    auto existing = info.colorPipelines.find(key);
    if (existing != info.colorPipelines.end()) { return existing->second; }
    vk::DynamicGraphicsPipelineBuilder builder{1};
    builder.defineRenderPass(overlayDrawRenderPass_, 0)
        .beginShaderStage()
        .defineShaderStage(info.shaders.vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
        .defineShaderStage(info.shaders.fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
        .endShaderStage();
    builder.defineVertexInputState(info.vertexLayout);
    auto pipeline = builder.defineInputAssemblyState(info.topology)
                        .definePipelineLayout(overlayDescriptorTables_[0])
                        .defineColorBlendAttachmentState(0, colorState)
                        .build(framework->device());
    info.colorPipelines.emplace(key, pipeline);
    std::fprintf(stderr, "[RADIANCE-E15] static color pipeline shader=%s variant=%zu blend=%u mask=0x%x factors=%u,%u,%u,%u ops=%u,%u\n",
                 info.key.c_str(), info.colorPipelines.size(), key[0], key[7], key[1], key[2], key[4], key[5], key[3], key[6]);
    return pipeline;
}

std::shared_ptr<vk::DescriptorTable> UIModule::createOverlayDescriptorTable() {
    auto framework = framework_.lock();

    return vk::DescriptorTableBuilder{}
        .beginDescriptorLayoutSet()
        .beginDescriptorLayoutSetBinding()
        .defineDescriptorLayoutSetBinding({
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = framework->device()->textureDescriptorCapacity(),
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        })
        .defineDescriptorLayoutSetBinding({
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        })
        .endDescriptorLayoutSetBinding()
        .endDescriptorLayoutSet()
        .beginDescriptorLayoutSet()
        .beginDescriptorLayoutSetBinding()
        .defineDescriptorLayoutSetBinding({
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        })
        .defineDescriptorLayoutSetBinding({
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        })
        .endDescriptorLayoutSetBinding()
        .endDescriptorLayoutSet()
        .build(framework->device());
}

void UIModule::bindOverlayDescriptorTableResources(std::shared_ptr<vk::DescriptorTable> descriptorTable,
                                                   uint32_t frameIndex) {
    for (auto &[index, binding] : overlayTextureBindings_) {
        descriptorTable->bindSamplerImage(binding.sampler, binding.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
                                          0, index);
    }

    descriptorTable->bindSamplerImage(overlayDrawColorImageSamplers_[frameIndex], overlayDrawColorImages_[frameIndex],
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1, 0);
    descriptorTable->bindBufferRange(Renderer::instance().buffers()->overlayDrawUniformBuffer(), 1, 0, 0,
                                     Renderer::instance().buffers()->overlayDrawUniformDescriptorRange());
    descriptorTable->bindBufferRange(Renderer::instance().buffers()->overlayPostUniformBuffer(), 1, 1, 0,
                                     Renderer::instance().buffers()->overlayPostUniformDescriptorRange());
}

void UIModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                           std::shared_ptr<vk::DeviceLocalImage> image,
                           int index) {
    auto framework = framework_.lock();
    if (index < 0 || uint32_t(index) >= framework->device()->textureDescriptorCapacity()) {
        throw std::runtime_error("UI texture ID exceeds supported descriptor capacity");
    }
    overlayTextureBindings_[index] = {
        .sampler = sampler,
        .image = image,
    };

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        overlayDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0,
                                                      index);
    }
}

void UIModule::refreshOverlayDescriptorTable(uint32_t frameIndex) {
    auto framework = framework_.lock();
    auto &frr = framework->frameResourceRetainer();

    auto oldDescriptorTable = overlayDescriptorTables_[frameIndex];
    auto descriptorTable = createOverlayDescriptorTable();
    bindOverlayDescriptorTableResources(descriptorTable, frameIndex);

    overlayDescriptorTables_[frameIndex] = descriptorTable;
    if (frameIndex < contexts_.size() && contexts_[frameIndex] != nullptr) {
        contexts_[frameIndex]->overlayDescriptorTable = descriptorTable;
    }

    frr.retain(oldDescriptorTable);
}

void UIModule::initOverlayDescriptorTablesAndFrameSamplers() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayDescriptorTables_.resize(size);
    overlayDrawColorImageSamplers_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayDescriptorTables_[i] = createOverlayDescriptorTable();

        overlayDrawColorImageSamplers_[i] = vk::Sampler::create(
            framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }
}

void UIModule::initOverlayDrawImages() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayDrawColorImages_.resize(size);
    overlayDrawDepthStencilImages_.resize(size);
#ifdef __APPLE__
    std::fprintf(stderr, "[RADIANCE-E06] UI offscreen images: RGBA8_UNORM %ux%u, COLOR_ATTACHMENT|SAMPLED|TRANSFER_SRC|TRANSFER_DST, images=%u\n",
                 framework->swapchain()->vkExtent().width, framework->swapchain()->vkExtent().height, size);
#endif

    for (int i = 0; i < size; i++) {
        overlayDrawColorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, framework->swapchain()->vkExtent().width,
            framework->swapchain()->vkExtent().height, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#if defined(USE_AMD) || defined(__APPLE__)
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
        );
        overlayDrawDepthStencilImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, framework->swapchain()->vkExtent().width,
            framework->swapchain()->vkExtent().height, 1, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

        overlayDescriptorTables_[i]->bindSamplerImage(overlayDrawColorImageSamplers_[i], overlayDrawColorImages_[i],
                                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1, 0);
    }
}

void UIModule::initOverlayDrawRenderPass() {
    auto framework = framework_.lock();

    overlayDrawRenderPass_ = vk::RenderPassBuilder{}
                                 .beginAttachmentDescription()
                                 .defineAttachmentDescription(VkAttachmentDescription{
                                     .format = overlayDrawColorImages_[0]->vkFormat(),
                                     .samples = VK_SAMPLE_COUNT_1_BIT,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
#if defined(USE_AMD) || defined(__APPLE__)
                                     .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                                     .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                     .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                                 })
                                 .defineAttachmentDescription(VkAttachmentDescription{
                                     .format = overlayDrawDepthStencilImages_[0]->vkFormat(),
                                     .samples = VK_SAMPLE_COUNT_1_BIT,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 })
                                 .endAttachmentDescription()
                                 .beginAttachmentReference()
                                 .defineAttachmentReference({
                                     .attachment = 0,
                                     .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 })
                                 .defineAttachmentReference({
                                     .attachment = 1,
                                     .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 })
                                 .endAttachmentReference()
                                 .beginSubpassDescription()
                                 .defineSubpassDescription({
                                     .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     .colorAttachmentIndices = {0},
                                     .depthStencilAttachmentIndex = 1,
                                 })
                                 .endSubpassDescription()
                                 .build(framework->device());
}

void UIModule::initOverlayDrawFrameBuffers() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayDrawFramebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayDrawFramebuffers_[i] = vk::FramebufferBuilder{}
                                          .beginAttachment()
                                          .defineAttachment(overlayDrawColorImages_[i])
                                          .defineAttachment(overlayDrawDepthStencilImages_[i])
                                          .endAttachment()
                                          .build(framework->device(), overlayDrawRenderPass_);
    }
}

void UIModule::initOverlayPostImages() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayPostColorImages_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayPostColorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, framework->swapchain()->vkExtent().width,
            framework->swapchain()->vkExtent().height, 1, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#if defined(USE_AMD) || defined(__APPLE__)
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
        );
    }
}

void UIModule::initOverlayPostRenderPass() {
    auto framework = framework_.lock();

    overlayPostRenderPass_ = vk::RenderPassBuilder{}
                                 .beginAttachmentDescription()
                                 .defineAttachmentDescription({
                                     .format = overlayPostColorImages_[0]->vkFormat(),
                                     .samples = VK_SAMPLE_COUNT_1_BIT,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                     .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#if defined(USE_AMD) || defined(__APPLE__)
                                     .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                                     .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                     .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
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
                                 .build(framework->device());
}

void UIModule::initOverlayPostFrameBuffers() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayPostFramebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayPostFramebuffers_[i] = vk::FramebufferBuilder{}
                                          .beginAttachment()
                                          .defineAttachment(overlayPostColorImages_[i])
                                          .endAttachment()
                                          .build(framework->device(), overlayPostRenderPass_);
    }
}

void UIModule::initOverlayPostPipelineTypes() {
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    overlayPostPipelineInfos_[BLUR] = {
        .vertexShaderFile = (shaderPath / "overlay/post/blur_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/post/blur_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
}

void UIModule::initOverlayPostPipelines() {
    auto framework = framework_.lock();

    for (auto overlayPipelineInfo : overlayPostPipelineInfos_) {
        auto [type, info] = overlayPipelineInfo;
        auto [vertexShaderFile, fragmentShaderFile, topology] = info;

        overlayPostPipelineShaders_[type] = {
            .vertexShader = vk::Shader::create(framework->device(), vertexShaderFile),
            .fragmentShader = vk::Shader::create(framework->device(), fragmentShaderFile),
        };

#ifdef DEBUG
        {
            std::stringstream ss;
            ss << "UI Post Vertex Shader Type " << type;
            std::string vertName = ss.str();

            VkDebugUtilsObjectNameInfoEXT nameInfo = {};
            nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
            nameInfo.objectHandle = (uint64_t)overlayPostPipelineShaders_[type].vertexShader->vkShaderModule();
            nameInfo.pObjectName = vertName.c_str();

            vkSetDebugUtilsObjectNameEXT(framework->device()->vkDevice(), &nameInfo);
        }

        {
            std::stringstream ss;
            ss << "UI Post Fragment Shader Type " << type;
            std::string fragName = ss.str();

            VkDebugUtilsObjectNameInfoEXT nameInfo = {};
            nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
            nameInfo.objectHandle = (uint64_t)overlayPostPipelineShaders_[type].fragmentShader->vkShaderModule();
            nameInfo.pObjectName = fragName.c_str();

            vkSetDebugUtilsObjectNameEXT(framework->device()->vkDevice(), &nameInfo);
        }
#endif

        overlayPostPipelines_[type] =
            vk::GraphicsPipelineBuilder{}
                .defineRenderPass(overlayPostRenderPass_, 0)
                .beginShaderStage()
                .defineShaderStage(overlayPostPipelineShaders_[type].vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
                .defineShaderStage(overlayPostPipelineShaders_[type].fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
                .endShaderStage()
                .defineVertexInputState<void>()
                .defineViewportScissorState({
                    .viewport =
                        {
                            .x = 0,
                            .y = 0,
                            .width = static_cast<float>(framework->swapchain()->vkExtent().width),
                            .height = static_cast<float>(framework->swapchain()->vkExtent().height),
                            .minDepth = 0.0,
                            .maxDepth = 1.0,
                        },
                    .scissor =
                        {
                            .offset = {.x = 0, .y = 0},
                            .extent = framework->swapchain()->vkExtent(),
                        },
                })
                .defineDepthStencilState({
                    .depthTestEnable = VK_TRUE,
                    .depthWriteEnable = VK_TRUE,
                    .depthCompareOp = VK_COMPARE_OP_LESS,
                    .depthBoundsTestEnable = VK_FALSE,
                    .stencilTestEnable = VK_FALSE,
                })
                .beginColorBlendAttachmentState()
                .defineDefaultColorBlendAttachmentState()
                .endColorBlendAttachmentState()
                .definePipelineLayout(overlayDescriptorTables_[0])
                .build(framework->device());
    }
}

UIModuleContext::UIModuleContext(std::shared_ptr<FrameworkContext> context, std::shared_ptr<UIModule> uiModule)
    : frameworkContext(context),
      uiModule(uiModule),
      overlayDescriptorTable(uiModule->overlayDescriptorTables_[context->frameIndex]),
      overlayDrawColorImage(uiModule->overlayDrawColorImages_[context->frameIndex]),
      overlayDrawDepthStencilImage(uiModule->overlayDrawDepthStencilImages_[context->frameIndex]),
      overlayDrawFramebuffer(uiModule->overlayDrawFramebuffers_[context->frameIndex]),
      overlayPostColorImage(uiModule->overlayPostColorImages_[context->frameIndex]),
      overlayDrawColorImageSampler(uiModule->overlayDrawColorImageSamplers_[context->frameIndex]),
      overlayPostFramebuffer(uiModule->overlayPostFramebuffers_[context->frameIndex]) {
    overlayScissorEnabled = VK_FALSE;
    overlayScissor = {
        .offset = {0, 0},
        .extent = context->swapchain->vkExtent(),
    };
    overlayViewport = {
        .x = 0,
        .y = 0,
        .width = static_cast<float>(context->swapchain->vkExtent().width),
        .height = static_cast<float>(context->swapchain->vkExtent().height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };

    overlayBlendEnabled = VK_FALSE;
    overlayColorBlendEquation = {
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
    };
    overlayColorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    overlayColorLogicOpEnable = VK_FALSE;
    overlayColorLogicOp = VK_LOGIC_OP_COPY;
    overlayBlendConstants = {0.f, 0.f, 0.f, 0.f};

    overlayDepthTestEnable = VK_FALSE;
    overlayDepthWriteEnable = VK_TRUE;
    overlayDepthCompareOp = VK_COMPARE_OP_LESS;
    overlayStencilTestEnable = VK_FALSE;
    overlayFailOp = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    overlayPassOp = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    overlayDepthFailOp = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    overlayCompareOp = {VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS};
    overlayReference = {0, 0};
    overlayCompareMask = {0xffffffff, 0xffffffff};
    overlayWriteMask = {0xffffffff, 0xffffffff};

    overlayCullMode = VK_CULL_MODE_NONE;
    overlayFrontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    overlayPolygonMode = VK_POLYGON_MODE_FILL;
    overlayDepthBiasEnable = VK_FALSE;
    overlayDepthBiasConstantFactor = {0.0, 0.0, 0.0};
    overlayDepthBiasClamp = {0.0, 0.0, 0.0};
    overlayDepthBiasSlopeFactor = {0.0, 0.0, 0.0};
    overlayLineWidth = 1.0;

    overlayClearColors = {1.0, 1.0, 1.0, 1.0};
    overlayClearDepth = 1.0;
    overlayClearStencil = 0xffffffff;
}

void UIModuleContext::syncToCommandBuffer() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    auto commandBuffer = context->overlayCommandBuffer->vkCommandBuffer();

    vkCmdSetViewport(commandBuffer, 0, 1, &overlayViewport);

    if (overlayScissorEnabled)
        vkCmdSetScissor(commandBuffer, 0, 1, &overlayScissor);
    else {
        VkRect2D full_scissor = {
            .offset = {0, 0},
            .extent = framework->swapchain()->vkExtent(),
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &full_scissor);
    }

    vkCmdSetDepthTestEnable(commandBuffer, overlayDepthTestEnable);

    vkCmdSetDepthWriteEnable(commandBuffer, overlayDepthWriteEnable);

    vkCmdSetDepthCompareOp(commandBuffer, overlayDepthCompareOp);

    vkCmdSetStencilTestEnable(commandBuffer, overlayStencilTestEnable);

    vkCmdSetStencilOp(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayFailOp[0], overlayPassOp[0],
                      overlayDepthFailOp[0], overlayCompareOp[0]);
    vkCmdSetStencilOp(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayFailOp[1], overlayPassOp[1],
                      overlayDepthFailOp[1], overlayCompareOp[1]);

    vkCmdSetStencilReference(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayReference[0]);
    vkCmdSetStencilReference(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayReference[1]);

    vkCmdSetStencilCompareMask(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayCompareMask[0]);
    vkCmdSetStencilCompareMask(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayCompareMask[1]);

    vkCmdSetStencilWriteMask(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayWriteMask[0]);
    vkCmdSetStencilWriteMask(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayWriteMask[1]);

    vkCmdSetCullMode(commandBuffer, overlayCullMode);

    vkCmdSetFrontFace(commandBuffer, overlayFrontFace);

    vkCmdSetPolygonModeEXT(commandBuffer, overlayPolygonMode);

    vkCmdSetDepthBiasEnable(commandBuffer, overlayDepthBiasEnable);

    vkCmdSetDepthBias(commandBuffer, overlayDepthBiasConstantFactor[overlayPolygonMode],
                      overlayDepthBiasClamp[overlayPolygonMode], overlayDepthBiasSlopeFactor[overlayPolygonMode]);

    vkCmdSetLineWidth(commandBuffer, overlayLineWidth);

    if (context->device->hasDynamicColorBlendState()) {
        vkCmdSetColorBlendEnableEXT(commandBuffer, 0, 1, &overlayBlendEnabled);
        vkCmdSetColorBlendEquationEXT(commandBuffer, 0, 1, &overlayColorBlendEquation);
        vkCmdSetColorWriteMaskEXT(commandBuffer, 0, 1, &overlayColorWriteMask);
    }

    if (context->device->hasExtendedDynamicState2LogicOp()) {
        vkCmdSetLogicOpEXT(commandBuffer, overlayColorLogicOp);
        vkCmdSetLogicOpEnableEXT(commandBuffer, overlayColorLogicOpEnable);
    }

    vkCmdSetBlendConstants(commandBuffer, overlayBlendConstants.data());
}

void UIModuleContext::syncFromContext(std::shared_ptr<UIModuleContext> other) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayScissorEnabled = other->overlayScissorEnabled;
    overlayScissor = other->overlayScissor;
    overlayViewport = other->overlayViewport;

    overlayBlendEnabled = other->overlayBlendEnabled;
    overlayColorBlendEquation = other->overlayColorBlendEquation;
    overlayColorWriteMask = other->overlayColorWriteMask;
    overlayColorLogicOpEnable = other->overlayColorLogicOpEnable;
    overlayColorLogicOp = other->overlayColorLogicOp;
    overlayBlendConstants = other->overlayBlendConstants;

    overlayDepthTestEnable = other->overlayDepthTestEnable;
    overlayDepthWriteEnable = other->overlayDepthWriteEnable;
    overlayDepthCompareOp = other->overlayDepthCompareOp;
    overlayStencilTestEnable = other->overlayStencilTestEnable;
    overlayFailOp = other->overlayFailOp;
    overlayPassOp = other->overlayPassOp;
    overlayDepthFailOp = other->overlayDepthFailOp;
    overlayCompareOp = other->overlayCompareOp;
    overlayReference = other->overlayReference;
    overlayCompareMask = other->overlayCompareMask;
    overlayWriteMask = other->overlayWriteMask;

    overlayCullMode = other->overlayCullMode;
    overlayFrontFace = other->overlayFrontFace;
    overlayPolygonMode = other->overlayPolygonMode;
    overlayDepthBiasEnable = other->overlayDepthBiasEnable;
    overlayDepthBiasConstantFactor = other->overlayDepthBiasConstantFactor;
    overlayDepthBiasClamp = other->overlayDepthBiasClamp;
    overlayDepthBiasSlopeFactor = other->overlayDepthBiasSlopeFactor;
    overlayLineWidth = other->overlayLineWidth;

    overlayClearColors = other->overlayClearColors;
    overlayClearDepth = other->overlayClearDepth;
    overlayClearStencil = other->overlayClearStencil;

    syncToCommandBuffer();
}

void UIModuleContext::setOverlayScissorEnabled(bool enabled) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayScissorEnabled = enabled;
    if (overlayScissorEnabled) {
        vkCmdSetScissor(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayScissor);
    } else {
        VkRect2D full_scissor = {
            .offset = {0, 0},
            .extent = framework->swapchain()->vkExtent(),
        };
        vkCmdSetScissor(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &full_scissor);
    }
}

void UIModuleContext::setOverlayScissor(int x, int y, int width, int height) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    y = context->swapchainImage->height() - (y + height);

    x = std::max(x, 0);
    y = std::max(y, 0);
    if (x + width > context->swapchainImage->width()) width = context->swapchainImage->width() - x;
    if (y + height > context->swapchainImage->height()) height = context->swapchainImage->height() - y;

    overlayScissor.offset.x = x;
    overlayScissor.offset.y = y;
    overlayScissor.extent.width = width;
    overlayScissor.extent.height = height;
    setOverlayScissorEnabled(overlayScissorEnabled);
}

void UIModuleContext::setOverlayViewport(int x, int y, int width, int height) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayViewport.x = x;
    overlayViewport.y = y;
    overlayViewport.width = width;
    overlayViewport.height = height;
    vkCmdSetViewport(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayViewport);
}

void UIModuleContext::setOverlayBlendEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayBlendEnabled = enable;
    if (context->device->hasDynamicColorBlendState()) {
        vkCmdSetColorBlendEnableEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayBlendEnabled);
    }
}

void UIModuleContext::setOverlayColorBlendConstants(float const1, float const2, float const3, float const4) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayBlendConstants[0] = const1;
    overlayBlendConstants[1] = const2;
    overlayBlendConstants[2] = const3;
    overlayBlendConstants[3] = const4;
    vkCmdSetBlendConstants(context->overlayCommandBuffer->vkCommandBuffer(), overlayBlendConstants.data());
}

void UIModuleContext::setOverlayColorLogicOpEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorLogicOpEnable = enable;
    if (context->device->hasExtendedDynamicState2LogicOp()) {
        vkCmdSetLogicOpEnableEXT(context->overlayCommandBuffer->vkCommandBuffer(), overlayColorLogicOpEnable);
    }
}

void UIModuleContext::setOverlayBlendFuncSeparate(int srcColorBlendFactor,
                                                  int srcAlphaBlendFactor,
                                                  int dstColorBlendFactor,
                                                  int dstAlphaBlendFactor) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorBlendEquation.srcColorBlendFactor = static_cast<VkBlendFactor>(srcColorBlendFactor);
    overlayColorBlendEquation.srcAlphaBlendFactor = static_cast<VkBlendFactor>(srcAlphaBlendFactor);
    overlayColorBlendEquation.dstColorBlendFactor = static_cast<VkBlendFactor>(dstColorBlendFactor);
    overlayColorBlendEquation.dstAlphaBlendFactor = static_cast<VkBlendFactor>(dstAlphaBlendFactor);
    if (context->device->hasDynamicColorBlendState()) {
        vkCmdSetColorBlendEquationEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayColorBlendEquation);
    }
}

void UIModuleContext::setOverlayBlendOpSeparate(int colorBlendOp, int alphaBlendOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorBlendEquation.colorBlendOp = static_cast<VkBlendOp>(colorBlendOp);
    overlayColorBlendEquation.alphaBlendOp = static_cast<VkBlendOp>(alphaBlendOp);
    if (context->device->hasDynamicColorBlendState()) {
        vkCmdSetColorBlendEquationEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayColorBlendEquation);
    }
}

void UIModuleContext::setOverlayColorWriteMask(int colorWriteMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorWriteMask = colorWriteMask;
    if (context->device->hasDynamicColorBlendState()) {
        vkCmdSetColorWriteMaskEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayColorWriteMask);
    }
}

void UIModuleContext::setOverlayColorLogicOp(int colorLogicOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorLogicOp = static_cast<VkLogicOp>(colorLogicOp);
    if (context->device->hasExtendedDynamicState2LogicOp()) {
        vkCmdSetLogicOpEXT(context->overlayCommandBuffer->vkCommandBuffer(), overlayColorLogicOp);
    }
}

void UIModuleContext::setOverlayDepthTestEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthTestEnable = enable;
    vkCmdSetDepthTestEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthTestEnable);
}

void UIModuleContext::setOverlayDepthWriteEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthWriteEnable = enable;
    vkCmdSetDepthWriteEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthWriteEnable);
}

void UIModuleContext::setOverlayStencilTestEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayStencilTestEnable = enable;
    vkCmdSetStencilTestEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayStencilTestEnable);
}

void UIModuleContext::setOverlayDepthCompareOp(int depthCompareOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthCompareOp = static_cast<VkCompareOp>(depthCompareOp);
    vkCmdSetDepthCompareOp(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthCompareOp);
}

void UIModuleContext::setOverlayStencilFrontFunc(int compareOp, int reference, int compareMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayCompareOp[0] = static_cast<VkCompareOp>(compareOp);
    overlayReference[0] = reference;
    overlayCompareMask[0] = compareMask;
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT, overlayFailOp[0],
                      overlayPassOp[0], overlayDepthFailOp[0], overlayCompareOp[0]);
    vkCmdSetStencilReference(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT,
                             overlayReference[0]);
    vkCmdSetStencilCompareMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT,
                               overlayCompareMask[0]);
}

void UIModuleContext::setOverlayStencilBackFunc(int compareOp, int reference, int compareMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayCompareOp[1] = static_cast<VkCompareOp>(compareOp);
    overlayReference[1] = reference;
    overlayCompareMask[1] = compareMask;
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT, overlayFailOp[1],
                      overlayPassOp[1], overlayDepthFailOp[1], overlayCompareOp[1]);
    vkCmdSetStencilReference(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT,
                             overlayReference[1]);
    vkCmdSetStencilCompareMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT,
                               overlayCompareMask[1]);
}

void UIModuleContext::setOverlayStencilFrontOp(int failOp, int depthFailOp, int passOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayFailOp[0] = static_cast<VkStencilOp>(failOp);
    overlayDepthFailOp[0] = static_cast<VkStencilOp>(depthFailOp);
    overlayPassOp[0] = static_cast<VkStencilOp>(passOp);
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT, overlayFailOp[0],
                      overlayPassOp[0], overlayDepthFailOp[0], overlayCompareOp[0]);
}

void UIModuleContext::setOverlayStencilBackOp(int failOp, int depthFailOp, int passOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayFailOp[1] = static_cast<VkStencilOp>(failOp);
    overlayDepthFailOp[1] = static_cast<VkStencilOp>(depthFailOp);
    overlayPassOp[1] = static_cast<VkStencilOp>(passOp);
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT, overlayFailOp[1],
                      overlayPassOp[1], overlayDepthFailOp[1], overlayCompareOp[1]);
}

void UIModuleContext::setOverlayStencilFrontWriteMask(int writeMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayWriteMask[0] = writeMask;
    vkCmdSetStencilWriteMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT,
                             overlayWriteMask[0]);
}

void UIModuleContext::setOverlayStencilBackWriteMask(int writeMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayWriteMask[1] = writeMask;
    vkCmdSetStencilWriteMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT,
                             overlayWriteMask[1]);
}

void UIModuleContext::setOverlayLineWidth(float lineWidth) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayLineWidth = lineWidth;
    vkCmdSetLineWidth(context->overlayCommandBuffer->vkCommandBuffer(), overlayLineWidth);
}

void UIModuleContext::setOverlayPolygonMode(int polygonMode) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayPolygonMode = static_cast<VkPolygonMode>(polygonMode);
    vkCmdSetPolygonModeEXT(context->overlayCommandBuffer->vkCommandBuffer(), overlayPolygonMode);
    setOverlayDepthBiasEnable(overlayPolygonMode, overlayDepthBiasEnable);
}

void UIModuleContext::setOverlayCullMode(int cullMode) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayCullMode = cullMode;
    vkCmdSetCullMode(context->overlayCommandBuffer->vkCommandBuffer(), overlayCullMode);
}

void UIModuleContext::setOverlayFrontFace(int frontFace) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayFrontFace = static_cast<VkFrontFace>(frontFace);
    vkCmdSetFrontFace(context->overlayCommandBuffer->vkCommandBuffer(), overlayFrontFace);
}

void UIModuleContext::setOverlayDepthBiasEnable(int polygonMode, bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthBiasEnable = enable;
    if (overlayDepthBiasEnable)
        vkCmdSetDepthBias(context->overlayCommandBuffer->vkCommandBuffer(),
                          overlayDepthBiasConstantFactor[overlayPolygonMode], overlayDepthBiasClamp[overlayPolygonMode],
                          overlayDepthBiasSlopeFactor[overlayPolygonMode]);
    vkCmdSetDepthBiasEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthBiasEnable);
}

void UIModuleContext::setOverlayDepthBias(float depthBiasSlopeFactor, float depthBiasConstantFactor) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthBiasSlopeFactor[overlayPolygonMode] = depthBiasSlopeFactor;
    overlayDepthBiasConstantFactor[overlayPolygonMode] = depthBiasConstantFactor;
    setOverlayDepthBiasEnable(overlayPolygonMode, overlayDepthBiasEnable);
}

void UIModuleContext::setOverlayClearColor(float red, float green, float blue, float alpha) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayClearColors[0] = red;
    overlayClearColors[1] = green;
    overlayClearColors[2] = blue;
    overlayClearColors[3] = alpha;
}

void UIModuleContext::setOverlayClearDepth(double depth) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayClearDepth = depth;
}

void UIModuleContext::setOverlayClearStencil(int stencil) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayClearStencil = stencil;
}

void UIModuleContext::switchOverlayDraw() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    auto mainQueueIndex = context->physicalDevice->mainQueueIndex();

    if (overlayMode == POST) {
        context->overlayCommandBuffer->endRenderPass();
#if defined(USE_AMD) || defined(__APPLE__)
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    }

    if (overlayMode == NONE || overlayMode == POST) {
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawColorImage->imageLayout(),
#if defined(USE_AMD) || defined(__APPLE__)
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawDepthStencilImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawDepthStencilImage,
                     .subresourceRange = vk::wholeDepthSubresourceRange,
                 }});
#if defined(USE_AMD) || defined(__APPLE__)
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawDepthStencilImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;

        context->overlayCommandBuffer->beginRenderPass({
            .renderPass = module->overlayDrawRenderPass_,
            .framebuffer = overlayDrawFramebuffer,
            .renderAreaExtent = {overlayDrawColorImage->width(), overlayDrawColorImage->height()},
            .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
        });

        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        overlayDrawDepthStencilImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        syncToCommandBuffer();
    }

    overlayMode = DRAW;
}

void UIModuleContext::switchOverlayPost() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    auto mainQueueIndex = context->physicalDevice->mainQueueIndex();

    if (overlayMode == DRAW) {
        context->overlayCommandBuffer->endRenderPass();
#if defined(USE_AMD) || defined(__APPLE__)
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawDepthStencilImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    if (overlayMode == NONE || overlayMode == DRAW) {
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayPostColorImage->imageLayout(),
#if defined(USE_AMD) || defined(__APPLE__)
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayPostColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawColorImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
#if defined(USE_AMD) || defined(__APPLE__)
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        context->overlayCommandBuffer->beginRenderPass({
            .renderPass = module->overlayPostRenderPass_,
            .framebuffer = overlayPostFramebuffer,
            .renderAreaExtent = {overlayPostColorImage->width(), overlayPostColorImage->height()},
            .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
        });
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    overlayMode = POST;
}

void UIModuleContext::clearOverlayEntireColorAttachment() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    switchOverlayDraw();
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*context->overlayCommandBuffer, "ui/clear_color", 0.95f, 0.55f,
                                                   0.25f);

    VkClearAttachment clearAttachment{};
    clearAttachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clearAttachment.colorAttachment = 0;
    clearAttachment.clearValue.color.float32[0] = overlayClearColors[0];
    clearAttachment.clearValue.color.float32[1] = overlayClearColors[1];
    clearAttachment.clearValue.color.float32[2] = overlayClearColors[2];
    clearAttachment.clearValue.color.float32[3] = overlayClearColors[3];

    VkClearRect clearRect{};
    clearRect.rect.offset = {0, 0};
    clearRect.rect.extent = framework->swapchain()->vkExtent();
    clearRect.baseArrayLayer = 0;
    clearRect.layerCount = 1;

    vkCmdClearAttachments(context->overlayCommandBuffer->vkCommandBuffer(), 1, &clearAttachment, 1, &clearRect);
}

void UIModuleContext::clearOverlayEntireDepthStencilAttachment(int aspectMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    switchOverlayDraw();
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*context->overlayCommandBuffer, "ui/clear_depth_stencil", 0.95f,
                                                   0.55f, 0.25f);

    VkClearAttachment clearAttachment{};
    clearAttachment.aspectMask = aspectMask;
    clearAttachment.clearValue.depthStencil.depth = overlayClearDepth;
    clearAttachment.clearValue.depthStencil.stencil = overlayClearStencil;

    VkClearRect clearRect{};
    clearRect.rect.offset = {0, 0};
    clearRect.rect.extent = framework->swapchain()->vkExtent();
    clearRect.baseArrayLayer = 0;
    clearRect.layerCount = 1;

    vkCmdClearAttachments(context->overlayCommandBuffer->vkCommandBuffer(), 1, &clearAttachment, 1, &clearRect);
}

void UIModuleContext::drawIndexed(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                  std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer,
                                  uint32_t shaderId,
                                  uint32_t uniformOffset,
                                  uint32_t indexCount,
                                  VkIndexType indexType) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    switchOverlayDraw();
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*context->overlayCommandBuffer, "ui/draw_indexed", 0.95f, 0.55f,
                                                   0.25f);

    auto &shaderInfo = module->overlayDrawShaderInfo(shaderId);
#ifdef __APPLE__
    static uint64_t diagnosticDraw = 0;
    ++diagnosticDraw;
    if (diagnosticDraw <= 24 || (diagnosticDraw <= 20000 && diagnosticDraw % 1000 == 0)) {
        std::fprintf(stderr, "[RADIANCE-E15] draw=%llu mode=%s shader=%s indices=%u indexType=%d uniformOffset=%u viewport=%.0f,%.0f,%.0f,%.0f scissorEnabled=%u scissor=%d,%d,%u,%u colorMask=0x%x depthTest=%u depthWrite=%u depthCompare=%d cull=0x%x blend=%u\n",
                     static_cast<unsigned long long>(diagnosticDraw), isFlatUiProbeEnabled() ? "flat-ui" : "normal",
                     shaderInfo.key.c_str(), indexCount, int(indexType), uniformOffset,
                     overlayViewport.x, overlayViewport.y, overlayViewport.width, overlayViewport.height,
                     overlayScissorEnabled, overlayScissor.offset.x, overlayScissor.offset.y,
                     overlayScissor.extent.width, overlayScissor.extent.height, overlayColorWriteMask,
                     overlayDepthTestEnable, overlayDepthWriteEnable, int(overlayDepthCompareOp), overlayCullMode,
                     overlayBlendEnabled);
    }
#endif
    vkCmdBindPipeline(context->overlayCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      module->overlayDrawPipeline(shaderId, {
                          .blendEnable = overlayBlendEnabled,
                          .srcColorBlendFactor = overlayColorBlendEquation.srcColorBlendFactor,
                          .dstColorBlendFactor = overlayColorBlendEquation.dstColorBlendFactor,
                          .colorBlendOp = overlayColorBlendEquation.colorBlendOp,
                          .srcAlphaBlendFactor = overlayColorBlendEquation.srcAlphaBlendFactor,
                          .dstAlphaBlendFactor = overlayColorBlendEquation.dstAlphaBlendFactor,
                          .alphaBlendOp = overlayColorBlendEquation.alphaBlendOp,
                          .colorWriteMask = overlayColorWriteMask,
                      })->vkPipeline());

    uint32_t dynamicOffsets[] = {uniformOffset, 0};
    vkCmdBindDescriptorSets(context->overlayCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                            overlayDescriptorTable->vkPipelineLayout(), 0,
                            overlayDescriptorTable->descriptorSet().size(),
                            overlayDescriptorTable->descriptorSet().data(), 2, dynamicOffsets);

    context->overlayCommandBuffer->bindVertexBuffers(vertexBuffer)
        ->bindIndexBuffer(indexBuffer, indexType)
        ->drawIndexed(indexCount, 1);
}

void UIModuleContext::postBlur(int times) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    vk::CommandBuffer::ScopedDebugLabel debugLabel(*context->overlayCommandBuffer, "ui/post_blur", 0.95f, 0.55f, 0.25f);
    int postID = Renderer::instance().buffers()->getPostID();

    for (int i = postID - times + 1; i <= postID; i++) {
        switchOverlayPost();

        vkCmdBindPipeline(context->overlayCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          module->overlayPostPipelines_[BLUR]->vkPipeline());
        uint32_t dynamicOffsets[] = {0, Renderer::instance().buffers()->overlayPostUniformOffset(i)};
        vkCmdBindDescriptorSets(context->overlayCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                overlayDescriptorTable->vkPipelineLayout(), 0,
                                overlayDescriptorTable->descriptorSet().size(),
                                overlayDescriptorTable->descriptorSet().data(), 2, dynamicOffsets);

        context->overlayCommandBuffer->draw(3, 1);

        context->overlayCommandBuffer->endRenderPass();
#if defined(USE_AMD) || defined(__APPLE__)
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayMode = NONE;

        auto mainQueueIndex = context->physicalDevice->mainQueueIndex();
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .oldLayout = overlayPostColorImage->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = overlayPostColorImage,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    },
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .oldLayout = overlayDrawColorImage->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = overlayDrawColorImage,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    },
                });

        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageBlit imageBlit{};
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.mipLevel = 0;
        imageBlit.srcSubresource.baseArrayLayer = 0;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcOffsets[0] = {0, 0, 0};
        imageBlit.srcOffsets[1] = {static_cast<int>(overlayPostColorImage->width()),
                                   static_cast<int>(overlayPostColorImage->height()), 1};
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.mipLevel = 0;
        imageBlit.dstSubresource.baseArrayLayer = 0;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstOffsets[0] = {0, 0, 0};
        imageBlit.dstOffsets[1] = {static_cast<int>(overlayDrawColorImage->width()),
                                   static_cast<int>(overlayDrawColorImage->height()), 1};

        vkCmdBlitImage(context->overlayCommandBuffer->vkCommandBuffer(), overlayPostColorImage->vkImage(),
                       overlayPostColorImage->imageLayout(), overlayDrawColorImage->vkImage(),
                       overlayDrawColorImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayPostColorImage->imageLayout(),
#if defined(USE_AMD) || defined(__APPLE__)
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayPostColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawColorImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
#if defined(USE_AMD) || defined(__APPLE__)
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

void UIModuleContext::refreshOverlayDescriptorTable() {
    auto context = frameworkContext.lock();
    auto module = uiModule.lock();
    module->refreshOverlayDescriptorTable(context->frameIndex);
}

void UIModuleContext::begin(std::shared_ptr<UIModuleContext> lastContext) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayMode = NONE;

    overlayDescriptorTable->bindBufferRange(Renderer::instance().buffers()->overlayPostUniformBuffer(), 1, 1, 0,
                                            Renderer::instance().buffers()->overlayPostUniformDescriptorRange());
    overlayDescriptorTable->bindBufferRange(Renderer::instance().buffers()->overlayDrawUniformBuffer(), 1, 0, 0,
                                            Renderer::instance().buffers()->overlayDrawUniformDescriptorRange());
    context->overlayCommandBuffer->bindDescriptorTable(overlayDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);

    if (lastContext != nullptr)
        syncFromContext(lastContext);
    else
        syncToCommandBuffer();
}

void UIModuleContext::end() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    if (overlayMode == DRAW) {
        context->overlayCommandBuffer->endRenderPass();
#if defined(USE_AMD) || defined(__APPLE__)
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    } else if (overlayMode == POST) {
        context->overlayCommandBuffer->endRenderPass();

        auto mainQueueIndex = context->physicalDevice->mainQueueIndex();
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = overlayDrawColorImage->imageLayout(),
#if defined(USE_AMD) || defined(__APPLE__)
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = overlayDrawColorImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }});
#if defined(USE_AMD) || defined(__APPLE__)
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    }

    overlayMode = NONE;
}
