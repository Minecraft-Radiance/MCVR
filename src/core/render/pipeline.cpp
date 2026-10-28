#include "core/render/pipeline.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#ifdef MCVR_UI_BACKEND_VULKAN
#    include "core/render/modules/ui_module.hpp"
#endif
#include "core/render/modules/world/dlss/dlss_rr_module.hpp"
#include "core/render/modules/world/dlss/dlss_sr_module.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr_upscaler_module.hpp"
#include "core/render/modules/world/main_render/main_render_module.hpp"
#include "core/render/modules/world/nrd/nrd_module.hpp"
#include "core/render/modules/world/post_render/post_render_module.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/render/modules/world/svgf/svgf_module.hpp"
#include "core/render/modules/world/temporal_accumulation/temporal_accumulation_module.hpp"
#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"
#include "core/render/modules/world/xess_upscaler/xess_sr_module.hpp"
#include "core/vulkan/resource_memory_tracker.hpp"

#include <cstdlib>
#include <set>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstring>

struct WorldTimingCapture {
    using Clock = std::chrono::steady_clock;
    static bool enabled() {
        const char *value = std::getenv("MCVR_PERFORMANCE_CAPTURE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }

    std::shared_ptr<vk::Device> device;
    VkQueryPool pool = VK_NULL_HANDLE;
    std::vector<std::string> names;
    std::vector<uint64_t> timestamps;
    std::vector<double> cpuMilliseconds;
    float timestampPeriod = 0;
    uint64_t timestampMask = UINT64_MAX;
    bool pending = false;
    uint32_t slot = 0;
    uint64_t samples = 0;
    Clock::time_point cpuStart;
    Clock::time_point lastReport{};

    WorldTimingCapture(std::shared_ptr<FrameworkContext> context, const std::vector<std::string> &moduleNames)
        : device(context->device), names(moduleNames), timestamps(names.size() * 2),
          cpuMilliseconds(names.size()), slot(context->frameIndex) {
        if (names.empty()) return;
        auto physical = context->physicalDevice;
        timestampPeriod = physical->properties().limits.timestampPeriod;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical->vkPhysicalDevice(), &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical->vkPhysicalDevice(), &count, queues.data());
        uint32_t bits = queues.at(physical->mainQueueIndex()).timestampValidBits;
        if (bits == 0 || timestampPeriod <= 0) {
            std::fprintf(stderr, "[RADIANCE-E15-PERF] GPU timestamps unavailable; capture disabled\n");
            return;
        }
        timestampMask = bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1;
        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = static_cast<uint32_t>(timestamps.size());
        VkResult result = vkCreateQueryPool(device->vkDevice(), &info, nullptr, &pool);
        if (result != VK_SUCCESS) {
            pool = VK_NULL_HANDLE;
            std::fprintf(stderr, "[RADIANCE-E15-PERF] query pool unavailable: %d\n", int(result));
        }
    }

    ~WorldTimingCapture() {
        if (pool != VK_NULL_HANDLE) vkDestroyQueryPool(device->vkDevice(), pool, nullptr);
    }

    void begin(VkCommandBuffer command, uint32_t width, uint32_t height) {
        if (pool == VK_NULL_HANDLE) return;
        // acquireContext already waited this slot's fence. Never wait again for
        // profiling: read completed queries, then reset them in the next frame.
        if (pending) {
            VkResult result = vkGetQueryPoolResults(device->vkDevice(), pool, 0,
                static_cast<uint32_t>(timestamps.size()), timestamps.size() * sizeof(uint64_t),
                timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            auto now = Clock::now();
            if (result == VK_SUCCESS && (samples == 0 || now - lastReport >= std::chrono::seconds(5))) {
                auto milliseconds = [&](uint64_t start, uint64_t end) {
                    return double((end - start) & timestampMask) * timestampPeriod / 1e6;
                };
                std::fprintf(stderr, "[RADIANCE-E15-PERF] sample=%llu slot=%u output=%ux%u gpu_world_span_ms=%.3f\n",
                    static_cast<unsigned long long>(samples), slot, width, height,
                    milliseconds(timestamps.front(), timestamps.back()));
                for (size_t i = 0; i < names.size(); ++i) {
                    std::fprintf(stderr, "[RADIANCE-E15-PERF] module=%s gpu_interval_ms=%.3f cpu_record_ms=%.3f\n",
                        names[i].c_str(), milliseconds(timestamps[2 * i], timestamps[2 * i + 1]), cpuMilliseconds[i]);
                }
                lastReport = now;
            }
            ++samples;
        }
        vkCmdResetQueryPool(command, pool, 0, static_cast<uint32_t>(timestamps.size()));
        pending = true;
    }

    void beginModule(VkCommandBuffer command, uint32_t index) {
        if (pool == VK_NULL_HANDLE) return;
        cpuStart = Clock::now();
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 2 * index);
    }

    void endModule(VkCommandBuffer command, uint32_t index) {
        if (pool == VK_NULL_HANDLE) return;
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 2 * index + 1);
        cpuMilliseconds[index] = std::chrono::duration<double, std::milli>(Clock::now() - cpuStart).count();
    }
};

WorldPipelineBlueprint::WorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    MCVR_PROFILE_SCOPE("native.WorldPipelineBlueprint.build");
    constexpr uint32_t maxSupportedImageIndex = 4095;
    std::set<uint32_t> imageIndices;
    for (int i = 0; i < params->moduleCount; i++) {
        std::string moduleName = params->moduleNames[i];
        moduleNames_.emplace_back(moduleName);
        auto &moduleInputIndices = modulesInputIndices_.emplace_back();
        auto &moduleOutputIndices = modulesOutputIndices_.emplace_back();

        attributeCounts_.emplace_back(params->attributeCounts[i]);
        auto &attributeKV = attributeKVs_.emplace_back();
        for (int j = 0; j < params->attributeCounts[i]; j++) {
            attributeKV.push_back(std::string{params->attributeKVs[i][2 * j + 0]});
            attributeKV.push_back(std::string{params->attributeKVs[i][2 * j + 1]});
        }

        auto moduleImageNumIter = Pipeline::worldModuleInOutImageNums.find(moduleName);
        if (moduleImageNumIter == Pipeline::worldModuleInOutImageNums.end()) {
            throw std::runtime_error("Unknown world module in pipeline blueprint: " + moduleName);
        }

        auto [inputImageNum, outputImageNum] = moduleImageNumIter->second;
        auto appendImageIndex = [&](int rawIndex, std::vector<uint32_t> &indices, const char *kind) {
            if (rawIndex < 0) {
                throw std::runtime_error("Invalid " + std::string(kind) + " image index for module " + moduleName);
            }
            uint32_t index = static_cast<uint32_t>(rawIndex);
            indices.emplace_back(index);
            imageIndices.insert(index);
        };

        for (int j = 0; j < inputImageNum; j++) {
            appendImageIndex(params->inputIndices[i][j], moduleInputIndices, "input");
        }
        for (int j = 0; j < outputImageNum; j++) {
            appendImageIndex(params->outputIndices[i][j], moduleOutputIndices, "output");
        }
    }

    if (imageIndices.empty() || *imageIndices.begin() != 0) {
        throw std::runtime_error("World pipeline image index 0 is required for the final output");
    }

    uint32_t maxImageIndex = *imageIndices.rbegin();
    if (maxImageIndex > maxSupportedImageIndex) { throw std::runtime_error("World pipeline image index is too large"); }

    imageFormats_.resize(static_cast<size_t>(maxImageIndex) + 1, VK_FORMAT_UNDEFINED);
    for (auto imageIndex : imageIndices) {
        imageFormats_[imageIndex] = static_cast<VkFormat>(params->imageFormats[imageIndex]);
    }
}

const std::vector<std::string> &WorldPipelineBlueprint::moduleNames() const {
    return moduleNames_;
}

const std::vector<std::vector<uint32_t>> &WorldPipelineBlueprint::moduleInputIndices() const {
    return modulesInputIndices_;
}

const std::vector<std::vector<uint32_t>> &WorldPipelineBlueprint::moduleOutputIndices() const {
    return modulesOutputIndices_;
}

const std::vector<VkFormat> &WorldPipelineBlueprint::imageFormats() const {
    return imageFormats_;
}

const std::vector<std::vector<std::string>> &WorldPipelineBlueprint::attributeKVs() const {
    return attributeKVs_;
}

WorldPipeline::WorldPipeline() {}
void WorldPipeline::init(std::shared_ptr<Framework> framework, std::shared_ptr<Pipeline> pipeline) {
    MCVR_PROFILE_SCOPE("native.WorldPipeline.init");
    vk::ResourceMemoryTagScope worldPipelineTag("world_pipeline");
    auto blueprint = pipeline->worldPipelineBlueprint();
    uint32_t frameNum = framework->swapchain()->imageCount();
    const bool reportNativeProgress = Pipeline::nativeRebuildActive();

    worldModules_.resize(blueprint->moduleNames_.size());
    worldModuleNames_ = blueprint->moduleNames_;
    sharedImages_.resize(frameNum,
                         std::vector<std::shared_ptr<vk::DeviceLocalImage>>(blueprint->imageFormats_.size(), nullptr));
    contexts_.resize(frameNum);
    VkExtent2D extent = framework->swapchain()->vkExtent();

    for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
        vk::ResourceMemoryTagScope memoryTag("world_pipeline.final_output");
        sharedImages_[frameIndex][0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, extent.width, extent.height, 1, blueprint->imageFormats_[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }

    shaderPack_ = nullptr;
    bool dlssRrEnabled = false;
    bool nrdEnabled = false;
    bool nrdShEnabled = false;
    bool nrdSeparateDirectEnabled = false;
    for (size_t i = 0; i < blueprint->moduleNames_.size(); ++i) {
        if (blueprint->moduleNames_[i] == DlssRRModule::NAME) { dlssRrEnabled = true; }
        if (blueprint->moduleNames_[i] != NrdModule::NAME) { continue; }
        nrdEnabled = true;
        nrdSeparateDirectEnabled = true;
        const auto &attributes = blueprint->attributeKVs_[i];
        for (size_t j = 0; j + 1 < attributes.size(); j += 2) {
            if (attributes[j] == "render_pipeline.module.nrd.attribute.enable_sh") {
                const std::string &value = attributes[j + 1];
                nrdShEnabled = value == "render_pipeline.true" || value == "true" || value == "True" ||
                               value == "TRUE" || value == "1";
            } else if (attributes[j] ==
                       "render_pipeline.module.nrd.attribute.enable_separate_direct_denoising") {
                const std::string &value = attributes[j + 1];
                nrdSeparateDirectEnabled = value == "render_pipeline.true" || value == "true" || value == "True" ||
                                           value == "TRUE" || value == "1";
            }
        }
    }
    if (dlssRrEnabled && nrdEnabled) {
        throw std::runtime_error("DLSS Ray Reconstruction and NRD cannot be enabled together");
    }
    for (size_t i = 0; i < blueprint->moduleNames_.size(); i++) {
        if (blueprint->moduleNames_[i] != MainRenderModule::NAME) { continue; }

        auto buildConfig = ShaderPack::buildConfigFromMainRenderAttributes(blueprint->attributeKVs_[i]);
        buildConfig.injectedDefinitions["MCVR_USE_DLSSRR"] = dlssRrEnabled ? "1" : "0";
        buildConfig.injectedDefinitions["MCVR_USE_NRD"] = nrdEnabled ? "1" : "0";
        buildConfig.injectedDefinitions["MCVR_USE_NRD_SEPARATE_DIRECT"] =
            nrdEnabled && nrdSeparateDirectEnabled ? "1" : "0";
        if (nrdShEnabled) { buildConfig.injectedDefinitions["MCVR_NRD_ENABLE_SH"] = "1"; }
        auto shaderPack = std::make_shared<ShaderPack>(framework);
        std::string error;
        {
            vk::ResourceMemoryTagScope memoryTag("world_module.main_render.shader_pack");
            if (!shaderPack->initialize(buildConfig, error)) {
              if (FILE *f = std::fopen("/tmp/mcvr-error.log", "a")) {
    std::fprintf(f, "[MCVR] Shader pack initialization FAILED: %s\n", error.c_str());
    std::fclose(f);
}
                throw std::runtime_error("failed to load shared shader pack");
            }
        }
        shaderPack_ = shaderPack;
        break;
    }

    for (int i = blueprint->moduleNames_.size() - 1; i >= 0; i--) {
        const std::string moduleMemoryTag = std::string("world_module.") + blueprint->moduleNames_[i];
        {
            vk::ResourceMemoryTagScope memoryTag(moduleMemoryTag);
            worldModules_[i] =
                Pipeline::worldModuleConstructors[blueprint->moduleNames_[i]](framework, shared_from_this());
        }

        auto &moduleInputIndices = blueprint->modulesInputIndices_[i];
        auto &moduleOutputIndices = blueprint->modulesOutputIndices_[i];

        {
            vk::ResourceMemoryTagScope memoryTag(moduleMemoryTag);
            worldModules_[i]->setAttributes(blueprint->attributeCounts_[i], blueprint->attributeKVs_[i]);
        }

        for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
            vk::ResourceMemoryTagScope memoryTag(moduleMemoryTag);
            {
                std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages;
                std::vector<VkFormat> outputFormats;
                for (int j = 0; j < moduleOutputIndices.size(); j++) {
                    outputImages.push_back(sharedImages_[frameIndex][moduleOutputIndices[j]]);
                    outputFormats.push_back(blueprint->imageFormats_[moduleOutputIndices[j]]);
                }
                bool result = worldModules_[i]->setOrCreateOutputImages(outputImages, outputFormats, frameIndex);
                if (!result) {
                    throw std::runtime_error("Output image not set properly for " + blueprint->moduleNames_[i]);
                }
                for (int j = 0; j < moduleOutputIndices.size(); j++) {
                    sharedImages_[frameIndex][moduleOutputIndices[j]] = outputImages[j];
                }
            }

            {
                std::vector<std::shared_ptr<vk::DeviceLocalImage>> inputImages;
                std::vector<VkFormat> inputFormats;
                for (int j = 0; j < moduleInputIndices.size(); j++) {
                    inputImages.push_back(sharedImages_[frameIndex][moduleInputIndices[j]]);
                    inputFormats.push_back(blueprint->imageFormats_[moduleInputIndices[j]]);
                }
                bool result = worldModules_[i]->setOrCreateInputImages(inputImages, inputFormats, frameIndex);
                if (!result) throw std::runtime_error("Input image not set properly");
                for (int j = 0; j < moduleInputIndices.size(); j++) {
                    sharedImages_[frameIndex][moduleInputIndices[j]] = inputImages[j];
                }
            }
        }

        {
            MCVR_PROFILE_SCOPE(std::string("native.WorldModule.build.") + blueprint->moduleNames_[i]);
            vk::ResourceMemoryTagScope memoryTag(moduleMemoryTag);
            worldModules_[i]->build();
        }
    }

    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        vk::ResourceMemoryTagScope memoryTag("world_pipeline.contexts");
        auto pipelineContext = WorldPipelineContext::create(framework->contexts()[i], shared_from_this());
        contexts_[i] = pipelineContext;

        for (const auto &worldModuleContext : pipelineContext->worldModuleContexts) {
            if (worldModuleContext == nullptr) { throw std::runtime_error("World module context is not initialized"); }
            worldModuleContext->setWorldPipelineContext(pipelineContext);
        }
    }
}

std::vector<std::shared_ptr<WorldModule>> &WorldPipeline::worldModules() {
    return worldModules_;
}

std::vector<std::shared_ptr<WorldPipelineContext>> &WorldPipeline::contexts() {
    return contexts_;
}

std::shared_ptr<ShaderPack> WorldPipeline::shaderPack() {
    return shaderPack_;
}

void WorldPipeline::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                std::shared_ptr<vk::DeviceLocalImage> image,
                                int index) {
    for (int i = 0; i < worldModules_.size(); i++) { worldModules_[i]->bindTexture(sampler, image, index); }
}

WorldPipelineContext::WorldPipelineContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                           std::shared_ptr<WorldPipeline> worldPipeline)
    : frameworkContext(frameworkContext),
      worldPipeline(worldPipeline),
      outputImage(worldPipeline->sharedImages_[frameworkContext->frameIndex][0]) {
    auto &worldModules = worldPipeline->worldModules();
    for (int i = 0; i < worldModules.size(); i++) {
        worldModuleContexts.push_back(worldModules[i]->contexts()[frameworkContext->frameIndex]);
    }
    if (WorldTimingCapture::enabled()) {
        timingCapture = std::make_shared<WorldTimingCapture>(frameworkContext, worldPipeline->worldModuleNames_);
    }
}

void WorldPipelineContext::render() {
    MCVR_PROFILE_SCOPE("native.WorldPipelineContext.render");
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    vk::CommandBuffer::ScopedDebugLabel worldDebugLabel(*worldCommandBuffer, "world", 0.30f, 0.85f, 0.45f);

    if (outputImage) {
        VkImageLayout targetLayout =
#ifdef USE_AMD
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        VkPipelineStageFlags2 dstStage =
#ifdef USE_AMD
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
#else
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
#endif
        VkAccessFlags2 dstAccess =
#ifdef USE_AMD
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
#else
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
#endif
        worldCommandBuffer->barriersBufferImage({}, {{
                                                        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                                        .srcAccessMask = 0,
                                                        .dstStageMask = dstStage,
                                                        .dstAccessMask = dstAccess,
                                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                        .newLayout = targetLayout,
                                                        .srcQueueFamilyIndex = mainQueueIndex,
                                                        .dstQueueFamilyIndex = mainQueueIndex,
                                                        .image = outputImage,
                                                        .subresourceRange = vk::wholeColorSubresourceRange,
                                                    }});
        outputImage->imageLayout() = targetLayout;
    }

    auto wp = worldPipeline.lock();
    if (timingCapture) {
        timingCapture->begin(worldCommandBuffer->vkCommandBuffer(), outputImage ? outputImage->width() : 0,
                             outputImage ? outputImage->height() : 0);
    }
    for (int i = 0; i < worldModuleContexts.size(); i++) {
        if (timingCapture) timingCapture->beginModule(worldCommandBuffer->vkCommandBuffer(), i);
        if (wp != nullptr && i < static_cast<int>(wp->worldModuleNames_.size())) {
            std::string debugLabelName = "world/" + wp->worldModuleNames_[i];
            vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, debugLabelName.c_str(), 0.30f, 0.85f,
                                                           0.45f);
            MCVR_PROFILE_SCOPE(std::string("native.WorldModule.render.") + wp->worldModuleNames_[i]);
            worldModuleContexts[i]->render();
        } else {
            vk::CommandBuffer::ScopedDebugLabel debugLabel(*worldCommandBuffer, "world/module", 0.30f, 0.85f, 0.45f);
            MCVR_PROFILE_SCOPE("native.WorldModule.render");
            worldModuleContexts[i]->render();
        }
        if (timingCapture) timingCapture->endModule(worldCommandBuffer->vkCommandBuffer(), i);
    }

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = outputImage->imageLayout(),
#ifdef USE_AMD
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = outputImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});

#ifdef USE_AMD
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}

std::map<std::string,
         std::function<std::shared_ptr<WorldModule>(std::shared_ptr<Framework>, std::shared_ptr<WorldPipeline>)>>
    Pipeline::worldModuleConstructors{};
std::map<std::string, std::pair<uint32_t, uint32_t>> Pipeline::worldModuleInOutImageNums{};
std::map<std::string, std::function<void()>> Pipeline::worldModuleStaticPreCloser{};
std::atomic<bool> Pipeline::nativeRebuildActive_{false};

void Pipeline::beginNativeRebuild() {
    nativeRebuildActive_.store(true, std::memory_order_release);
}

void Pipeline::endNativeRebuild() {
    nativeRebuildActive_.store(false, std::memory_order_release);
}

bool Pipeline::nativeRebuildActive() {
    return nativeRebuildActive_.load(std::memory_order_acquire);
}

void Pipeline::collectWorldModules() {
    MCVR_PROFILE_SCOPE("native.Pipeline.collectWorldModules");
    worldModuleConstructors.clear();
    worldModuleInOutImageNums.clear();
    worldModuleStaticPreCloser.clear();

    auto framework = Renderer::instance().framework();

    worldModuleConstructors.insert(std::make_pair(
        MainRenderModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return MainRenderModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
        MainRenderModule::NAME, std::make_pair(MainRenderModule::inputImageNum, MainRenderModule::outputImageNum)));

    worldModuleConstructors.insert(std::make_pair(
        NrdModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return NrdModule::create(framework, worldPipeline);
        }));

    worldModuleInOutImageNums.insert(
        std::make_pair(NrdModule::NAME, std::make_pair(NrdModule::inputImageNum, NrdModule::outputImageNum)));

    worldModuleConstructors.insert(
        std::make_pair(TemporalAccumulationModule::NAME,
                       [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                           return TemporalAccumulationModule::create(framework, worldPipeline);
                       }));
    worldModuleInOutImageNums.insert(
        std::make_pair(TemporalAccumulationModule::NAME, std::make_pair(TemporalAccumulationModule::inputImageNum,
                                                                        TemporalAccumulationModule::outputImageNum)));

    worldModuleConstructors.insert(
        std::make_pair(FSRUpscalerModule::NAME,
                       [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                           return FSRUpscalerModule::create(framework, worldPipeline);
                       }));
    worldModuleInOutImageNums.insert(std::make_pair(
        FSRUpscalerModule::NAME, std::make_pair(FSRUpscalerModule::inputImageNum, FSRUpscalerModule::outputImageNum)));

#ifdef MCVR_ENABLE_XESS
    if (framework != nullptr && framework->device() != nullptr &&
        framework->device()->isXessDeviceExtensionsCompatible()) {
        worldModuleConstructors.insert(std::make_pair(
            XessSrModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                return XessSrModule::create(framework, worldPipeline);
            }));
        worldModuleInOutImageNums.insert(std::make_pair(
            XessSrModule::NAME, std::make_pair(XessSrModule::inputImageNum, XessSrModule::outputImageNum)));
    } else {
        std::cerr << "[Pipeline] xess module skipped: incompatible instance/device extension requirements."
                  << std::endl;
    }
#endif

    worldModuleConstructors.insert(
        std::make_pair(ToneMappingModule::NAME,
                       [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                           return ToneMappingModule::create(framework, worldPipeline);
                       }));
    worldModuleInOutImageNums.insert(std::make_pair(
        ToneMappingModule::NAME, std::make_pair(ToneMappingModule::inputImageNum, ToneMappingModule::outputImageNum)));

    if (framework != nullptr && framework->device() != nullptr &&
        framework->device()->isDlssRRDeviceExtensionsCompatible()) {
        bool result = DlssRRModule::initNGXContext();
        if (result) {
            worldModuleConstructors.insert(
                std::make_pair(DlssRRModule::NAME,
                               [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                                   return DlssRRModule::create(framework, worldPipeline);
                               }));
            worldModuleInOutImageNums.insert(std::make_pair(
                DlssRRModule::NAME, std::make_pair(DlssRRModule::inputImageNum, DlssRRModule::outputImageNum)));
            worldModuleStaticPreCloser.insert(std::make_pair(DlssRRModule::NAME, DlssRRModule::deinitNGXContext));
        } else {
            std::cerr << "[Pipeline] dlss ray reconstruction module skipped: NGX initialization/query failed."
                      << std::endl;
        }
    } else {
        std::cerr << "[Pipeline] dlss ray reconstruction module skipped: incompatible instance/device extension "
                     "requirements."
                  << std::endl;
    }

    if (framework != nullptr && framework->device() != nullptr &&
        framework->device()->isDlssSRDeviceExtensionsCompatible()) {
        bool result = DlssSRModule::initNGXContext();
        if (result) {
            worldModuleConstructors.insert(
                std::make_pair(DlssSRModule::NAME,
                               [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                                   return DlssSRModule::create(framework, worldPipeline);
                               }));
            worldModuleInOutImageNums.insert(std::make_pair(
                DlssSRModule::NAME, std::make_pair(DlssSRModule::inputImageNum, DlssSRModule::outputImageNum)));
            worldModuleStaticPreCloser.insert(std::make_pair(DlssSRModule::NAME, DlssSRModule::deinitNGXContext));
        } else {
            std::cerr << "[Pipeline] dlss super resolution module skipped: NGX initialization/query failed."
                      << std::endl;
        }
    } else {
        std::cerr << "[Pipeline] dlss super resolution module skipped: incompatible instance/device extension "
                     "requirements."
                  << std::endl;
    }

    worldModuleConstructors.insert(
        std::make_pair(TemporalAccumulationModule::NAME,
                       [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                           return TemporalAccumulationModule::create(framework, worldPipeline);
                       }));
    worldModuleInOutImageNums.insert(
        std::make_pair(TemporalAccumulationModule::NAME, std::make_pair(TemporalAccumulationModule::inputImageNum,
                                                                        TemporalAccumulationModule::outputImageNum)));

    worldModuleConstructors.insert(std::make_pair(
        PostRenderModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return PostRenderModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
        PostRenderModule::NAME, std::make_pair(PostRenderModule::inputImageNum, PostRenderModule::outputImageNum)));
}

Pipeline::Pipeline() {
}

Pipeline::~Pipeline() {
}

void Pipeline::init(std::shared_ptr<Framework> framework) {
    MCVR_PROFILE_SCOPE("native.Pipeline.init");
    framework_ = framework;
#ifdef MCVR_UI_BACKEND_VULKAN
    uiModule_ = UIModule::create(framework);
#endif
    contexts_ = std::make_shared<std::vector<std::shared_ptr<PipelineContext>>>();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_->resize(size);

    for (int i = 0; i < size; i++) {
        contexts_->at(i) = PipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Pipeline::buildWorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    MCVR_PROFILE_SCOPE("native.Pipeline.buildWorldPipelineBlueprint");
    beginNativeRebuild();
    try {
        auto blueprint = WorldPipelineBlueprint::create(params);
        {
            std::scoped_lock lock(blueprintMtx_);
            pendingWorldPipelineBlueprint_ = std::move(blueprint);
            recreationNeeded_.store(true, std::memory_order_release);
        }
    } catch (...) {
        endNativeRebuild();
        throw;
    }
}

void Pipeline::recreate(std::shared_ptr<Framework> framework) {
    MCVR_PROFILE_SCOPE("native.Pipeline.recreate");
    const bool reportNativeProgress = Pipeline::nativeRebuildActive();
    {
        std::scoped_lock lock(blueprintMtx_);
        if (pendingWorldPipelineBlueprint_ != nullptr) {
            worldPipelineBlueprint_ = std::move(pendingWorldPipelineBlueprint_);
        }
        recreationNeeded_.store(false, std::memory_order_release);
    }
    auto &frr = framework->frameResourceRetainer();
#ifdef MCVR_UI_BACKEND_VULKAN
    std::vector<OverlayDynamicDrawShaderInfo> overlayDynamicDrawShaders;
    if (uiModule_ != nullptr) { overlayDynamicDrawShaders = uiModule_->overlayDynamicDrawShaders(); }

    frr.retain(uiModule_);
    uiModule_ = UIModule::create(framework);
    for (const auto &shaderInfo : overlayDynamicDrawShaders) {
        uiModule_->registerOverlayDrawShader(shaderInfo.key, shaderInfo.vertexFormatType, shaderInfo.drawMode,
                                             shaderInfo.uniformSize, shaderInfo.vertexShaderPath,
                                             shaderInfo.fragmentShaderPath, shaderInfo.definitions);
    }
#endif

    if (worldPipeline_ != nullptr)
        for (auto &module : worldPipeline_->worldModules()) { module->preClose(); }
    frr.retain(worldPipeline_);
    worldPipeline_ =
        worldPipelineBlueprint_ == nullptr ? nullptr : WorldPipeline::create(framework, shared_from_this());

    frr.retain(contexts_);
    contexts_ = std::make_shared<std::vector<std::shared_ptr<PipelineContext>>>();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_->resize(size);

    for (int i = 0; i < size; i++) {
        contexts_->at(i) = PipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Pipeline::close() {
    if (worldPipeline_ != nullptr) {
        for (auto &module : worldPipeline_->worldModules()) { module->preClose(); }
    }

    for (auto &destructor : worldModuleStaticPreCloser) { destructor.second(); }
}

std::shared_ptr<PipelineContext> Pipeline::acquirePipelineContext(std::shared_ptr<FrameworkContext> context) {
    return contexts_->at(context->frameIndex);
}

std::vector<std::shared_ptr<PipelineContext>> &Pipeline::contexts() {
    return *contexts_;
}

void Pipeline::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                           std::shared_ptr<vk::DeviceLocalImage> image,
                           int index) {
    MCVR_PROFILE_SCOPE("native.Pipeline.bindTexture");
    if (worldPipeline_ != nullptr) worldPipeline_->bindTexture(sampler, image, index);
#ifdef MCVR_UI_BACKEND_VULKAN
    uiModule_->bindTexture(sampler, image, index);
#endif
}

#ifdef MCVR_UI_BACKEND_VULKAN
std::shared_ptr<UIModule> Pipeline::uiModule() {
    return uiModule_;
}
#endif

std::shared_ptr<WorldPipeline> Pipeline::worldPipeline() {
    return worldPipeline_;
}

std::shared_ptr<WorldPipelineBlueprint> Pipeline::worldPipelineBlueprint() {
    std::scoped_lock lock(blueprintMtx_);
    return worldPipelineBlueprint_;
}

bool Pipeline::isRecreationNeeded() const {
    return recreationNeeded_.load(std::memory_order_acquire);
}

PipelineContext::PipelineContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<Pipeline> pipeline)
    : frameworkContext(frameworkContext),
#ifdef MCVR_UI_BACKEND_VULKAN
      uiModuleContext(pipeline->uiModule()->contexts()[frameworkContext->frameIndex]),
#endif
      worldPipelineContext(pipeline->worldPipeline() == nullptr ?
                               nullptr :
                               pipeline->worldPipeline()->contexts()[frameworkContext->frameIndex]) {
}

#ifdef MCVR_UI_BACKEND_VULKAN
void PipelineContext::fuseWorld() {
    MCVR_PROFILE_SCOPE("native.PipelineContext.fuseWorld");
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    if (!framework->isRunning()) return;

    uiModuleContext->end();

    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    auto overlayCommandBuffer = context->overlayCommandBuffer;
    vk::CommandBuffer::ScopedDebugLabel debugLabel(*overlayCommandBuffer, "ui/fuse_world", 0.95f, 0.55f, 0.25f);

    overlayCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
#    ifdef USE_AMD
                    .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#    else
                    .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#    endif
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = worldPipelineContext->outputImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = uiModuleContext->overlayDrawColorImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = uiModuleContext->overlayDrawColorImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });

    uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(worldPipelineContext->outputImage->width()),
                               static_cast<int>(worldPipelineContext->outputImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(uiModuleContext->overlayDrawColorImage->width()),
                               static_cast<int>(uiModuleContext->overlayDrawColorImage->height()), 1};

    vkCmdBlitImage(overlayCommandBuffer->vkCommandBuffer(), worldPipelineContext->outputImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, uiModuleContext->overlayDrawColorImage->vkImage(),
                   uiModuleContext->overlayDrawColorImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

    overlayCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
#    ifdef USE_AMD
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#    else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#    endif
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = worldPipelineContext->outputImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = uiModuleContext->overlayDrawColorImage->imageLayout(),
#    ifdef USE_AMD
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#    else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#    endif
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = uiModuleContext->overlayDrawColorImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });

#    ifdef USE_AMD
    uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#    else
    uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#    endif
#    ifdef USE_AMD
    worldPipelineContext->outputImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#    else
    worldPipelineContext->outputImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#    endif
}
#endif
