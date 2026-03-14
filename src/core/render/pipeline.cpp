#include "core/render/pipeline.hpp"

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include "core/render/modules/ui_module.hpp"
#include "core/render/modules/world/dlss/dlss_module.hpp"
#include "core/render/modules/world/fsr_upscaler/upscaler_module.hpp"
#include "core/render/modules/world/nrd/nrd_module.hpp"
#include "core/render/modules/world/post_render/post_render_module.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/modules/world/svgf/svgf_module.hpp"
#include "core/render/modules/world/temporal_accumulation/temporal_accumulation_module.hpp"
#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <set>

WorldPipelineBlueprint::WorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    std::set<uint32_t> imageIndices;
    auto framework = Renderer::instance().framework();
    auto pipeline = framework->pipeline();

    eyeCount_ = static_cast<uint32_t>(params->eyeCount > 0 ? params->eyeCount : 1);

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

        auto [inputImageNum, outputImageNum] = Pipeline::worldModuleInOutImageNums[moduleName];
        for (int j = 0; j < inputImageNum; j++) {
            moduleInputIndices.emplace_back(params->inputIndices[i][j]);
            imageIndices.insert(params->inputIndices[i][j]);
        }
        for (int j = 0; j < outputImageNum; j++) {
            moduleOutputIndices.emplace_back(params->outputIndices[i][j]);
            imageIndices.insert(params->outputIndices[i][j]);
        }
    }

    int cnt = 0;
    for (auto i : imageIndices) {
        if (i != cnt) { throw std::runtime_error("Indices are not continous!"); }
        cnt++;
        imageFormats_.push_back(static_cast<VkFormat>(params->imageFormats[i]));
    }
}

WorldPipeline::WorldPipeline() {}

void WorldPipeline::dumpSharedImages(const char *label) const {
    std::cerr << label << std::endl;
    for (size_t frameIndex = 0; frameIndex < sharedImages_.size(); frameIndex++) {
        for (size_t idx = 0; idx < sharedImages_[frameIndex].size(); idx++) {
            auto &img = sharedImages_[frameIndex][idx];
            if (!img) continue;
            std::cerr << "  frame=" << frameIndex << " idx=" << idx << " size=" << img->width() << "x" << img->height()
                      << " fmt=" << img->vkFormat() << " image=0x" << std::hex << (uint64_t)img->vkImage() << std::dec
                      << std::endl;
        }
    }
}

void WorldPipeline::init(std::shared_ptr<Framework> framework, std::shared_ptr<Pipeline> pipeline) {
    auto blueprint = pipeline->worldPipelineBlueprint();
    uint32_t frameNum = framework->swapchain()->imageCount();

    bool useStereo = false;
#ifdef MCVR_ENABLE_OPENXR
    if (auto *xr = framework->xrContext(); xr != nullptr) {
        useStereo = Renderer::options.vrEnabled && xr->isSessionRunning();
    }
#endif
    eyeCount_ = useStereo ? 2u : 1u;

    VkExtent2D mirrorExtent = framework->swapchain()->vkExtent();
    uint32_t targetWidth = mirrorExtent.width;
    uint32_t targetHeight = mirrorExtent.height;

    // Sync VRSystem state from static options
    {
        auto &vr = Renderer::instance().vrSystem();
        auto &opts = Renderer::options;
        vr.enabled = useStereo;
        vr.config.renderScale = opts.vrRenderScale;
        vr.config.ipd = opts.vrIPD;
        vr.config.worldScale = opts.vrWorldScale;
        vr.eyeCount = eyeCount_;

        if (vr.enabled && eyeCount_ > 1) {
            bool xrResolutionReady = false;
#ifdef MCVR_ENABLE_OPENXR
            if (auto *xr = framework->xrContext(); xr != nullptr) {
                const uint32_t xrW = xr->swapchainWidth();
                const uint32_t xrH = xr->swapchainHeight();
                if (xrW > 0 && xrH > 0) {
                    for (uint32_t eye = 0; eye < 2; eye++) {
                        vr.eyes[eye].recommendedWidth = xrW;
                        vr.eyes[eye].recommendedHeight = xrH;
                    }
                    xrResolutionReady = true;
                }
            }
#endif
            if (!xrResolutionReady) {
                // No XR runtime/session yet: keep legacy simulation fallback for startup safety.
                vr.updateSimulation(mirrorExtent.width, mirrorExtent.height, glm::radians(70.0f));
            }

            const uint32_t baseWidth = std::max(vr.eyes[0].recommendedWidth, vr.eyes[1].recommendedWidth);
            const uint32_t baseHeight = std::max(vr.eyes[0].recommendedHeight, vr.eyes[1].recommendedHeight);
            const float scale = std::max(0.1f, vr.config.renderScale);
            targetWidth = std::max(1u, static_cast<uint32_t>(static_cast<float>(baseWidth) * scale));
            targetHeight = std::max(1u, static_cast<uint32_t>(static_cast<float>(baseHeight) * scale));
        } else if (vr.enabled) {
            vr.updateSimulation(mirrorExtent.width, mirrorExtent.height, glm::radians(70.0f));
        }
    }

    worldModules_.resize(blueprint->moduleNames_.size());
    sharedImages_.resize(frameNum,
                         std::vector<std::shared_ptr<vk::DeviceLocalImage>>(blueprint->imageFormats_.size(), nullptr));
    contexts_.resize(frameNum);

    // Determine initial render resolution from upscaler quality (if present)
    VkExtent2D extent{targetWidth, targetHeight};
    uint32_t renderWidth = extent.width;
    uint32_t renderHeight = extent.height;
    size_t upscalerIndex = std::numeric_limits<size_t>::max();
    UpscalerModule::QualityMode upscalerMode = UpscalerModule::QualityMode::NativeAA;
    for (size_t i = 0; i < blueprint->moduleNames_.size(); i++) {
        if (blueprint->moduleNames_[i] != UpscalerModule::NAME) continue;
        upscalerIndex = i;
        const auto &kvs = blueprint->attributeKVs_[i];
        for (size_t k = 0; k + 1 < kvs.size(); k += 2) {
            const std::string &key = kvs[k];
            const std::string &value = kvs[k + 1];
            if (UpscalerModule::isQualityModeAttributeKey(key)) {
                UpscalerModule::parseQualityModeValue(value, upscalerMode);
            }
        }
        if (upscalerMode != UpscalerModule::QualityMode::NativeAA) {
            UpscalerModule::getRenderResolution(extent.width, extent.height, upscalerMode, &renderWidth, &renderHeight);
        }
        break;
    }

    for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
        // Keep the primary output at display resolution
        sharedImages_[frameIndex][0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, extent.width, extent.height, eyeCount_, blueprint->imageFormats_[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#ifdef USE_AMD
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
        );
        if (eyeCount_ > 1) sharedImages_[frameIndex][0]->createPerLayerViews();
    }

    // Pre-create outputs for modules before upscaler at render resolution
    if (upscalerIndex != std::numeric_limits<size_t>::max() &&
        (renderWidth != extent.width || renderHeight != extent.height)) {
        std::set<uint32_t> renderIndices;
        for (size_t i = 0; i < upscalerIndex; i++) {
            for (uint32_t idx : blueprint->modulesOutputIndices_[i]) renderIndices.insert(idx);
        }

        for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
            for (uint32_t idx : renderIndices) {
                if (sharedImages_[frameIndex][idx] != nullptr) continue;
                sharedImages_[frameIndex][idx] = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, renderWidth, renderHeight, eyeCount_,
                    blueprint->imageFormats_[idx],
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#ifdef USE_AMD
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
                );
                if (eyeCount_ > 1) sharedImages_[frameIndex][idx]->createPerLayerViews();
            }
        }
    }

    for (int i = blueprint->moduleNames_.size() - 1; i >= 0; i--) {
        worldModules_[i] = Pipeline::worldModuleConstructors[blueprint->moduleNames_[i]](framework, shared_from_this());
        worldModules_[i]->setEyeCount(eyeCount_);

        auto &moduleInputIndices = blueprint->modulesInputIndices_[i];
        auto &moduleOutputIndices = blueprint->modulesOutputIndices_[i];

        for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
            { // output
                std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages;
                std::vector<VkFormat> outputFormats;
                for (int j = 0; j < moduleOutputIndices.size(); j++) {
                    outputImages.push_back(sharedImages_[frameIndex][moduleOutputIndices[j]]);
                    outputFormats.push_back(blueprint->imageFormats_[moduleOutputIndices[j]]);
                }
                bool result = worldModules_[i]->setOrCreateOutputImages(outputImages, outputFormats, frameIndex);
                if (!result) {
                    std::cout << blueprint->moduleNames_[i] << std::endl;
                    throw std::runtime_error("Output image not set properly");
                }
                for (int j = 0; j < moduleOutputIndices.size(); j++) {
                    sharedImages_[frameIndex][moduleOutputIndices[j]] = outputImages[j];
                }
            }

            { // input
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

        worldModules_[i]->setAttributes(blueprint->attributeCounts_[i], blueprint->attributeKVs_[i]);

        worldModules_[i]->build();
    }

    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        contexts_[i] = WorldPipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModule>> &WorldPipeline::worldModules() {
    return worldModules_;
}

std::vector<std::shared_ptr<WorldPipelineContext>> &WorldPipeline::contexts() {
    return contexts_;
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
      eyeCount_(worldPipeline->eyeCount()),
      outputImage(worldPipeline->sharedImages_[frameworkContext->frameIndex][0]) {
    auto &worldModules = worldPipeline->worldModules();
    for (int i = 0; i < worldModules.size(); i++) {
        worldModuleContexts.push_back(worldModules[i]->contexts()[frameworkContext->frameIndex]);
    }
}

void WorldPipelineContext::render() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    // Preflight: ensure output image has a valid initial layout (avoid UNDEFINED on AMD)
    if (outputImage && outputImage->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
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

    for (int i = 0; i < worldModuleContexts.size(); i++) {
        if (eyeCount_ > 1) {
            auto mode = worldModuleContexts[i]->stereoMode();
            switch (mode) {
                case StereoMode::SingleInstance3DDispatch:
                    worldModuleContexts[i]->render3D(eyeCount_);
                    break;
                case StereoMode::SingleInstanceMultiDispatch:
                case StereoMode::DualInstance:
                    for (uint32_t eye = 0; eye < eyeCount_; eye++) {
                        worldModuleContexts[i]->renderEye(eye);
                    }
                    break;
            }
        } else {
            worldModuleContexts[i]->render();
        }
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

void Pipeline::collectWorldModules() {
    worldModuleConstructors.insert(std::make_pair(
        RayTracingModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return RayTracingModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
        RayTracingModule::NAME, std::make_pair(RayTracingModule::inputImageNum, RayTracingModule::outputImageNum)));

    worldModuleConstructors.insert(std::make_pair(
        NrdModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return NrdModule::create(framework, worldPipeline);
        }));

    worldModuleInOutImageNums.insert(
        std::make_pair(NrdModule::NAME, std::make_pair(NrdModule::inputImageNum, NrdModule::outputImageNum)));

    // Not working well, just leave it here
    // worldModuleConstructors.insert(std::make_pair(
    //     SvgfModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    //         return SvgfModule::create(framework, worldPipeline);
    //     }));

    // worldModuleInOutImageNums.insert(
    //     std::make_pair(SvgfModule::NAME, std::make_pair(SvgfModule::inputImageNum, SvgfModule::outputImageNum)));

    worldModuleConstructors.insert(
        std::make_pair(TemporalAccumulationModule::NAME,
                       [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                           return TemporalAccumulationModule::create(framework, worldPipeline);
                       }));
    worldModuleInOutImageNums.insert(
        std::make_pair(TemporalAccumulationModule::NAME, std::make_pair(TemporalAccumulationModule::inputImageNum,
                                                                        TemporalAccumulationModule::outputImageNum)));

    worldModuleConstructors.insert(std::make_pair(
        UpscalerModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return UpscalerModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
        UpscalerModule::NAME, std::make_pair(UpscalerModule::inputImageNum, UpscalerModule::outputImageNum)));

    worldModuleConstructors.insert(
        std::make_pair(ToneMappingModule::NAME,
                       [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                           return ToneMappingModule::create(framework, worldPipeline);
                       }));
    worldModuleInOutImageNums.insert(std::make_pair(
        ToneMappingModule::NAME, std::make_pair(ToneMappingModule::inputImageNum, ToneMappingModule::outputImageNum)));

    bool result = DLSSModule::initNGXContext();
    if (result) {
        worldModuleConstructors.insert(std::make_pair(
            DLSSModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
                return DLSSModule::create(framework, worldPipeline);
            }));
        worldModuleInOutImageNums.insert(
            std::make_pair(DLSSModule::NAME, std::make_pair(DLSSModule::inputImageNum, DLSSModule::outputImageNum)));
        worldModuleStaticPreCloser.insert(std::make_pair(DLSSModule::NAME, DLSSModule::deinitNGXContext));
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

    // TODO: invoke extension's collection
}

Pipeline::Pipeline() {}

Pipeline::~Pipeline() {
#ifdef DEBUG
    std::cout << "Pipeline deconstruct" << std::endl;
#endif
}

void Pipeline::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;
    uiModule_ = UIModule::create(framework);
    contexts_ = std::make_shared<std::vector<std::shared_ptr<PipelineContext>>>();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_->resize(size);

    for (int i = 0; i < size; i++) {
        contexts_->at(i) = PipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Pipeline::buildWorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    worldPipelineBlueprint_ = WorldPipelineBlueprint::create(params);
    needRecreate = true;
}

void Pipeline::recreate(std::shared_ptr<Framework> framework) {
    auto &gc = framework->gc();

    gc.collect(uiModule_);
    uiModule_ = UIModule::create(framework);

    if (worldPipeline_ != nullptr)
        for (auto &module : worldPipeline_->worldModules()) { module->preClose(); }
    gc.collect(worldPipeline_);
    worldPipeline_ =
        worldPipelineBlueprint_ == nullptr ? nullptr : WorldPipeline::create(framework, shared_from_this());

    gc.collect(contexts_);
    contexts_ = std::make_shared<std::vector<std::shared_ptr<PipelineContext>>>();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_->resize(size);

    for (int i = 0; i < size; i++) {
        contexts_->at(i) = PipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Pipeline::close() {
    for (auto &module : worldPipeline_->worldModules()) { module->preClose(); }

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
    if (worldPipeline_ != nullptr) worldPipeline_->bindTexture(sampler, image, index);
    uiModule_->bindTexture(sampler, image, index);
}

std::shared_ptr<UIModule> Pipeline::uiModule() {
    return uiModule_;
}

std::shared_ptr<WorldPipeline> Pipeline::worldPipeline() {
    return worldPipeline_;
}

std::shared_ptr<WorldPipelineBlueprint> Pipeline::worldPipelineBlueprint() {
    return worldPipelineBlueprint_;
}

PipelineContext::PipelineContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<Pipeline> pipeline)
    : frameworkContext(frameworkContext),
      uiModuleContext(pipeline->uiModule()->contexts()[frameworkContext->frameIndex]),
      worldPipelineContext(pipeline->worldPipeline() == nullptr ?
                               nullptr :
                               pipeline->worldPipeline()->contexts()[frameworkContext->frameIndex]) {}

void PipelineContext::fuseWorld() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    if (!framework->isRunning()) return;

    uiModuleContext->end();

    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    auto overlayCommandBuffer = context->overlayCommandBuffer;

    overlayCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
#ifdef USE_AMD
                    .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                    .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
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

    auto srcImage = worldPipelineContext->outputImage;
    auto dstImage = uiModuleContext->overlayDrawColorImage;
    int srcW = static_cast<int>(srcImage->width());
    int srcH = static_cast<int>(srcImage->height());
    int dstW = static_cast<int>(dstImage->width());
    int dstH = static_cast<int>(dstImage->height());

    if (worldPipelineContext->eyeCount_ > 1) {
        // SBS: layer 0 → left half, layer 1 → right half
        int halfW = dstW / 2;
        VkImageBlit blits[2]{};
        for (uint32_t eye = 0; eye < 2; eye++) {
            blits[eye].srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eye, 1};
            blits[eye].srcOffsets[0] = {0, 0, 0};
            blits[eye].srcOffsets[1] = {srcW, srcH, 1};
            blits[eye].dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blits[eye].dstOffsets[0] = {static_cast<int>(eye * halfW), 0, 0};
            blits[eye].dstOffsets[1] = {static_cast<int>((eye + 1) * halfW), dstH, 1};
        }
        vkCmdBlitImage(overlayCommandBuffer->vkCommandBuffer(), srcImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage->vkImage(),
                       dstImage->imageLayout(), 2, blits, VK_FILTER_LINEAR);
    } else {
        VkImageBlit imageBlit{};
        imageBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageBlit.srcOffsets[0] = {0, 0, 0};
        imageBlit.srcOffsets[1] = {srcW, srcH, 1};
        imageBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageBlit.dstOffsets[0] = {0, 0, 0};
        imageBlit.dstOffsets[1] = {dstW, dstH, 1};
        vkCmdBlitImage(overlayCommandBuffer->vkCommandBuffer(), srcImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage->vkImage(),
                       dstImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
    }

    overlayCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
#ifdef USE_AMD
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
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
#ifdef USE_AMD
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = uiModuleContext->overlayDrawColorImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });

#ifdef USE_AMD
    uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
#ifdef USE_AMD
    worldPipelineContext->outputImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    worldPipelineContext->outputImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}
