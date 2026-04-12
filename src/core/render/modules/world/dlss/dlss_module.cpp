#include "core/render/modules/world/dlss/dlss_module.hpp"

#include <cmath>
#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

std::shared_ptr<NgxContext> DLSSModule::ngxContext_ = nullptr;

bool DLSSModule::initNGXContext() {
    std::filesystem::path dlssPath = Renderer::folderPath / "dlss";
    std::error_code ec;
    if (!std::filesystem::create_directories(dlssPath, ec)) {
        if (ec) {
            std::cerr << "Failed to create directory: " << ec.message() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    auto framework = Renderer::instance().framework();
    ngxContext_ = NgxContext::create();

    NgxContext::NgxInitInfo ngxInitInfo{};
    ngxInitInfo.instance = framework->instance();
    ngxInitInfo.physicalDevice = framework->physicalDevice();
    ngxInitInfo.device = framework->device();
    ngxInitInfo.applicationPath = dlssPath.string();
    if (ngxContext_->init(ngxInitInfo) != NVSDK_NGX_Result_Success) {
        ngxContext_ = nullptr;
        return false;
    }

    if (ngxContext_->queryDlssRRAvailable() != NVSDK_NGX_Result_Success) {
        ngxContext_->deinit();
        ngxContext_ = nullptr;
        return false;
    }

    return true;
}

void DLSSModule::deinitNGXContext() {
    if (ngxContext_ != nullptr) {
        ngxContext_->deinit();
        ngxContext_ = nullptr;
    }
}

bool DLSSModule::isFrameGenAvailable() {
    if (ngxContext_ == nullptr) return false;
    return ngxContext_->queryFrameGenAvailable() == NVSDK_NGX_Result_Success;
}

DLSSModule::DLSSModule() {}

void DLSSModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    specularHitDepthImages_.resize(size);
    firstHitDepthImages_.resize(size);
    processedImages_.resize(size);
    upscaledFirstHitDepthImages_.resize(size);
    interpFrameImages_.resize(size);

    dlss_ = DlssRR::create();
}

bool DLSSModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (ngxContext_ == nullptr) return false;

    if (images.size() != inputImageNum) return false;

    NgxContext::QuerySizeInfo querySizeInfo{};
    querySizeInfo.outputSize.width = outputWidth_;
    querySizeInfo.outputSize.height = outputHeight_;
    querySizeInfo.quality = mode_;
    ngxContext_->querySupportedDlssInputSizes(querySizeInfo, supportedSizes_);
#ifdef DEBUG
    std::cout << "DLSS sizes:" << std::endl;
    std::cout << "\tminSize: [" << supportedSizes_.minSize.width << ", " << supportedSizes_.minSize.height << "]"
              << std::endl;
    std::cout << "\tmaxSize: [" << supportedSizes_.maxSize.width << ", " << supportedSizes_.maxSize.height << "]"
              << std::endl;
    std::cout << "\toptimalSize: [" << supportedSizes_.optimalSize.width << ", " << supportedSizes_.optimalSize.height
              << "]" << std::endl;
#endif

    inputWidth_ = supportedSizes_.optimalSize.width;
    inputHeight_ = supportedSizes_.optimalSize.height;

    if (images[0] == nullptr) {
        hdrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[0]->width() != inputWidth_ || images[0]->height() != inputHeight_) return false;
    }

    if (images[1] == nullptr) {
        diffuseAlbedoImages_[frameIndex] = images[1] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[1],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[1]->width() != inputWidth_ || images[1]->height() != inputHeight_) return false;
    }

    if (images[2] == nullptr) {
        specularAlbedoImages_[frameIndex] = images[2] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[2],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[2]->width() != inputWidth_ || images[2]->height() != inputHeight_) return false;
    }

    if (images[3] == nullptr) {
        normalRoughnessImages_[frameIndex] = images[3] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[3],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[3]->width() != inputWidth_ || images[3]->height() != inputHeight_) return false;
    }

    if (images[4] == nullptr) {
        motionVectorImages_[frameIndex] = images[4] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[4],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[4]->width() != inputWidth_ || images[4]->height() != inputHeight_) return false;
    }

    if (images[5] == nullptr) {
        linearDepthImages_[frameIndex] = images[5] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[5],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[5]->width() != inputWidth_ || images[5]->height() != inputHeight_) return false;
    }

    if (images[6] == nullptr) {
        specularHitDepthImages_[frameIndex] = images[6] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[6],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[6]->width() != inputWidth_ || images[6]->height() != inputHeight_) return false;
    }

    if (images[7] == nullptr) {
        firstHitDepthImages_[frameIndex] = images[7] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[7],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[7]->width() != inputWidth_ || images[7]->height() != inputHeight_) return false;
    }

    return true;
}

bool DLSSModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (ngxContext_ == nullptr) return false;

    if (images.size() != outputImageNum || images[0] == nullptr) return false;

    outputWidth_ = images[0]->width();
    outputHeight_ = images[0]->height();

    processedImages_[frameIndex] = images[0];

    if (images[1] == nullptr) {
        upscaledFirstHitDepthImages_[frameIndex] = images[1] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, outputWidth_, outputHeight_, 1, formats[1],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[1]->width() != outputWidth_ || images[1]->height() != outputHeight_) return false;
        upscaledFirstHitDepthImages_[frameIndex] = images[1];
    }

    return true;
}

void DLSSModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        if (attributeKVs[2 * i] == "render_pipeline.module.dlss.attribute.mode") {
            if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.ultra_performance") {
                mode_ = NVSDK_NGX_PerfQuality_Value_UltraPerformance;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.performance") {
                mode_ = NVSDK_NGX_PerfQuality_Value_MaxPerf;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.balanced") {
                mode_ = NVSDK_NGX_PerfQuality_Value_Balanced;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.quality") {
                mode_ = NVSDK_NGX_PerfQuality_Value_MaxQuality;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.dlaa") {
                mode_ = NVSDK_NGX_PerfQuality_Value_DLAA;
            }
        } else if (attributeKVs[2 * i] == "render_pipeline.module.dlss.attribute.frame_generation") {
            const auto &val = attributeKVs[2 * i + 1];
            frameGenDynamic_ = false;
            if (val == "render_pipeline.module.dlss.attribute.frame_generation.off") {
                frameGenMultiFrameCount_ = 0;
            } else if (val == "render_pipeline.module.dlss.attribute.frame_generation.x2") {
                frameGenMultiFrameCount_ = 1;
            } else if (val == "render_pipeline.module.dlss.attribute.frame_generation.x3") {
                frameGenMultiFrameCount_ = 2;
            } else if (val == "render_pipeline.module.dlss.attribute.frame_generation.x4") {
                frameGenMultiFrameCount_ = 3;
            } else if (val == "render_pipeline.module.dlss.attribute.frame_generation.x5") {
                frameGenMultiFrameCount_ = 4;
            } else if (val == "render_pipeline.module.dlss.attribute.frame_generation.x6") {
                frameGenMultiFrameCount_ = 5;
            } else if (val == "render_pipeline.module.dlss.attribute.frame_generation.dynamic") {
                frameGenDynamic_ = true;
                // Sentinel value: will be resolved to hardware max in build()
                frameGenMultiFrameCount_ = UINT32_MAX;
            }
        }
    }
}

void DLSSModule::build() {
    // ngxContext_ must not be nullptr

    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    NgxContext::DlssRRInitInfo dlssRRInitInfo{};
    dlssRRInitInfo.inputSize = {inputWidth_, inputHeight_};
    dlssRRInitInfo.outputSize = {outputWidth_, outputHeight_};
    dlssRRInitInfo.quality = mode_;
    ngxContext_->initDlssRR(dlssRRInitInfo, framework->mainCommandPool(), dlss_);

    // Initialize Frame Generation if enabled and available
    if (frameGenMultiFrameCount_ > 0 && ngxContext_->queryFrameGenAvailable() == NVSDK_NGX_Result_Success) {
        // Clamp requested count to hardware max (also resolves dynamic/sentinel mode)
        uint32_t maxCount = ngxContext_->queryMaxMultiFrameCount();
        frameGenMaxMultiFrameCount_ = maxCount;
        if (frameGenDynamic_ || frameGenMultiFrameCount_ >= UINT32_MAX) {
            // Dynamic mode: allocate to hardware max, actual count adjusted per-frame
            frameGenMultiFrameCount_ = maxCount;
            std::cout << "[DLSS] Dynamic Frame Generation: hardware max " << maxCount
                      << " (" << (maxCount + 1) << "x)" << std::endl;
        } else if (frameGenMultiFrameCount_ > maxCount) {
            std::cout << "[DLSS] Requested multi-frame count " << frameGenMultiFrameCount_
                      << " exceeds hardware max " << maxCount << ", clamping" << std::endl;
            frameGenMultiFrameCount_ = maxCount;
        }

        dlssFG_ = DlssFG::create();
        NgxContext::DlssFGInitInfo fgInitInfo{};
        fgInitInfo.width = outputWidth_;
        fgInitInfo.height = outputHeight_;
        fgInitInfo.backbufferFormat = framework->swapchain()->vkSurfaceFormat().format;
        if (ngxContext_->initFrameGen(fgInitInfo, framework->mainCommandPool(), dlssFG_) != NVSDK_NGX_Result_Success) {
            std::cerr << "[DLSS] Frame Generation initialization failed" << std::endl;
            dlssFG_ = nullptr;
            frameGenMultiFrameCount_ = 0;
        } else {
            // Create interpolated frame images: one per multi-frame index per swapchain image
            for (uint32_t i = 0; i < size; i++) {
                interpFrameImages_[i].resize(frameGenMultiFrameCount_);
                for (uint32_t j = 0; j < frameGenMultiFrameCount_; j++) {
                    interpFrameImages_[i][j] = vk::DeviceLocalImage::create(
                        framework->device(), framework->vma(), false, outputWidth_, outputHeight_, 1,
                        framework->swapchain()->vkSurfaceFormat().format,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT);
                }
            }
            std::cout << "[DLSS] Frame Generation initialized (multi-frame count: " << frameGenMultiFrameCount_
                      << ", multiplier: " << (frameGenMultiFrameCount_ + 1) << "x)" << std::endl;
        }
    }

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] =
            DLSSModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &DLSSModule::contexts() {
    return contexts_;
}

void DLSSModule::updateDynamicFrameCount(double frameTimeSec, int targetFps) {
    if (!frameGenDynamic_ || frameGenMaxMultiFrameCount_ == 0) return;

    double baseFps = 1.0 / frameTimeSec;
    // Total frames needed to hit target = ceil(targetFps / baseFps)
    // Interp frames = total - 1 (the real frame counts as one)
    uint32_t totalNeeded = static_cast<uint32_t>(std::ceil(static_cast<double>(targetFps) / baseFps));
    uint32_t interpNeeded = (totalNeeded > 1) ? (totalNeeded - 1) : 1;
    frameGenMultiFrameCount_ = std::min(interpNeeded, frameGenMaxMultiFrameCount_);
}

void DLSSModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                             std::shared_ptr<vk::DeviceLocalImage> image,
                             int index) {}

void DLSSModule::preClose() {
    if (dlssFG_) { dlssFG_->deinit(); dlssFG_ = nullptr; }
    dlss_->deinit();
}

DLSSModuleContext::DLSSModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                     std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                     std::shared_ptr<DLSSModule> dlssModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      dLSSModule(dlssModule),
      hdrImage(dlssModule->hdrImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(dlssModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(dlssModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(dlssModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(dlssModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(dlssModule->linearDepthImages_[frameworkContext->frameIndex]),
      specularHitDepthImage(dlssModule->specularHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(dlssModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      processedImage(dlssModule->processedImages_[frameworkContext->frameIndex]),
      upscaledFirstHitDepthImage(dlssModule->upscaledFirstHitDepthImages_[frameworkContext->frameIndex]) {}

void DLSSModuleContext::render() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = dLSSModule.lock();

    {
        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = hdrImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = hdrImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = diffuseAlbedoImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = diffuseAlbedoImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = specularAlbedoImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = specularAlbedoImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = normalRoughnessImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = normalRoughnessImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = motionVectorImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = motionVectorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = linearDepthImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = linearDepthImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = processedImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = processedImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
        hdrImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        diffuseAlbedoImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        specularAlbedoImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        normalRoughnessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        motionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        linearDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        processedImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        module->dlss_->setResource(DlssRR::RESOURCE_COLOR_IN, hdrImage);
        module->dlss_->setResource(DlssRR::RESOURCE_COLOR_OUT, processedImage);
        module->dlss_->setResource(DlssRR::RESOURCE_DIFFUSE_ALBEDO, diffuseAlbedoImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_ALBEDO, specularAlbedoImage);
        module->dlss_->setResource(DlssRR::RESOURCE_NORMALROUGHNESS, normalRoughnessImage);
        module->dlss_->setResource(DlssRR::RESOURCE_MOTIONVECTOR, motionVectorImage);
        module->dlss_->setResource(DlssRR::RESOURCE_LINEARDEPTH, linearDepthImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_HITDISTANCE, specularHitDepthImage);

        auto worldUBOBuffer = Renderer::instance().buffers()->worldUniformBuffer();
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
        if (worldUBO != nullptr) {
            glm::vec2 jitter = worldUBO->cameraJitter;
            module->dlss_->denoise(worldCommandBuffer, glm::uvec2{module->inputWidth_, module->inputHeight_}, jitter,
                                   worldUBO->cameraViewMat, worldUBO->cameraProjMat);
        }
    }

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = firstHitDepthImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = firstHitDepthImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = upscaledFirstHitDepthImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = upscaledFirstHitDepthImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             }});
    firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // TODO: add to command buffer
    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(firstHitDepthImage->width()),
                               static_cast<int>(firstHitDepthImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(upscaledFirstHitDepthImage->width()),
                               static_cast<int>(upscaledFirstHitDepthImage->height()), 1};

    vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), firstHitDepthImage->vkImage(),
                   firstHitDepthImage->imageLayout(), upscaledFirstHitDepthImage->vkImage(),
                   upscaledFirstHitDepthImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
}