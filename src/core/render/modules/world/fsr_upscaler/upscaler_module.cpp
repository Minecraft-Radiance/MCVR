#include "upscaler_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr3_upscaler.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <iostream>

UpscalerModule::UpscalerModule() {}

bool UpscalerModule::isQualityModeAttributeKey(const std::string &key) {
    return key == "render_pipeline.module.fsr3_upscaler.attribute.quality_mode";
}

bool UpscalerModule::parseQualityModeValue(const std::string &value, QualityMode &outMode) {
    if (value == "0" || value == "native" || value == "native_aa" || value == "1x") {
        outMode = QualityMode::NativeAA;
        return true;
    }
    if (value == "1" || value == "quality") {
        outMode = QualityMode::Quality;
        return true;
    }
    if (value == "2" || value == "balanced") {
        outMode = QualityMode::Balanced;
        return true;
    }
    if (value == "3" || value == "performance") {
        outMode = QualityMode::Performance;
        return true;
    }
    if (value == "4" || value == "ultra" || value == "ultra_performance") {
        outMode = QualityMode::UltraPerformance;
        return true;
    }
    return false;
}

void UpscalerModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();
    deviceDepthImages_.resize(size);
    fsrMotionVectorImages_.resize(size);
    inputImages_.resize(size);
    outputImages_.resize(size);
}

bool UpscalerModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                            std::vector<VkFormat> &formats, uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto fw = framework_.lock();
    if (!fw) return false;

    if (displayWidth_ == 0 || displayHeight_ == 0) {
        const auto &vr = Renderer::instance().vrSystem();
        if (vr.enabled && eyeCount_ > 1) {
            displayWidth_ = vr.eyeRenderWidth();
            displayHeight_ = vr.eyeRenderHeight();
        }
        if (displayWidth_ == 0 || displayHeight_ == 0) {
            VkExtent2D extent = fw->swapchain()->vkExtent();
            displayWidth_ = extent.width;
            displayHeight_ = extent.height;
        }
    }

    if (renderWidth_ == 0 || renderHeight_ == 0) {
        for (const auto &img : images) {
            if (img != nullptr) {
                renderWidth_ = img->width();
                renderHeight_ = img->height();
                break;
            }
        }
        if (renderWidth_ == 0 || renderHeight_ == 0) {
            getRenderResolution(displayWidth_, displayHeight_, qualityMode_, &renderWidth_, &renderHeight_);
        }
    }

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                fw->device(), fw->vma(), false, renderWidth_, renderHeight_, eyeCount_, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            if (eyeCount_ > 1) images[i]->createPerLayerViews();
        } else if (images[i]->width() != renderWidth_ || images[i]->height() != renderHeight_) {
            return false;
        }
        inputImages_[frameIndex][i] = images[i];
    }

    return true;
}

bool UpscalerModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                             std::vector<VkFormat> &formats, uint32_t frameIndex) {
    if (images.size() != outputImageNum) return false;

    auto fw = framework_.lock();
    if (!fw) return false;

    if (displayWidth_ == 0 || displayHeight_ == 0) {
        for (const auto &img : images) {
            if (img != nullptr) {
                displayWidth_ = img->width();
                displayHeight_ = img->height();
                break;
            }
        }
        if (displayWidth_ == 0 || displayHeight_ == 0) {
            const auto &vr = Renderer::instance().vrSystem();
            if (vr.enabled && eyeCount_ > 1) {
                displayWidth_ = vr.eyeRenderWidth();
                displayHeight_ = vr.eyeRenderHeight();
            }
            if (displayWidth_ == 0 || displayHeight_ == 0) {
                VkExtent2D extent = fw->swapchain()->vkExtent();
                displayWidth_ = extent.width;
                displayHeight_ = extent.height;
            }
        }
    }

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                fw->device(), fw->vma(), false, displayWidth_, displayHeight_, eyeCount_, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            if (eyeCount_ > 1) images[i]->createPerLayerViews();
        } else if (images[i]->width() != displayWidth_ || images[i]->height() != displayHeight_) {
            return false;
        }
        outputImages_[frameIndex][i] = images[i];
    }

    return true;
}

void UpscalerModule::build() {
    auto fw = framework_.lock();
    auto wp = worldPipeline_.lock();
    uint32_t size = fw->swapchain()->imageCount();

    // Create per-eye FSR3 instances (DualInstance)
    fsr3Upscalers_.resize(eyeCount_);

    mcvr::UpscalerConfig config{};
    config.device = fw->device()->vkDevice();
    config.physicalDevice = fw->physicalDevice()->vkPhysicalDevice();
    config.commandPool = fw->mainCommandPool()->vkCommandPool();
    config.graphicsQueue = fw->device()->mainVkQueue();
    config.graphicsQueueFamily = fw->physicalDevice()->mainQueueIndex();
    config.maxRenderWidth = renderWidth_;
    config.maxRenderHeight = renderHeight_;
    config.maxDisplayWidth = displayWidth_;
    config.maxDisplayHeight = displayHeight_;
    config.qualityMode = static_cast<mcvr::UpscalerQualityMode>(qualityMode_);
    config.hdr = true;
    config.depthInverted = false;
    config.depthInfinite = true;
    config.autoExposure = false;
    config.enableSharpening = true;
    config.sharpness = sharpness_;

    if (!fsr3Enabled_) {
        initialized_ = false;
    } else {
        initialized_ = true;
        for (uint32_t eye = 0; eye < eyeCount_; eye++) {
            fsr3Upscalers_[eye] = std::make_shared<mcvr::FSR3Upscaler>();
            if (!fsr3Upscalers_[eye]->initialize(config)) {
                std::cerr << "UpscalerModule: Failed to initialize FSR3 for eye " << eye << std::endl;
                initialized_ = false;
            }
        }
    }

    uint32_t totalCtx = size * eyeCount_;
    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        contexts_[i] =
            std::make_shared<UpscalerModuleContext>(fw->contexts()[i], wp->contexts()[i], shared_from_this());

        contexts_[i]->inputColorImage = inputImages_[i][0];
        contexts_[i]->inputDepthImage = inputImages_[i][1];
        contexts_[i]->inputMotionVectorImage = inputImages_[i][2];
        contexts_[i]->inputFirstHitDepthImage = inputImages_[i][3];

        contexts_[i]->outputImage = outputImages_[i][0];
        contexts_[i]->upscaledFirstHitDepthImage = outputImages_[i][1];

        // Store eye-0 references for mono compatibility
        contexts_[i]->deviceDepthImage = deviceDepthImages_[i * eyeCount_];
        contexts_[i]->fsrMotionVectorImage = fsrMotionVectorImages_[i * eyeCount_];
        contexts_[i]->depthDescriptorTable = depthDescriptorTables_[i * eyeCount_];
    }
}

void UpscalerModule::initDescriptorTables() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;
    depthDescriptorTables_.resize(totalCtx);

    for (uint32_t i = 0; i < totalCtx; i++) {
        depthDescriptorTables_[i] = vk::DescriptorTableBuilder{}
                                        .beginDescriptorLayoutSet()
                                        .beginDescriptorLayoutSetBinding()
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 0,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 2,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .defineDescriptorLayoutSetBinding({
                                            .binding = 3,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                        })
                                        .endDescriptorLayoutSetBinding()
                                        .endDescriptorLayoutSet()
                                        .definePushConstant({
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                            .offset = 0,
                                            .size = sizeof(float) * 4 + sizeof(uint32_t) * 2,
                                        })
                                        .build(fw->device());
    }
}

void UpscalerModule::initImages() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;

    deviceDepthImages_.resize(totalCtx);
    fsrMotionVectorImages_.resize(totalCtx);
    fsr3ColorImages_.resize(totalCtx);
    fsr3OutputImages_.resize(totalCtx);

    for (uint32_t i = 0; i < totalCtx; i++) {
        deviceDepthImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        depthDescriptorTables_[i]->bindImage(deviceDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);

        fsrMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        // Per-eye single-layer intermediates for FSR3 SDK (can't handle array images)
        fsr3ColorImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        fsr3OutputImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, displayWidth_, displayHeight_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }
}

void UpscalerModule::initPipeline() {
    auto fw = framework_.lock();
    auto shader = vk::Shader::create(fw->device(), (Renderer::folderPath / "shaders/world/upscaler/linear_to_device_depth_comp.spv").string());

    depthConversionPipeline_ = vk::ComputePipelineBuilder{}
                                   .defineShader(shader)
                                   .definePipelineLayout(depthDescriptorTables_[0])
                                   .build(fw->device());
}

void UpscalerModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) {
        if (value == "1" || value == "true" || value == "True" || value == "TRUE") return true;
        if (value == "0" || value == "false" || value == "False" || value == "FALSE") return false;
        return true;
    };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.fsr3_upscaler.attribute.enable") {
            fsr3Enabled_ = parseBool(value);
        } else if (isQualityModeAttributeKey(key)) {
            QualityMode mode = qualityMode_;
            if (parseQualityModeValue(value, mode)) {
                qualityMode_ = mode;
                if (displayWidth_ > 0 && displayHeight_ > 0) {
                    getRenderResolution(displayWidth_, displayHeight_, qualityMode_, &renderWidth_, &renderHeight_);
                } else {
                    renderWidth_ = 0;
                    renderHeight_ = 0;
                }
            }
        } else if (key == "render_pipeline.module.fsr3_upscaler.attribute.sharpness") {
            sharpness_ = std::stof(value);
        } else if (key == "render_pipeline.module.fsr3_upscaler.attribute.pre_exposure") {
            preExposure_ = std::stof(value);
        }
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &UpscalerModule::contexts() {
    return reinterpret_cast<std::vector<std::shared_ptr<WorldModuleContext>> &>(contexts_);
}

void UpscalerModule::bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image,
                                 int index) {}

void UpscalerModule::preClose() {
    for (auto &fsr : fsr3Upscalers_) {
        if (fsr) {
            fsr->destroy();
            fsr.reset();
        }
    }
    fsr3Upscalers_.clear();
    fsr3ColorImages_.clear();
    fsr3OutputImages_.clear();
    initialized_ = false;
}

void UpscalerModule::getRenderResolution(uint32_t displayWidth, uint32_t displayHeight, QualityMode mode,
                                         uint32_t *outRenderWidth, uint32_t *outRenderHeight) {
    float ratio = 1.0f;
    switch (mode) {
    case QualityMode::NativeAA: ratio = 1.0f; break;
    case QualityMode::Quality: ratio = 1.5f; break;
    case QualityMode::Balanced: ratio = 1.7f; break;
    case QualityMode::Performance: ratio = 2.0f; break;
    case QualityMode::UltraPerformance: ratio = 3.0f; break;
    }

    *outRenderWidth = static_cast<uint32_t>(static_cast<float>(displayWidth) / ratio);
    *outRenderHeight = static_cast<uint32_t>(static_cast<float>(displayHeight) / ratio);
}

UpscalerModuleContext::UpscalerModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                             std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                             std::shared_ptr<UpscalerModule> upscalerModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext), upscalerModule_(upscalerModule) {
    lastFrameTime_ = std::chrono::high_resolution_clock::now();
}

bool UpscalerModuleContext::checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir) {
    auto module = upscalerModule_.lock();
    if (module->firstFrame_) {
        module->firstFrame_ = false;
        module->lastCameraPos_ = cameraPos;
        module->lastCameraDir_ = cameraDir;
        lastResetResult_ = true;
        return true;
    }

    float positionDelta = glm::length(cameraPos - module->lastCameraPos_);
    float directionDot = glm::dot(glm::normalize(cameraDir), glm::normalize(module->lastCameraDir_));
    bool shouldReset = (positionDelta > 1.0f) || (directionDot < 0.866f);

    module->lastCameraPos_ = cameraPos;
    module->lastCameraDir_ = cameraDir;
    lastResetResult_ = shouldReset;
    return shouldReset;
}

float UpscalerModuleContext::getSmoothDeltaTime() {
    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaMs = std::chrono::duration<float, std::milli>(currentTime - lastFrameTime_).count();
    lastFrameTime_ = currentTime;

    if (timingFirstFrame_) {
        timingFirstFrame_ = false;
        return 16.67f;
    }

    if (deltaMs < 1.0f || deltaMs > 100.0f) deltaMs = 16.67f;

    frameTimes_.push_back(deltaMs);
    if (frameTimes_.size() > 10) frameTimes_.pop_front();

    float sum = 0.0f;
    for (float t : frameTimes_) sum += t;
    return sum / frameTimes_.size();
}

void UpscalerModuleContext::render() {
    renderEye(0);
}

void UpscalerModuleContext::renderEye(uint32_t eyeIndex) {
    auto module = upscalerModule_.lock();
    if (!module) return;

    auto fwContext = frameworkContext.lock();
    auto worldCommandBuffer = fwContext->worldCommandBuffer;
    auto mainQueueIndex = fwContext->framework.lock()->physicalDevice()->mainQueueIndex();

    uint32_t dtIdx = fwContext->frameIndex * module->eyeCount() + eyeIndex;
    auto eyeDeviceDepth = module->deviceDepthImages_[dtIdx];
    auto eyeFsrMotionVector = module->fsrMotionVectorImages_[dtIdx];
    auto eyeDepthDT = module->depthDescriptorTables_[dtIdx];
    auto eyeColorIntermediate = module->fsr3ColorImages_[dtIdx];
    auto eyeOutputIntermediate = module->fsr3OutputImages_[dtIdx];

    if (!module->fsr3Enabled_) {
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                  .oldLayout = inputColorImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = inputColorImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .oldLayout = outputImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = outputImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                  .oldLayout = inputFirstHitDepthImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = inputFirstHitDepthImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .oldLayout = upscaledFirstHitDepthImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = upscaledFirstHitDepthImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        inputColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        outputImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        inputFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageBlit colorBlit{};
        colorBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
        colorBlit.srcOffsets[1] = {static_cast<int32_t>(inputColorImage->width()),
                                   static_cast<int32_t>(inputColorImage->height()), 1};
        colorBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
        colorBlit.dstOffsets[1] = {static_cast<int32_t>(outputImage->width()),
                                   static_cast<int32_t>(outputImage->height()), 1};

        vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), inputColorImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &colorBlit, VK_FILTER_LINEAR);

        VkImageBlit depthBlit{};
        depthBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
        depthBlit.srcOffsets[1] = {static_cast<int32_t>(inputFirstHitDepthImage->width()),
                                   static_cast<int32_t>(inputFirstHitDepthImage->height()), 1};
        depthBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
        depthBlit.dstOffsets[1] = {static_cast<int32_t>(upscaledFirstHitDepthImage->width()),
                                   static_cast<int32_t>(upscaledFirstHitDepthImage->height()), 1};

        vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), inputFirstHitDepthImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, upscaledFirstHitDepthImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &depthBlit, VK_FILTER_LINEAR);

        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = outputImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        return;
    }

    if (!module->initialized_ || module->fsr3Upscalers_.size() <= eyeIndex || !module->fsr3Upscalers_[eyeIndex]) return;

    auto buffers = Renderer::instance().buffers();
    auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());

    // Barrier shared pipeline inputs + per-eye intermediates to GENERAL
    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
              .oldLayout = inputColorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputColorImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = inputDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = inputMotionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputMotionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = eyeDeviceDepth->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = eyeDeviceDepth,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = eyeFsrMotionVector->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = eyeFsrMotionVector,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = eyeOutputIntermediate->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = eyeOutputIntermediate,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    inputColorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    inputDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    inputMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    eyeDeviceDepth->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    eyeFsrMotionVector->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    eyeOutputIntermediate->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    // Copy inputColorImage layer[eyeIndex] → per-eye single-layer intermediate for FSR3
    {
        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
        copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.extent = {module->renderWidth_, module->renderHeight_, 1};

        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .oldLayout = eyeColorIntermediate->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = eyeColorIntermediate,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        vkCmdCopyImage(worldCommandBuffer->vkCommandBuffer(),
                       inputColorImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                       eyeColorIntermediate->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &copyRegion);

        eyeColorIntermediate->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }

    // Depth conversion: bind per-layer views for array inputs
    uint32_t viewIdx = (module->eyeCount() > 1) ? (1 + eyeIndex) : 0;
    eyeDepthDT->bindImage(inputDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0, viewIdx);
    eyeDepthDT->bindImage(eyeDeviceDepth, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    eyeDepthDT->bindImage(inputMotionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 2, viewIdx);
    eyeDepthDT->bindImage(eyeFsrMotionVector, VK_IMAGE_LAYOUT_GENERAL, 0, 3);
    struct PushConstants {
        float cameraNear;
        float cameraFar;
        uint32_t width;
        uint32_t height;
        float jitterX;
        float jitterY;
    } pushConstants{0.1f, 10000.0f, module->renderWidth_, module->renderHeight_,
                    worldUBO->cameraJitter.x, worldUBO->cameraJitter.y};

    worldCommandBuffer->bindDescriptorTable(eyeDepthDT, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->depthConversionPipeline_);

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), eyeDepthDT->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->renderWidth_ + 15) / 16,
                  (module->renderHeight_ + 15) / 16, 1);

    worldCommandBuffer->barriersMemory({{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT}});

    // Transition per-eye intermediates to shader read-only for FSR3
    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = eyeColorIntermediate->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = eyeColorIntermediate,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = eyeDeviceDepth->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = eyeDeviceDepth,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = eyeFsrMotionVector->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = eyeFsrMotionVector,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    eyeColorIntermediate->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    eyeDeviceDepth->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    eyeFsrMotionVector->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // FSR3 Dispatch with per-eye single-layer intermediates
    mcvr::UpscalerInput input{};
    input.commandBuffer = worldCommandBuffer->vkCommandBuffer();
    input.colorImage = eyeColorIntermediate->vkImage();
    input.colorImageView = eyeColorIntermediate->vkImageView();
    input.colorLayout = eyeColorIntermediate->imageLayout();
    input.depthImage = eyeDeviceDepth->vkImage();
    input.depthImageView = eyeDeviceDepth->vkImageView();
    input.depthFormat = eyeDeviceDepth->vkFormat();
    input.motionVectorImage = eyeFsrMotionVector->vkImage();
    input.motionVectorImageView = eyeFsrMotionVector->vkImageView();
    input.outputImage = eyeOutputIntermediate->vkImage();
    input.outputImageView = eyeOutputIntermediate->vkImageView();
    input.outputLayout = VK_IMAGE_LAYOUT_GENERAL;
    input.renderWidth = module->renderWidth_;
    input.renderHeight = module->renderHeight_;
    input.displayWidth = module->displayWidth_;
    input.displayHeight = module->displayHeight_;
    input.jitterOffsetX = -worldUBO->cameraJitter.x;
    input.jitterOffsetY = -worldUBO->cameraJitter.y;
    input.motionVectorScaleX = 1.0f;
    input.motionVectorScaleY = 1.0f;
    input.cameraNear = 0.1f;
    input.cameraFar = 10000.0f;
    {
        glm::mat4 eyeProj = worldUBO->eyeProjOffsets[eyeIndex] * worldUBO->cameraProjMat;
        input.cameraFovVertical = 2.0f * atan(1.0f / eyeProj[1][1]);
    }
    input.frameTimeDelta = getSmoothDeltaTime();
    input.preExposure = module->preExposure_;
    input.reset = (eyeIndex == 0)
                      ? checkCameraReset(glm::vec3(worldUBO->cameraPos),
                                         glm::vec3(worldUBO->cameraViewMat[0][2], worldUBO->cameraViewMat[1][2],
                                                   worldUBO->cameraViewMat[2][2]))
                      : lastResetResult_;
    input.enableSharpening = true;
    input.sharpness = module->sharpness_;

    module->fsr3Upscalers_[eyeIndex]->dispatch(input);
    eyeOutputIntermediate->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    // Copy per-eye FSR3 output → outputImage layer[eyeIndex]
    {
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                  .oldLayout = eyeOutputIntermediate->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = eyeOutputIntermediate,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .oldLayout = outputImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = outputImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        eyeOutputIntermediate->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        outputImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
        copyRegion.extent = {module->displayWidth_, module->displayHeight_, 1};

        vkCmdCopyImage(worldCommandBuffer->vkCommandBuffer(),
                       eyeOutputIntermediate->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       outputImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &copyRegion);
    }

    // Transition output to GENERAL for downstream
    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = outputImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = outputImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    // Blit first hit depth per-layer
    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
              .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
              .oldLayout = inputFirstHitDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputFirstHitDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
              .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .oldLayout = upscaledFirstHitDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = upscaledFirstHitDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    inputFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(module->renderWidth_), static_cast<int32_t>(module->renderHeight_), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eyeIndex, 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(module->displayWidth_), static_cast<int32_t>(module->displayHeight_), 1};

    vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), inputFirstHitDepthImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, upscaledFirstHitDepthImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
}
