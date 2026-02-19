#include "xess_sr_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>

XessSrModule::XessSrModule() {}

bool XessSrModule::isQualityModeAttributeKey(const std::string &key) {
    return key == "render_pipeline.module.xess_sr.attribute.quality_mode";
}

bool XessSrModule::parseQualityModeValue(const std::string &value, QualityMode &outMode) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Numeric mapping follows QualityMode enum values:
    // 0 native_aa, 1 ultra_quality_plus, 2 ultra_quality, 3 quality,
    // 4 balanced, 5 performance, 6 ultra_performance.
    if (v == "0" || v == "native" || v == "native_aa" || v == "1x" ||
        v == "render_pipeline.module.xess_sr.attribute.quality_mode.native_aa" ||
        v == "render_pipeline.module.xess_sr.attribute.quality_mode.native_anti_aliasing") {
        outMode = QualityMode::NativeAA;
        return true;
    }
    if (v == "ultra_quality_plus" || v == "uq_plus" || v == "uqp" || v == "ultraqualityplus" ||
        v == "render_pipeline.module.xess_sr.attribute.quality_mode.ultra_quality_plus" || v == "1") {
        outMode = QualityMode::UltraQualityPlus;
        return true;
    }
    if (v == "ultra_quality" || v == "ultraquality" ||
        v == "render_pipeline.module.xess_sr.attribute.quality_mode.ultra_quality" || v == "2") {
        outMode = QualityMode::UltraQuality;
        return true;
    }
    if (v == "3" || v == "quality" || v == "render_pipeline.module.xess_sr.attribute.quality_mode.quality") {
        outMode = QualityMode::Quality;
        return true;
    }
    if (v == "4" || v == "balanced" || v == "render_pipeline.module.xess_sr.attribute.quality_mode.balanced") {
        outMode = QualityMode::Balanced;
        return true;
    }
    if (v == "5" || v == "performance" || v == "render_pipeline.module.xess_sr.attribute.quality_mode.performance") {
        outMode = QualityMode::Performance;
        return true;
    }
    if (v == "6" || v == "ultra" || v == "ultra_performance" || v == "ultra_performance_3x" ||
        v == "render_pipeline.module.xess_sr.attribute.quality_mode.ultra_performance") {
        outMode = QualityMode::UltraPerformance;
        return true;
    }
    return false;
}

void XessSrModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();
    deviceDepthImages_.resize(size);
    xessMotionVectorImages_.resize(size);
    inputImages_.resize(size);
    outputImages_.resize(size);
}

bool XessSrModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                          std::vector<VkFormat> &formats,
                                          uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto fw = framework_.lock();
    if (!fw) return false;

    if (displayWidth_ == 0 || displayHeight_ == 0) {
        VkExtent2D extent = fw->swapchain()->vkExtent();
        displayWidth_ = extent.width;
        displayHeight_ = extent.height;
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
            bool gotOptimal = false;
            if (xessEnabled_) {
                gotOptimal = mcvr::XeSSWrapper::queryOptimalInputResolution(
                    fw->instance()->vkInstance(), fw->physicalDevice()->vkPhysicalDevice(), fw->device()->vkDevice(),
                    displayWidth_, displayHeight_, static_cast<mcvr::XeSSQualityMode>(qualityMode_), &renderWidth_,
                    &renderHeight_);
            }
            if (!gotOptimal) {
                getRenderResolution(displayWidth_, displayHeight_, qualityMode_, &renderWidth_, &renderHeight_);
            }
        }
    }

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        } else if (images[i]->width() != renderWidth_ || images[i]->height() != renderHeight_) {
            return false;
        }
        inputImages_[frameIndex][i] = images[i];
    }

    return true;
}

bool XessSrModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                           std::vector<VkFormat> &formats,
                                           uint32_t frameIndex) {
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
            VkExtent2D extent = fw->swapchain()->vkExtent();
            displayWidth_ = extent.width;
            displayHeight_ = extent.height;
        }
    }

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                fw->device(), fw->vma(), false, displayWidth_, displayHeight_, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        } else if (images[i]->width() != displayWidth_ || images[i]->height() != displayHeight_) {
            return false;
        }
        outputImages_[frameIndex][i] = images[i];
    }

    return true;
}

void XessSrModule::build() {
    auto fw = framework_.lock();
    auto wp = worldPipeline_.lock();
    uint32_t size = fw->swapchain()->imageCount();

    xess_ = std::make_shared<mcvr::XeSSWrapper>();

    mcvr::XeSSConfig config{};
    config.instance = fw->instance()->vkInstance();
    config.physicalDevice = fw->physicalDevice()->vkPhysicalDevice();
    config.device = fw->device()->vkDevice();
    config.renderWidth = renderWidth_;
    config.renderHeight = renderHeight_;
    config.displayWidth = displayWidth_;
    config.displayHeight = displayHeight_;
    config.qualityMode = static_cast<mcvr::XeSSQualityMode>(qualityMode_);
    config.initFlags = 0;
    config.velocityScaleX = 1.0f;
    config.velocityScaleY = 1.0f;

    if (!xessEnabled_) {
        initialized_ = false;
    } else if (!xess_->initialize(config)) {
        std::cerr << "XessSrModule: Failed to initialize XeSS, fallback to blit" << std::endl;
        initialized_ = false;
    } else {
        initialized_ = true;
    }

    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        contexts_[i] =
            std::make_shared<XessSrModuleContext>(fw->contexts()[i], wp->contexts()[i], shared_from_this());

        contexts_[i]->inputColorImage = inputImages_[i][0];
        contexts_[i]->inputDepthImage = inputImages_[i][1];
        contexts_[i]->inputMotionVectorImage = inputImages_[i][2];
        contexts_[i]->inputFirstHitDepthImage = inputImages_[i][3];

        contexts_[i]->outputImage = outputImages_[i][0];
        contexts_[i]->upscaledFirstHitDepthImage = outputImages_[i][1];
        contexts_[i]->depthDescriptorTable = depthDescriptorTables_[i];
        contexts_[i]->deviceDepthImage = deviceDepthImages_[i];
        contexts_[i]->xessMotionVectorImage = xessMotionVectorImages_[i];
    }
}

void XessSrModule::initDescriptorTables() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();
    depthDescriptorTables_.resize(size);

    for (uint32_t i = 0; i < size; i++) {
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

void XessSrModule::initImages() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();

    for (uint32_t i = 0; i < size; i++) {
        deviceDepthImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        depthDescriptorTables_[i]->bindImage(deviceDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);

        xessMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }
}

void XessSrModule::initPipeline() {
    auto fw = framework_.lock();
    auto shader = vk::Shader::create(
        fw->device(), (Renderer::folderPath / "shaders/world/upscaler/linear_to_device_depth_comp.spv").string());

    depthConversionPipeline_ = vk::ComputePipelineBuilder{}
                                   .defineShader(shader)
                                   .definePipelineLayout(depthDescriptorTables_[0])
                                   .build(fw->device());
}

void XessSrModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) {
        if (value == "1" || value == "true" || value == "True" || value == "TRUE") return true;
        if (value == "0" || value == "false" || value == "False" || value == "FALSE") return false;
        return true;
    };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.xess_sr.attribute.enable") {
            xessEnabled_ = parseBool(value);
        } else if (isQualityModeAttributeKey(key)) {
            QualityMode mode = qualityMode_;
            if (parseQualityModeValue(value, mode)) {
                qualityMode_ = mode;
                if (displayWidth_ > 0 && displayHeight_ > 0) {
                    bool gotOptimal = false;
                    auto fw = framework_.lock();
                    if (xessEnabled_ && fw) {
                        gotOptimal = mcvr::XeSSWrapper::queryOptimalInputResolution(
                            fw->instance()->vkInstance(), fw->physicalDevice()->vkPhysicalDevice(),
                            fw->device()->vkDevice(), displayWidth_, displayHeight_,
                            static_cast<mcvr::XeSSQualityMode>(qualityMode_), &renderWidth_, &renderHeight_);
                    }
                    if (!gotOptimal) {
                        getRenderResolution(displayWidth_, displayHeight_, qualityMode_, &renderWidth_, &renderHeight_);
                    }
                } else {
                    renderWidth_ = 0;
                    renderHeight_ = 0;
                }
            }
        } else if (key == "render_pipeline.module.xess_sr.attribute.pre_exposure") {
            preExposure_ = std::stof(value);
        }
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &XessSrModule::contexts() {
    return reinterpret_cast<std::vector<std::shared_ptr<WorldModuleContext>> &>(contexts_);
}

void XessSrModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                               std::shared_ptr<vk::DeviceLocalImage> image,
                               int index) {}

void XessSrModule::preClose() {
    if (xess_) {
        xess_->destroy();
        xess_.reset();
    }
    initialized_ = false;
}

void XessSrModule::getRenderResolution(uint32_t displayWidth,
                                       uint32_t displayHeight,
                                       QualityMode mode,
                                       uint32_t *outRenderWidth,
                                       uint32_t *outRenderHeight) {
    float ratio = 1.0f;
    switch (mode) {
        case QualityMode::NativeAA: ratio = 1.0f; break;
        case QualityMode::UltraQualityPlus: ratio = 1.3f; break;
        case QualityMode::UltraQuality: ratio = 1.5f; break;
        case QualityMode::Quality: ratio = 1.7f; break;
        case QualityMode::Balanced: ratio = 2.0f; break;
        case QualityMode::Performance: ratio = 2.3f; break;
        case QualityMode::UltraPerformance: ratio = 3.0f; break;
    }

    *outRenderWidth = static_cast<uint32_t>(static_cast<float>(displayWidth) / ratio);
    *outRenderHeight = static_cast<uint32_t>(static_cast<float>(displayHeight) / ratio);
}

XessSrModuleContext::XessSrModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                         std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                         std::shared_ptr<XessSrModule> xessModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext), xessModule_(xessModule) {}

bool XessSrModuleContext::checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir) {
    auto module = xessModule_.lock();
    if (module->firstFrame_) {
        module->firstFrame_ = false;
        module->lastCameraPos_ = cameraPos;
        module->lastCameraDir_ = cameraDir;
        return true;
    }

    float positionDelta = glm::length(cameraPos - module->lastCameraPos_);
    float directionDot = glm::dot(glm::normalize(cameraDir), glm::normalize(module->lastCameraDir_));
    bool shouldReset = (positionDelta > 1.0f) || (directionDot < 0.866f);

    module->lastCameraPos_ = cameraPos;
    module->lastCameraDir_ = cameraDir;
    return shouldReset;
}

void XessSrModuleContext::render() {
    auto module = xessModule_.lock();
    if (!module) return;

    auto fwContext = frameworkContext.lock();
    auto worldCommandBuffer = fwContext->worldCommandBuffer;
    auto mainQueueIndex = fwContext->framework.lock()->physicalDevice()->mainQueueIndex();

    auto fallbackBlit = [&]() {
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
        colorBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        colorBlit.srcOffsets[1] = {static_cast<int32_t>(inputColorImage->width()),
                                   static_cast<int32_t>(inputColorImage->height()), 1};
        colorBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        colorBlit.dstOffsets[1] = {static_cast<int32_t>(outputImage->width()),
                                   static_cast<int32_t>(outputImage->height()), 1};

        vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), inputColorImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outputImage->vkImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &colorBlit, VK_FILTER_LINEAR);

        VkImageBlit depthBlit{};
        depthBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        depthBlit.srcOffsets[1] = {static_cast<int32_t>(inputFirstHitDepthImage->width()),
                                   static_cast<int32_t>(inputFirstHitDepthImage->height()), 1};
        depthBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
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
    };

    if (!module->xessEnabled_ || !module->initialized_ || !module->xess_) {
        fallbackBlit();
        return;
    }

    auto buffers = Renderer::instance().buffers();
    auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());

    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
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
              .oldLayout = deviceDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = deviceDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
              .oldLayout = xessMotionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = xessMotionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .oldLayout = outputImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = outputImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    inputColorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    inputDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    inputMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    deviceDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    xessMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    depthDescriptorTable->bindImage(inputDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
    depthDescriptorTable->bindImage(deviceDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    depthDescriptorTable->bindImage(inputMotionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    depthDescriptorTable->bindImage(xessMotionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 3);

    struct PushConstants {
        float cameraNear;
        float cameraFar;
        uint32_t width;
        uint32_t height;
        float jitterX;
        float jitterY;
    } pushConstants{0.1f, 10000.0f, module->renderWidth_, module->renderHeight_, worldUBO->cameraJitter.x,
                    worldUBO->cameraJitter.y};

    worldCommandBuffer->bindDescriptorTable(depthDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->depthConversionPipeline_);

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), depthDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->renderWidth_ + 15) / 16,
                  (module->renderHeight_ + 15) / 16, 1);

    worldCommandBuffer->barriersMemory({{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT}});

    worldCommandBuffer->barriersBufferImage(
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = inputColorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = inputColorImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = deviceDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = deviceDepthImage,
              .subresourceRange = vk::wholeColorSubresourceRange},
             {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = xessMotionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = xessMotionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    inputColorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    deviceDepthImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    xessMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    mcvr::XeSSInput input{};
    input.commandBuffer = worldCommandBuffer->vkCommandBuffer();

    input.colorTexture.image = inputColorImage->vkImage();
    input.colorTexture.imageView = inputColorImage->vkImageView();
    input.colorTexture.format = inputColorImage->vkFormat();
    input.colorTexture.width = inputColorImage->width();
    input.colorTexture.height = inputColorImage->height();

    input.velocityTexture.image = xessMotionVectorImage->vkImage();
    input.velocityTexture.imageView = xessMotionVectorImage->vkImageView();
    input.velocityTexture.format = xessMotionVectorImage->vkFormat();
    input.velocityTexture.width = xessMotionVectorImage->width();
    input.velocityTexture.height = xessMotionVectorImage->height();

    input.depthTexture.image = deviceDepthImage->vkImage();
    input.depthTexture.imageView = deviceDepthImage->vkImageView();
    input.depthTexture.format = deviceDepthImage->vkFormat();
    input.depthTexture.width = deviceDepthImage->width();
    input.depthTexture.height = deviceDepthImage->height();

    input.outputTexture.image = outputImage->vkImage();
    input.outputTexture.imageView = outputImage->vkImageView();
    input.outputTexture.format = outputImage->vkFormat();
    input.outputTexture.width = outputImage->width();
    input.outputTexture.height = outputImage->height();

    input.jitterOffsetX = -worldUBO->cameraJitter.x;
    input.jitterOffsetY = -worldUBO->cameraJitter.y;
    input.exposureScale = module->preExposure_;
    input.resetHistory = checkCameraReset(glm::vec3(worldUBO->cameraPos),
                                          glm::vec3(worldUBO->cameraViewMat[0][2], worldUBO->cameraViewMat[1][2],
                                                    worldUBO->cameraViewMat[2][2]));
    input.inputWidth = module->renderWidth_;
    input.inputHeight = module->renderHeight_;

    if (!module->xess_->dispatch(input)) {
        fallbackBlit();
        return;
    }

    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

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
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(module->renderWidth_), static_cast<int32_t>(module->renderHeight_), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(module->displayWidth_), static_cast<int32_t>(module->displayHeight_), 1};

    vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), inputFirstHitDepthImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, upscaledFirstHitDepthImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
}
