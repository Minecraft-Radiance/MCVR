#include "fsr_upscaler_module.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/frame_generation.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr3_upscaler.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

FSRUpscalerModule::FSRUpscalerModule() {
#ifndef MCVR_ENABLE_FFX_UPSCALER
    fsr3Enabled_ = false;
    qualityMode_ = QualityMode::NativeAA;
#endif
}

bool FSRUpscalerModule::isHdrUpscaleFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return true;
        default: return false;
    }
}

bool FSRUpscalerModule::isQualityModeAttributeKey(const std::string &key) {
    return key == "render_pipeline.module.fsr_upscaler.attribute.quality_mode";
}

bool FSRUpscalerModule::parseQualityModeValue(const std::string &value, QualityMode &outMode) {
    if (value == "render_pipeline.module.fsr_upscaler.attribute.quality_mode.native") {
        outMode = QualityMode::NativeAA;
        return true;
    }
    if (value == "render_pipeline.module.fsr_upscaler.attribute.quality_mode.quality") {
        outMode = QualityMode::Quality;
        return true;
    }
    if (value == "render_pipeline.module.fsr_upscaler.attribute.quality_mode.balanced") {
        outMode = QualityMode::Balanced;
        return true;
    }
    if (value == "render_pipeline.module.fsr_upscaler.attribute.quality_mode.performance") {
        outMode = QualityMode::Performance;
        return true;
    }
    if (value == "render_pipeline.module.fsr_upscaler.attribute.quality_mode.ultra") {
        outMode = QualityMode::UltraPerformance;
        return true;
    }
    return false;
}

void FSRUpscalerModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();
    deviceDepthImages_.resize(size);
    fsrMotionVectorImages_.resize(size);
    inputImages_.resize(size);
    outputImages_.resize(size);
}

bool FSRUpscalerModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
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
            getRenderResolution(displayWidth_, displayHeight_, qualityMode_, &renderWidth_, &renderHeight_);
        }
    }

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        } else if (images[i]->width() != renderWidth_ || images[i]->height() != renderHeight_) {
            return false;
        }
        inputImages_[frameIndex][i] = images[i];
    }

    return true;
}

bool FSRUpscalerModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
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

void FSRUpscalerModule::build() {
    auto fw = framework_.lock();
    auto wp = worldPipeline_.lock();
    uint32_t size = fw->swapchain()->imageCount();

    fsr3_ = std::make_shared<mcvr::FSR3Upscaler>();
    frameClock_.reset();
    firstFrame_ = true;
    hasDepthHistory_ = false;
    historyDepthIndex_ = 0;

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
    config.hdr =
        !inputImages_.empty() && inputImages_[0][0] != nullptr && isHdrUpscaleFormat(inputImages_[0][0]->vkFormat());
    config.depthInverted = false;
    config.depthInfinite = false;
    config.autoExposure = false;
    config.enableSharpening = true;
    config.sharpness = sharpness_;

    if (!fsr3Enabled_) {
        initialized_ = false;
    } else if (!fsr3_->initialize(config)) {
        std::cerr << "FSRUpscalerModule: Failed to initialize FSR3" << std::endl;
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
            std::make_shared<FSRUpscalerModuleContext>(fw->contexts()[i], wp->contexts()[i], shared_from_this());

        contexts_[i]->inputColorImage = inputImages_[i][0];
        contexts_[i]->inputDepthImage = inputImages_[i][1];
        contexts_[i]->inputMotionVectorImage = inputImages_[i][2];
        contexts_[i]->inputFirstHitDepthImage = inputImages_[i][3];
        contexts_[i]->inputNormalRoughnessImage = inputImages_[i][4];

        contexts_[i]->outputImage = outputImages_[i][0];
        contexts_[i]->upscaledFirstHitDepthImage = outputImages_[i][1];
        contexts_[i]->upscaledMotionVectorImage = outputImages_[i][2];
        contexts_[i]->upscaledNormalRoughnessImage = outputImages_[i][3];

        contexts_[i]->deviceDepthImage = deviceDepthImages_[i];
        contexts_[i]->fsrMotionVectorImage = fsrMotionVectorImages_[i];
        contexts_[i]->depthDescriptorTable = depthDescriptorTables_[i];
        contexts_[i]->guideDescriptorTable = guideDescriptorTables_[i];
    }
}

void FSRUpscalerModule::initDescriptorTables() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();
    depthDescriptorTables_.resize(size);
    guideDescriptorTables_.resize(size);
    reactiveDescriptorTables_.resize(size);
    reactiveImages_.resize(size);

    auto createTable = [&]() {
        vk::DescriptorTableBuilder builder;
        auto &bindings = builder.beginDescriptorLayoutSet().beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < 7; ++binding) {
            bindings.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
        }
        bindings.endDescriptorLayoutSetBinding().endDescriptorLayoutSet().definePushConstant({
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(MotionUpscalePushConstants),
        });
        return builder.build(fw->device());
    };

    for (uint32_t i = 0; i < size; ++i) {
        depthDescriptorTables_[i] = createTable();
        guideDescriptorTables_[i] = createTable();
        vk::DescriptorTableBuilder reactiveBuilder;
        auto &images = reactiveBuilder.beginDescriptorLayoutSet().beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < 5; ++binding) {
            images.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
        }
        images.endDescriptorLayoutSetBinding().endDescriptorLayoutSet();
        auto &uniforms = reactiveBuilder.beginDescriptorLayoutSet().beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < 2; ++binding) {
            uniforms.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
        }
        uniforms.endDescriptorLayoutSetBinding().endDescriptorLayoutSet().definePushConstant({
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
        });
        reactiveDescriptorTables_[i] = reactiveBuilder.build(fw->device());
    }
}

void FSRUpscalerModule::initImages() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();

    for (auto &image : historyDepthImages_) {
        image = vk::DeviceLocalImage::create(fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1,
                                             VK_FORMAT_R16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
    }
    for (uint32_t i = 0; i < size; i++) {
        reactiveImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        deviceDepthImages_[i] =
            vk::DeviceLocalImage::create(fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1,
                                         VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        fsrMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        depthDescriptorTables_[i]->bindImage(inputImages_[i][1], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        depthDescriptorTables_[i]->bindImage(deviceDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        depthDescriptorTables_[i]->bindImage(inputImages_[i][2], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        depthDescriptorTables_[i]->bindImage(fsrMotionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);

        guideDescriptorTables_[i]->bindImage(inputImages_[i][3], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        guideDescriptorTables_[i]->bindImage(outputImages_[i][1], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        guideDescriptorTables_[i]->bindImage(inputImages_[i][2], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        guideDescriptorTables_[i]->bindImage(outputImages_[i][2], VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        guideDescriptorTables_[i]->bindImage(inputImages_[i][4], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        guideDescriptorTables_[i]->bindImage(outputImages_[i][3], VK_IMAGE_LAYOUT_GENERAL, 0, 6);
    }
}

void FSRUpscalerModule::initPipeline() {
    auto fw = framework_.lock();
    auto shader = vk::Shader::create(
        fw->device(), (Renderer::folderPath / "shaders/world/upscaler/linear_to_device_depth_comp.spv").string());
    auto firstHitDepthShader = vk::Shader::create(
        fw->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_first_hit_depth_comp.spv").string());
    auto motionShader = vk::Shader::create(
        fw->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_motion_vector_comp.spv").string());
    auto normalRoughnessShader = vk::Shader::create(
        fw->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_normal_roughness_comp.spv").string());

    auto reactiveShader = vk::Shader::create(
        fw->device(), (Renderer::folderPath / "shaders/world/upscaler/fsr_reactive_mask_comp.spv").string());
    reactivePipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(reactiveShader)
                            .definePipelineLayout(reactiveDescriptorTables_[0])
                            .build(fw->device());

    depthConversionPipeline_ = vk::ComputePipelineBuilder{}
                                   .defineShader(shader)
                                   .definePipelineLayout(depthDescriptorTables_[0])
                                   .build(fw->device());
    firstHitDepthUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                        .defineShader(firstHitDepthShader)
                                        .definePipelineLayout(guideDescriptorTables_[0])
                                        .build(fw->device());
    motionUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                 .defineShader(motionShader)
                                 .definePipelineLayout(guideDescriptorTables_[0])
                                 .build(fw->device());
    normalRoughnessUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                          .defineShader(normalRoughnessShader)
                                          .definePipelineLayout(guideDescriptorTables_[0])
                                          .build(fw->device());
}

void FSRUpscalerModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) { return value == "render_pipeline.true"; };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.fsr_upscaler.attribute.enable") {
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
        } else if (key == "render_pipeline.module.fsr_upscaler.attribute.sharpness") {
            sharpness_ = std::stof(value);
        } else if (key == "render_pipeline.module.fsr_upscaler.attribute.pre_exposure") {
            preExposure_ = std::stof(value);
        }
    }
#ifndef MCVR_ENABLE_FFX_UPSCALER
    // Keep the existing native-resolution blit path when FFX is not compiled.
    fsr3Enabled_ = false;
    qualityMode_ = QualityMode::NativeAA;
    renderWidth_ = displayWidth_;
    renderHeight_ = displayHeight_;
#endif
}

std::vector<std::shared_ptr<WorldModuleContext>> &FSRUpscalerModule::contexts() {
    return reinterpret_cast<std::vector<std::shared_ptr<WorldModuleContext>> &>(contexts_);
}

void FSRUpscalerModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                    std::shared_ptr<vk::DeviceLocalImage> image,
                                    int index) {}

void FSRUpscalerModule::preClose() {
    if (auto fw = framework_.lock()) { fw->waitRenderQueueIdle(); }
    if (fsr3_) {
        fsr3_->destroy();
        fsr3_.reset();
    }
    initialized_ = false;
}

void FSRUpscalerModule::getRenderResolution(uint32_t displayWidth,
                                            uint32_t displayHeight,
                                            QualityMode mode,
                                            uint32_t *outRenderWidth,
                                            uint32_t *outRenderHeight) {
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

FSRUpscalerModuleContext::FSRUpscalerModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                   std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                   std::shared_ptr<FSRUpscalerModule> upscalerModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext), upscalerModule_(upscalerModule) {}

bool FSRUpscalerModuleContext::populateFrameGenerationInputs(mcvr::FrameGenerationInputs &inputs) {
    auto module = upscalerModule_.lock();
    if (module == nullptr || !module->initialized_ || deviceDepthImage == nullptr || fsrMotionVectorImage == nullptr) {
        return false;
    }

    inputs.depth = deviceDepthImage;
    inputs.motionVector = fsrMotionVectorImage;
    inputs.renderWidth = module->renderWidth_;
    inputs.renderHeight = module->renderHeight_;
    inputs.motionVectorScale = glm::vec2(1.0f);
    inputs.depthInverted = false;
    inputs.cameraMotionIncluded = true;
    return true;
}

bool FSRUpscalerModuleContext::checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir) {
    auto module = upscalerModule_.lock();
    if (module->firstFrame_ || !Renderer::instance().buffers()->hasWorldHistory()) {
        module->firstFrame_ = false;
        module->frameClock_.reset();
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

void FSRUpscalerModuleContext::render() {
    MCVR_PROFILE_SCOPE("native.FSRUpscalerModuleContext.render");
    auto module = upscalerModule_.lock();
    if (!module) return;

    auto fwContext = frameworkContext.lock();
    auto worldCommandBuffer = fwContext->worldCommandBuffer;
    auto mainQueueIndex = fwContext->framework.lock()->physicalDevice()->mainQueueIndex();

    auto worldUBO =
        static_cast<vk::Data::WorldUBO *>(Renderer::instance().buffers()->worldUniformBuffer()->mappedPtr());
    if (!worldUBO) { return; }

    auto dispatchUpscaledFirstHitDepth = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                  .oldLayout = inputFirstHitDepthImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = inputFirstHitDepthImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                  .oldLayout = upscaledFirstHitDepthImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = upscaledFirstHitDepthImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        inputFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;


        MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->displayWidth_) / static_cast<float>(std::max(1u, module->renderWidth_)),
            static_cast<float>(module->displayHeight_) / static_cast<float>(std::max(1u, module->renderHeight_)),
            module->renderWidth_,
            module->renderHeight_,
            module->displayWidth_,
            module->displayHeight_,
            worldUBO->cameraJitter.x,
            worldUBO->cameraJitter.y,
        };

        worldCommandBuffer->bindDescriptorTable(guideDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->firstHitDepthUpscalePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), guideDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->displayWidth_ + 15) / 16,
                      (module->displayHeight_ + 15) / 16, 1);
    };

    auto dispatchUpscaledMotion = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                  .oldLayout = inputMotionVectorImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = inputMotionVectorImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                  .oldLayout = upscaledMotionVectorImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = upscaledMotionVectorImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        inputMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        upscaledMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;


        MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->displayWidth_) / static_cast<float>(std::max(1u, module->renderWidth_)),
            static_cast<float>(module->displayHeight_) / static_cast<float>(std::max(1u, module->renderHeight_)),
            module->renderWidth_,
            module->renderHeight_,
            module->displayWidth_,
            module->displayHeight_,
            worldUBO->cameraJitter.x,
            worldUBO->cameraJitter.y,
        };

        worldCommandBuffer->bindDescriptorTable(guideDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->motionUpscalePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), guideDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->displayWidth_ + 15) / 16,
                      (module->displayHeight_ + 15) / 16, 1);
    };

    auto dispatchUpscaledNormalRoughness = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                  .oldLayout = inputNormalRoughnessImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = inputNormalRoughnessImage,
                  .subresourceRange = vk::wholeColorSubresourceRange},
                 {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                  .oldLayout = upscaledNormalRoughnessImage->imageLayout(),
                  .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = upscaledNormalRoughnessImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});

        inputNormalRoughnessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        upscaledNormalRoughnessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;


        MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->displayWidth_) / static_cast<float>(std::max(1u, module->renderWidth_)),
            static_cast<float>(module->displayHeight_) / static_cast<float>(std::max(1u, module->renderHeight_)),
            module->renderWidth_,
            module->renderHeight_,
            module->displayWidth_,
            module->displayHeight_,
            worldUBO->cameraJitter.x,
            worldUBO->cameraJitter.y,
        };

        worldCommandBuffer->bindDescriptorTable(guideDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->normalRoughnessUpscalePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), guideDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->displayWidth_ + 15) / 16,
                      (module->displayHeight_ + 15) / 16, 1);
    };

    auto fallbackBlit = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {},
            {{.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
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
              .subresourceRange = vk::wholeColorSubresourceRange}});

        inputColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        outputImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

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
        dispatchUpscaledFirstHitDepth();
        dispatchUpscaledMotion();
        dispatchUpscaledNormalRoughness();
    };

    if (!module->fsr3Enabled_) {
        fallbackBlit();
        return;
    }

    if (!module->initialized_ || !module->fsr3_) return;


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
              .oldLayout = fsrMotionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = fsrMotionVectorImage,
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
    fsrMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    struct PushConstants {
        float cameraNear;
        float cameraFar;
        uint32_t width;
        uint32_t height;
        float jitterX;
        float jitterY;
    } pushConstants{0.1f,
                    10000.0f,
                    module->renderWidth_,
                    module->renderHeight_,
                    worldUBO->cameraJitter.x,
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
        {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                               VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
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
              .oldLayout = fsrMotionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = fsrMotionVectorImage,
              .subresourceRange = vk::wholeColorSubresourceRange}});

    inputColorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    deviceDepthImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    fsrMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    mcvr::UpscalerInput input{};
    input.commandBuffer = worldCommandBuffer->vkCommandBuffer();
    input.colorImage = inputColorImage->vkImage();
    input.colorImageView = inputColorImage->vkImageView();
    input.colorLayout = inputColorImage->imageLayout();
    input.colorFormat = inputColorImage->vkFormat();
    input.depthImage = deviceDepthImage->vkImage();
    input.depthImageView = deviceDepthImage->vkImageView();
    input.depthFormat = deviceDepthImage->vkFormat();
    input.motionVectorImage = fsrMotionVectorImage->vkImage();
    input.motionVectorImageView = fsrMotionVectorImage->vkImageView();
    input.motionVectorFormat = fsrMotionVectorImage->vkFormat();
    input.outputImage = outputImage->vkImage();
    input.outputImageView = outputImage->vkImageView();
    input.outputLayout = VK_IMAGE_LAYOUT_GENERAL;
    input.outputFormat = outputImage->vkFormat();
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
    input.cameraFovVertical = 2.0f * std::atan(1.0f / std::abs(worldUBO->cameraProjMat[1][1]));
    input.preExposure = module->preExposure_;
    input.reset = checkCameraReset(
        glm::vec3(worldUBO->cameraPos),
        glm::vec3(worldUBO->cameraViewMat[0][2], worldUBO->cameraViewMat[1][2], worldUBO->cameraViewMat[2][2]));
    prepareReactiveMask(input.reset);
    auto reactiveImage = module->reactiveImages_[fwContext->frameIndex];
    input.reactiveImage = reactiveImage->vkImage();
    input.reactiveImageView = reactiveImage->vkImageView();
    input.frameTimeDelta = module->frameClock_.advance();
    input.enableSharpening = true;
    input.sharpness = module->sharpness_;

    module->fsr3_->dispatch(input);
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    dispatchUpscaledFirstHitDepth();
    dispatchUpscaledMotion();
    dispatchUpscaledNormalRoughness();
}

void FSRUpscalerModuleContext::prepareReactiveMask(bool reset) {
    auto module = upscalerModule_.lock();
    auto context = frameworkContext.lock();
    auto command = context->worldCommandBuffer;
    auto queueIndex = context->framework.lock()->physicalDevice()->mainQueueIndex();
    auto previousDepth = module->historyDepthImages_[module->historyDepthIndex_];
    auto historyDepth = module->historyDepthImages_[module->historyDepthIndex_ ^ 1u];
    auto reactiveImage = module->reactiveImages_[context->frameIndex];
    auto table = module->reactiveDescriptorTables_[context->frameIndex];
    auto transition = [&](const std::shared_ptr<vk::DeviceLocalImage> &image, VkAccessFlags2 access,
                          VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
        const bool undefined = image->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED;
        command->barriersBufferImage({}, {{
            .srcStageMask = undefined ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = undefined ? 0 : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = access,
            .oldLayout = image->imageLayout(),
            .newLayout = layout,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .image = image,
            .subresourceRange = vk::wholeColorSubresourceRange,
        }});
        image->imageLayout() = layout;
    };
    transition(inputFirstHitDepthImage, VK_ACCESS_2_SHADER_READ_BIT);
    transition(inputMotionVectorImage, VK_ACCESS_2_SHADER_READ_BIT);
    transition(previousDepth, VK_ACCESS_2_SHADER_READ_BIT);
    transition(historyDepth, VK_ACCESS_2_SHADER_WRITE_BIT);
    transition(reactiveImage, VK_ACCESS_2_SHADER_WRITE_BIT);
    table->bindImage(inputFirstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
    table->bindImage(inputMotionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    table->bindImage(previousDepth, VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    table->bindImage(reactiveImage, VK_IMAGE_LAYOUT_GENERAL, 0, 3);
    table->bindImage(historyDepth, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
    table->bindBuffer(Renderer::instance().buffers()->worldUniformBuffer(), 1, 0);
    table->bindBuffer(Renderer::instance().buffers()->lastWorldUniformBuffer(), 1, 1);
    uint32_t hasHistory = module->hasDepthHistory_ && !reset ? 1u : 0u;
    command->bindDescriptorTable(table, VK_PIPELINE_BIND_POINT_COMPUTE)->bindComputePipeline(module->reactivePipeline_);
    vkCmdPushConstants(command->vkCommandBuffer(), table->vkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(hasHistory), &hasHistory);
    vkCmdDispatch(command->vkCommandBuffer(), (module->renderWidth_ + 15) / 16, (module->renderHeight_ + 15) / 16, 1);
    transition(reactiveImage, VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    module->historyDepthIndex_ ^= 1u;
    module->hasDepthHistory_ = true;
}
