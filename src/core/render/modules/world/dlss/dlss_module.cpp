#include "core/render/modules/world/dlss/dlss_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>

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
    upscaledMotionVectorImages_.resize(size);
    upscaledNormalRoughnessImages_.resize(size);
    motionDescriptorTables_.resize(size);

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
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    } else {
        if (images[1]->width() != outputWidth_ || images[1]->height() != outputHeight_) return false;
        upscaledFirstHitDepthImages_[frameIndex] = images[1];
    }

    if (images[2] == nullptr) {
        upscaledMotionVectorImages_[frameIndex] = images[2] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, outputWidth_, outputHeight_, 1, formats[2],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[2]->width() != outputWidth_ || images[2]->height() != outputHeight_) return false;
        upscaledMotionVectorImages_[frameIndex] = images[2];
    }

    if (images[3] == nullptr) {
        upscaledNormalRoughnessImages_[frameIndex] = images[3] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, outputWidth_, outputHeight_, 1, formats[3],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[3]->width() != outputWidth_ || images[3]->height() != outputHeight_) return false;
        upscaledNormalRoughnessImages_[frameIndex] = images[3];
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

    auto firstHitDepthShader = vk::Shader::create(
        framework->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_first_hit_depth_comp.spv").string());
    auto motionShader = vk::Shader::create(
        framework->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_motion_vector_comp.spv").string());
    auto normalRoughnessShader = vk::Shader::create(
        framework->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_normal_roughness_comp.spv").string());
    for (uint32_t i = 0; i < size; i++) {
        motionDescriptorTables_[i] = vk::DescriptorTableBuilder{}
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
                                             .binding = 4,
                                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                             .descriptorCount = 1,
                                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                         })
                                         .defineDescriptorLayoutSetBinding({
                                             .binding = 5,
                                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                             .descriptorCount = 1,
                                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                         })
                                         .defineDescriptorLayoutSetBinding({
                                             .binding = 6,
                                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                             .descriptorCount = 1,
                                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                         })
                                         .endDescriptorLayoutSetBinding()
                                         .endDescriptorLayoutSet()
                                         .definePushConstant({
                                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                             .offset = 0,
                                             .size = sizeof(MotionUpscalePushConstants),
                                         })
                                         .build(framework->device());
        motionDescriptorTables_[i]->bindImage(motionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        motionDescriptorTables_[i]->bindImage(upscaledMotionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        motionDescriptorTables_[i]->bindImage(normalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        motionDescriptorTables_[i]->bindImage(upscaledNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 6);
    }
    firstHitDepthUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                        .defineShader(firstHitDepthShader)
                                        .definePipelineLayout(motionDescriptorTables_[0])
                                        .build(framework->device());
    motionUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                 .defineShader(motionShader)
                                 .definePipelineLayout(motionDescriptorTables_[0])
                                 .build(framework->device());
    normalRoughnessUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                          .defineShader(normalRoughnessShader)
                                          .definePipelineLayout(motionDescriptorTables_[0])
                                          .build(framework->device());

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] =
            DLSSModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &DLSSModule::contexts() {
    return contexts_;
}

void DLSSModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                             std::shared_ptr<vk::DeviceLocalImage> image,
                             int index) {}

void DLSSModule::preClose() {
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
      upscaledFirstHitDepthImage(dlssModule->upscaledFirstHitDepthImages_[frameworkContext->frameIndex]),
      upscaledMotionVectorImage(dlssModule->upscaledMotionVectorImages_[frameworkContext->frameIndex]),
      upscaledNormalRoughnessImage(dlssModule->upscaledNormalRoughnessImages_[frameworkContext->frameIndex]),
      motionDescriptorTable(dlssModule->motionDescriptorTables_[frameworkContext->frameIndex]) {}

void DLSSModuleContext::render() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = dLSSModule.lock();

    auto dispatchUpscaledFirstHitDepth = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {},
            {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = firstHitDepthImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = firstHitDepthImage,
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

        firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        motionDescriptorTable->bindImage(firstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        motionDescriptorTable->bindImage(upscaledFirstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);

        MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->outputWidth_) / static_cast<float>(std::max(1u, module->inputWidth_)),
            static_cast<float>(module->outputHeight_) / static_cast<float>(std::max(1u, module->inputHeight_)),
            module->inputWidth_,
            module->inputHeight_,
            module->outputWidth_,
            module->outputHeight_,
        };

        worldCommandBuffer->bindDescriptorTable(motionDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->firstHitDepthUpscalePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), motionDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->outputWidth_ + 15) / 16,
                      (module->outputHeight_ + 15) / 16, 1);
    };

    auto dispatchUpscaledMotion = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {},
            {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = motionVectorImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = motionVectorImage,
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

        motionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        upscaledMotionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        motionDescriptorTable->bindImage(motionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        motionDescriptorTable->bindImage(upscaledMotionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 4);

        MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->outputWidth_) / static_cast<float>(std::max(1u, module->inputWidth_)),
            static_cast<float>(module->outputHeight_) / static_cast<float>(std::max(1u, module->inputHeight_)),
            module->inputWidth_,
            module->inputHeight_,
            module->outputWidth_,
            module->outputHeight_,
        };

        worldCommandBuffer->bindDescriptorTable(motionDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->motionUpscalePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), motionDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->outputWidth_ + 15) / 16,
                      (module->outputHeight_ + 15) / 16, 1);
    };

    auto dispatchUpscaledNormalRoughness = [&]() {
        worldCommandBuffer->barriersBufferImage(
            {},
            {{.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
              .oldLayout = normalRoughnessImage->imageLayout(),
              .newLayout = VK_IMAGE_LAYOUT_GENERAL,
              .srcQueueFamilyIndex = mainQueueIndex,
              .dstQueueFamilyIndex = mainQueueIndex,
              .image = normalRoughnessImage,
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

        normalRoughnessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        upscaledNormalRoughnessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        motionDescriptorTable->bindImage(normalRoughnessImage, VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        motionDescriptorTable->bindImage(upscaledNormalRoughnessImage, VK_IMAGE_LAYOUT_GENERAL, 0, 6);

        MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->outputWidth_) / static_cast<float>(std::max(1u, module->inputWidth_)),
            static_cast<float>(module->outputHeight_) / static_cast<float>(std::max(1u, module->inputHeight_)),
            module->inputWidth_,
            module->inputHeight_,
            module->outputWidth_,
            module->outputHeight_,
        };

        worldCommandBuffer->bindDescriptorTable(motionDescriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->normalRoughnessUpscalePipeline_);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), motionDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->outputWidth_ + 15) / 16,
                      (module->outputHeight_ + 15) / 16, 1);
    };

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

    dispatchUpscaledFirstHitDepth();
    dispatchUpscaledMotion();
    dispatchUpscaledNormalRoughness();
}
