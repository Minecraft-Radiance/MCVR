#include "core/render/modules/world/temporal_accumulation/temporal_accumulation_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

TemporalAccumulationModule::TemporalAccumulationModule() {}

void TemporalAccumulationModule::init(std::shared_ptr<Framework> framework,
                                      std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrNoisyImages_.resize(size);
    motionVectorImages_.resize(size);
    normalRoughnessImages_.resize(size);
    accumulatedRadianceOutImages_.resize(size);
    accumulatedNormalOutImages_.resize(size);
}

bool TemporalAccumulationModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                                        std::vector<VkFormat> &formats,
                                                        uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        hdrNoisyImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, eyeCount_, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (eyeCount_ > 1) images[0]->createPerLayerViews();
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        hdrNoisyImages_[frameIndex] = images[0];
    }

    if (images[1] == nullptr) {
        motionVectorImages_[frameIndex] = images[1] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, eyeCount_, formats[1],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (eyeCount_ > 1) images[1]->createPerLayerViews();
    } else {
        if (images[1]->width() != width_ || images[1]->height() != height_) return false;
        motionVectorImages_[frameIndex] = images[1];
    }

    if (images[2] == nullptr) {
        normalRoughnessImages_[frameIndex] = images[2] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, eyeCount_, formats[2],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (eyeCount_ > 1) images[2]->createPerLayerViews();
    } else {
        if (images[2]->width() != width_ || images[2]->height() != height_) return false;
        normalRoughnessImages_[frameIndex] = images[2];
    }

    return true;
}

bool TemporalAccumulationModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                                         std::vector<VkFormat> &formats,
                                                         uint32_t frameIndex) {
    if (images.size() != 1 || images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    accumulatedRadianceOutImages_[frameIndex] = images[0];

    return true;
}

void TemporalAccumulationModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {}

void TemporalAccumulationModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    initDescriptorTables();
    initImages();
    initRenderPass();
    initFrameBuffers();
    initPipeline();

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] = TemporalAccumulationModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                                 shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &TemporalAccumulationModule::contexts() {
    return contexts_;
}

void TemporalAccumulationModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                             std::shared_ptr<vk::DeviceLocalImage> image,
                                             int index) {}

void TemporalAccumulationModule::preClose() {}

void TemporalAccumulationModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;

    descriptorTables_.resize(totalCtx);

    for (uint32_t i = 0; i < totalCtx; i++) {
        descriptorTables_[i] = vk::DescriptorTableBuilder{}
                                   .beginDescriptorLayoutSet() // set 0
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 3,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 4,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant({
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                       .offset = 0,
                                       .size = sizeof(TemporalAccumulationPushConstant),
                                   })
                                   .build(framework->device());
    }

    sampler_ = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                   VK_SAMPLER_ADDRESS_MODE_REPEAT);
}

void TemporalAccumulationModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;

    accumulatedRadianceImage_ = vk::DeviceLocalImage::create(
        framework->device(), framework->vma(), false, hdrNoisyImages_[0]->width(), hdrNoisyImages_[0]->height(),
        eyeCount_, hdrNoisyImages_[0]->vkFormat(),
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (eyeCount_ > 1) accumulatedRadianceImage_->createPerLayerViews();

    accumulatedNormalImage_ = vk::DeviceLocalImage::create(
        framework->device(), framework->vma(), false, hdrNoisyImages_[0]->width(), hdrNoisyImages_[0]->height(),
        eyeCount_, normalRoughnessImages_[0]->vkFormat(),
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (eyeCount_ > 1) accumulatedNormalImage_->createPerLayerViews();

    accumulatedNormalOutImages_.resize(totalCtx);

    for (uint32_t i = 0; i < totalCtx; i++) {
        uint32_t frameIdx = i / eyeCount_;
        uint32_t eyeIdx = i % eyeCount_;
        uint32_t viewIdx = (eyeCount_ > 1) ? (1 + eyeIdx) : 0;

        descriptorTables_[i]->bindSamplerImageForShader(sampler_, hdrNoisyImages_[frameIdx], 0, 0, viewIdx);
        descriptorTables_[i]->bindSamplerImageForShader(sampler_, accumulatedRadianceImage_, 0, 1, viewIdx);
        descriptorTables_[i]->bindSamplerImageForShader(sampler_, motionVectorImages_[frameIdx], 0, 2, viewIdx);
        descriptorTables_[i]->bindSamplerImageForShader(sampler_, normalRoughnessImages_[frameIdx], 0, 3, viewIdx);
        descriptorTables_[i]->bindSamplerImageForShader(sampler_, accumulatedNormalImage_, 0, 4, viewIdx);

        accumulatedNormalOutImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, hdrNoisyImages_[0]->width(), hdrNoisyImages_[0]->height(), 1,
            normalRoughnessImages_[0]->vkFormat(),
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }
}

void TemporalAccumulationModule::initRenderPass() {
    renderPass_ = vk::RenderPassBuilder{}
                      .beginAttachmentDescription()
                      .defineAttachmentDescription({
                          // color
                          .format = accumulatedRadianceOutImages_[0]->vkFormat(),
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                          .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                          .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      })
                      .defineAttachmentDescription({
                          // color
                          .format = accumulatedNormalOutImages_[0]->vkFormat(),
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                          .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                          .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
                      .build(framework_.lock()->device());
}

void TemporalAccumulationModule::initFrameBuffers() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;

    framebuffers_.resize(totalCtx);

    for (uint32_t i = 0; i < totalCtx; i++) {
        uint32_t frameIdx = i / eyeCount_;
        uint32_t eyeIdx = i % eyeCount_;
        int viewIdx = (eyeCount_ > 1) ? static_cast<int>(1 + eyeIdx) : 0;

        framebuffers_[i] = vk::FramebufferBuilder{}
                               .beginAttachment()
                               .defineAttachment(accumulatedRadianceOutImages_[frameIdx], viewIdx)
                               .defineAttachment(accumulatedNormalOutImages_[i])
                               .endAttachment()
                               .build(framework->device(), renderPass_);
    }
}

void TemporalAccumulationModule::initPipeline() {
    auto framework = framework_.lock();
    uint32_t targetWidth = framework->swapchain()->vkExtent().width;
    uint32_t targetHeight = framework->swapchain()->vkExtent().height;
    if (!accumulatedRadianceOutImages_.empty() && accumulatedRadianceOutImages_[0] != nullptr) {
        targetWidth = accumulatedRadianceOutImages_[0]->width();
        targetHeight = accumulatedRadianceOutImages_[0]->height();
    }
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    vertShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/temporal_accumulation/tmp_acc_vert.spv").string());
    fragShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/temporal_accumulation/tmp_acc_frag.spv").string());

    pipeline_ = vk::GraphicsPipelineBuilder{}
                    .defineRenderPass(renderPass_, 0)
                    .beginShaderStage()
                    .defineShaderStage(vertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                    .defineShaderStage(fragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                    .endShaderStage()
                    .defineVertexInputState<void>()
                    .defineViewportScissorState({
                        .viewport =
                            {
                                .x = 0,
                                .y = 0,
                                .width = static_cast<float>(targetWidth),
                                .height = static_cast<float>(targetHeight),
                                .minDepth = 0.0,
                                .maxDepth = 1.0,
                            },
                        .scissor =
                            {
                                .offset = {.x = 0, .y = 0},
                                .extent = {targetWidth, targetHeight},
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
                    .defineDefaultColorBlendAttachmentState() // color
                    .endColorBlendAttachmentState()
                    .definePipelineLayout(descriptorTables_[0])
                    .build(framework->device());
}

TemporalAccumulationModuleContext::TemporalAccumulationModuleContext(
    std::shared_ptr<FrameworkContext> frameworkContext,
    std::shared_ptr<WorldPipelineContext> worldPipelineContext,
    std::shared_ptr<TemporalAccumulationModule> temporalAccumulationModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      temporalAccumulationModule(temporalAccumulationModule),
      hdrNoisyImage(temporalAccumulationModule->hdrNoisyImages_[frameworkContext->frameIndex]),
      motionVectorImage(temporalAccumulationModule->motionVectorImages_[frameworkContext->frameIndex]),
      descriptorTable(temporalAccumulationModule->descriptorTables_[frameworkContext->frameIndex * temporalAccumulationModule->eyeCount()]),
      framebuffer(temporalAccumulationModule->framebuffers_[frameworkContext->frameIndex * temporalAccumulationModule->eyeCount()]),
      accumulatedRadianceImage(temporalAccumulationModule->accumulatedRadianceImage_),
      accumulatedNormalImage(temporalAccumulationModule->accumulatedNormalImage_),
      accumulatedNormalOutImage(temporalAccumulationModule->accumulatedNormalOutImages_[frameworkContext->frameIndex * temporalAccumulationModule->eyeCount()]),
      accumulatedRadianceOutImage(
          temporalAccumulationModule->accumulatedRadianceOutImages_[frameworkContext->frameIndex]) {}

void TemporalAccumulationModuleContext::render() {
    renderEye(0);
}

void TemporalAccumulationModuleContext::renderEye(uint32_t eyeIndex) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = temporalAccumulationModule.lock();
    uint32_t dtIdx = context->frameIndex * module->eyeCount() + eyeIndex;

    auto eyeDT = module->descriptorTables_[dtIdx];
    auto eyeFB = module->framebuffers_[dtIdx];
    auto eyeNormalOut = module->accumulatedNormalOutImages_[dtIdx];

    TemporalAccumulationPushConstant pc{};
    pc.alpha = module->alpha_;
    pc.threshold = module->threshould_;

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), eyeDT->vkPipelineLayout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(TemporalAccumulationPushConstant), &pc);

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = hdrNoisyImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = hdrNoisyImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = motionVectorImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = motionVectorImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = accumulatedRadianceImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = accumulatedRadianceImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = accumulatedRadianceOutImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = accumulatedRadianceOutImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             }});
    hdrNoisyImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    motionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    accumulatedRadianceImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    accumulatedRadianceOutImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->renderPass_,
        .framebuffer = eyeFB,
        .renderAreaExtent = {accumulatedRadianceOutImage->width(), accumulatedRadianceOutImage->height()},
        .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
    });
    accumulatedRadianceOutImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    worldCommandBuffer->bindGraphicsPipeline(module->pipeline_)
        ->bindDescriptorTable(eyeDT, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->draw(3, 1)
        ->endRenderPass();
    accumulatedRadianceOutImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = accumulatedRadianceOutImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = accumulatedRadianceOutImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = eyeNormalOut->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = eyeNormalOut,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = accumulatedRadianceImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = accumulatedRadianceImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = accumulatedNormalImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = accumulatedNormalImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             }});
    accumulatedRadianceOutImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    eyeNormalOut->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    accumulatedRadianceImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    accumulatedNormalImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // blit accumulatedRadianceOutImage layer[eyeIndex] → accumulatedRadianceImage layer[eyeIndex]
    {
        VkImageBlit imageBlit{};
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.mipLevel = 0;
        imageBlit.srcSubresource.baseArrayLayer = eyeIndex;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcOffsets[0] = {0, 0, 0};
        imageBlit.srcOffsets[1] = {static_cast<int>(accumulatedRadianceOutImage->width()),
                                   static_cast<int>(accumulatedRadianceOutImage->height()), 1};
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.mipLevel = 0;
        imageBlit.dstSubresource.baseArrayLayer = eyeIndex;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstOffsets[0] = {0, 0, 0};
        imageBlit.dstOffsets[1] = {static_cast<int>(accumulatedRadianceImage->width()),
                                   static_cast<int>(accumulatedRadianceImage->height()), 1};

        vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), accumulatedRadianceOutImage->vkImage(),
                       accumulatedRadianceOutImage->imageLayout(), accumulatedRadianceImage->vkImage(),
                       accumulatedRadianceImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
    }

    // blit eyeNormalOut (single-layer) → accumulatedNormalImage layer[eyeIndex]
    {
        VkImageBlit imageBlit{};
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.mipLevel = 0;
        imageBlit.srcSubresource.baseArrayLayer = 0;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcOffsets[0] = {0, 0, 0};
        imageBlit.srcOffsets[1] = {static_cast<int>(eyeNormalOut->width()),
                                   static_cast<int>(eyeNormalOut->height()), 1};
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.mipLevel = 0;
        imageBlit.dstSubresource.baseArrayLayer = eyeIndex;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstOffsets[0] = {0, 0, 0};
        imageBlit.dstOffsets[1] = {static_cast<int>(accumulatedNormalImage->width()),
                                   static_cast<int>(accumulatedNormalImage->height()), 1};

        vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), eyeNormalOut->vkImage(),
                       eyeNormalOut->imageLayout(), accumulatedNormalImage->vkImage(),
                       accumulatedNormalImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
    }
}