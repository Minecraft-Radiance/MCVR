#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

ToneMappingModule::ToneMappingModule() {}

void ToneMappingModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrImages_.resize(size);
    ldrImages_.resize(size);
}

bool ToneMappingModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.size() == 0) return false;

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        hdrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, eyeCount_, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (eyeCount_ > 1) images[0]->createPerLayerViews();
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        hdrImages_[frameIndex] = images[0];
    }

    return true;
}

bool ToneMappingModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                                std::vector<VkFormat> &formats,
                                                uint32_t frameIndex) {
    if (images.size() == 0 || images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    ldrImages_[frameIndex] = images[0];

    return true;
}

void ToneMappingModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.middle_grey") {
            middleGrey_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.exposure_up_speed") {
            speedUp_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.exposure_down_speed") {
            speedDown_ = std::stof(attributeKVs[2 * i + 1]);
        }
    }
}

void ToneMappingModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    initDescriptorTables();
    initImages();
    initBuffers();
    initRenderPass();
    initFrameBuffers();
    initPipeline();

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] = ToneMappingModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                        shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &ToneMappingModule::contexts() {
    return contexts_;
}

void ToneMappingModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                    std::shared_ptr<vk::DeviceLocalImage> image,
                                    int index) {}

void ToneMappingModule::preClose() {}

void ToneMappingModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;

    descriptorTables_.resize(totalCtx);
    samplers_.resize(totalCtx);

    for (int i = 0; i < totalCtx; i++) {
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
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant(VkPushConstantRange{
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                       .offset = 0,
                                       .size = sizeof(ToneMappingModulePushConstant),
                                   })
                                   .build(framework->device());

        samplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }
}

void ToneMappingModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    for (uint32_t i = 0; i < size; i++) {
        for (uint32_t e = 0; e < eyeCount_; e++) {
            uint32_t dtIdx = i * eyeCount_ + e;
            uint32_t viewIdx = eyeCount_ > 1 ? 1 + e : 0;
            descriptorTables_[dtIdx]->bindSamplerImageForShader(samplers_[dtIdx], hdrImages_[i], 0, 0, viewIdx);
        }
    }
}

void ToneMappingModule::initBuffers() {
    auto framework = framework_.lock();
    auto vma = framework->vma();
    auto device = framework->device();
    uint32_t size = framework->swapchain()->imageCount();

    histBuffers_.resize(size);

    exposureData_ =
        vk::DeviceLocalBuffer::create(vma, device, sizeof(ToneMappingModuleExposureData),
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    for (uint32_t i = 0; i < size; i++) {
        histBuffers_[i] =
            vk::DeviceLocalBuffer::create(vma, device, histSize * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (uint32_t e = 0; e < eyeCount_; e++) {
            uint32_t dtIdx = i * eyeCount_ + e;
            descriptorTables_[dtIdx]->bindBuffer(histBuffers_[i], 0, 1);
            descriptorTables_[dtIdx]->bindBuffer(exposureData_, 0, 2);
        }
    }
}

void ToneMappingModule::initRenderPass() {
    renderPass_ = vk::RenderPassBuilder{}
                      .beginAttachmentDescription()
                      .defineAttachmentDescription({
                          // color
                          .format = ldrImages_[0]->vkFormat(),
                      .samples = VK_SAMPLE_COUNT_1_BIT,
                      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#ifdef USE_AMD
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
                      .build(framework_.lock()->device());
}

void ToneMappingModule::initFrameBuffers() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;

    framebuffers_.resize(totalCtx);

    for (uint32_t i = 0; i < size; i++) {
        for (uint32_t e = 0; e < eyeCount_; e++) {
            uint32_t idx = i * eyeCount_ + e;
            uint32_t viewIdx = eyeCount_ > 1 ? 1 + e : 0;
            framebuffers_[idx] = vk::FramebufferBuilder{}
                                     .beginAttachment()
                                     .defineAttachment(ldrImages_[i], viewIdx)
                                     .endAttachment()
                                     .build(framework->device(), renderPass_);
        }
    }
}

void ToneMappingModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    uint32_t targetWidth = framework->swapchain()->vkExtent().width;
    uint32_t targetHeight = framework->swapchain()->vkExtent().height;
    if (!hdrImages_.empty() && hdrImages_[0] != nullptr) {
        targetWidth = hdrImages_[0]->width();
        targetHeight = hdrImages_[0]->height();
    }
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    histShader_ = vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/hist_comp.spv").string());
    histPipeline_ =
        vk::ComputePipelineBuilder{}.defineShader(histShader_).definePipelineLayout(descriptorTables_[0]).build(device);

    exposureShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/exposure_comp.spv").string());
    exposurePipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(exposureShader_)
                            .definePipelineLayout(descriptorTables_[0])
                            .build(device);

    vertShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/tone_mapping_vert.spv").string());
    fragShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/tone_mapping_frag.spv").string());

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
                    .build(device);
}

ToneMappingModuleContext::ToneMappingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                   std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                   std::shared_ptr<ToneMappingModule> toneMappingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      toneMappingModule(toneMappingModule),
      eyeCount_(toneMappingModule->eyeCount()),
      hdrImage(toneMappingModule->hdrImages_[frameworkContext->frameIndex]),
      histBuffer(toneMappingModule->histBuffers_[frameworkContext->frameIndex]),
      ldrImage(toneMappingModule->ldrImages_[frameworkContext->frameIndex]) {
    uint32_t base = frameworkContext->frameIndex * eyeCount_;
    for (uint32_t e = 0; e < eyeCount_; e++) {
        descriptorTables.push_back(toneMappingModule->descriptorTables_[base + e]);
        framebuffers.push_back(toneMappingModule->framebuffers_[base + e]);
    }
}

void ToneMappingModuleContext::render() {
    render3D(1);
}

void ToneMappingModuleContext::render3D(uint32_t eyeCount) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = toneMappingModule.lock();

    auto chooseSrc = [](VkImageLayout oldLayout,
                        VkPipelineStageFlags2 fallbackStage,
                        VkAccessFlags2 fallbackAccess,
                        VkPipelineStageFlags2 &outStage,
                        VkAccessFlags2 &outAccess) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            outStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            outAccess = 0;
        } else {
            outStage = fallbackStage;
            outAccess = fallbackAccess;
        }
    };

    VkPipelineStageFlags2 hdrSrcStage = 0;
    VkAccessFlags2 hdrSrcAccess = 0;
    chooseSrc(hdrImage->imageLayout(),
              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              hdrSrcStage, hdrSrcAccess);

    VkPipelineStageFlags2 ldrSrcStage = 0;
    VkAccessFlags2 ldrSrcAccess = 0;
    chooseSrc(ldrImage->imageLayout(),
              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              ldrSrcStage, ldrSrcAccess);

    worldCommandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = histBuffer,
        }},
        {{
             .srcStageMask = hdrSrcStage,
             .srcAccessMask = hdrSrcAccess,
             .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
             .oldLayout = hdrImage->imageLayout(),
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .srcQueueFamilyIndex = mainQueueIndex,
             .dstQueueFamilyIndex = mainQueueIndex,
             .image = hdrImage,
             .subresourceRange = vk::wholeColorSubresourceRange,
         },
         {
             .srcStageMask = ldrSrcStage,
             .srcAccessMask = ldrSrcAccess,
             .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
             .oldLayout = ldrImage->imageLayout(),
#ifdef USE_AMD
                 .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                 .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
             .srcQueueFamilyIndex = mainQueueIndex,
             .dstQueueFamilyIndex = mainQueueIndex,
             .image = ldrImage,
             .subresourceRange = vk::wholeColorSubresourceRange,
         }});
    hdrImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
#ifdef USE_AMD
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif

    vkCmdFillBuffer(worldCommandBuffer->vkCommandBuffer(), histBuffer->vkBuffer(), 0, VK_WHOLE_SIZE, 0);

    worldCommandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = histBuffer,
        }},
        {});

    std::chrono::time_point<std::chrono::high_resolution_clock> currentTimePoint =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = currentTimePoint - module->lastTimePoint_;
    module->lastTimePoint_ = currentTimePoint;

    ToneMappingModulePushConstant pc{};
    pc.log2Min = -12.0f;
    pc.log2Max = +4.0f;
    pc.epsilon = 1e-6f;
    pc.lowPercent = 0.005f;
    pc.highPercent = 0.99f;
    pc.middleGrey = module->middleGrey_;
    pc.dt = elapsedTime.count();
    pc.speedUp = module->speedUp_;
    pc.speedDown = module->speedDown_;
    pc.minExposure = 1e-4f;
    pc.maxExposure = 2.0f;

    // Eye 0: histogram → exposure → tone mapping
    // Eye 1+: tone mapping only (reuse eye 0's exposure for consistent brightness)
    for (uint32_t eye = 0; eye < eyeCount; eye++) {
        auto &eyeDT = descriptorTables[eye];
        auto &eyeFB = framebuffers[eye];

        if (eye == 0) {
            vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), eyeDT->vkPipelineLayout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ToneMappingModulePushConstant), &pc);

            worldCommandBuffer->bindDescriptorTable(eyeDT, VK_PIPELINE_BIND_POINT_COMPUTE)
                ->bindComputePipeline(module->histPipeline_);

            uint32_t groupX = (module->width_ + 16 - 1) / 16;
            uint32_t groupY = (module->height_ + 16 - 1) / 16;
            vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

            worldCommandBuffer->barriersBufferImage(
                {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .buffer = histBuffer,
                }},
                {});

            worldCommandBuffer->bindComputePipeline(module->exposurePipeline_);
            vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), 1, 1, 1);

            worldCommandBuffer->barriersBufferImage(
                {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .buffer = module->exposureData_,
                }},
                {});
        }

        worldCommandBuffer->beginRenderPass({
            .renderPass = module->renderPass_,
            .framebuffer = eyeFB,
            .renderAreaExtent = {ldrImage->width(), ldrImage->height()},
            .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
        });

        worldCommandBuffer->bindGraphicsPipeline(module->pipeline_)
            ->bindDescriptorTable(eyeDT, VK_PIPELINE_BIND_POINT_GRAPHICS)
            ->draw(3, 1)
            ->endRenderPass();
    }

#ifdef USE_AMD
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}
