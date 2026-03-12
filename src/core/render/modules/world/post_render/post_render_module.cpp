#include "core/render/modules/world/post_render/post_render_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/entities.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <random>

PostRenderModule::PostRenderModule() {}

void PostRenderModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    ldrImages_.resize(size);
    firstHitDepthImages_.resize(size);
    postRenderedImages_.resize(size);
    postRenderedInitialized_.assign(size, 0);
}

bool PostRenderModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        ldrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, eyeCount_, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (eyeCount_ > 1) images[0]->createPerLayerViews();
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        ldrImages_[frameIndex] = images[0];
    }

    if (images[1] == nullptr) {
        firstHitDepthImages_[frameIndex] = images[1] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, eyeCount_, formats[1],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (eyeCount_ > 1) images[1]->createPerLayerViews();
    } else {
        if (images[1]->width() != width_ || images[1]->height() != height_) return false;
        firstHitDepthImages_[frameIndex] = images[1];
    }

    return true;
}

bool PostRenderModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.size() != outputImageNum || images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    postRenderedImages_[frameIndex] = images[0];

    return true;
}

void PostRenderModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {}

void PostRenderModule::build() {
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
        contexts_[i] =
            PostRenderModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &PostRenderModule::contexts() {
    return contexts_;
}

void PostRenderModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    uint32_t totalCtx = size * eyeCount_;
    for (int i = 0; i < totalCtx; i++) {
        if (descriptorTables_[i] != nullptr)
            descriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0,
                                                   index);
    }
}

void PostRenderModule::preClose() {}

void PostRenderModule::initDescriptorTables() {
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
                                       .descriptorCount = 4096, // a very big number
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1, // light map
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2, // first hit depth
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .beginDescriptorLayoutSet() // set 1
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0, // binding 0: current world ubo
                                       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1, // binding 1: sky ubo
                                       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2, // binding 2: light map ubo
                                       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .beginDescriptorLayoutSet() // set 2
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0, // binding 0: mapping
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant({
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       .offset = 0,
                                       .size = sizeof(PostRenderPushConstant),
                                   })
                                   .build(framework->device());

        samplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }
}

void PostRenderModule::initImages() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t size = framework->swapchain()->imageCount();

    worldLightMapImages_.resize(size);
    uint32_t totalCtx = size * eyeCount_;
    worldPostDepthImages_.resize(totalCtx);

    for (int i = 0; i < size; i++) {
        worldLightMapImages_[i] =
            vk::DeviceLocalImage::create(device, vma, false, 16, 16, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        for (uint32_t e = 0; e < eyeCount_; e++) {
            uint32_t dtIdx = i * eyeCount_ + e;
            uint32_t viewIdx = eyeCount_ > 1 ? 1 + e : 0;

            descriptorTables_[dtIdx]->bindSamplerImageForShader(samplers_[dtIdx], worldLightMapImages_[i], 0, 1);
            descriptorTables_[dtIdx]->bindImage(firstHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2, viewIdx);

            worldPostDepthImages_[dtIdx] = vk::DeviceLocalImage::create(
                device, vma, false, width_, height_, 1, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        }
    }
}

static inline float rand01(std::mt19937 &rng) {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

static inline glm ::vec3 sampleUnitSphere(std::mt19937 &rng) {
    float u = rand01(rng);
    float v = rand01(rng);

    float z = 1.0f - 2.0f * u;
    float a = 2.0f * glm::pi<float>() * v;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));

    float x = r * std::cos(a);
    float y = r * std::sin(a);
    return glm::vec3(x, y, z);
}

static inline vk::VertexFormat::PBRTriangle makeStarVertex(const glm ::vec3 dir, const glm ::vec4 color) {
    vk::VertexFormat::PBRTriangle v{};
    v.pos = dir;
    v.useColorLayer = 1;
    v.colorLayer = color;
    v.coordinate = 2;
    return v;
}

void PostRenderModule::initBuffers() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();

    const float sunCone = 0.05f;
    const float moonCone = 0.08f;
    const float cosSun = std::cos(sunCone);
    const float cosMoon = std::cos(moonCone);
    uint32_t seed = 12345;
    uint32_t starCount = 3000;
    float starSizeMin = 0.5;
    float starSizeMax = 0.7;
    float starRadius = 400;

    std::mt19937 rng(seed);

    std::vector<vk::VertexFormat::PBRTriangle> verts;
    verts.reserve((size_t)starCount * 6);

    const float half = 0.5f * (starSizeMin + rand01(rng) * (starSizeMax - starSizeMin));

    for (int i = 0; i < starCount; i++) {
        glm ::vec3 dir{};
        for (;;) {
            dir = sampleUnitSphere(rng);
            if (dir.x > cosSun) continue;
            if (dir.x < -cosMoon) continue;
            break;
        }

        float u = rand01(rng);
        float brightness;
        if (u < 0.35f) {
            brightness = 0.75f + 0.25f * rand01(rng);
        } else {
            brightness = 0.02f + 0.35f * std::pow(rand01(rng), 3.0f);
        }

        float tint = rand01(rng);
        glm::vec3 baseRGB;
        if (tint < 0.75f)
            baseRGB = glm::vec3(1.0f, 1.0f, 1.0f);
        else if (tint < 0.9f)
            baseRGB = glm::vec3(1.0f, 0.95f, 0.85f); // 暖
        else
            baseRGB = glm::vec3(0.85f, 0.9f, 1.0f); // 冷

        glm::vec4 color(baseRGB * brightness, 1.0f);

        glm::vec3 center = dir * starRadius;

        glm::vec3 ref = (std::abs(dir.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
        glm::vec3 t1 = glm::normalize(glm::cross(ref, dir));
        glm::vec3 t2 = glm::cross(dir, t1);

        glm::vec3 dx = t1 * half;
        glm::vec3 dy = t2 * half;

        glm::vec3 p0 = center - dx - dy;
        glm::vec3 p1 = center + dx - dy;
        glm::vec3 p2 = center + dx + dy;
        glm::vec3 p3 = center - dx + dy;

        verts.push_back(makeStarVertex(p0, color));
        verts.push_back(makeStarVertex(p1, color));
        verts.push_back(makeStarVertex(p2, color));

        verts.push_back(makeStarVertex(p2, color));
        verts.push_back(makeStarVertex(p3, color));
        verts.push_back(makeStarVertex(p0, color));
    }

    starFieldVertexBuffer = vk::DeviceLocalBuffer::create(
        vma, device, verts.size() * sizeof(vk::VertexFormat::PBRTriangle), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    starFieldVertexBuffer->uploadToStagingBuffer(verts.data());

    std::shared_ptr<vk::Fence> fence = vk::Fence::create(device);
    std::shared_ptr<vk::CommandBuffer> oneTimeBuffer = vk::CommandBuffer::create(device, framework->mainCommandPool());
    oneTimeBuffer->begin();
    starFieldVertexBuffer->uploadToBuffer(oneTimeBuffer);
    oneTimeBuffer->end();

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = 0;
    vkSubmitInfo.pWaitSemaphores = nullptr;
    vkSubmitInfo.pWaitDstStageMask = nullptr;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &oneTimeBuffer->vkCommandBuffer();
    vkSubmitInfo.signalSemaphoreCount = 0;
    vkSubmitInfo.pSignalSemaphores = nullptr;
    vkQueueSubmit(device->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence());
    vkWaitForFences(device->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
}

void PostRenderModule::initRenderPass() {
    auto device = framework_.lock()->device();
    worldLightMapRenderPass_ = vk::RenderPassBuilder{}
                                   .beginAttachmentDescription()
                                   .defineAttachmentDescription({
                                       // color
                                       .format = worldLightMapImages_[0]->vkFormat(),
                                       .samples = VK_SAMPLE_COUNT_1_BIT,
                                       .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
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
                                   .build(device);

    worldPostColorToDepthRenderPass_ = vk::RenderPassBuilder{}
                                           .beginAttachmentDescription()
                                           .defineAttachmentDescription({
                                               // depth
                                               .format = worldPostDepthImages_[0]->vkFormat(),
                                               .samples = VK_SAMPLE_COUNT_1_BIT,
                                               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                               .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                               .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                               .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                               .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                           })
                                           .endAttachmentDescription()
                                           .beginAttachmentReference()
                                           .defineAttachmentReference({
                                               .attachment = 0,
                                               .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                           })
                                           .endAttachmentReference()
                                           .beginSubpassDescription()
                                           .defineSubpassDescription({
                                               .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                               .colorAttachmentIndices = {},
                                               .depthStencilAttachmentIndex = 0,
                                           })
                                           .endSubpassDescription()
                                           .build(device);

    worldPostRenderPass_ = vk::RenderPassBuilder{}
                               .beginAttachmentDescription()
                               .defineAttachmentDescription({
                                   .format = postRenderedImages_[0]->vkFormat(),
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
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
                               .defineAttachmentDescription({
                                   // depth
                                   .format = worldPostDepthImages_[0]->vkFormat(),
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
                               .build(device);
}

void PostRenderModule::initFrameBuffers() {
    auto framework = framework_.lock();
    auto device = framework->device();
    uint32_t size = framework->swapchain()->imageCount();

    uint32_t totalCtx = size * eyeCount_;

    worldLightMapFramebuffers_.resize(size);
    worldPostColorToDepthFramebuffers_.resize(totalCtx);
    worldPostFramebuffers_.resize(totalCtx);

    for (int i = 0; i < size; i++) {
        worldLightMapFramebuffers_[i] = vk::FramebufferBuilder{}
                                            .beginAttachment()
                                            .defineAttachment(worldLightMapImages_[i])
                                            .endAttachment()
                                            .build(device, worldLightMapRenderPass_);

        for (uint32_t e = 0; e < eyeCount_; e++) {
            uint32_t idx = i * eyeCount_ + e;
            uint32_t viewIdx = eyeCount_ > 1 ? 1 + e : 0;

            worldPostColorToDepthFramebuffers_[idx] = vk::FramebufferBuilder{}
                                                          .beginAttachment()
                                                          .defineAttachment(worldPostDepthImages_[idx])
                                                          .endAttachment()
                                                          .build(device, worldPostColorToDepthRenderPass_);

            worldPostFramebuffers_[idx] = vk::FramebufferBuilder{}
                                              .beginAttachment()
                                              .defineAttachment(postRenderedImages_[i], viewIdx)
                                              .defineAttachment(worldPostDepthImages_[idx])
                                              .endAttachment()
                                              .build(device, worldPostRenderPass_);
        }
    }
}

void PostRenderModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    worldLightMapVertShader_ =
        vk::Shader::create(device, (shaderPath / "world/post_render/light_map_vert.spv").string());
    worldLightMapFragShader_ =
        vk::Shader::create(device, (shaderPath / "world/post_render/light_map_frag.spv").string());

    worldLightMapPipeline_ = vk::GraphicsPipelineBuilder{}
                                 .defineRenderPass(worldLightMapRenderPass_, 0)
                                 .beginShaderStage()
                                 .defineShaderStage(worldLightMapVertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                                 .defineShaderStage(worldLightMapFragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                                 .endShaderStage()
                                 .defineVertexInputState<void>()
                                 .defineViewportScissorState({
                                     .viewport =
                                         {
                                             .x = 0,
                                             .y = 0,
                                             .width = static_cast<float>(worldLightMapImages_[0]->width()),
                                             .height = static_cast<float>(worldLightMapImages_[0]->height()),
                                             .minDepth = 0.0,
                                             .maxDepth = 1.0,
                                         },
                                     .scissor =
                                         {
                                             .offset = {.x = 0, .y = 0},
                                             .extent =
                                                 {
                                                     .width = worldLightMapImages_[0]->width(),
                                                     .height = worldLightMapImages_[0]->height(),
                                                 },
                                         },
                                 })
                                 .beginColorBlendAttachmentState()
                                 .defineDefaultColorBlendAttachmentState() // color
                                 .endColorBlendAttachmentState()
                                 .definePipelineLayout(descriptorTables_[0])
                                 .build(device);

    worldPostColorToDepthVertShader_ =
        vk::Shader::create(device, (shaderPath / "world/post_render/color_to_depth_vert.spv").string());
    worldPostColorToDepthFragShader_ =
        vk::Shader::create(device, (shaderPath / "world/post_render/color_to_depth_frag.spv").string());

    worldPostColorToDepthPipeline_ =
        vk::GraphicsPipelineBuilder{}
            .defineRenderPass(worldPostColorToDepthRenderPass_, 0)
            .beginShaderStage()
            .defineShaderStage(worldPostColorToDepthVertShader_, VK_SHADER_STAGE_VERTEX_BIT)
            .defineShaderStage(worldPostColorToDepthFragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
            .endShaderStage()
            .defineVertexInputState<void>()
            .defineViewportScissorState({
                .viewport =
                    {
                        .x = 0,
                        .y = 0,
                        .width = static_cast<float>(worldPostDepthImages_[0]->width()),
                        .height = static_cast<float>(worldPostDepthImages_[0]->height()),
                        .minDepth = 0.0,
                        .maxDepth = 1.0,
                    },
                .scissor =
                    {
                        .offset = {.x = 0, .y = 0},
                        .extent =
                            {
                                .width = worldPostDepthImages_[0]->width(),
                                .height = worldPostDepthImages_[0]->height(),
                            },
                    },
            })
            .defineDepthStencilState({
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .minDepthBounds = 0.0,
                .maxDepthBounds = 1.0,
            })
            .beginColorBlendAttachmentState()
            .endColorBlendAttachmentState()
            .definePipelineLayout(descriptorTables_[0])
            .build(device);

    VkPipelineColorBlendAttachmentState postColorBlendAttachmentState{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    worldPostVertShader_ = vk::Shader::create(device, (shaderPath / "world/post_render/world_post_vert.spv").string());
    worldPostFragShader_ = vk::Shader::create(device, (shaderPath / "world/post_render/world_post_frag.spv").string());

    worldPostPipeline_ = vk::GraphicsPipelineBuilder{}
                             .defineRenderPass(worldPostRenderPass_, 0)
                             .beginShaderStage()
                             .defineShaderStage(worldPostVertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                             .defineShaderStage(worldPostFragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                             .endShaderStage()
                             .defineVertexInputState<vk::VertexFormat::PBRTriangle>()
                             .defineViewportScissorState({
                                 .viewport =
                                     {
                                         .x = 0,
                                         .y = 0,
                                         .width = static_cast<float>(worldPostDepthImages_[0]->width()),
                                         .height = static_cast<float>(worldPostDepthImages_[0]->height()),
                                         .minDepth = 0.0,
                                         .maxDepth = 1.0,
                                     },
                                 .scissor =
                                     {
                                         .offset = {.x = 0, .y = 0},
                                         .extent =
                                             {
                                                 .width = worldPostDepthImages_[0]->width(),
                                                 .height = worldPostDepthImages_[0]->height(),
                                             },
                                     },
                             })
                             .defineDepthStencilState({
                                 .depthTestEnable = VK_TRUE,
                                 .depthWriteEnable = VK_TRUE,
                                 .depthCompareOp = VK_COMPARE_OP_LESS,
                                 .depthBoundsTestEnable = VK_FALSE,
                                 .stencilTestEnable = VK_FALSE,
                                 .minDepthBounds = 0.0,
                                 .maxDepthBounds = 1.0,
                             })
                             .defineRasterizationState(VkPipelineRasterizationStateCreateInfo{
                                 .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                 .depthClampEnable = VK_FALSE,
                                 .rasterizerDiscardEnable = VK_FALSE,
                                 .polygonMode = VK_POLYGON_MODE_FILL,
                                 .cullMode = VK_CULL_MODE_NONE,
                                 .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                 .depthBiasEnable = VK_FALSE,
                                 .depthBiasConstantFactor = 0.0f,
                                 .depthBiasClamp = 0.0f,
                                 .depthBiasSlopeFactor = 0.0f,
                                 .lineWidth = 1.0f,
                             })
                             .beginColorBlendAttachmentState()
                             .defineColorBlendAttachmentState(postColorBlendAttachmentState)
                             .endColorBlendAttachmentState()
                             .definePipelineLayout(descriptorTables_[0])
                             .build(device);

    worldPostStarFieldVertShader_ =
        vk::Shader::create(device, (shaderPath / "world/post_render/world_post_star_vert.spv").string());
    worldPostStarFieldFragShader_ =
        vk::Shader::create(device, (shaderPath / "world/post_render/world_post_star_frag.spv").string());

    worldPostStarFieldPipeline_ = vk::GraphicsPipelineBuilder{}
                                      .defineRenderPass(worldPostRenderPass_, 0)
                                      .beginShaderStage()
                                      .defineShaderStage(worldPostStarFieldVertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                                      .defineShaderStage(worldPostStarFieldFragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                                      .endShaderStage()
                                      .defineVertexInputState<vk::VertexFormat::PBRTriangle>()
                                      .defineViewportScissorState({
                                          .viewport =
                                              {
                                                  .x = 0,
                                                  .y = 0,
                                                  .width = static_cast<float>(worldPostDepthImages_[0]->width()),
                                                  .height = static_cast<float>(worldPostDepthImages_[0]->height()),
                                                  .minDepth = 0.0,
                                                  .maxDepth = 1.0,
                                              },
                                          .scissor =
                                              {
                                                  .offset = {.x = 0, .y = 0},
                                                  .extent =
                                                      {
                                                          .width = worldPostDepthImages_[0]->width(),
                                                          .height = worldPostDepthImages_[0]->height(),
                                                      },
                                              },
                                      })
                                      .defineDepthStencilState({
                                          .depthTestEnable = VK_TRUE,
                                          .depthWriteEnable = VK_TRUE,
                                          .depthCompareOp = VK_COMPARE_OP_LESS,
                                          .depthBoundsTestEnable = VK_FALSE,
                                          .stencilTestEnable = VK_FALSE,
                                          .minDepthBounds = 0.0,
                                          .maxDepthBounds = 1.0,
                                      })
                                      .defineRasterizationState(VkPipelineRasterizationStateCreateInfo{
                                          .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                          .depthClampEnable = VK_FALSE,
                                          .rasterizerDiscardEnable = VK_FALSE,
                                          .polygonMode = VK_POLYGON_MODE_FILL,
                                          .cullMode = VK_CULL_MODE_NONE,
                                          .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                          .depthBiasEnable = VK_FALSE,
                                          .depthBiasConstantFactor = 0.0f,
                                          .depthBiasClamp = 0.0f,
                                          .depthBiasSlopeFactor = 0.0f,
                                          .lineWidth = 1.0f,
                                      })
                                      .beginColorBlendAttachmentState()
                                      .defineColorBlendAttachmentState(postColorBlendAttachmentState)
                                      .endColorBlendAttachmentState()
                                      .definePipelineLayout(descriptorTables_[0])
                                      .build(device);
}

PostRenderModuleContext::PostRenderModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<PostRenderModule> postRenderModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      postRenderModule(postRenderModule),
      ldrImage(postRenderModule->ldrImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(postRenderModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      worldLightMapImage(postRenderModule->worldLightMapImages_[frameworkContext->frameIndex]),
      worldLightMapFramebuffer(postRenderModule->worldLightMapFramebuffers_[frameworkContext->frameIndex]),
      postRenderedImage(postRenderModule->postRenderedImages_[frameworkContext->frameIndex]),
      eyeCount_(postRenderModule->eyeCount_) {
    uint32_t fi = frameworkContext->frameIndex;
    uint32_t base = fi * eyeCount_;
    for (uint32_t e = 0; e < eyeCount_; e++) {
        descriptorTables.push_back(postRenderModule->descriptorTables_[base + e]);
        worldPostDepthImages.push_back(postRenderModule->worldPostDepthImages_[base + e]);
        worldPostColorToDepthFramebuffers.push_back(postRenderModule->worldPostColorToDepthFramebuffers_[base + e]);
        worldPostFramebuffers.push_back(postRenderModule->worldPostFramebuffers_[base + e]);
    }
}

void PostRenderModuleContext::render() {
    render3D(1);
}

void PostRenderModuleContext::render3D(uint32_t eyeCount) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = postRenderModule.lock();

    auto buffers = Renderer::instance().buffers();

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

    // Force init for postRenderedImage on first use (layout tracking can be stale)
    if (module && module->postRenderedInitialized_.size() > context->frameIndex &&
        module->postRenderedInitialized_[context->frameIndex] == 0) {
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
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                  .srcAccessMask = 0,
                  .dstStageMask = dstStage,
                  .dstAccessMask = dstAccess,
                  .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                  .newLayout = targetLayout,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = postRenderedImage,
                  .subresourceRange = vk::wholeColorSubresourceRange}});
        postRenderedImage->imageLayout() = targetLayout;
        module->postRenderedInitialized_[context->frameIndex] = 1;
    }

    auto ensureLayout = [&](const std::shared_ptr<vk::DeviceLocalImage> &img,
                            VkImageLayout targetLayout,
                            VkPipelineStageFlags2 dstStage,
                            VkAccessFlags2 dstAccess) {
        if (!img || img->imageLayout() == targetLayout) return;
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(img->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        worldCommandBuffer->barriersBufferImage(
            {}, {{.srcStageMask = srcStage,
                  .srcAccessMask = srcAccess,
                  .dstStageMask = dstStage,
                  .dstAccessMask = dstAccess,
                  .oldLayout = img->imageLayout(),
                  .newLayout = targetLayout,
                  .srcQueueFamilyIndex = mainQueueIndex,
                  .dstQueueFamilyIndex = mainQueueIndex,
                  .image = img,
                  .subresourceRange = vk::wholeColorSubresourceRange}});
        img->imageLayout() = targetLayout;
    };

    // Preflight: make sure first-use layouts are valid
    ensureLayout(postRenderedImage,
#ifdef USE_AMD
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    ensureLayout(worldLightMapImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    ensureLayout(firstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    for (uint32_t e = 0; e < eyeCount; e++) {
        auto &eyeDepth = worldPostDepthImages[e];
        if (eyeDepth && eyeDepth->imageLayout() != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            VkPipelineStageFlags2 srcStage = 0;
            VkAccessFlags2 srcAccess = 0;
            chooseSrc(eyeDepth->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
            worldCommandBuffer->barriersBufferImage(
                {}, {{.srcStageMask = srcStage,
                      .srcAccessMask = srcAccess,
                      .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                      .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      .oldLayout = eyeDepth->imageLayout(),
                      .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                      .srcQueueFamilyIndex = mainQueueIndex,
                      .dstQueueFamilyIndex = mainQueueIndex,
                      .image = eyeDepth,
                      .subresourceRange = vk::wholeDepthSubresourceRange}});
            eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
    }

    for (uint32_t e = 0; e < eyeCount; e++) {
        descriptorTables[e]->bindBuffer(buffers->worldUniformBuffer(), 1, 0);
        descriptorTables[e]->bindBuffer(buffers->skyUniformBuffer(), 1, 1);
        descriptorTables[e]->bindBuffer(buffers->lightMapUniformBuffer(), 1, 2);
        descriptorTables[e]->bindBuffer(Renderer::instance().buffers()->textureMappingBuffer(), 2, 0);
    }

    // render light map
    {
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(worldLightMapImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStage, srcAccess);
        worldCommandBuffer->barriersBufferImage(
            {}, {{
                    .srcStageMask = srcStage,
                    .srcAccessMask = srcAccess,
                    .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = worldLightMapImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = worldLightMapImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }});
    }
    worldLightMapImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->worldLightMapRenderPass_,
        .framebuffer = worldLightMapFramebuffer,
        .renderAreaExtent = {worldLightMapImage->width(), worldLightMapImage->height()},
        .clearValues = {},
    });
    worldLightMapImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    worldCommandBuffer->bindGraphicsPipeline(module->worldLightMapPipeline_)
        ->bindDescriptorTable(descriptorTables[0], VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->draw(3, 1)
        ->endRenderPass();
    worldLightMapImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // ---- PER-EYE LOOP ----
    for (uint32_t eye = 0; eye < eyeCount; eye++) {
        auto &eyeDT = descriptorTables[eye];
        auto &eyeDepth = worldPostDepthImages[eye];
        auto &eyeC2DFB = worldPostColorToDepthFramebuffers[eye];
        auto &eyePostFB = worldPostFramebuffers[eye];
        uint32_t eyeLayer = eyeCount > 1 ? eye : 0;

    // cast linear depth from color to depth format
    {
        VkPipelineStageFlags2 srcStageDepth = 0;
        VkAccessFlags2 srcAccessDepth = 0;
        chooseSrc(eyeDepth->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStageDepth, srcAccessDepth);

        VkPipelineStageFlags2 srcStageFirstHit = 0;
        VkAccessFlags2 srcAccessFirstHit = 0;
        chooseSrc(firstHitDepthImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStageFirstHit, srcAccessFirstHit);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = srcStageDepth,
                     .srcAccessMask = srcAccessDepth,
                     .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = eyeDepth->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = eyeDepth,
                     .subresourceRange = vk::wholeDepthSubresourceRange,
                 },
                 {
                     .srcStageMask = srcStageFirstHit,
                     .srcAccessMask = srcAccessFirstHit,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = firstHitDepthImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = firstHitDepthImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
    }
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->worldPostColorToDepthRenderPass_,
        .framebuffer = eyeC2DFB,
        .renderAreaExtent = {eyeDepth->width(), eyeDepth->height()},
        .clearValues = {{.depthStencil = {.depth = 1.0f}}},
    });
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    worldCommandBuffer->bindDescriptorTable(eyeDT, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->bindGraphicsPipeline(module->worldPostColorToDepthPipeline_)
        ->draw(3, 1)
        ->endRenderPass();
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // copy input to output
    {
        VkPipelineStageFlags2 srcStageLdr = 0;
        VkAccessFlags2 srcAccessLdr = 0;
        chooseSrc(ldrImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStageLdr, srcAccessLdr);

        VkPipelineStageFlags2 srcStagePost = 0;
        VkAccessFlags2 srcAccessPost = 0;
        chooseSrc(postRenderedImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStagePost, srcAccessPost);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = srcStageLdr,
                     .srcAccessMask = srcAccessLdr,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = ldrImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = ldrImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = srcStagePost,
                     .srcAccessMask = srcAccessPost,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = postRenderedImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = postRenderedImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
    }
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // TODO: add to command buffer
    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = eyeLayer;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(ldrImage->width()), static_cast<int>(ldrImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = eyeLayer;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(postRenderedImage->width()),
                               static_cast<int>(postRenderedImage->height()), 1};

    vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), ldrImage->vkImage(), ldrImage->imageLayout(),
                   postRenderedImage->vkImage(), postRenderedImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

    // post render
    {
        VkPipelineStageFlags2 srcStagePost = 0;
        VkAccessFlags2 srcAccessPost = 0;
        chooseSrc(postRenderedImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStagePost, srcAccessPost);

        VkPipelineStageFlags2 srcStageLight = 0;
        VkAccessFlags2 srcAccessLight = 0;
        chooseSrc(worldLightMapImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStageLight, srcAccessLight);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = srcStagePost,
                     .srcAccessMask = srcAccessPost,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = postRenderedImage->imageLayout(),
#ifdef USE_AMD
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = postRenderedImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = srcStageLight,
                     .srcAccessMask = srcAccessLight,
                     .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = worldLightMapImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = worldLightMapImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
    }
#ifdef USE_AMD
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    worldLightMapImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->worldPostRenderPass_,
        .framebuffer = eyePostFB,
        .renderAreaExtent = {eyeDepth->width(), eyeDepth->height()},
        .clearValues = {},
    });
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::shared_ptr<EntityPostBatch> entityPostRenderDataBatch =
        Renderer::instance().world()->entities()->entityPostBatch();
    if (entityPostRenderDataBatch != nullptr) {
        PostRenderPushConstant pc{};
        pc.eyeIndex = eye;
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), eyeDT->vkPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PostRenderPushConstant), &pc);

        worldCommandBuffer->bindDescriptorTable(eyeDT, VK_PIPELINE_BIND_POINT_GRAPHICS)
            ->bindGraphicsPipeline(module->worldPostPipeline_);

        for (int i = 0; i < entityPostRenderDataBatch->entities.size(); i++) {
            auto &entity = entityPostRenderDataBatch->entities[i];

            for (int j = 0; j < entity->geometryCount; j++) {
                auto &vertexBuffer = entity->vertexBuffers[j];
                auto &indexBuffer = entity->indexBuffers[j];

                worldCommandBuffer->bindVertexBuffers(vertexBuffer)
                    ->bindIndexBuffer(indexBuffer)
                    ->drawIndexed(entity->indices[j].size(), 1);
            }
        }
    }

    worldCommandBuffer->endRenderPass();
#ifdef USE_AMD
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // render post star field
    {
        VkPipelineStageFlags2 srcStagePost = 0;
        VkAccessFlags2 srcAccessPost = 0;
        chooseSrc(postRenderedImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStagePost, srcAccessPost);

        VkPipelineStageFlags2 srcStageLight = 0;
        VkAccessFlags2 srcAccessLight = 0;
        chooseSrc(worldLightMapImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  srcStageLight, srcAccessLight);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = srcStagePost,
                     .srcAccessMask = srcAccessPost,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = postRenderedImage->imageLayout(),
#ifdef USE_AMD
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = postRenderedImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = srcStageLight,
                     .srcAccessMask = srcAccessLight,
                     .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = worldLightMapImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = worldLightMapImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
    }
#ifdef USE_AMD
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    worldLightMapImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->worldPostRenderPass_,
        .framebuffer = eyePostFB,
        .renderAreaExtent = {eyeDepth->width(), eyeDepth->height()},
        .clearValues = {},
    });
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    {
        PostRenderPushConstant pc{};
        pc.eyeIndex = eye;
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), eyeDT->vkPipelineLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PostRenderPushConstant), &pc);
    }
    worldCommandBuffer->bindDescriptorTable(eyeDT, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->bindGraphicsPipeline(module->worldPostStarFieldPipeline_)

        ->bindVertexBuffers(module->starFieldVertexBuffer)
        ->draw(module->starFieldVertexBuffer->size() / sizeof(vk::VertexFormat::PBRTriangle), 1);

    worldCommandBuffer->endRenderPass();
#ifdef USE_AMD
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    eyeDepth->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    } // end per-eye loop
}
