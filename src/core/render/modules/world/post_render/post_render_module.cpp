#include "core/render/modules/world/post_render/post_render_module.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/entities.hpp"
#include "core/render/modules/world/main_render/main_render_module.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/util/parallel.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/vertex.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string_view>
#include <utility>

using json = nlohmann::json;

static constexpr uint32_t segmentationMaskDescriptorSet = 3;
static constexpr uint32_t segmentationMaskDescriptorBinding = 5;

static VkCullModeFlags parseCullMode(const std::string &value, const std::string &context) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower == "none") { return VK_CULL_MODE_NONE; }
    if (lower == "front") { return VK_CULL_MODE_FRONT_BIT; }
    if (lower == "back") { return VK_CULL_MODE_BACK_BIT; }
    if (lower == "front_and_back" || lower == "all") { return VK_CULL_MODE_FRONT_AND_BACK; }
    throw std::runtime_error("unsupported " + context + " cull mode: " + value);
}

static VkFrontFace parseFrontFace(const std::string &value, const std::string &context) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower == "counter_clockwise" || lower == "ccw") { return VK_FRONT_FACE_COUNTER_CLOCKWISE; }
    if (lower == "clockwise" || lower == "cw") { return VK_FRONT_FACE_CLOCKWISE; }
    throw std::runtime_error("unsupported " + context + " front_face: " + value);
}

const RenderPass::ShaderVariant *PostRenderModule::findShaderVariant(const RenderPass &pass,
                                                                     std::string_view contentName) {
    auto variantIter = pass.shaderVariants.find(std::string(contentName));
    if (variantIter != pass.shaderVariants.end()) { return &variantIter->second; }

    variantIter = pass.shaderVariants.find("default");
    if (variantIter != pass.shaderVariants.end()) { return &variantIter->second; }

    return nullptr;
}

struct PostRenderModule::PostRenderExecutionRuntime final : ShaderPack::ExecutionRuntime {
    std::shared_ptr<PostRenderModule> postRenderModule;
    std::shared_ptr<PostRenderModuleContext> postRenderContext;
};

struct PostRenderModule::PostRenderPassExecutionOp final : ShaderPack::ExecutionOp {
    explicit PostRenderPassExecutionOp(PostRenderPassVariant postRenderPassVariant)
        : postRenderPassVariant(std::move(postRenderPassVariant)) {}

    PostRenderPassVariant postRenderPassVariant;

    void operator()(ShaderPack::ExecutionRuntime &executionRuntime) const override {
        MCVR_PROFILE_SCOPE("native.PostRenderModule.executePass");
        auto &runtime = dynamic_cast<PostRenderExecutionRuntime &>(executionRuntime);
        auto postRenderModule = runtime.postRenderModule;
        auto postRenderContext = runtime.postRenderContext;
        if (postRenderModule == nullptr || postRenderContext == nullptr) { return; }

        auto frameworkContext = postRenderContext->frameworkContext.lock();
        auto framework = frameworkContext->framework.lock();
        auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
        const uint32_t frameIndex = frameworkContext->frameIndex;
        const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();

        auto addColorImageBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &image, VkImageLayout newLayout) {
            if (image == nullptr) { return; }
            VkPipelineStageFlags2 destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            VkAccessFlags2 destinationAccess = VK_ACCESS_2_SHADER_READ_BIT;
            if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                destinationStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                destinationAccess = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ||
                       newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                destinationAccess = newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ? VK_ACCESS_2_TRANSFER_READ_BIT :
                                                                                        VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                        .srcStageMask = image->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED ?
                                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        .srcAccessMask = image->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED ?
                                             0 :
                                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = destinationStage,
                        .dstAccessMask = destinationAccess,
                        .oldLayout = image->imageLayout(),
                        .newLayout = newLayout,
                        .srcQueueFamilyIndex = queueIndex,
                        .dstQueueFamilyIndex = queueIndex,
                        .image = image,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    }});
            image->imageLayout() = newLayout;
        };

        auto addDepthImageBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &image, VkImageLayout newLayout) {
            if (image == nullptr || image->imageLayout() == newLayout) { return; }
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                        .srcStageMask = image->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED ?
                                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT :
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        .srcAccessMask = image->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED ?
                                             0 :
                                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask =
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = image->imageLayout(),
                        .newLayout = newLayout,
                        .srcQueueFamilyIndex = queueIndex,
                        .dstQueueFamilyIndex = queueIndex,
                        .image = image,
                        .subresourceRange = vk::wholeDepthSubresourceRange,
                    }});
            image->imageLayout() = newLayout;
        };

        auto transitionInputImages = [&](const ShaderPackLoader::ResourceList &inputs) {
            for (const std::string &input : inputs.images) {
                if (auto builtInImage = postRenderModule->findBuiltInImage(input, frameIndex);
                    builtInImage != nullptr) {
                    addColorImageBarrier(builtInImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    continue;
                }
                auto runtimeTexture = postRenderModule->shaderPack_->findRuntimeTexture(input);
                if (!runtimeTexture.has_value()) { throw std::runtime_error("unknown post_render input: " + input); }
                VkImageLayout layout =
                    runtimeTexture->get().config.imported || !runtimeTexture->get().config.storageBinding.has_value() ?
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL :
                        VK_IMAGE_LAYOUT_GENERAL;
                addColorImageBarrier(
                    postRenderModule->shaderPack_->findRuntimeVKTexture(runtimeTexture->get(), frameIndex), layout);
            }
        };

        std::visit(
            [&](const auto &compiledPass) {
                using PassType = std::decay_t<decltype(compiledPass)>;
                if (compiledPass == nullptr) { return; }

                if constexpr (std::is_same_v<PassType, std::shared_ptr<FullScreenPass>>) {
                    auto targetImage = postRenderModule->findTargetImage(compiledPass->config.target, frameIndex);
                    if (targetImage == nullptr) {
                        throw std::runtime_error("unknown post_render target: " + compiledPass->config.target);
                    }

                    transitionInputImages(compiledPass->config.inputs);

                    if (compiledPass->config.target == PostRenderModule::TARGET_LDR ||
                        compiledPass->config.target == PostRenderModule::TARGET_FIRST_HIT_DEPTH) {
                        addColorImageBarrier(targetImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                    } else {
                        auto runtimeTexture =
                            postRenderModule->shaderPack_->findRuntimeTexture(compiledPass->config.target);
                        if (!runtimeTexture.has_value() || runtimeTexture->get().config.imported) {
                            throw std::runtime_error("invalid post_render output target: " +
                                                     compiledPass->config.target);
                        }
                        addColorImageBarrier(targetImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                    }

                    if (compiledPass->executionBuffer != nullptr) {
                        postRenderModule->uploadExecutionBuffer(compiledPass->executionBuffer, *postRenderContext,
                                                                runtime.executionVariables);
                    }

                    worldCommandBuffer->beginRenderPass({
                        .renderPass = compiledPass->renderPass,
                        .framebuffer = compiledPass->framebuffers[frameIndex][0],
                        .renderAreaExtent = {targetImage->width(), targetImage->height()},
                        .clearValues = {},
                    });
                    targetImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    worldCommandBuffer->bindGraphicsPipeline(compiledPass->pipelines[0])
                        ->bindDescriptorTable(postRenderContext->descriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
                        ->draw(3, 1)
                        ->endRenderPass();
                    targetImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    return;
                } else if constexpr (std::is_same_v<PassType, std::shared_ptr<RenderPass>>) {
                    if (compiledPass->target == RenderPass::Target::Star &&
                        postRenderContext->segmentationMaskImage == nullptr) {
                        return;
                    }
                    transitionInputImages(compiledPass->config.inputs);
                    addColorImageBarrier(postRenderContext->postRenderedImage,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                    if (compiledPass->writesFirstHitDepth) {
                        addColorImageBarrier(postRenderContext->firstHitDepthImage,
                                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                    }
                    addDepthImageBarrier(postRenderContext->worldPostDepthImage,
                                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

                    if (compiledPass->executionBuffer != nullptr) {
                        postRenderModule->uploadExecutionBuffer(compiledPass->executionBuffer, *postRenderContext,
                                                                runtime.executionVariables);
                    }

                    bool hasRenderableContent = false;
                    if (compiledPass->target == RenderPass::Target::Star) {
                        hasRenderableContent = postRenderModule->starFieldVertexBuffer != nullptr &&
                                               postRenderModule->starFieldVertexBuffer->size() > 0;
                    } else {
                        auto entityPostRenderDataBatch = Renderer::instance().world()->entities()->entityPostBatch();
                        const int postRenderFlag = PostRenderModule::renderTargetPostFlag(compiledPass->target);
                        if (entityPostRenderDataBatch != nullptr) {
                            for (const auto &entity : entityPostRenderDataBatch->entities) {
                                if (entity->postRenderFlag != postRenderFlag) { continue; }

                                for (uint32_t geometryIndex = 0; geometryIndex < entity->geometryCount;
                                     geometryIndex++) {
                                    const std::string &contentName =
                                        geometryIndex < entity->geometryContentNames.size() ?
                                            entity->geometryContentNames[geometryIndex] :
                                            std::string{};
                                    if (PostRenderModule::findShaderVariant(*compiledPass, contentName) != nullptr) {
                                        hasRenderableContent = true;
                                        break;
                                    }
                                }
                                if (hasRenderableContent) { break; }
                            }
                        }
                    }
                    if (!hasRenderableContent) { return; }

                    worldCommandBuffer->beginRenderPass({
                        .renderPass = compiledPass->renderPass,
                        .framebuffer = compiledPass->framebuffers[frameIndex],
                        .renderAreaExtent = {postRenderContext->postRenderedImage->width(),
                                             postRenderContext->postRenderedImage->height()},
                        .clearValues = {},
                    });
                    postRenderContext->postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    if (compiledPass->writesFirstHitDepth) {
                        postRenderContext->firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    }
                    postRenderContext->worldPostDepthImage->imageLayout() =
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

                    worldCommandBuffer->bindDescriptorTable(postRenderContext->descriptorTable,
                                                            VK_PIPELINE_BIND_POINT_GRAPHICS);
                    if (compiledPass->target == RenderPass::Target::Star) {
                        const RenderPass::ShaderVariant *variant =
                            PostRenderModule::findShaderVariant(*compiledPass, compiledPass->config.content);
                        if (variant == nullptr || variant->pipeline == nullptr) {
                            worldCommandBuffer->endRenderPass();
                            postRenderContext->postRenderedImage->imageLayout() =
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                            if (compiledPass->writesFirstHitDepth) {
                                postRenderContext->firstHitDepthImage->imageLayout() =
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                            }
                            postRenderContext->worldPostDepthImage->imageLayout() =
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                            return;
                        }
                        worldCommandBuffer->bindGraphicsPipeline(variant->pipeline)
                            ->bindVertexBuffers(postRenderModule->starFieldVertexBuffer)
                            ->draw(postRenderModule->starFieldVertexBuffer->size() /
                                       sizeof(vk::VertexFormat::EntityPBRVertex),
                                   1);
                    } else {
                        auto entityPostRenderDataBatch = Renderer::instance().world()->entities()->entityPostBatch();
                        const int postRenderFlag = PostRenderModule::renderTargetPostFlag(compiledPass->target);
                        const RenderPass::ShaderVariant *boundVariant = nullptr;
                        for (const auto &entity : entityPostRenderDataBatch->entities) {
                            if (entity->postRenderFlag != postRenderFlag) { continue; }

                            for (uint32_t j = 0; j < entity->geometryCount; j++) {
                                const std::string &contentName = j < entity->geometryContentNames.size() ?
                                                                     entity->geometryContentNames[j] :
                                                                     std::string{};
                                const RenderPass::ShaderVariant *variant =
                                    PostRenderModule::findShaderVariant(*compiledPass, contentName);
                                if (variant == nullptr || variant->pipeline == nullptr) { continue; }
                                if (variant != boundVariant) {
                                    worldCommandBuffer->bindGraphicsPipeline(variant->pipeline);
                                    boundVariant = variant;
                                }

                                auto &vertexBuffer = entity->vertexBuffers[j];
                                auto &indexBuffer = entity->indexBuffers[j];

                                worldCommandBuffer->bindVertexBuffers(vertexBuffer)
                                    ->bindIndexBuffer(indexBuffer)
                                    ->drawIndexed(entity->indexCounts[j], 1);
                            }
                        }
                    }
                    worldCommandBuffer->endRenderPass();
                    postRenderContext->postRenderedImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    if (compiledPass->writesFirstHitDepth) {
                        postRenderContext->firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    }
                    postRenderContext->worldPostDepthImage->imageLayout() =
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                }
            },
            postRenderPassVariant);
    }
};

PostRenderModule::PostRenderModule() {}

void PostRenderModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    ldrImages_.resize(size);
    firstHitDepthImages_.resize(size);
    hdrImages_.resize(size);
    motionVectorImages_.resize(size);
    normalRoughnessImages_.resize(size);
    postRenderedImages_.resize(size);
    postRefractionSourceImages_.resize(size);
    outputImages_.resize(size);
}

uint32_t PostRenderModule::scaledInputExtent(uint32_t outputExtent) const {
    const float precision = std::isfinite(renderPrecision_) ? std::clamp(renderPrecision_, 0.5f, 2.0f) : 1.0f;
    return static_cast<uint32_t>(std::max(1.0, std::round(static_cast<double>(outputExtent) * precision)));
}

void PostRenderModule::updateInputExtent() {
    inputWidth_ = scaledInputExtent(width_);
    inputHeight_ = scaledInputExtent(height_);
}

bool PostRenderModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;
    if (inputWidth_ == 0 || inputHeight_ == 0) { updateInputExtent(); }

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        ldrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    } else {
        if (images[0]->width() != inputWidth_ || images[0]->height() != inputHeight_) return false;
        ldrImages_[frameIndex] = images[0];
    }

    if (images[1] == nullptr) {
        firstHitDepthImages_[frameIndex] = images[1] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[1],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[1]->width() != inputWidth_ || images[1]->height() != inputHeight_) return false;
        firstHitDepthImages_[frameIndex] = images[1];
    }

    if (images[2] == nullptr) {
        hdrImages_[frameIndex] = images[2] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[2],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    } else {
        if (images[2]->width() != inputWidth_ || images[2]->height() != inputHeight_) return false;
        hdrImages_[frameIndex] = images[2];
    }

    if (images[3] == nullptr) {
        motionVectorImages_[frameIndex] = images[3] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[3],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    } else {
        if (images[3]->width() != inputWidth_ || images[3]->height() != inputHeight_) return false;
        motionVectorImages_[frameIndex] = images[3];
    }

    if (images[4] == nullptr) {
        normalRoughnessImages_[frameIndex] = images[4] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[4],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    } else {
        if (images[4]->width() != inputWidth_ || images[4]->height() != inputHeight_) return false;
        normalRoughnessImages_[frameIndex] = images[4];
    }

    return true;
}

bool PostRenderModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.size() != outputImageNum || images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();
    updateInputExtent();

    outputImages_[frameIndex] = images[0];

    return true;
}

void PostRenderModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseUint = [](const std::string &value, uint32_t fallback) {
        try {
            const long long parsed = std::stoll(value);
            if (parsed < 0LL) { return fallback; }
            return static_cast<uint32_t>(std::min<long long>(parsed, std::numeric_limits<uint32_t>::max()));
        } catch (...) { return fallback; }
    };
    auto parseFloat = [](const std::string &value, float fallback) {
        try {
            return std::stof(value);
        } catch (...) { return fallback; }
    };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.post_render.attribute.star_count") {
            starCount_ = std::max(1u, parseUint(value, starCount_));
        } else if (key == "render_pipeline.module.post_render.attribute.render_precision") {
            const float parsedPrecision = parseFloat(value, renderPrecision_);
            if (std::isfinite(parsedPrecision)) { renderPrecision_ = std::clamp(parsedPrecision, 0.5f, 2.0f); }
        } else if (key == "render_pipeline.module.post_render.attribute.star_min_size") {
            starSizeMin_ = std::max(0.001f, parseFloat(value, starSizeMin_));
        } else if (key == "render_pipeline.module.post_render.attribute.star_max_size") {
            starSizeMax_ = std::max(0.001f, parseFloat(value, starSizeMax_));
        } else if (key == "render_pipeline.module.post_render.attribute.star_radius") {
            starRadius_ = std::max(1.0f, parseFloat(value, starRadius_));
        }
    }

    if (starSizeMin_ > starSizeMax_) { std::swap(starSizeMin_, starSizeMax_); }
    updateInputExtent();
}

void PostRenderModule::initExecutionVariables() {
    executionVariableConfigs_.clear();
    globalVariables_.clear();

    if (shaderPack_ == nullptr) { return; }
    shaderPack_->copyStageExecutionState(ShaderPackLoader::Stage::PostRender, executionVariableConfigs_,
                                         globalVariables_);
}

void PostRenderModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    shaderPack_ = worldPipeline != nullptr ? worldPipeline->shaderPack() : nullptr;
    initExecutionVariables();
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

    ensureDynamicPipelines();
}

std::vector<std::shared_ptr<WorldModuleContext>> &PostRenderModule::contexts() {
    return contexts_;
}

void PostRenderModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        if (descriptorTables_[i] != nullptr)
            descriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0,
                                                   index);
    }
}

void PostRenderModule::preClose() {
    fullScreenPasses_.clear();
    passNameToPass_.clear();
    renderPasses_.clear();
    renderPassNameToPass_.clear();
    compiledExecutionProgram_ = nullptr;
    executionVariableConfigs_.clear();
    globalVariables_.clear();
    shaderPack_ = nullptr;
    isDynamicPipelinesReady_ = false;
}

RenderPass::Target PostRenderModule::parseRenderContent(const std::string &content) {
    const std::string lower = ShaderPackLoader::toLower(content);
    if (lower == "weather") { return RenderPass::Target::Weather; }
    if (lower == "particle" || lower == "particles") { return RenderPass::Target::Particle; }
    if (lower == "world_overlay") { return RenderPass::Target::WorldOverlay; }
    if (lower == "text") { return RenderPass::Target::Text; }
    if (lower == "name_tag" || lower == "nametag" || lower == "name-tag") { return RenderPass::Target::NameTag; }
    if (lower == "star" || lower == "stars") { return RenderPass::Target::Star; }
    throw std::runtime_error("unsupported post_render render content: " + content);
}

int PostRenderModule::renderTargetPostFlag(RenderPass::Target target) {
    switch (target) {
        case RenderPass::Target::Weather: return weatherPostFlag;
        case RenderPass::Target::Particle: return particlePostFlag;
        case RenderPass::Target::WorldOverlay: return worldOverlayPostFlag;
        case RenderPass::Target::Text: return textPostFlag;
        case RenderPass::Target::NameTag: return nameTagPostFlag;
        case RenderPass::Target::Star: return 0;
    }
    return 0;
}

VkCompareOp PostRenderModule::parseDepthCompare(const std::string &value) {
    const std::string lower = ShaderPackLoader::toLower(value);
    if (lower == "never") { return VK_COMPARE_OP_NEVER; }
    if (lower == "less") { return VK_COMPARE_OP_LESS; }
    if (lower == "equal") { return VK_COMPARE_OP_EQUAL; }
    if (lower == "less_or_equal" || lower == "lequal") { return VK_COMPARE_OP_LESS_OR_EQUAL; }
    if (lower == "greater") { return VK_COMPARE_OP_GREATER; }
    if (lower == "not_equal") { return VK_COMPARE_OP_NOT_EQUAL; }
    if (lower == "greater_or_equal" || lower == "gequal") { return VK_COMPARE_OP_GREATER_OR_EQUAL; }
    if (lower == "always") { return VK_COMPARE_OP_ALWAYS; }
    throw std::runtime_error("unsupported post_render depth_compare: " + value);
}

void PostRenderModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();
    descriptorTables_.resize(size);
    samplers_.resize(size);
    postPassColorSamplers_.resize(size);
    postPassDepthSamplers_.resize(size);
    postRefractionSourceSamplers_.resize(size);

    for (int i = 0; i < size; i++) {
        vk::DescriptorTableBuilder builder;
        builder.beginDescriptorLayoutSet()
            .beginDescriptorLayoutSetBinding()
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = framework->device()->textureDescriptorCapacity(),
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .endDescriptorLayoutSetBinding()
            .endDescriptorLayoutSet()
            .beginDescriptorLayoutSet()
            .beginStaticDescriptorLayoutSetBinding()
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .endDescriptorLayoutSetBinding()
            .endDescriptorLayoutSet()
            .beginDescriptorLayoutSet()
            .beginDescriptorLayoutSetBinding()
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .endDescriptorLayoutSetBinding()
            .endDescriptorLayoutSet()
            .beginDescriptorLayoutSet()
            .beginDescriptorLayoutSetBinding()
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = segmentationMaskDescriptorBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .endDescriptorLayoutSetBinding()
            .endDescriptorLayoutSet();

        if (shaderPack_ != nullptr) {
            shaderPack_->defineRuntimeResourceDescriptorSet(builder,
                                                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            shaderPack_->defineExecutionDescriptorSet(builder, ShaderPackLoader::Stage::PostRender,
                                                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        }

        descriptorTables_[i] = builder.build(framework->device());

        samplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_REPEAT);
        postPassColorSamplers_[i] =
            vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        postPassDepthSamplers_[i] =
            vk::Sampler::create(framework->device(), VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        postRefractionSourceSamplers_[i] =
            vk::Sampler::create(framework->device(), VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
}

void PostRenderModule::initImages() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t size = framework->swapchain()->imageCount();

    worldPostDepthImages_.resize(size);

    for (int i = 0; i < size; i++) {
        postRenderedImages_[i] = vk::DeviceLocalImage::create(
            device, vma, false, inputWidth_, inputHeight_, 1, outputImages_[i]->vkFormat(),
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        postRefractionSourceImages_[i] = vk::DeviceLocalImage::create(
            device, vma, false, inputWidth_, inputHeight_, 1, postRenderedImages_[i]->vkFormat(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        descriptorTables_[i]->bindImage(firstHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        descriptorTables_[i]->bindSamplerImage(postPassColorSamplers_[i], postRenderedImages_[i],
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3, 0, 0);
        descriptorTables_[i]->bindSamplerImage(postPassDepthSamplers_[i], firstHitDepthImages_[i],
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3, 1, 0);
        descriptorTables_[i]->bindSamplerImage(postPassColorSamplers_[i], hdrImages_[i],
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3, 2, 0);
        descriptorTables_[i]->bindSamplerImage(postPassColorSamplers_[i], motionVectorImages_[i],
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3, 3, 0);
        descriptorTables_[i]->bindSamplerImage(postPassColorSamplers_[i], normalRoughnessImages_[i],
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 3, 4, 0);

        worldPostDepthImages_[i] =
            vk::DeviceLocalImage::create(device, vma, false, inputWidth_, inputHeight_, 1, VK_FORMAT_D32_SFLOAT,
                                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

        if (shaderPack_ != nullptr && shaderPack_->hasRuntimeResources() && shaderPack_->runtimeResourcesReady()) {
            shaderPack_->bindRuntimeResources(descriptorTables_[i], 4, i);
            bindPostRefractionSourceImage(i);
        }
    }
}

float PostRenderModule::sampleUnitFloat(std::mt19937 &randomGenerator) {
    static std::uniform_real_distribution<float> unitDistribution(0.0f, 1.0f);
    return unitDistribution(randomGenerator);
}

glm::vec3 PostRenderModule::sampleUnitSphereDirection(std::mt19937 &randomGenerator) {
    float heightSample = sampleUnitFloat(randomGenerator);
    float angleSample = sampleUnitFloat(randomGenerator);

    float z = 1.0f - 2.0f * heightSample;
    float azimuth = 2.0f * glm::pi<float>() * angleSample;
    float radius = std::sqrt(std::max(0.0f, 1.0f - z * z));

    float x = radius * std::cos(azimuth);
    float y = radius * std::sin(azimuth);
    return glm::vec3(x, y, z);
}

vk::VertexFormat::EntityPBRVertex PostRenderModule::buildStarVertex(glm::vec3 direction, glm::vec4 color) {
    vk::VertexFormat::EntityPBRVertex starVertex{};
    starVertex.pos = direction;
    vk::Vertex::setColorLayer(starVertex, color);
    starVertex.packedData = vk::Vertex::setCoordinate(starVertex.packedData, 2);
    return starVertex;
}

void PostRenderModule::initBuffers() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();

    uint32_t seed = 12345;
    constexpr float starSizeReferenceHeight = 1440.0f;
    const float starResolutionScale =
        std::clamp(starSizeReferenceHeight / static_cast<float>(std::max(inputHeight_, 1u)), 0.25f, 8.0f);

    const uint32_t starCount = std::max(1u, starCount_);
    const float starSizeMin = std::max(0.001f, starSizeMin_ * starResolutionScale);
    const float starSizeMax = std::max(starSizeMin, starSizeMax_ * starResolutionScale);
    const float starRadius = std::max(1.0f, starRadius_);

    std::mt19937 randomGenerator(seed);

    std::vector<vk::VertexFormat::EntityPBRVertex> starVertices;
    starVertices.reserve((size_t)starCount * 6);

    for (uint32_t starIndex = 0; starIndex < starCount; starIndex++) {
        glm::vec3 direction = sampleUnitSphereDirection(randomGenerator);

        float brightnessSelector = sampleUnitFloat(randomGenerator);
        float brightness;
        if (brightnessSelector < 0.35f) {
            brightness = 0.75f + 0.25f * sampleUnitFloat(randomGenerator);
        } else {
            brightness = 0.02f + 0.35f * std::pow(sampleUnitFloat(randomGenerator), 3.0f);
        }

        float tint = sampleUnitFloat(randomGenerator);
        glm::vec3 baseColor;
        if (tint < 0.75f)
            baseColor = glm::vec3(1.0f, 1.0f, 1.0f);
        else if (tint < 0.9f)
            baseColor = glm::vec3(1.0f, 0.95f, 0.85f);
        else
            baseColor = glm::vec3(0.85f, 0.9f, 1.0f);

        glm::vec4 color(baseColor, brightness);

        glm::vec3 center = direction * starRadius;
        const float halfSize = 0.5f * (starSizeMin + sampleUnitFloat(randomGenerator) * (starSizeMax - starSizeMin));

        glm::vec3 tangentReference = (std::abs(direction.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
        glm::vec3 tangentX = glm::normalize(glm::cross(tangentReference, direction));
        glm::vec3 tangentY = glm::cross(direction, tangentX);

        glm::vec3 xOffset = tangentX * halfSize;
        glm::vec3 yOffset = tangentY * halfSize;

        glm::vec3 cornerA = center - xOffset - yOffset;
        glm::vec3 cornerB = center + xOffset - yOffset;
        glm::vec3 cornerC = center + xOffset + yOffset;
        glm::vec3 cornerD = center - xOffset + yOffset;

        starVertices.push_back(buildStarVertex(cornerA, color));
        starVertices.push_back(buildStarVertex(cornerB, color));
        starVertices.push_back(buildStarVertex(cornerC, color));

        starVertices.push_back(buildStarVertex(cornerC, color));
        starVertices.push_back(buildStarVertex(cornerD, color));
        starVertices.push_back(buildStarVertex(cornerA, color));
    }

    starFieldVertexBuffer =
        vk::DeviceLocalBuffer::create(vma, device, starVertices.size() * sizeof(vk::VertexFormat::EntityPBRVertex),
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    starFieldVertexBuffer->uploadToStagingBuffer(starVertices.data());

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
    {
        std::unique_lock<std::recursive_mutex> queueLock(device->mainQueueMutex());
        vk::requireSuccess(vkQueueSubmit(device->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence()),
                           "vkQueueSubmit(star field upload)");
    }
    vk::waitForFence(device->vkDevice(), fence->vkFence(), "vkWaitForFences(star field upload)");
}

void PostRenderModule::initRenderPass() {
    auto device = framework_.lock()->device();
    worldPostColorToDepthRenderPass_ = vk::RenderPassBuilder{}
                                           .beginAttachmentDescription()
                                           .defineAttachmentDescription({
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
}

void PostRenderModule::initFrameBuffers() {
    auto framework = framework_.lock();
    auto device = framework->device();
    uint32_t size = framework->swapchain()->imageCount();

    worldPostColorToDepthFramebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        worldPostColorToDepthFramebuffers_[i] = vk::FramebufferBuilder{}
                                                    .beginAttachment()
                                                    .defineAttachment(worldPostDepthImages_[i])
                                                    .endAttachment()
                                                    .build(device, worldPostColorToDepthRenderPass_);
    }
}

void PostRenderModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    fullScreenVertexShader_ = vk::Shader::create(device, (shaderPath / "full_screen_vert.spv").string());

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
}

std::vector<ExpressionEvaluator::Variable> PostRenderModule::executionExpressionVariables() const {
    return {
        {.name = "RENDER_WIDTH", .value = static_cast<double>(inputWidth_)},
        {.name = "RENDER_HEIGHT", .value = static_cast<double>(inputHeight_)},
    };
}

double PostRenderModule::evaluateNumericExpression(const std::string &expression,
                                                   const PostRenderModule::ExecutionVariables &variables) {
    return shaderPack_->evaluateNumericExpression(ShaderPackLoader::Stage::PostRender, expression, variables,
                                                  executionExpressionVariables(), true);
}

std::optional<std::reference_wrapper<ShaderPackLoader::VariableConfig>>
PostRenderModule::findExecutionVariableConfig(std::string_view name) {
    auto executionIter = executionVariableConfigs_.find(std::string(name));
    if (executionIter != executionVariableConfigs_.end()) { return std::ref(executionIter->second); }
    return std::nullopt;
}

void PostRenderModule::uploadExecutionBuffer(const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                                             PostRenderModuleContext &context,
                                             const std::unordered_map<std::string, ExecutionVariable> &variables) {
    auto frameworkContext = context.frameworkContext.lock();
    auto framework = frameworkContext->framework.lock();
    shaderPack_->uploadExecutionBuffer(ShaderPackLoader::Stage::PostRender, executionBuffer, variables,
                                       frameworkContext->worldCommandBuffer, context.descriptorTable,
                                       shaderPack_->executionSet(4u), framework->physicalDevice()->mainQueueIndex(),
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
}

void PostRenderModule::bindPostRefractionSourceImage(uint32_t frameIndex) {
    if (shaderPack_ == nullptr || !shaderPack_->hasRuntimeResources()) { return; }
    auto runtimeTexture = shaderPack_->findRuntimeTexture(TARGET_POST_SCENE_COLOR);
    if (!runtimeTexture.has_value() || !runtimeTexture->get().config.sampledBinding.has_value()) { return; }
    if (frameIndex >= descriptorTables_.size() || frameIndex >= postRefractionSourceImages_.size() ||
        frameIndex >= postRefractionSourceSamplers_.size()) {
        return;
    }
    if (descriptorTables_[frameIndex] == nullptr || postRefractionSourceImages_[frameIndex] == nullptr ||
        postRefractionSourceSamplers_[frameIndex] == nullptr) {
        return;
    }
    descriptorTables_[frameIndex]->bindSamplerImage(
        postRefractionSourceSamplers_[frameIndex], postRefractionSourceImages_[frameIndex],
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 4, *runtimeTexture->get().config.sampledBinding, 0);
}

std::shared_ptr<vk::DeviceLocalImage> PostRenderModule::findBuiltInImage(const std::string &name, uint32_t frameIndex) {
    if (name == TARGET_LDR) { return postRenderedImages_[frameIndex]; }
    if (name == TARGET_FIRST_HIT_DEPTH) { return firstHitDepthImages_[frameIndex]; }
    if (name == TARGET_HDR) { return hdrImages_[frameIndex]; }
    if (name == TARGET_MOTION_VECTOR) { return motionVectorImages_[frameIndex]; }
    if (name == TARGET_NORMAL_ROUGHNESS) { return normalRoughnessImages_[frameIndex]; }
    if (name == TARGET_POST_SCENE_COLOR) { return postRefractionSourceImages_[frameIndex]; }
    return nullptr;
}

std::shared_ptr<vk::DeviceLocalImage> PostRenderModule::findTargetImage(const std::string &target,
                                                                        uint32_t frameIndex) {
    if (target == TARGET_LDR) { return postRenderedImages_[frameIndex]; }
    if (target == TARGET_FIRST_HIT_DEPTH) { return firstHitDepthImages_[frameIndex]; }
    if (target == TARGET_POST_SCENE_COLOR) { return postRefractionSourceImages_[frameIndex]; }

    if (shaderPack_ != nullptr) {
        if (auto runtimeTexture = shaderPack_->findRuntimeTexture(target); runtimeTexture.has_value()) {
            if (runtimeTexture->get().config.imported) { return nullptr; }
            return shaderPack_->findRuntimeVKTexture(runtimeTexture->get(), frameIndex);
        }
    }
    return nullptr;
}

void PostRenderModule::ensureDynamicPipelines() {
    MCVR_PROFILE_SCOPE("native.PostRenderModule.ensureDynamicPipelines");
    if (isDynamicPipelinesReady_) { return; }
    if (shaderPack_ == nullptr || !shaderPack_->runtimeResourcesReady()) { return; }

    auto framework = framework_.lock();
    auto device = framework->device();
    uint32_t frameCount = framework->swapchain()->imageCount();
    const auto &shaderPack = shaderPack_->shaderPack();
    const uint32_t executionSet = shaderPack_->executionSet(4u);
    const size_t executionBufferSize = shaderPack.postRenderExecution.variables.size() * sizeof(float);
    const std::unordered_set<std::string> activePassNames =
        shaderPack_->activePassNames(ShaderPackLoader::Stage::PostRender, true);

    fullScreenPasses_.clear();
    passNameToPass_.clear();
    renderPasses_.clear();
    renderPassNameToPass_.clear();
    compiledExecutionProgram_ = nullptr;
    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        if (shaderPack_->hasRuntimeResources()) {
            shaderPack_->bindRuntimeResources(descriptorTables_[frameIndex], 4, frameIndex);
        }
    }

    struct ShaderAssignment {
        enum class Type {
            FullScreenFragment,
            RenderVertex,
            RenderFragment,
        };
        Type type;
        size_t index;
        std::string variantName;
    };

    std::vector<ShaderPack::ShaderCreateInfo> shaderRequests;
    std::vector<ShaderAssignment> shaderAssignments;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> fullScreenTargetImages;

    auto hasOutputImage = [](const ShaderPackLoader::ResourceList &resources, std::string_view name) {
        return std::find(resources.images.begin(), resources.images.end(), std::string(name)) != resources.images.end();
    };

    auto colorBlendAttachment = [](bool enableBlend) {
        VkPipelineColorBlendAttachmentState state{
            .blendEnable = enableBlend ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
        };
        if (!enableBlend) {
            state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        }
        return state;
    };

    auto firstHitDepthBlendAttachment = []() {
        return VkPipelineColorBlendAttachmentState{
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT,
        };
    };

    for (const auto &passConfig : shaderPack.passes) {
        if (passConfig.stage != ShaderPackLoader::Stage::PostRender) { continue; }
        const std::string *passName = nullptr;
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen: passName = &passConfig.fullScreen.name; break;
            case ShaderPackLoader::PassConfig::Type::Render: passName = &passConfig.render.name; break;
            case ShaderPackLoader::PassConfig::Type::RayTracing: passName = &passConfig.rayTracing.name; break;
            case ShaderPackLoader::PassConfig::Type::Compute: passName = &passConfig.compute.name; break;
        }
        if (passName != nullptr && activePassNames.find(*passName) == activePassNames.end()) { continue; }
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::FullScreen: {
                if (passConfig.fullScreen.outputs.images.size() != 1 ||
                    passConfig.fullScreen.outputs.images[0] != passConfig.fullScreen.target) {
                    throw std::runtime_error("post_render full_screen pass must only write to its target: " +
                                             passConfig.fullScreen.name);
                }

                auto targetImage = findTargetImage(passConfig.fullScreen.target, 0);
                if (targetImage == nullptr) {
                    throw std::runtime_error("invalid post_render target: " + passConfig.fullScreen.target);
                }
                uint32_t targetViewIndex = 0;
                if (targetImage->layer() > 1) {
                    if (!passConfig.fullScreen.layer.has_value()) {
                        throw std::runtime_error("post_render full_screen array target requires layer: " +
                                                 passConfig.fullScreen.name);
                    }
                    if (*passConfig.fullScreen.layer >= targetImage->layer()) {
                        throw std::runtime_error("post_render full_screen layer out of range: " +
                                                 passConfig.fullScreen.name);
                    }
                    targetViewIndex = *passConfig.fullScreen.layer + 1;
                }

                auto pass = std::make_shared<FullScreenPass>();
                pass->config = passConfig.fullScreen;
                if (executionBufferSize > 0) {
                    pass->executionBuffer =
                        ShaderPack::createPassExecutionBuffer(device, framework->vma(), executionBufferSize);
                }
                pass->renderPass = vk::RenderPassBuilder{}
                                       .beginAttachmentDescription()
                                       .defineAttachmentDescription({
                                           .format = targetImage->vkFormat(),
                                           .samples = VK_SAMPLE_COUNT_1_BIT,
                                           .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                           .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                           .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                           .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                           .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
                pass->framebuffers.resize(frameCount);
                for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
                    pass->framebuffers[frameIndex].push_back(
                        vk::FramebufferBuilder{}
                            .beginAttachment()
                            .defineAttachment(findTargetImage(pass->config.target, frameIndex), targetViewIndex)
                            .endAttachment()
                            .build(device, pass->renderPass));
                }
                shaderAssignments.push_back({ShaderAssignment::Type::FullScreenFragment, fullScreenPasses_.size()});
                shaderRequests.push_back({
                    .path = pass->config.fragmentShaderPath,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .definitions = pass->config.definitions,
                    .executionStage = ShaderPackLoader::Stage::PostRender,
                    .executionSet = executionSet,
                });
                fullScreenTargetImages.push_back(targetImage);
                fullScreenPasses_.push_back(pass);
                break;
            }
            case ShaderPackLoader::PassConfig::Type::Render: {
                if (!hasOutputImage(passConfig.render.outputs, TARGET_LDR)) {
                    throw std::runtime_error("post_render render pass must output out:ldr: " + passConfig.render.name);
                }
                for (const auto &output : passConfig.render.outputs.images) {
                    if (output != TARGET_LDR && output != TARGET_FIRST_HIT_DEPTH) {
                        throw std::runtime_error("post_render render pass only supports out:ldr and "
                                                 "out:first_hit_depth outputs: " +
                                                 passConfig.render.name);
                    }
                }

                auto pass = std::make_shared<RenderPass>();
                pass->config = passConfig.render;
                pass->target = parseRenderContent(pass->config.content);
                pass->writesFirstHitDepth = hasOutputImage(pass->config.outputs, TARGET_FIRST_HIT_DEPTH);
                if (executionBufferSize > 0) {
                    pass->executionBuffer =
                        ShaderPack::createPassExecutionBuffer(device, framework->vma(), executionBufferSize);
                }

                vk::RenderPassBuilder renderPassBuilder;
                auto &attachmentBuilder = renderPassBuilder.beginAttachmentDescription();
                attachmentBuilder.defineAttachmentDescription({
                    .format = postRenderedImages_[0]->vkFormat(),
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });
                if (pass->writesFirstHitDepth) {
                    attachmentBuilder.defineAttachmentDescription({
                        .format = firstHitDepthImages_[0]->vkFormat(),
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                        .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    });
                }
                attachmentBuilder.defineAttachmentDescription({
                    .format = worldPostDepthImages_[0]->vkFormat(),
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                });
                attachmentBuilder.endAttachmentDescription();
                auto &referenceBuilder = renderPassBuilder.beginAttachmentReference().defineAttachmentReference({
                    .attachment = 0,
                    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });
                if (pass->writesFirstHitDepth) {
                    referenceBuilder.defineAttachmentReference({
                        .attachment = 1,
                        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    });
                }
                const uint32_t depthAttachmentIndex = pass->writesFirstHitDepth ? 2u : 1u;
                pass->renderPass =
                    referenceBuilder
                        .defineAttachmentReference({
                            .attachment = depthAttachmentIndex,
                            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        })
                        .endAttachmentReference()
                        .beginSubpassDescription()
                        .defineSubpassDescription({
                            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                            .colorAttachmentIndices =
                                pass->writesFirstHitDepth ? std::vector<uint32_t>{0, 1} : std::vector<uint32_t>{0},
                            .depthStencilAttachmentIndex = depthAttachmentIndex,
                        })
                        .endSubpassDescription()
                        .build(device);

                pass->framebuffers.resize(frameCount);
                for (uint32_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
                    vk::FramebufferBuilder framebufferBuilder;
                    auto &attachments =
                        framebufferBuilder.beginAttachment().defineAttachment(postRenderedImages_[frameIndex]);
                    if (pass->writesFirstHitDepth) { attachments.defineAttachment(firstHitDepthImages_[frameIndex]); }
                    pass->framebuffers[frameIndex] = attachments.defineAttachment(worldPostDepthImages_[frameIndex])
                                                         .endAttachment()
                                                         .build(device, pass->renderPass);
                }

                for (const auto &[variantName, pipelineConfig] : pass->config.pipelineConfigs) {
                    pass->shaderVariants[variantName].config = pipelineConfig;
                    shaderAssignments.push_back({
                        ShaderAssignment::Type::RenderVertex,
                        renderPasses_.size(),
                        variantName,
                    });
                    shaderRequests.push_back({
                        .path = pipelineConfig.shader.vertexShaderPath,
                        .stage = VK_SHADER_STAGE_VERTEX_BIT,
                        .definitions = pass->config.definitions,
                        .executionStage = ShaderPackLoader::Stage::PostRender,
                        .executionSet = executionSet,
                    });
                    shaderAssignments.push_back({
                        ShaderAssignment::Type::RenderFragment,
                        renderPasses_.size(),
                        variantName,
                    });
                    shaderRequests.push_back({
                        .path = pipelineConfig.shader.fragmentShaderPath,
                        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                        .definitions = pass->config.definitions,
                        .executionStage = ShaderPackLoader::Stage::PostRender,
                        .executionSet = executionSet,
                    });
                }
                renderPasses_.push_back(pass);
                break;
            }
            case ShaderPackLoader::PassConfig::Type::RayTracing:
            case ShaderPackLoader::PassConfig::Type::Compute:
                throw std::runtime_error("post_render stage only supports full_screen and render passes");
        }
    }

    auto shaders = shaderPack_->createShaders(device, shaderRequests);

    for (size_t shaderIndex = 0; shaderIndex < shaders.size(); shaderIndex++) {
        const auto &assignment = shaderAssignments[shaderIndex];
        switch (assignment.type) {
            case ShaderAssignment::Type::FullScreenFragment:
                fullScreenPasses_[assignment.index]->fragmentShaders.push_back(shaders[shaderIndex]);
                break;
            case ShaderAssignment::Type::RenderVertex:
                renderPasses_[assignment.index]->shaderVariants[assignment.variantName].vertexShader =
                    shaders[shaderIndex];
                break;
            case ShaderAssignment::Type::RenderFragment:
                renderPasses_[assignment.index]->shaderVariants[assignment.variantName].fragmentShader =
                    shaders[shaderIndex];
                break;
        }
    }

    mcvr::parallelFor(
        fullScreenPasses_.size(),
        [&](size_t passIndex) {
            auto &pass = fullScreenPasses_[passIndex];
            auto &targetImage = fullScreenTargetImages[passIndex];
            std::vector<std::shared_ptr<vk::GraphicsPipeline>> pipelines;
            pipelines.push_back(
                vk::GraphicsPipelineBuilder{}
                    .defineRenderPass(pass->renderPass, 0)
                    .beginShaderStage()
                    .defineShaderStage(fullScreenVertexShader_, VK_SHADER_STAGE_VERTEX_BIT)
                    .defineShaderStage(pass->fragmentShaders[0], VK_SHADER_STAGE_FRAGMENT_BIT)
                    .endShaderStage()
                    .defineVertexInputState<void>()
                    .defineViewportScissorState({
                        .viewport =
                            {
                                .x = 0.0f,
                                .y = 0.0f,
                                .width = static_cast<float>(targetImage->width()),
                                .height = static_cast<float>(targetImage->height()),
                                .minDepth = 0.0f,
                                .maxDepth = 1.0f,
                            },
                        .scissor =
                            {
                                .offset = {.x = 0, .y = 0},
                                .extent = {.width = targetImage->width(), .height = targetImage->height()},
                            },
                    })
                    .beginColorBlendAttachmentState()
                    .defineDefaultColorBlendAttachmentState()
                    .endColorBlendAttachmentState()
                    .definePipelineLayout(descriptorTables_[0])
                    .build(device));
            pass->pipelines = std::move(pipelines);
        },
        "MCVR_POST_PIPELINE_BUILD_THREADS");
    mcvr::parallelFor(
        renderPasses_.size(),
        [&](size_t passIndex) {
            auto &pass = renderPasses_[passIndex];
            for (auto &[variantName, variant] : pass->shaderVariants) {
                const ShaderPackLoader::RenderPipelineConfig &pipelineConfig = variant.config;
                const bool depthTest = pipelineConfig.depthTest;
                const bool depthWrite = pipelineConfig.depthWrite;
                const VkCompareOp depthCompare = parseDepthCompare(pipelineConfig.depthCompare);
                const bool colorBlend = pipelineConfig.colorBlend;
                const VkCullModeFlags cullMode =
                    parseCullMode(pipelineConfig.cull, pass->config.name + "." + variantName);
                const VkFrontFace frontFace =
                    parseFrontFace(pipelineConfig.frontFace, pass->config.name + "." + variantName);
                vk::GraphicsPipelineBuilder pipelineBuilder;
                auto &colorBlendBuilder =
                    pipelineBuilder.defineRenderPass(pass->renderPass, 0)
                        .beginShaderStage()
                        .defineShaderStage(variant.vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
                        .defineShaderStage(variant.fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
                        .endShaderStage()
                        .defineVertexInputState<vk::VertexFormat::EntityPBRVertex>()
                        .defineViewportScissorState({
                            .viewport =
                                {
                                    .x = 0.0f,
                                    .y = 0.0f,
                                    .width = static_cast<float>(postRenderedImages_[0]->width()),
                                    .height = static_cast<float>(postRenderedImages_[0]->height()),
                                    .minDepth = 0.0f,
                                    .maxDepth = 1.0f,
                                },
                            .scissor =
                                {
                                    .offset = {.x = 0, .y = 0},
                                    .extent = {.width = postRenderedImages_[0]->width(),
                                               .height = postRenderedImages_[0]->height()},
                                },
                        })
                        .defineDepthStencilState({
                            .depthTestEnable = depthTest ? VK_TRUE : VK_FALSE,
                            .depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE,
                            .depthCompareOp = depthCompare,
                            .depthBoundsTestEnable = VK_FALSE,
                            .stencilTestEnable = VK_FALSE,
                            .minDepthBounds = 0.0f,
                            .maxDepthBounds = 1.0f,
                        })
                        .defineRasterizationState(VkPipelineRasterizationStateCreateInfo{
                            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                            .depthClampEnable = VK_FALSE,
                            .rasterizerDiscardEnable = VK_FALSE,
                            .polygonMode = VK_POLYGON_MODE_FILL,
                            .cullMode = cullMode,
                            .frontFace = frontFace,
                            .depthBiasEnable = VK_FALSE,
                            .depthBiasConstantFactor = 0.0f,
                            .depthBiasClamp = 0.0f,
                            .depthBiasSlopeFactor = 0.0f,
                            .lineWidth = 1.0f,
                        })
                        .beginColorBlendAttachmentState()
                        .defineColorBlendAttachmentState(colorBlendAttachment(colorBlend));
                if (pass->writesFirstHitDepth) {
                    colorBlendBuilder.defineColorBlendAttachmentState(firstHitDepthBlendAttachment());
                }
                variant.pipeline = colorBlendBuilder.endColorBlendAttachmentState()
                                       .definePipelineLayout(descriptorTables_[0])
                                       .build(device);
            }
        },
        "MCVR_POST_PIPELINE_BUILD_THREADS");
    for (size_t passIndex = 0; passIndex < fullScreenPasses_.size(); passIndex++) {
        auto &pass = fullScreenPasses_[passIndex];
        if (!passNameToPass_.emplace(pass->config.name, pass).second) {
            throw std::runtime_error("duplicate post_render pass name: " + pass->config.name);
        }
    }

    for (size_t passIndex = 0; passIndex < renderPasses_.size(); passIndex++) {
        auto &pass = renderPasses_[passIndex];
        if (!renderPassNameToPass_.emplace(pass->config.name, pass).second ||
            passNameToPass_.find(pass->config.name) != passNameToPass_.end()) {
            throw std::runtime_error("duplicate post_render pass name: " + pass->config.name);
        }
    }

    compileExecutionProgram();
    isDynamicPipelinesReady_ = true;
}

void PostRenderModule::compileExecutionProgram() {
    compiledExecutionProgram_ = nullptr;
    if (shaderPack_ == nullptr) { return; }

    auto passExecutionOpBuilder = [this](const std::string &passName) -> std::unique_ptr<ShaderPack::ExecutionOp> {
        if (auto passIter = passNameToPass_.find(passName); passIter != passNameToPass_.end()) {
            return std::make_unique<PostRenderPassExecutionOp>(PostRenderPassVariant(passIter->second));
        }
        if (auto passIter = renderPassNameToPass_.find(passName); passIter != renderPassNameToPass_.end()) {
            return std::make_unique<PostRenderPassExecutionOp>(PostRenderPassVariant(passIter->second));
        }
        return std::make_unique<ShaderPack::SequentialOp>();
    };

    compiledExecutionProgram_ =
        shaderPack_->compileExecutionProgram(ShaderPackLoader::Stage::PostRender, true, passExecutionOpBuilder);
}

PostRenderModuleContext::PostRenderModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<PostRenderModule> postRenderModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      postRenderModule(postRenderModule),
      ldrImage(postRenderModule->ldrImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(postRenderModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      hdrImage(postRenderModule->hdrImages_[frameworkContext->frameIndex]),
      motionVectorImage(postRenderModule->motionVectorImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(postRenderModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      segmentationMaskImage(nullptr),
      worldPostDepthImage(postRenderModule->worldPostDepthImages_[frameworkContext->frameIndex]),
      descriptorTable(postRenderModule->descriptorTables_[frameworkContext->frameIndex]),
      worldPostColorToDepthFramebuffer(
          postRenderModule->worldPostColorToDepthFramebuffers_[frameworkContext->frameIndex]),
      postRenderedImage(postRenderModule->postRenderedImages_[frameworkContext->frameIndex]),
      outputImage(postRenderModule->outputImages_[frameworkContext->frameIndex]) {}

void PostRenderModuleContext::render() {
    MCVR_PROFILE_SCOPE("native.PostRenderModuleContext.render");
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = postRenderModule.lock();
    if (!module) { return; }
    const uint32_t frameIndex = context->frameIndex;

    segmentationMaskImage = nullptr;
    std::shared_ptr<MainRenderModuleContext> mainRenderContext;
    if (auto pipelineContext = worldPipelineContext.lock(); pipelineContext != nullptr) {
        for (const auto &worldModuleContext : pipelineContext->worldModuleContexts) {
            auto candidate = std::dynamic_pointer_cast<MainRenderModuleContext>(worldModuleContext);
            if (candidate == nullptr) { continue; }
            mainRenderContext = std::move(candidate);
            segmentationMaskImage = mainRenderContext->segmentationMaskImage;
            break;
        }
    }

    auto buffers = Renderer::instance().buffers();
    auto chooseSrc = [](VkImageLayout oldLayout, VkPipelineStageFlags2 fallbackStage, VkAccessFlags2 fallbackAccess,
                        VkPipelineStageFlags2 &outStage, VkAccessFlags2 &outAccess) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            outStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            outAccess = 0;
        } else {
            outStage = fallbackStage;
            outAccess = fallbackAccess;
        }
    };

    auto ensureLayout = [&](const std::shared_ptr<vk::DeviceLocalImage> &img, VkImageLayout targetLayout,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
        if (!img || img->imageLayout() == targetLayout) return;
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(img->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        worldCommandBuffer->barriersBufferImage({}, {{.srcStageMask = srcStage,
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

    if (segmentationMaskImage != nullptr) {
        descriptorTable->bindImage(segmentationMaskImage, VK_IMAGE_LAYOUT_GENERAL, segmentationMaskDescriptorSet,
                                   segmentationMaskDescriptorBinding);

        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(segmentationMaskImage->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        worldCommandBuffer->barriersBufferImage({}, {{.srcStageMask = srcStage,
                                                      .srcAccessMask = srcAccess,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                      .oldLayout = segmentationMaskImage->imageLayout(),
                                                      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                      .srcQueueFamilyIndex = mainQueueIndex,
                                                      .dstQueueFamilyIndex = mainQueueIndex,
                                                      .image = segmentationMaskImage,
                                                      .subresourceRange = vk::wholeColorSubresourceRange}});
        segmentationMaskImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        if (mainRenderContext != nullptr) {
            auto &trackedState = mainRenderContext->trackedImageStates[segmentationMaskImage.get()];
            trackedState.layout = VK_IMAGE_LAYOUT_GENERAL;
            trackedState.stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            trackedState.accessMask = VK_ACCESS_2_SHADER_READ_BIT;
            trackedState.initialized = true;
        }
    }

    auto blitColorImage = [&](const std::shared_ptr<vk::DeviceLocalImage> &source,
                              const std::shared_ptr<vk::DeviceLocalImage> &target) {
        if (!source || !target || source == target) { return; }

        VkPipelineStageFlags2 srcStageSource = 0;
        VkAccessFlags2 srcAccessSource = 0;
        chooseSrc(source->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStageSource, srcAccessSource);

        VkPipelineStageFlags2 srcStageTarget = 0;
        VkAccessFlags2 srcAccessTarget = 0;
        chooseSrc(target->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStageTarget, srcAccessTarget);

        worldCommandBuffer->barriersBufferImage({}, {{
                                                         .srcStageMask = srcStageSource,
                                                         .srcAccessMask = srcAccessSource,
                                                         .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                         .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                                                         .oldLayout = source->imageLayout(),
                                                         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                         .srcQueueFamilyIndex = mainQueueIndex,
                                                         .dstQueueFamilyIndex = mainQueueIndex,
                                                         .image = source,
                                                         .subresourceRange = vk::wholeColorSubresourceRange,
                                                     },
                                                     {
                                                         .srcStageMask = srcStageTarget,
                                                         .srcAccessMask = srcAccessTarget,
                                                         .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                         .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                         .oldLayout = target->imageLayout(),
                                                         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                         .srcQueueFamilyIndex = mainQueueIndex,
                                                         .dstQueueFamilyIndex = mainQueueIndex,
                                                         .image = target,
                                                         .subresourceRange = vk::wholeColorSubresourceRange,
                                                     }});
        source->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        target->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageBlit imageBlit{};
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.mipLevel = 0;
        imageBlit.srcSubresource.baseArrayLayer = 0;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcOffsets[0] = {0, 0, 0};
        imageBlit.srcOffsets[1] = {static_cast<int32_t>(source->width()), static_cast<int32_t>(source->height()), 1};
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.mipLevel = 0;
        imageBlit.dstSubresource.baseArrayLayer = 0;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstOffsets[0] = {0, 0, 0};
        imageBlit.dstOffsets[1] = {static_cast<int32_t>(target->width()), static_cast<int32_t>(target->height()), 1};

        vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), source->vkImage(), source->imageLayout(),
                       target->vkImage(), target->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
    };

    ensureLayout(firstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    if (worldPostDepthImage && worldPostDepthImage->imageLayout() != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(worldPostDepthImage->imageLayout(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        worldCommandBuffer->barriersBufferImage({}, {{.srcStageMask = srcStage,
                                                      .srcAccessMask = srcAccess,
                                                      .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                                                      .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                                                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                                      .oldLayout = worldPostDepthImage->imageLayout(),
                                                      .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                                      .srcQueueFamilyIndex = mainQueueIndex,
                                                      .dstQueueFamilyIndex = mainQueueIndex,
                                                      .image = worldPostDepthImage,
                                                      .subresourceRange = vk::wholeDepthSubresourceRange}});
        worldPostDepthImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    descriptorTable->bindBuffer(buffers->worldUniformBuffer(), 1, 0);
    descriptorTable->bindBuffer(buffers->skyUniformBuffer(), 1, 1);
    descriptorTable->bindBuffer(Renderer::instance().buffers()->textureMappingBuffer(), 2, 0);

    {
        VkPipelineStageFlags2 srcStageDepth = 0;
        VkAccessFlags2 srcAccessDepth = 0;
        chooseSrc(worldPostDepthImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStageDepth, srcAccessDepth);

        VkPipelineStageFlags2 srcStageFirstHit = 0;
        VkAccessFlags2 srcAccessFirstHit = 0;
        chooseSrc(firstHitDepthImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStageFirstHit, srcAccessFirstHit);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = srcStageDepth,
                     .srcAccessMask = srcAccessDepth,
                     .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = worldPostDepthImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = worldPostDepthImage,
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
    worldPostDepthImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->worldPostColorToDepthRenderPass_,
        .framebuffer = worldPostColorToDepthFramebuffer,
        .renderAreaExtent = {worldPostDepthImage->width(), worldPostDepthImage->height()},
        .clearValues = {{.depthStencil = {.depth = 1.0f}}},
    });
    worldPostDepthImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->bindGraphicsPipeline(module->worldPostColorToDepthPipeline_)
        ->draw(3, 1)
        ->endRenderPass();
    worldPostDepthImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    blitColorImage(ldrImage, postRenderedImage);

    module->ensureDynamicPipelines();
    if (module->shaderPack_ != nullptr && module->shaderPack_->hasRuntimeResources()) {
        MCVR_PROFILE_SCOPE("native.PostRenderModule.bindRuntimeResources");
        module->shaderPack_->bindRuntimeResources(descriptorTable, 4, context->frameIndex);
        module->bindPostRefractionSourceImage(context->frameIndex);
    }
    if (module->isDynamicPipelinesReady_) {
        PostRenderModule::ExecutionVariables variables;
        variables.reserve(module->globalVariables_.size());
        for (const auto &[name, value] : module->globalVariables_) {
            auto config = module->findExecutionVariableConfig(name);
            variables.emplace(name, PostRenderModule::ExecutionVariable{
                                        .name = name,
                                        .value = value,
                                        .type = config.has_value() ? config->get().type : "",
                                    });
        }

        if (module->compiledExecutionProgram_ != nullptr) {
            MCVR_PROFILE_SCOPE("native.PostRenderModule.executeProgram");
            PostRenderModule::PostRenderExecutionRuntime runtime;
            runtime.executionVariables = std::move(variables);
            runtime.runtimeExpressionVariables = module->executionExpressionVariables();
            runtime.postRenderModule = module;
            runtime.postRenderContext = shared_from_this();
            (*module->compiledExecutionProgram_)(runtime);
            variables = std::move(runtime.executionVariables);
        } else if (!module->shaderPack_->execution(ShaderPackLoader::Stage::PostRender).commands.empty()) {
            throw std::runtime_error("post_render execution program was not compiled");
        }
        for (auto &[name, value] : module->globalVariables_) {
            auto iter = variables.find(name);
            if (iter != variables.end()) { value = iter->second.value; }
        }
    }

    blitColorImage(postRenderedImage, outputImage);

    auto finalizePostRenderTarget = [&]() {
        VkPipelineStageFlags2 srcStagePost = 0;
        VkAccessFlags2 srcAccessPost = 0;
        chooseSrc(outputImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStagePost, srcAccessPost);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                    .srcStageMask = srcStagePost,
                    .srcAccessMask = srcAccessPost,
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
    };
    finalizePostRenderTarget();
}
