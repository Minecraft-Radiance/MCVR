#include "core/render/pipeline.hpp"

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include "core/render/modules/ui_module.hpp"
#include "core/render/modules/world/dlss/dlss_module.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr3_upscaler_module.hpp"
#include "core/render/modules/world/nrd/nrd_module.hpp"
#include "core/render/modules/world/post_render/post_render_module.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/modules/world/temporal_accumulation/temporal_accumulation_module.hpp"
#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"
#include "core/render/modules/world/xess_upscaler/xess_sr_module.hpp"

#include <cstdlib>
#include <iomanip>
#include <optional>
#include <set>

WorldPipelineBlueprint::WorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    std::set<uint32_t> imageIndices;
    auto framework = Renderer::instance().framework();
    auto pipeline = framework->pipeline();

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

    worldModules_.resize(blueprint->moduleNames_.size());
    sharedImages_.resize(frameNum,
                         std::vector<std::shared_ptr<vk::DeviceLocalImage>>(blueprint->imageFormats_.size(), nullptr));
    contexts_.resize(frameNum);

    VkExtent2D extent = framework->swapchain()->vkExtent();
    auto inferModuleExtent = [&](int frameIndex,
                                 const std::vector<uint32_t> &outputIndices,
                                 const std::vector<uint32_t> &inputIndices) -> std::pair<uint32_t, uint32_t> {
        auto tryIndices = [&](const std::vector<uint32_t> &indices) -> std::optional<std::pair<uint32_t, uint32_t>> {
            for (uint32_t idx : indices) {
                if (idx >= sharedImages_[frameIndex].size()) continue;
                auto &img = sharedImages_[frameIndex][idx];
                if (img) return {{img->width(), img->height()}};
            }
            return std::nullopt;
        };

        if (auto size = tryIndices(outputIndices)) return *size;
        if (auto size = tryIndices(inputIndices)) return *size;
        return {extent.width, extent.height};
    };
    auto createImageAt = [&](int frameIndex, uint32_t idx, uint32_t width, uint32_t height, VkFormat format) {
        if (sharedImages_[frameIndex][idx]) return;
        sharedImages_[frameIndex][idx] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width, height, 1, format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#ifdef USE_AMD
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
        );
    };

    for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
        // Keep the primary output at display resolution
        sharedImages_[frameIndex][0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, extent.width, extent.height, 1, blueprint->imageFormats_[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#ifdef USE_AMD
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
        );
    }

    for (int i = blueprint->moduleNames_.size() - 1; i >= 0; i--) {
        worldModules_[i] = Pipeline::worldModuleConstructors[blueprint->moduleNames_[i]](framework, shared_from_this());
        worldModules_[i]->setAttributes(blueprint->attributeCounts_[i], blueprint->attributeKVs_[i]);

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
                    auto [w, h] = inferModuleExtent(frameIndex, moduleOutputIndices, moduleInputIndices);
                    for (int j = 0; j < moduleOutputIndices.size(); j++) {
                        createImageAt(frameIndex, moduleOutputIndices[j], w, h, blueprint->imageFormats_[moduleOutputIndices[j]]);
                        outputImages[j] = sharedImages_[frameIndex][moduleOutputIndices[j]];
                    }

                    result = worldModules_[i]->setOrCreateOutputImages(outputImages, outputFormats, frameIndex);
                    if (!result) {
                        std::cerr << "[WorldPipeline] Output image setup failed after fallback for module: "
                                  << blueprint->moduleNames_[i] << std::endl;
                        throw std::runtime_error("Output image not set properly");
                    }
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
                if (!result) {
                    auto [w, h] = inferModuleExtent(frameIndex, moduleOutputIndices, moduleInputIndices);
                    for (int j = 0; j < moduleInputIndices.size(); j++) {
                        createImageAt(frameIndex, moduleInputIndices[j], w, h, blueprint->imageFormats_[moduleInputIndices[j]]);
                        inputImages[j] = sharedImages_[frameIndex][moduleInputIndices[j]];
                    }
                    result = worldModules_[i]->setOrCreateInputImages(inputImages, inputFormats, frameIndex);
                    if (!result) {
                        std::cerr << "[WorldPipeline] Input image setup failed after fallback for module: "
                                  << blueprint->moduleNames_[i] << std::endl;
                        throw std::runtime_error("Input image not set properly");
                    }
                }
                for (int j = 0; j < moduleInputIndices.size(); j++) {
                    sharedImages_[frameIndex][moduleInputIndices[j]] = inputImages[j];
                }
            }
        }

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

    for (int i = 0; i < worldModuleContexts.size(); i++) { worldModuleContexts[i]->render(); }

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
        FSR3UpscalerModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return FSR3UpscalerModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
        FSR3UpscalerModule::NAME, std::make_pair(FSR3UpscalerModule::inputImageNum, FSR3UpscalerModule::outputImageNum)));
    
    worldModuleConstructors.insert(std::make_pair(
        XessSrModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return XessSrModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
        XessSrModule::NAME, std::make_pair(XessSrModule::inputImageNum, XessSrModule::outputImageNum)));


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

    // TODO: add to command buffer
    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(worldPipelineContext->outputImage->width()),
                               static_cast<int>(worldPipelineContext->outputImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(uiModuleContext->overlayDrawColorImage->width()),
                               static_cast<int>(uiModuleContext->overlayDrawColorImage->height()), 1};

    vkCmdBlitImage(overlayCommandBuffer->vkCommandBuffer(), worldPipelineContext->outputImage->vkImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, uiModuleContext->overlayDrawColorImage->vkImage(),
                   uiModuleContext->overlayDrawColorImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

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
