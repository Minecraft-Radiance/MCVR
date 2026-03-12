#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct TemporalAccumulationPushConstant {
    float alpha;
    float threshold;
};

struct TemporalAccumulationModuleContext;

class TemporalAccumulationModule : public WorldModule, public SharedObject<TemporalAccumulationModule> {
    friend TemporalAccumulationModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.temporal_accumulation.name";
    constexpr static uint32_t inputImageNum = 3;
    constexpr static uint32_t outputImageNum = 1;

    TemporalAccumulationModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;
    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

    StereoMode stereoMode() const override { return StereoMode::SingleInstanceMultiDispatch; }

  private:
    void initDescriptorTables();
    void initImages();
    void initRenderPass();
    void initFrameBuffers();
    void initPipeline();

  private:
    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrNoisyImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;

    // accumulation
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::shared_ptr<vk::Shader> vertShader_;
    std::shared_ptr<vk::Shader> fragShader_;
    std::shared_ptr<vk::RenderPass> renderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> pipeline_;

    std::shared_ptr<vk::DeviceLocalImage> accumulatedRadianceImage_;
    std::shared_ptr<vk::DeviceLocalImage> accumulatedNormalImage_;
    std::shared_ptr<vk::Sampler> sampler_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> accumulatedNormalOutImages_;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> accumulatedRadianceOutImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t width_, height_;

    float alpha_ = 0.12;
    float threshould_ = 0.9;
};

struct TemporalAccumulationModuleContext : public WorldModuleContext, SharedObject<TemporalAccumulationModuleContext> {
    std::weak_ptr<TemporalAccumulationModule> temporalAccumulationModule;

    // input
    std::shared_ptr<vk::DeviceLocalImage> hdrNoisyImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;

    // accumulation
    std::shared_ptr<vk::DescriptorTable> descriptorTable;
    std::shared_ptr<vk::Framebuffer> framebuffer;

    std::shared_ptr<vk::DeviceLocalImage> accumulatedRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> accumulatedNormalImage;
    std::shared_ptr<vk::DeviceLocalImage> accumulatedNormalOutImage;

    // output
    std::shared_ptr<vk::DeviceLocalImage> accumulatedRadianceOutImage;

    TemporalAccumulationModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                      std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                      std::shared_ptr<TemporalAccumulationModule> temporalAccumulationModule);

    StereoMode stereoMode() const override { return StereoMode::SingleInstanceMultiDispatch; }
    void render() override;
    void renderEye(uint32_t eyeIndex) override;
};