#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"
#include <chrono>

#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct ToneMappingModuleContext;

struct ToneMappingModuleExposureData {
    float exposure;
    float avgLogLum;
    float padding0;
    float padding1;
};

struct ToneMappingModulePushConstant {
    float log2Min;     // 例如 -12
    float log2Max;     // 例如 +4
    float epsilon;     // 例如 1e-6
    float lowPercent;  // 例如 0.005 (0.5%)
    float highPercent; // 例如 0.99  (99%)
    float middleGrey;  // 例如 0.18
    float dt;          // 本帧 delta time（秒）
    float speedUp;     // 变亮适应速度（1/秒），例如 3.0
    float speedDown;   // 变暗适应速度（1/秒），例如 1.0
    float minExposure; // 可选 clamp，例如 0.0001
    float maxExposure; // 可选 clamp，例如 10000.0
};

class ToneMappingModule : public WorldModule, public SharedObject<ToneMappingModule> {
    friend ToneMappingModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.tone_mapping.name";
    constexpr static uint32_t inputImageNum = 1;
    constexpr static uint32_t outputImageNum = 1;

    ToneMappingModule();

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

    StereoMode stereoMode() const override { return StereoMode::SingleInstance3DDispatch; }

  private:
    static constexpr uint32_t histSize = 256;

    void initDescriptorTables();
    void initImages();
    void initBuffers();
    void initRenderPass();
    void initFrameBuffers();
    void initPipeline();

  private:
    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;

    // tone mapping
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> histBuffers_;
    std::shared_ptr<vk::DeviceLocalBuffer> exposureData_;

    std::shared_ptr<vk::Shader> histShader_;
    std::shared_ptr<vk::ComputePipeline> histPipeline_;

    std::shared_ptr<vk::Shader> exposureShader_;
    std::shared_ptr<vk::ComputePipeline> exposurePipeline_;

    std::shared_ptr<vk::Shader> vertShader_;
    std::shared_ptr<vk::Shader> fragShader_;
    std::shared_ptr<vk::RenderPass> renderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> pipeline_;
    std::vector<std::shared_ptr<vk::Sampler>> samplers_;

    float middleGrey_ = 0.18;
    float speedUp_ = 3.0;
    float speedDown_ = 3.0;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> ldrImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    std::chrono::time_point<std::chrono::high_resolution_clock> lastTimePoint_;

    uint32_t width_, height_;
};

struct ToneMappingModuleContext : public WorldModuleContext, SharedObject<ToneMappingModuleContext> {
    std::weak_ptr<ToneMappingModule> toneMappingModule;

    StereoMode stereoMode() const override { return StereoMode::SingleInstance3DDispatch; }
    void render3D(uint32_t eyeCount) override;

    // input
    std::shared_ptr<vk::DeviceLocalImage> hdrImage;

    // tone mapping (per-eye indexed)
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers;
    std::shared_ptr<vk::DeviceLocalBuffer> histBuffer;

    // output
    std::shared_ptr<vk::DeviceLocalImage> ldrImage;

    ToneMappingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                             std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                             std::shared_ptr<ToneMappingModule> toneMappingModule);

    void render() override;

  private:
    uint32_t eyeCount_;
};