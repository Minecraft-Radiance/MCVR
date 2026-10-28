#pragma once

#include "core/render/modules/world/world_module.hpp"
#include "frame_clock.hpp"
#include <array>

class FSRUpscalerModuleContext;

namespace mcvr {
class FSR3Upscaler;
}

class FSRUpscalerModule : public WorldModule, public SharedObject<FSRUpscalerModule> {
    friend FSRUpscalerModuleContext;

  public:
    enum class QualityMode { NativeAA = 0, Quality = 1, Balanced = 2, Performance = 3, UltraPerformance = 4 };

    static constexpr const char *NAME = "render_pipeline.module.fsr_upscaler.name";
    static constexpr uint32_t inputImageNum = 5;
    static constexpr uint32_t outputImageNum = 4;

    static bool isQualityModeAttributeKey(const std::string &key);
    static bool parseQualityModeValue(const std::string &value, QualityMode &outMode);

    FSRUpscalerModule();
    ~FSRUpscalerModule() = default;

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;

    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void build() override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

    static void getRenderResolution(uint32_t displayWidth,
                                    uint32_t displayHeight,
                                    QualityMode mode,
                                    uint32_t *outRenderWidth,
                                    uint32_t *outRenderHeight);

  private:
    static bool isHdrUpscaleFormat(VkFormat format);

    void initDescriptorTables();
    void initImages();
    void initPipeline();

    std::vector<std::shared_ptr<FSRUpscalerModuleContext>> contexts_;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t displayWidth_ = 0;
    uint32_t displayHeight_ = 0;
    QualityMode qualityMode_ = QualityMode::Balanced;
    float sharpness_ = 0.7f;
    float preExposure_ = 1.0f;
    bool fsr3Enabled_ = true;

    std::shared_ptr<mcvr::FSR3Upscaler> fsr3_;
    bool initialized_ = false;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> deviceDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> fsrMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> depthDescriptorTables_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> guideDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> depthConversionPipeline_;
    std::shared_ptr<vk::ComputePipeline> firstHitDepthUpscalePipeline_;
    std::shared_ptr<vk::ComputePipeline> motionUpscalePipeline_;
    std::shared_ptr<vk::ComputePipeline> normalRoughnessUpscalePipeline_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> fsr1DescriptorTables_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> fsr1IntermediateImages_;
    std::shared_ptr<vk::Sampler> fsr1Sampler_;
    std::shared_ptr<vk::ComputePipeline> fsr1EasuPipeline_;
    std::shared_ptr<vk::ComputePipeline> fsr1RcasPipeline_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> reactiveDescriptorTables_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> reactiveImages_;
    std::array<std::shared_ptr<vk::DeviceLocalImage>, 2> historyDepthImages_;
    std::shared_ptr<vk::ComputePipeline> reactivePipeline_;
    uint32_t historyDepthIndex_ = 0;
    bool hasDepthHistory_ = false;

    glm::vec3 lastCameraPos_ = glm::vec3(0.0f);
    glm::vec3 lastCameraDir_ = glm::vec3(0.0f, 0.0f, -1.0f);
    bool firstFrame_ = true;
    mcvr::fsr::FrameClock frameClock_;

    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, 5>> inputImages_;
    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, 4>> outputImages_;
};

class FSRUpscalerModuleContext : public WorldModuleContext {
  public:
    FSRUpscalerModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                             std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                             std::shared_ptr<FSRUpscalerModule> upscalerModule);

    void render() override;
    bool populateFrameGenerationInputs(mcvr::FrameGenerationInputs &inputs) override;

    std::shared_ptr<vk::DeviceLocalImage> inputColorImage;
    std::shared_ptr<vk::DeviceLocalImage> inputDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> inputMotionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> inputFirstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> inputNormalRoughnessImage;

    std::shared_ptr<vk::DeviceLocalImage> outputImage;
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> upscaledMotionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> upscaledNormalRoughnessImage;

    std::shared_ptr<vk::DescriptorTable> depthDescriptorTable;
    std::shared_ptr<vk::DescriptorTable> guideDescriptorTable;
    std::shared_ptr<vk::DeviceLocalImage> deviceDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> fsrMotionVectorImage;

  private:
    bool checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir);
    void prepareReactiveMask(bool reset);

    std::weak_ptr<FSRUpscalerModule> upscalerModule_;

};
