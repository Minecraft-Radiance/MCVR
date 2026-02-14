#pragma once

#include "core/render/modules/world/world_module.hpp"
#include <array>
#include <deque>
#include <chrono>

class FSR3UpscalerModuleContext;

namespace mcvr {
    class FSR3Upscaler;
}

class FSR3UpscalerModule : public WorldModule, public SharedObject<FSR3UpscalerModule> {
    friend FSR3UpscalerModuleContext;

  public:
    enum class QualityMode {
        NativeAA = 0,           // 1.0x (no upscaling)
        Quality = 1,            // 1.5x
        Balanced = 2,           // 1.7x
        Performance = 3,        // 2.0x
        UltraPerformance = 4    // 3.0x
    };

    static constexpr const char *NAME = "render_pipeline.module.fsr3_upscaler.name";
    static constexpr uint32_t inputImageNum = 4;  // color, depth, motion vectors, firstHitDepth
    static constexpr uint32_t outputImageNum = 2; // upscaled HDR output, upscaled firstHitDepth

    static bool isQualityModeAttributeKey(const std::string &key);
    static bool parseQualityModeValue(const std::string &value, QualityMode &outMode);

    FSR3UpscalerModule();
    ~FSR3UpscalerModule() = default;

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

    static void getRenderResolution(uint32_t displayWidth, uint32_t displayHeight,
                                   QualityMode mode,
                                   uint32_t* outRenderWidth, uint32_t* outRenderHeight);

  private:
    void initDescriptorTables();
    void initImages();
    void initPipeline();

    std::vector<std::shared_ptr<FSR3UpscalerModuleContext>> contexts_;
    
    // Resolution state
    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t displayWidth_ = 0;
    uint32_t displayHeight_ = 0;
    QualityMode qualityMode_ = QualityMode::NativeAA;
    float sharpness_ = 0.7f;
    float preExposure_ = 1.0f;
    bool fsr3Enabled_ = true;

    // FSR3 implementation
    std::shared_ptr<mcvr::FSR3Upscaler> fsr3_;
    bool initialized_ = false;

    // Depth conversion resources
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> deviceDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> fsrMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> depthDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> depthConversionPipeline_;

    // Camera state for reset detection
    glm::vec3 lastCameraPos_ = glm::vec3(0.0f);
    glm::vec3 lastCameraDir_ = glm::vec3(0.0f, 0.0f, -1.0f);
    bool firstFrame_ = true;

    // Temporal storage for images during setOrCreate
    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, 4>> inputImages_;
    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, 2>> outputImages_;
};

class FSR3UpscalerModuleContext : public WorldModuleContext {
  public:
    FSR3UpscalerModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                          std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                          std::shared_ptr<FSR3UpscalerModule> upscalerModule);

    void render() override;

    // Inputs (render resolution)
    std::shared_ptr<vk::DeviceLocalImage> inputColorImage;
    std::shared_ptr<vk::DeviceLocalImage> inputDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> inputMotionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> inputFirstHitDepthImage;

    // Outputs (display resolution)
    std::shared_ptr<vk::DeviceLocalImage> outputImage;
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage;

    std::shared_ptr<vk::DescriptorTable> depthDescriptorTable;
    std::shared_ptr<vk::DeviceLocalImage> deviceDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> fsrMotionVectorImage;

  private:
    bool checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir);
    float getSmoothDeltaTime();

    std::weak_ptr<FSR3UpscalerModule> upscalerModule_;
    
    // Timing
    std::deque<float> frameTimes_;
    std::chrono::high_resolution_clock::time_point lastFrameTime_;
    bool timingFirstFrame_ = true;
};
