#pragma once

#include "core/render/modules/world/world_module.hpp"
#include "core/render/modules/world/xess_upscaler/xess_wrapper.hpp"

#include <array>
#include <deque>
#include <chrono>

class XessSrModuleContext;

class XessSrModule : public WorldModule, public SharedObject<XessSrModule> {
    friend XessSrModuleContext;

  public:
    enum class QualityMode {
        NativeAA = 0,
        UltraQualityPlus = 1,
        UltraQuality = 2,
        Quality = 3,
        Balanced = 4,
        Performance = 5,
        UltraPerformance = 6
    };

    static constexpr const char *NAME = "render_pipeline.module.xess_sr.name";
    static constexpr uint32_t inputImageNum = 4;  // color, depth, motion vectors, firstHitDepth
    static constexpr uint32_t outputImageNum = 2; // upscaled HDR output, upscaled firstHitDepth

    static bool isQualityModeAttributeKey(const std::string &key);
    static bool parseQualityModeValue(const std::string &value, QualityMode &outMode);

    XessSrModule();
    ~XessSrModule() = default;

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
    void initDescriptorTables();
    void initImages();
    void initPipeline();

    std::vector<std::shared_ptr<XessSrModuleContext>> contexts_;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t displayWidth_ = 0;
    uint32_t displayHeight_ = 0;
    QualityMode qualityMode_ = QualityMode::Quality;
    float preExposure_ = 1.0f;
    bool xessEnabled_ = true;

    std::shared_ptr<mcvr::XeSSWrapper> xess_;
    bool initialized_ = false;

    // Input conversion resources (match FSR3 prepare path)
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> deviceDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> xessMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> depthDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> depthConversionPipeline_;

    glm::vec3 lastCameraPos_ = glm::vec3(0.0f);
    glm::vec3 lastCameraDir_ = glm::vec3(0.0f, 0.0f, -1.0f);
    bool firstFrame_ = true;

    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, 4>> inputImages_;
    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, 2>> outputImages_;
};

class XessSrModuleContext : public WorldModuleContext {
  public:
    XessSrModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                        std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                        std::shared_ptr<XessSrModule> xessModule);

    void render() override;

    std::shared_ptr<vk::DeviceLocalImage> inputColorImage;
    std::shared_ptr<vk::DeviceLocalImage> inputDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> inputMotionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> inputFirstHitDepthImage;

    std::shared_ptr<vk::DeviceLocalImage> outputImage;
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage;

    std::shared_ptr<vk::DescriptorTable> depthDescriptorTable;
    std::shared_ptr<vk::DeviceLocalImage> deviceDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> xessMotionVectorImage;

  private:
    bool checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir);

    std::weak_ptr<XessSrModule> xessModule_;
};
