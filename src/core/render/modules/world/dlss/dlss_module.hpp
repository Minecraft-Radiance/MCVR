#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/dlss/dlssg_wrapper.hpp"
#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct DLSSModuleContext;

class DLSSModule : public WorldModule, public SharedObject<DLSSModule> {
    friend DLSSModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.dlss.name";
    constexpr static uint32_t inputImageNum = 8;
    constexpr static uint32_t outputImageNum = 2;

    static bool initNGXContext();
    static void deinitNGXContext();
    static std::shared_ptr<NgxContext> ngxContext() { return ngxContext_; }
    static bool isFrameGenAvailable();

    DLSSModule();

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

    bool isFrameGenEnabled() const { return frameGenMultiFrameCount_ > 0 && dlssFG_ != nullptr; }
    uint32_t frameGenMultiFrameCount() const { return frameGenMultiFrameCount_; }
    bool isFrameGenDynamic() const { return frameGenDynamic_; }
    void updateDynamicFrameCount(double frameTimeSec, int targetFps);
    std::shared_ptr<DlssFG> dlssFG() { return dlssFG_; }
    std::shared_ptr<vk::DeviceLocalImage> interpFrameImage(uint32_t frameIndex, uint32_t interpIndex) { return interpFrameImages_[frameIndex][interpIndex]; }
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage(uint32_t frameIndex) { return motionVectorImages_[frameIndex]; }
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage(uint32_t frameIndex) { return linearDepthImages_[frameIndex]; }
    uint32_t outputWidth() const { return outputWidth_; }
    uint32_t outputHeight() const { return outputHeight_; }

  private:
    static std::shared_ptr<NgxContext> ngxContext_;

    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;

    // dlss
    std::shared_ptr<DlssRR> dlss_;
    NgxContext::SupportedSizes supportedSizes_{};
    NVSDK_NGX_PerfQuality_Value mode_ = NVSDK_NGX_PerfQuality_Value_Balanced;

    // frame generation
    // 0 = disabled, 1 = 2x, 2 = 3x, 3 = 4x, etc.
    uint32_t frameGenMultiFrameCount_ = 0;
    uint32_t frameGenMaxMultiFrameCount_ = 0;
    bool frameGenDynamic_ = false;
    std::shared_ptr<DlssFG> dlssFG_;
    // interpFrameImages_[swapchainIndex][interpIndex] — one per interpolated frame
    std::vector<std::vector<std::shared_ptr<vk::DeviceLocalImage>>> interpFrameImages_;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> processedImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> upscaledFirstHitDepthImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t inputWidth_, inputHeight_;
    uint32_t outputWidth_, outputHeight_;
};

struct DLSSModuleContext : public WorldModuleContext, SharedObject<DLSSModuleContext> {
    std::weak_ptr<DLSSModule> dLSSModule;

    // input
    std::shared_ptr<vk::DeviceLocalImage> hdrImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> specularHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;

    // output
    std::shared_ptr<vk::DeviceLocalImage> processedImage;
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage;

    DLSSModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                      std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                      std::shared_ptr<DLSSModule> dLSSModule);

    void render() override;
};