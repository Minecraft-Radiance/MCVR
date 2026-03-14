#pragma once

#include "core/render/modules/world/nrd/nrd_wrapper.hpp"
#include "core/render/modules/world/world_module.hpp"
#include "core/render/renderer.hpp"
#include <array>
#include <map>
#include <memory>
#include <vector>

struct NrdModuleContext;

class NrdModule : public WorldModule, public SharedObject<NrdModule> {
  public:
    static constexpr auto NAME = "render_pipeline.module.nrd.name";
    static constexpr uint32_t inputImageNum = 12;
    static constexpr uint32_t outputImageNum = 1;

    NrdModule();
    virtual ~NrdModule() = default;

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

    std::shared_ptr<NrdWrapper> wrapper(uint32_t eyeIndex = 0) {
        return m_wrappers.empty() ? nullptr : m_wrappers[eyeIndex];
    }

    StereoMode stereoMode() const override { return StereoMode::DualInstance; }

    void dispatchComposition(std::shared_ptr<vk::CommandBuffer> cmd,
                             uint32_t frameIndex,
                             uint32_t eyeIndex,
                             const std::map<std::string, std::shared_ptr<vk::DeviceLocalImage>> &images,
                             std::shared_ptr<vk::HostVisibleBuffer> worldUBO,
                             std::shared_ptr<vk::DeviceLocalImage> outputImage);

  private:
    friend class NrdModuleContext;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::shared_ptr<vk::Device> m_device;
    std::shared_ptr<vk::VMA> m_vma;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
    // m_wrappers[eyeIndex] — one NRD instance per eye for stereo
    std::vector<std::shared_ptr<NrdWrapper>> m_wrappers;

    uint32_t maxAccumulatedFrameNum_ = 31;
    uint32_t maxFastAccumulatedFrameNum_ = 4;
    float maxBlurRadius_ = 20.0f;
    bool enableAntiFirefly_ = true;
    nrd::HitDistanceReconstructionMode hitDistanceReconstructionMode_ = nrd::HitDistanceReconstructionMode::AREA_5X5;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> clearRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> baseEmissionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularHitDepthImages_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedSpecularRadianceImages_;

    std::shared_ptr<vk::ComputePipeline> composePipeline_;
    // composeDescriptorTables_[frameIndex * eyeCount_ + eyeIndex]
    std::vector<std::shared_ptr<vk::DescriptorTable>> composeDescriptorTables_;
    std::array<std::shared_ptr<vk::Sampler>, 2> composeSamplers_;

    std::shared_ptr<vk::ComputePipeline> preparePipeline_;
    // prepareDescriptorTables_[frameIndex * eyeCount_ + eyeIndex]
    std::vector<std::shared_ptr<vk::DescriptorTable>> prepareDescriptorTables_;
    // NRD intermediate images: [frameIndex * eyeCount_ + eyeIndex]
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_nrdMotionVectorImages;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_nrdNormalRoughnessImages;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_nrdDiffuseRadianceImages;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_nrdSpecularRadianceImages;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_nrdLinearDepthImages;

    void createCompositionPipeline(std::shared_ptr<vk::Device> device, uint32_t contextCount);
    void createPreparePipeline(std::shared_ptr<vk::Device> device, uint32_t contextCount);
};

class NrdModuleContext : public WorldModuleContext, public SharedObject<NrdModuleContext> {
  public:
    NrdModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                     std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                     std::shared_ptr<NrdModule> nrdModule);

    StereoMode stereoMode() const override { return StereoMode::DualInstance; }
    void render() override;
    void renderEye(uint32_t eyeIndex) override;

  private:
    std::weak_ptr<NrdModule> nrdModule;

    std::shared_ptr<vk::DeviceLocalImage> diffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> specularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> directRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> clearRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> baseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> specularHitDepthImage;

    std::shared_ptr<vk::DeviceLocalImage> denoisedRadianceImage; // Final HDR output
    std::shared_ptr<vk::DeviceLocalImage> denoisedDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> denoisedSpecularRadianceImage;
};
