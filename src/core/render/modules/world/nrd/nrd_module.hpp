#pragma once

#include "core/render/modules/world/nrd/nrd_wrapper.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/render/modules/world/world_module.hpp"
#include "core/render/renderer.hpp"
#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct NrdModuleContext;

class NrdModule : public WorldModule, public SharedObject<NrdModule> {
    friend class NrdModuleContext;

  public:
    static constexpr auto NAME = "render_pipeline.module.nrd.name";
    static constexpr uint32_t inputImageNum = 20;
    static constexpr uint32_t outputImageNum = 1;

    NrdModule();

    ~NrdModule();

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

  private:
    enum InputImageIndex : uint32_t {
        DIRECT_DIFFUSE_RADIANCE_INPUT,
        DIRECT_SPECULAR_RADIANCE_INPUT,
        DIRECT_HIT_DISTANCE_INPUT,
        DIRECT_LIGHT_GUIDE_NORMAL_ROUGHNESS_INPUT,
        DIRECT_LIGHT_GUIDE_MOTION_DEPTH_INPUT,
        INDIRECT_DIFFUSE_RADIANCE_INPUT,
        INDIRECT_SPECULAR_RADIANCE_INPUT,
        INDIRECT_DIFFUSE_DIRECTION_INPUT,
        INDIRECT_SPECULAR_DIRECTION_INPUT,
        INDIRECT_DIFFUSE_HIT_DISTANCE_INPUT,
        INDIRECT_SPECULAR_HIT_DISTANCE_INPUT,
        INDIRECT_GUIDE_NORMAL_ROUGHNESS_INPUT,
        INDIRECT_GUIDE_MOTION_INPUT,
        INDIRECT_GUIDE_DEPTH_INPUT,
        SHARED_DIFFUSE_ALBEDO_METALLIC_INPUT,
        SHARED_SPECULAR_ALBEDO_INPUT,
        COMPOSE_CLEAR_RADIANCE_INPUT,
        COMPOSE_FIRST_HIT_REFRACTION_INPUT,
        COMPOSE_BASE_EMISSION_INPUT,
        COMPOSE_FOG_INPUT,
        INPUT_IMAGE_COUNT,
    };
    static_assert(INPUT_IMAGE_COUNT == inputImageNum);

    static nrd::ReblurSettings makeDefaultReblurSettings();
    static nrd::RelaxSettings makeDefaultRelaxSettings();
    static std::shared_ptr<vk::DeviceLocalImage>
    createInputImage(const std::shared_ptr<Framework> &framework, uint32_t width, uint32_t height, VkFormat format);
    static bool parseBool(const std::string &value);
    static uint32_t parseUint(const std::string &value, uint32_t fallback);
    static float parseFloat(const std::string &value, float fallback);
    static nrd::HitDistanceReconstructionMode
    parseHitDistanceReconstructionMode(const std::string &value, nrd::HitDistanceReconstructionMode fallback);
    void updateReblurSettings();
    void updateRelaxSettings();
    void applyFrameRateTemporalSettings(float framesPerSecond);
    void updateFrameTime(float milliseconds, bool restart);

    void initDescriptorTables();
    void initImages();
    void initPipeline();

  private:
    nrd::ReblurSettings reblurSettings_ = {};
    nrd::RelaxSettings relaxSettings_ = {};
    nrd::HitDistanceReconstructionMode configuredReblurHitDistanceReconstructionMode_ =
        nrd::HitDistanceReconstructionMode::AREA_3X3;
    nrd::ReblurSettings effectiveReblurSettings_ = {};
    nrd::RelaxSettings effectiveRelaxSettings_ = {};
    bool isShEnabled_ = false;
    bool isSeparateDirectDenoisingEnabled_ = true;
    std::chrono::steady_clock::time_point lastRenderTimestamp_ = {};
    bool hasLastRenderTimestamp_ = false;
    float smoothedFramesPerSecond_ = 60.0f;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directHitDistanceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directLightGuideNormalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directLightGuideMotionDepthImages_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectDiffuseDirectionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectSpecularDirectionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectDiffuseHitDistanceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectSpecularHitDistanceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectGuideNormalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectGuideMotionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectGuideDepthImages_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> sharedDiffuseAlbedoMetallicImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> sharedSpecularAlbedoImages_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> composeClearRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> composeFirstHitRefractionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> composeBaseEmissionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> composeFogImages_;

    std::shared_ptr<vk::ComputePipeline> preparePipeline_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> prepareDescriptorTables_;
    std::vector<std::array<std::shared_ptr<vk::DescriptorTable>, 2>> lightingResponseTables_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> lightingResponseImages_;
    std::array<std::shared_ptr<vk::DeviceLocalBuffer>, 2> rawLightingHistory_;
    std::array<std::shared_ptr<vk::DeviceLocalBuffer>, 2> lightingResponseHistory_;
    std::shared_ptr<vk::ComputePipeline> lightingResponsePipeline_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectDiffuseSh0Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectDiffuseSh1Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectSpecularSh0Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectSpecularSh1Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectNormalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectLinearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectBaseColorMetalnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectNormalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectLinearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectDisocclusionThresholdMixImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectDisocclusionThresholdMixImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdDirectConfidenceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> nrdIndirectConfidenceImages_;

    struct HalfRateDiffuseGuide {
        std::shared_ptr<vk::DeviceLocalImage> motion;
        std::shared_ptr<vk::DeviceLocalImage> normalRoughness;
        std::shared_ptr<vk::DeviceLocalImage> depth;
    };
    std::vector<HalfRateDiffuseGuide> halfRateDiffuseGuides_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedIndirectDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedIndirectSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedIndirectDiffuseSh0Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedIndirectDiffuseSh1Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedIndirectSpecularSh0Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedIndirectSpecularSh1Images_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedDirectDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedDirectSpecularRadianceImages_;

    std::shared_ptr<vk::ComputePipeline> composePipeline_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> composeDescriptorTables_;
    uint32_t nrdFrameIndex_ = 0;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> composedLinearImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> halfRateIndirectImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> halfRateSecondaryImages_;

    std::array<std::shared_ptr<vk::ComputePipeline>, 2> halfRateResolvePipelines_;
    std::vector<std::array<std::shared_ptr<vk::DescriptorTable>, 2>> halfRateResolveTables_;
    std::vector<std::array<std::shared_ptr<vk::DescriptorTable>, 2>> halfRateSecondaryResolveTables_;
    std::array<std::shared_ptr<vk::DeviceLocalBuffer>, 2> halfRateResolveHistory_;
    std::array<std::shared_ptr<vk::DeviceLocalBuffer>, 2> halfRateSecondaryHistory_;
    uint32_t halfRateResolveParity_ = 0;
    bool halfRateResolveHasHistory_ = false;

    uint32_t directWidth_ = 0;
    uint32_t directHeight_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::shared_ptr<NrdWrapper> indirectWrapper_;
    std::shared_ptr<NrdWrapper> directWrapper_;
    ShaderPackLoader::NrdMode nrdMode_ = ShaderPackLoader::NrdMode::Probabilistic;
    uint32_t indirectWidth_ = 0;
    uint32_t indirectHeight_ = 0;

    std::vector<std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>>
        indirectTexturePools_;
    std::vector<std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>>
        directTexturePools_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

struct NrdModuleContext : public WorldModuleContext, public SharedObject<NrdModuleContext> {
  public:
    NrdModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                     std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                     std::shared_ptr<NrdModule> nrdModule);

    void render() override;

  private:
    void recordHalfRateResolve(const std::shared_ptr<vk::CommandBuffer> &command,
                               const vk::Data::WorldUBO &world,
                               const vk::Data::WorldUBO &previousWorld,
                               bool resetHistory = false,
                               bool secondaryPass = false);
    void recordPasses(const std::shared_ptr<vk::CommandBuffer> &command,
                      const std::shared_ptr<vk::HostVisibleBuffer> &world,
                      const std::shared_ptr<vk::HostVisibleBuffer> &previousWorld,
                      const std::shared_ptr<vk::HostVisibleBuffer> &sky,
                      uint32_t frameIndex,
                      bool resetHistory);
    static void addComputeImageBarrier(std::vector<vk::CommandBuffer::ImageMemoryBarrier> &barriers,
                                       const std::shared_ptr<vk::DeviceLocalImage> &image,
                                       VkImageLayout targetLayout,
                                       uint32_t queueIndex);

    std::weak_ptr<NrdModule> nrdModule;

    std::shared_ptr<vk::DeviceLocalImage> directDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> directSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> directHitDistanceImage;
    std::shared_ptr<vk::DeviceLocalImage> directLightGuideNormalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> directLightGuideMotionDepthImage;

    std::shared_ptr<vk::DeviceLocalImage> indirectDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectDiffuseDirectionImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectSpecularDirectionImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectDiffuseHitDistanceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectSpecularHitDistanceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectGuideNormalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectGuideMotionImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectGuideDepthImage;

    std::shared_ptr<vk::DeviceLocalImage> sharedDiffuseAlbedoMetallicImage;
    std::shared_ptr<vk::DeviceLocalImage> sharedSpecularAlbedoImage;

    std::shared_ptr<vk::DeviceLocalImage> composeClearRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> composeFirstHitRefractionImage;
    std::shared_ptr<vk::DeviceLocalImage> composeBaseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> composeFogImage;

    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectDiffuseSh0Image;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectDiffuseSh1Image;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectSpecularSh0Image;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectSpecularSh1Image;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectMotionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectNormalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectLinearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectBaseColorMetalnessImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectMotionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectNormalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectLinearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectDisocclusionThresholdMixImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectDisocclusionThresholdMixImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdDirectConfidenceImage;
    std::shared_ptr<vk::DeviceLocalImage> nrdIndirectConfidenceImage;

    std::shared_ptr<vk::DeviceLocalImage> denoisedIndirectDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> denoisedIndirectSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> denoisedIndirectDiffuseSh0Image;
    std::shared_ptr<vk::DeviceLocalImage> denoisedIndirectDiffuseSh1Image;
    std::shared_ptr<vk::DeviceLocalImage> denoisedIndirectSpecularSh0Image;
    std::shared_ptr<vk::DeviceLocalImage> denoisedIndirectSpecularSh1Image;
    std::shared_ptr<vk::DeviceLocalImage> denoisedDirectDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> denoisedDirectSpecularRadianceImage;

    std::shared_ptr<vk::DescriptorTable> prepareDescriptorTable;
    std::shared_ptr<vk::DescriptorTable> composeDescriptorTable;

    std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>
        indirectTexturePool;
    std::shared_ptr<std::array<std::shared_ptr<vk::DeviceLocalImage>, size_t(nrd::ResourceType::MAX_NUM)>>
        directTexturePool;

    std::shared_ptr<vk::DeviceLocalImage> denoisedRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> composedLinearImage;
    std::shared_ptr<vk::DeviceLocalImage> halfRateIndirectImage;
};
