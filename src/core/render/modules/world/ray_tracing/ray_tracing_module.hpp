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

struct RayTracingModuleContext;

class Atmosphere;
class AtmosphereContext;
class WorldPrepare;
class WorldPrepareContext;

struct RayTracingPushConstant {
    int numRayBounces;
    int useJitter;
};

// Descriptor set and binding constants
constexpr uint32_t TEXTURES_SET = 0;
constexpr uint32_t ACCELERATION_SET = 1;
constexpr uint32_t UNIFORMS_SET = 2;
constexpr uint32_t OUTPUT_IMAGES_SET = 3;

class RayTracingModule : public WorldModule, public SharedObject<RayTracingModule> {
    friend RayTracingModuleContext;
    friend Atmosphere;
    friend AtmosphereContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.ray_tracing.name";
    constexpr static uint32_t inputImageNum = 0;
    constexpr static uint32_t outputImageNum = 14;

    RayTracingModule();

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
    void initDescriptorTables();
    void initImages();
    void initPipeline();
    void initSBT();
    void createVisibilityMaskImage();

  private:
    // input
    // none

    // ray tracing
    std::shared_ptr<vk::Shader> worldRayGenShader_;

    std::shared_ptr<vk::Shader> worldRayMissShader_;
    std::shared_ptr<vk::Shader> handRayMissShader_;
    std::shared_ptr<vk::Shader> shadowRayMissShader_;

    std::shared_ptr<vk::Shader> shadowRayClosestHitShader_;
    std::shared_ptr<vk::Shader> shadowAnyHitShader_;

    std::shared_ptr<vk::Shader> worldSolidTransparentClosestHitShader_;
    std::shared_ptr<vk::Shader> worldTransparentAnyHitShader_;

    std::shared_ptr<vk::Shader> worldNoReflectClosestHitShader_;
    std::shared_ptr<vk::Shader> worldNoReflectAnyHitShader_;

    std::shared_ptr<vk::Shader> worldCloudClosestHitShader_;
    std::shared_ptr<vk::Shader> worldCloudAnyHitShader_;

    std::shared_ptr<vk::Shader> boatWaterMaskClosestHitShader_;
    std::shared_ptr<vk::Shader> boatWaterMaskAnyHitShader_;

    std::shared_ptr<vk::Shader> endPortalClosestHitShader_;
    std::shared_ptr<vk::Shader> endPortalAnyHitShader_;

    std::shared_ptr<vk::Shader> endGatewayClosestHitShader_;
    std::shared_ptr<vk::Shader> endGatewayAnyHitShader_;

    std::shared_ptr<vk::Shader> worldPostColorToDepthVertShader_;
    std::shared_ptr<vk::Shader> worldPostColorToDepthFragShader_;
    std::shared_ptr<vk::Shader> worldPostVertShader_;
    std::shared_ptr<vk::Shader> worldPostFragShader_;
    std::shared_ptr<vk::Shader> worldToneMappingVertShader_;
    std::shared_ptr<vk::Shader> worldToneMappingFragShader_;
    std::shared_ptr<vk::Shader> radianceHistCompShader_;
    std::shared_ptr<vk::Shader> worldLightMapVertShader_;
    std::shared_ptr<vk::Shader> worldLightMapFragShader_;

    // rayTracingDescriptorTables_[frameIndex] - unified descriptor table for 3D dispatch
    std::vector<std::shared_ptr<vk::DescriptorTable>> rayTracingDescriptorTables_;
    std::shared_ptr<vk::RayTracingPipeline> rayTracingPipeline_;
    std::vector<std::shared_ptr<vk::SBT>> sbts_;

    uint32_t numRayBounces_ = 4;
    bool useJitter_ = true;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrNoisyOutputImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDiffuseDirectLightImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDiffuseIndirectLightImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitSpecularImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitClearImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitBaseEmissionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directLightDepthImages_;

    // visibility mask (static, shared across frames)
    std::shared_ptr<vk::DeviceLocalImage> visMaskImage_;

    // submodules
    std::shared_ptr<Atmosphere> atmosphere_;
    std::shared_ptr<WorldPrepare> worldPrepare_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

struct RayTracingModuleContext : public WorldModuleContext, SharedObject<RayTracingModuleContext> {
    std::weak_ptr<RayTracingModule> rayTracingModule;

    // input
    // none

    // ray tracing
    // unified descriptor table for 3D dispatch
    std::shared_ptr<vk::DescriptorTable> rayTracingDescriptorTable;
    std::shared_ptr<vk::SBT> sbt;

    // output
    std::shared_ptr<vk::DeviceLocalImage> hdrNoisyOutputImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> specularHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDiffuseDirectLightImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDiffuseIndirectLightImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitSpecularImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitClearImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitBaseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> directLightDepthImage;

    // submodule
    std::shared_ptr<AtmosphereContext> atmosphereContext;
    std::shared_ptr<WorldPrepareContext> worldPrepareContext;

    RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                            std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                            std::shared_ptr<RayTracingModule> rayTracingModule);

    StereoMode stereoMode() const override { return StereoMode::SingleInstance3DDispatch; }
    void render() override;
    void render3D(uint32_t eyeCount) override;
};