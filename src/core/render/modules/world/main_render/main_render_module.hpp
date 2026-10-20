#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

class Framework;
struct FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct MainRenderModuleContext;
struct MainRenderTimingCapture;

class WorldPrepare;
struct WorldPrepareContext;

class MainRenderModule : public WorldModule, public SharedObject<MainRenderModule> {
    friend MainRenderModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.main_render.name";
    constexpr static uint32_t inputImageNum = 0;
    constexpr static uint32_t outputImageNum = 22;

    constexpr static std::string_view TARGET_RADIANCE = "out:radiance";
    constexpr static std::string_view TARGET_DIFFUSE_ALBEDO_METALLIC = "out:diffuse_albedo_metallic";
    constexpr static std::string_view TARGET_SPECULAR_ALBEDO = "out:specular_albedo";
    constexpr static std::string_view TARGET_NORMAL_ROUGHNESS = "out:normal_roughness";
    constexpr static std::string_view TARGET_MOTION_VECTOR = "out:motion_vector";
    constexpr static std::string_view TARGET_LINEAR_DEPTH = "out:linear_depth";
    constexpr static std::string_view TARGET_FIRST_HIT_DEPTH = "out:first_hit_depth";
    constexpr static std::string_view TARGET_INDIRECT_SPECULAR_HIT_DISTANCE = "out:indirect_specular_hit_distance";
    constexpr static std::string_view TARGET_INDIRECT_DIFFUSE_HIT_DISTANCE = "out:indirect_diffuse_hit_distance";
    constexpr static std::string_view TARGET_INDIRECT_DIFFUSE_RADIANCE = "out:indirect_diffuse_radiance";
    constexpr static std::string_view TARGET_INDIRECT_DIFFUSE_DIRECTION = "out:indirect_diffuse_direction";
    constexpr static std::string_view TARGET_INDIRECT_SPECULAR_RADIANCE = "out:indirect_specular_radiance";
    constexpr static std::string_view TARGET_INDIRECT_SPECULAR_DIRECTION = "out:indirect_specular_direction";
    constexpr static std::string_view TARGET_FIRST_HIT_CLEAR = "out:first_hit_clear";
    constexpr static std::string_view TARGET_FIRST_HIT_REFRACTION = "out:first_hit_refraction";
    constexpr static std::string_view TARGET_FIRST_HIT_BASE_EMISSION = "out:first_hit_base_emission";
    constexpr static std::string_view TARGET_FOG_IMAGE = "out:fog_image";
    constexpr static std::string_view TARGET_DIRECT_DIFFUSE_RADIANCE = "out:direct_diffuse_radiance";
    constexpr static std::string_view TARGET_DIRECT_SPECULAR_RADIANCE = "out:direct_specular_radiance";
    constexpr static std::string_view TARGET_DIRECT_HIT_DISTANCE = "out:direct_hit_distance";
    constexpr static std::string_view TARGET_DIRECT_LIGHT_GUIDE_NORMAL_ROUGHNESS =
        "out:direct_light_guide_normal_roughness";
    constexpr static std::string_view TARGET_DIRECT_LIGHT_GUIDE_MOTION_DEPTH = "out:direct_light_guide_motion_depth";

    MainRenderModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    std::string getAttributes(const std::vector<std::string> &attributes, const std::string &language);

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;

    const std::unordered_map<std::string, std::string> &captureExecutionVariables() const;
    const std::unordered_map<std::string, std::string> &captureStaticAttributes() const;

    bool hasSegmentationMask() const;
    std::shared_ptr<vk::DeviceLocalImage> segmentationMaskImage(uint32_t frameIndex) const;


    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

  private:
    enum OutputImageIndex : uint32_t {
        RADIANCE_OUTPUT,
        DIFFUSE_ALBEDO_METALLIC_OUTPUT,
        SPECULAR_ALBEDO_OUTPUT,
        NORMAL_ROUGHNESS_OUTPUT,
        MOTION_VECTOR_OUTPUT,
        LINEAR_DEPTH_OUTPUT,
        FIRST_HIT_DEPTH_OUTPUT,
        INDIRECT_SPECULAR_HIT_DISTANCE_OUTPUT,
        INDIRECT_DIFFUSE_HIT_DISTANCE_OUTPUT,
        INDIRECT_DIFFUSE_RADIANCE_OUTPUT,
        INDIRECT_DIFFUSE_DIRECTION_OUTPUT,
        INDIRECT_SPECULAR_RADIANCE_OUTPUT,
        INDIRECT_SPECULAR_DIRECTION_OUTPUT,
        FIRST_HIT_CLEAR_OUTPUT,
        FIRST_HIT_REFRACTION_OUTPUT,
        FIRST_HIT_BASE_EMISSION_OUTPUT,
        FOG_OUTPUT,
        DIRECT_DIFFUSE_RADIANCE_OUTPUT,
        DIRECT_SPECULAR_RADIANCE_OUTPUT,
        DIRECT_HIT_DISTANCE_OUTPUT,
        DIRECT_LIGHT_GUIDE_NORMAL_ROUGHNESS_OUTPUT,
        DIRECT_LIGHT_GUIDE_MOTION_DEPTH_OUTPUT,
        OUTPUT_IMAGE_COUNT,
    };
    static_assert(OUTPUT_IMAGE_COUNT == outputImageNum);

    constexpr static uint32_t defaultSharcCapacityLog2 = 23;
    constexpr static uint32_t defaultSharcCapacity = 1u << defaultSharcCapacityLog2;
    constexpr static float defaultSharcSceneScale = 64.0f;
    constexpr static uint32_t defaultSharcAccumulationFrameNum = 64;
    constexpr static uint32_t defaultSharcStaleFrameNumMax = 256;
    constexpr static uint32_t defaultSharcUpdateDownsampleFactor = 5;
    constexpr static uint32_t defaultSharcResolveFrameStride = 4;
    constexpr static uint32_t sharcResolveWorkgroupSize = 64;
    constexpr static uint32_t executionLoopLimit = 1u << 16;
    constexpr static uint32_t directionalShadowResolution = 2048;

    struct SharcConfigData {
        uint32_t hashEntriesAddress[2];
        uint32_t lockAddress[2];
        uint32_t accumulationAddress[2];
        uint32_t resolvedAddress[2];
        glm::vec4 cameraPosition;
        glm::vec4 cameraPositionPrev;
        float sceneScale;
        float radianceScale;
        uint32_t accumulationFrameNum;
        uint32_t staleFrameNumMax;
        uint32_t capacity;
        uint32_t frameIndex;
        uint32_t enableAntiFireflyFilter;
        uint32_t useLockBuffer;
        uint32_t debugMode;
        uint32_t updateDownsampleFactor;
        uint32_t resolveBaseEntry;
        uint32_t resolveEntryCount;
        uint32_t resolveFrameStride;
        uint32_t reserved;
    };

    struct DirectionalShadowData {
        glm::mat4 lightViewProj;
        glm::vec4 lightDirectionAndStrength;
        glm::vec4 shadowParams;
        glm::vec4 shadowDistanceParams;
    };

    struct DirectionalShadowConfigUpdate {
        bool valid = false;
        bool updated = false;
        uint64_t signature = 0;
    };

    struct DirectionalShadowPushConstants {
        glm::mat4 lightViewProj;
    };

    using ExecutionVariable = ShaderPack::ExecutionVariable;
    using ExecutionVariables = ShaderPack::ExecutionVariables;

    using PassVariant = std::variant<std::shared_ptr<FullScreenPass>,
                                     std::shared_ptr<RenderPass>,
                                     std::shared_ptr<RayTracingPass>,
                                     std::shared_ptr<ComputePass>>;

    struct ResolvedImageResource {
        const std::vector<std::shared_ptr<vk::DeviceLocalImage>> *targetImages = nullptr;
        ShaderPack::RuntimeTexture *runtimeTexture = nullptr;
        bool isImportedRuntimeTexture = false;
    };

    struct ResolvedImageAccess {
        ResolvedImageResource resource;
        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
        VkAccessFlags2 accessMask = 0;
        bool reads = false;
        bool writes = false;
        bool needsPreviousContent = false;
    };

    struct ResolvedBufferAccess {
        ShaderPack::RuntimeBuffer *runtimeBuffer = nullptr;
        VkAccessFlags2 accessMask = 0;
        bool reads = false;
        bool writes = false;
        bool needsPreviousContent = false;
    };

    struct ResolvedPassResources {
        std::vector<ResolvedImageAccess> imageAccesses;
        std::vector<ResolvedBufferAccess> bufferAccesses;
        std::optional<ResolvedImageAccess> fullScreenTarget;
        std::vector<ResolvedImageAccess> colorAttachmentTargets;
        ShaderPack::RuntimeBuffer *indirectArgsBuffer = nullptr;
    };

  private:
    struct MainRenderExecutionRuntime;
    struct MainRenderPassExecutionOp;
    struct SharcUpdateExecutionOp;
    struct MainRenderBufferUpdateExecutionOp;
    struct MainRenderFrameGraph;

    static std::filesystem::path builtInShaderPackPath();
    static std::string buildMainRenderDebugLabel(std::string_view passKind, std::string_view passName);
    static VkCompareOp parseDepthCompare(const std::string &value);
    static std::shared_ptr<vk::Shader> createShader(std::shared_ptr<vk::Device> device,
                                                    const std::filesystem::path &path,
                                                    VkShaderStageFlagBits stage,
                                                    const std::unordered_map<std::string, std::string> &definitions,
                                                    const std::vector<std::string> &includeDirectories,
                                                    const std::string &injectedSource);
    void loadShaderPack();
    void initDescriptorTables();
    void initRuntimeTextures();
    void initRuntimeBuffers();
    void refreshRuntimeBuffers(const std::vector<ExpressionEvaluator::Variable> &runtimeExpressionVariables);
    void loadRuntimeResources();
    void initDirectionalShadowResources();
    void initSharc();
    void initPipelines();
    void initDirectionalShadowPipelines();
    void initSBTs();
    void initContexts();
    void compileExecutionProgram();
    void updateSharcConfig(uint32_t frameIndex);
    DirectionalShadowConfigUpdate updateDirectionalShadowConfig(MainRenderModuleContext &context, uint32_t frameIndex);
    void initExecutionVariables();
    std::vector<ExpressionEvaluator::Variable> executionExpressionVariables() const;
    std::unordered_map<std::string, std::string>
    shaderDefinitionsWithRuntimeExpressions(const std::unordered_map<std::string, std::string> &definitions) const;
    double evaluateNumericExpression(const std::string &expression,
                                     const std::vector<ExpressionEvaluator::Variable> &expressionVariables);
    std::optional<std::reference_wrapper<ShaderPackLoader::VariableConfig>>
    findExecutionVariableConfig(std::string_view name);
    std::optional<std::reference_wrapper<ShaderPack::RuntimeTexture>> findRuntimeTexture(std::string_view name);
    std::optional<std::reference_wrapper<ShaderPack::RuntimeBuffer>> findRuntimeBuffer(std::string_view name);
    std::shared_ptr<vk::DeviceLocalImage> findRuntimeVKTexture(ShaderPack::RuntimeTexture &runtimeTexture,
                                                               uint32_t frameIndex);
    std::shared_ptr<vk::DeviceLocalBuffer> findRuntimeVKBuffer(ShaderPack::RuntimeBuffer &runtimeBuffer,
                                                               uint32_t frameIndex);
    const std::vector<std::shared_ptr<vk::DeviceLocalImage>> *findTargetImages(std::string_view target);
    std::shared_ptr<vk::DeviceLocalImage> findTargetImage(const std::string &target, uint32_t frameIndex);
    std::shared_ptr<vk::DeviceLocalImage> resolveImageResource(const ResolvedImageResource &resource,
                                                               uint32_t frameIndex);
    std::vector<std::shared_ptr<vk::Framebuffer>> buildFramebuffers(std::shared_ptr<vk::DeviceLocalImage> image,
                                                                    std::shared_ptr<vk::RenderPass> renderPass);

    ResolvedImageResource resolveImageResource(std::string_view name);
    ShaderPack::RuntimeBuffer *resolveRuntimeBuffer(std::string_view name);
    ResolvedPassResources resolvePassResources(const ShaderPackLoader::ResourceList &inputs,
                                               const ShaderPackLoader::ResourceList &outputs,
                                               const std::optional<std::string> &indirectArgsBuffer,
                                               std::optional<std::string_view> fullScreenTarget,
                                               bool outputsAreColorAttachments = false);
    const ResolvedPassResources &resolvedResourcesForPass(const FullScreenPass &pass) const;
    const ResolvedPassResources &resolvedResourcesForPass(const RenderPass &pass) const;
    const ResolvedPassResources &resolvedResourcesForPass(const RayTracingPass &pass) const;
    const ResolvedPassResources &resolvedResourcesForPass(const ComputePass &pass) const;
    void addFrameGraphResourceBarriers(const ResolvedPassResources &resources,
                                       std::vector<vk::CommandBuffer::BufferMemoryBarrier> &bufferBarriers,
                                       std::vector<vk::CommandBuffer::ImageMemoryBarrier> &imageBarriers,
                                       MainRenderModuleContext &context,
                                       VkPipelineStageFlags2 passStageMask,
                                       uint32_t frameIndex,
                                       uint32_t queueIndex);
    void addFrameGraphImageBarrier(const ResolvedImageAccess &imageAccess,
                                   std::vector<vk::CommandBuffer::ImageMemoryBarrier> &imageBarriers,
                                   MainRenderModuleContext &context,
                                   VkPipelineStageFlags2 passStageMask,
                                   uint32_t frameIndex,
                                   uint32_t queueIndex);
    void addFrameGraphBufferBarrier(const ResolvedBufferAccess &bufferAccess,
                                    std::vector<vk::CommandBuffer::BufferMemoryBarrier> &bufferBarriers,
                                    MainRenderModuleContext &context,
                                    VkPipelineStageFlags2 passStageMask,
                                    uint32_t frameIndex,
                                    uint32_t queueIndex,
                                    VkDeviceSize offset = 0,
                                    VkDeviceSize size = VK_WHOLE_SIZE);
    void addRenderDepthBarrier(std::vector<vk::CommandBuffer::ImageMemoryBarrier> &imageBarriers,
                               MainRenderModuleContext &context,
                               uint32_t frameIndex,
                               uint32_t queueIndex,
                               bool depthLoad);
    void uploadPassExecutionConstants(const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                                      const ExecutionVariables &executionVariables,
                                      MainRenderModuleContext &context,
                                      VkShaderStageFlags shaderStages);
    void recordBufferUpdate(const ShaderPackLoader::ExecutionCommand::BufferUpdate &update,
                            MainRenderModuleContext &context);
    void clearSegmentationMask(MainRenderModuleContext &context);
    void prepareSegmentationMaskForToneMapping(MainRenderModuleContext &context);

    void initFullScreenPassTargets(FullScreenPass &pass, std::shared_ptr<vk::Device> device, uint32_t frameCount);
    std::vector<ShaderPack::ShaderCreateInfo>
    collectFullScreenPassShaderRequests(const FullScreenPass &pass,
                                        const std::unordered_map<std::string, std::string> &definitions);
    void buildFullScreenPassPipelines(FullScreenPass &pass,
                                      std::shared_ptr<vk::Device> device,
                                      const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                      size_t &shaderOffset);
    void renderFullScreenPass(const FullScreenPass &pass, MainRenderModuleContext &context);

    void initRenderPassTargets(RenderPass &pass, std::shared_ptr<vk::Device> device, uint32_t frameCount);
    std::vector<ShaderPack::ShaderCreateInfo>
    collectRenderPassShaderRequests(const RenderPass &pass,
                                    const std::unordered_map<std::string, std::string> &definitions);
    void buildRenderPassPipelines(RenderPass &pass,
                                  std::shared_ptr<vk::Device> device,
                                  const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                  size_t &shaderOffset);
    void addRenderIndirectResources(RenderPass &pass, ResolvedPassResources &resources);
    void renderDirectionalShadowMaps(MainRenderModuleContext &context,
                                     const std::shared_ptr<vk::DeviceLocalBuffer> &indirectCommandBuffer,
                                     uint32_t solidDrawCount,
                                     uint32_t alphaOpaqueDrawCount);
    void renderHandOnlyPass(RenderPass &pass, MainRenderModuleContext &context);
    void renderRenderPass(RenderPass &pass, MainRenderModuleContext &context);

    std::vector<ShaderPack::ShaderCreateInfo>
    collectRayTracingPassShaderRequests(RayTracingPass &pass,
                                        const std::unordered_map<std::string, std::string> &definitions);
    void buildRayTracingPassPipelines(RayTracingPass &pass,
                                      std::shared_ptr<vk::Device> device,
                                      const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                      size_t &shaderOffset);
    void renderSharcUpdateAndResolve(RayTracingPass &pass,
                                     MainRenderModuleContext &context,
                                     const std::vector<ExpressionEvaluator::Variable> &expressionVariables);
    void prepareSharcResources(MainRenderModuleContext &context, bool writes);
    void renderRayTracingPass(RayTracingPass &pass,
                              MainRenderModuleContext &context,
                              const std::vector<ExpressionEvaluator::Variable> &expressionVariables);

    std::vector<ShaderPack::ShaderCreateInfo>
    collectComputePassShaderRequests(const ComputePass &pass,
                                     const std::unordered_map<std::string, std::string> &definitions);
    void buildComputePassPipelines(ComputePass &pass,
                                   std::shared_ptr<vk::Device> device,
                                   const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
                                   size_t &shaderOffset);
    void renderComputePass(const ComputePass &pass,
                           MainRenderModuleContext &context,
                           const std::vector<ExpressionEvaluator::Variable> &expressionVariables);
    size_t shaderRequestCountForPass(PassVariant &passVariant);
    void uploadStaticRayTracingPassSbts(std::shared_ptr<vk::Device> device);

  private:
    std::shared_ptr<vk::Shader> fullScreenVertexShader_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> mainRenderDescriptorTables_;

    std::shared_ptr<ShaderPack> shaderPack_;
    std::vector<PassVariant> passes_;
    std::unordered_map<std::string, PassVariant> passNameToPass_;
    std::unordered_map<const FullScreenPass *, ResolvedPassResources> fullScreenPassResources_;
    std::unordered_map<const RenderPass *, ResolvedPassResources> renderPassResources_;
    std::unordered_map<const RayTracingPass *, ResolvedPassResources> rayTracingPassResources_;
    std::unordered_map<const ComputePass *, ResolvedPassResources> computePassResources_;
    std::shared_ptr<RayTracingPass> sharcUpdatePass_;
    std::unique_ptr<ShaderPack::SequentialOp> compiledExecutionProgram_;

    std::shared_ptr<vk::DeviceLocalBuffer> sharcHashEntriesBuffer_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcLockBuffer_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcAccumulationBuffer_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcResolvedBuffer_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> sharcConfigBuffers_;

    bool hasRasterIndirectPass_ = false;
    bool directionalShadowRasterEnabled_ = false;
    std::shared_ptr<vk::RenderPass> directionalShadowRenderPass_;
    std::shared_ptr<vk::Shader> directionalShadowSolidVertexShader_;
    std::shared_ptr<vk::Shader> directionalShadowSolidFragmentShader_;
    std::shared_ptr<vk::Shader> directionalShadowVertexShader_;
    std::shared_ptr<vk::Shader> directionalShadowOpaqueFragmentShader_;
    std::shared_ptr<vk::Shader> directionalShadowIndirectComputeShader_;
    std::shared_ptr<vk::GraphicsPipeline> directionalShadowSolidPipeline_;
    std::shared_ptr<vk::GraphicsPipeline> directionalShadowOpaquePipeline_;
    std::shared_ptr<vk::ComputePipeline> directionalShadowIndirectComputePipeline_;
    std::shared_ptr<vk::Sampler> directionalShadowDepthSampler_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directionalShadowDepthImages_;
    std::vector<std::shared_ptr<vk::Framebuffer>> directionalShadowFramebuffers_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> directionalShadowDataBuffers_;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> directionalShadowMetadataBuffers_;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> directionalShadowCommandBuffers_;

    glm::dvec3 sharcPrevCameraPos_ = glm::dvec3(0.0);
    uint32_t sharcFrameIndex_ = 0;
    bool isFirstSharcFrame_ = true;
    bool hasSharcRuntime_ = false;

    bool isJitterEnabled_ = true;
    bool isSharcEnabled_ = true;
    uint32_t sharcDebugMode_ = 0;
    std::string shaderPackPath_;
    float sharcSceneScale_ = defaultSharcSceneScale;
    uint32_t sharcAccumulationFrameNum_ = defaultSharcAccumulationFrameNum;
    uint32_t sharcStaleFrameNumMax_ = defaultSharcStaleFrameNumMax;
    uint32_t sharcUpdateDownsampleFactor_ = defaultSharcUpdateDownsampleFactor;
    uint32_t sharcCapacityLog2_ = defaultSharcCapacityLog2;
    uint32_t sharcCapacity_ = defaultSharcCapacity;
    uint32_t sharcResolveFrameStride_ = defaultSharcResolveFrameStride;
    uint32_t currentSharcResolveEntryCount_ = defaultSharcCapacity;

    std::unordered_map<std::string, ShaderPackLoader::VariableConfig> executionVariableConfigs_;
    std::unordered_map<std::string, std::string> globalVariables_;
    std::unordered_map<std::string, std::string> staticAttributes_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrNoisyOutputImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectSpecularHitDistanceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectDiffuseHitDistanceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectDiffuseDirectionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> indirectSpecularDirectionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitClearImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitRefractionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitBaseEmissionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> fogImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directSpecularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directHitDistanceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directLightGuideNormalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directLightGuideMotionDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> segmentationMaskImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> rasterDepthImages_;

    std::shared_ptr<WorldPrepare> worldPrepare_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

struct MainRenderModuleContext : public WorldModuleContext, SharedObject<MainRenderModuleContext> {
    struct AsyncBranchResources {
        std::shared_ptr<vk::CommandBuffer> commandBuffer;
        std::shared_ptr<vk::Semaphore> readySemaphore;
        std::shared_ptr<vk::Semaphore> completeSemaphore;
    };
    struct TrackedImageState {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stageMask = 0;
        VkAccessFlags2 accessMask = 0;
        bool initialized = false;
    };

    struct TrackedBufferState {
        VkPipelineStageFlags2 stageMask = 0;
        VkAccessFlags2 accessMask = 0;
        bool initialized = false;
    };

    std::weak_ptr<MainRenderModule> mainRenderModule;

    std::shared_ptr<vk::DescriptorTable> mainRenderDescriptorTable;

    std::shared_ptr<vk::DeviceLocalImage> hdrNoisyOutputImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectSpecularHitDistanceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectDiffuseHitDistanceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectDiffuseDirectionImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> indirectSpecularDirectionImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitClearImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitRefractionImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitBaseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> fogImage;
    std::shared_ptr<vk::DeviceLocalImage> directDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> directSpecularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> directHitDistanceImage;
    std::shared_ptr<vk::DeviceLocalImage> directLightGuideNormalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> directLightGuideMotionDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> segmentationMaskImage;

    std::shared_ptr<WorldPrepareContext> worldPrepareContext;
    std::shared_ptr<MainRenderTimingCapture> timingCapture;
    bool worldPrepareBuffersBound = false;
    std::shared_ptr<vk::TLAS> boundWorldPrepareTlas;
#if defined(MCVR_ENABLE_PROFILER)
    uint64_t worldPrepareBufferBindCount = 0;
    uint64_t worldPrepareBufferBindSkippedCount = 0;
    uint64_t worldPrepareTlasBindCount = 0;
    uint64_t worldPrepareTlasBindSkippedCount = 0;
#endif
    std::unordered_map<const void *, TrackedImageState> trackedImageStates;
    std::unordered_map<const void *, TrackedBufferState> trackedBufferStates;
    bool sharcQueryReady = false;
    std::shared_ptr<vk::CommandPool> asyncCommandPool;
    std::vector<AsyncBranchResources> asyncBranchResources;

    MainRenderModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                            std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                            std::shared_ptr<MainRenderModule> mainRenderModule);

    void bindWorldPrepareResources(bool includeAccelerationStructure);
    bool ensureRayTracingWorldPrepareResources();
    void render() override;
    bool populateFrameGenerationInputs(mcvr::FrameGenerationInputs &inputs) override;
};
