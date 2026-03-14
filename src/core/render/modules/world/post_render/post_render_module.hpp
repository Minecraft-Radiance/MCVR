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

struct PostRenderPushConstant {
    uint32_t eyeIndex;
};

struct PostRenderModuleContext;

class PostRenderModule : public WorldModule, public SharedObject<PostRenderModule> {
    friend PostRenderModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.post_render.name";
    constexpr static uint32_t inputImageNum = 2;
    constexpr static uint32_t outputImageNum = 1;

    PostRenderModule();

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
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> ldrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;

    // post render
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> worldLightMapImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> worldPostDepthImages_;
    std::vector<std::shared_ptr<vk::Sampler>> worldPostDepthImageSamplers_;

    std::vector<std::shared_ptr<vk::Sampler>> samplers_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    std::shared_ptr<vk::Shader> worldLightMapVertShader_;
    std::shared_ptr<vk::Shader> worldLightMapFragShader_;
    std::shared_ptr<vk::RenderPass> worldLightMapRenderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> worldLightMapFramebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> worldLightMapPipeline_;

    std::shared_ptr<vk::Shader> worldPostColorToDepthVertShader_;
    std::shared_ptr<vk::Shader> worldPostColorToDepthFragShader_;
    std::shared_ptr<vk::RenderPass> worldPostColorToDepthRenderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> worldPostColorToDepthFramebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> worldPostColorToDepthPipeline_;

    std::shared_ptr<vk::Shader> worldPostVertShader_;
    std::shared_ptr<vk::Shader> worldPostFragShader_;
    std::shared_ptr<vk::RenderPass> worldPostRenderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> worldPostFramebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> worldPostPipeline_;

    // world star field
    std::shared_ptr<vk::DeviceLocalBuffer> starFieldVertexBuffer;
    std::shared_ptr<vk::Shader> worldPostStarFieldVertShader_;
    std::shared_ptr<vk::Shader> worldPostStarFieldFragShader_;
    std::shared_ptr<vk::GraphicsPipeline> worldPostStarFieldPipeline_;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> postRenderedImages_;
    std::vector<uint8_t> postRenderedInitialized_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t width_, height_;
};

struct PostRenderModuleContext : public WorldModuleContext, SharedObject<PostRenderModuleContext> {
    std::weak_ptr<PostRenderModule> postRenderModule;

    StereoMode stereoMode() const override { return StereoMode::SingleInstance3DDispatch; }
    void render3D(uint32_t eyeCount) override;

    // input
    std::shared_ptr<vk::DeviceLocalImage> ldrImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;

    // post render (per-eye indexed)
    std::shared_ptr<vk::DeviceLocalImage> worldLightMapImage;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> worldPostDepthImages;
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables;
    std::shared_ptr<vk::Framebuffer> worldLightMapFramebuffer;
    std::vector<std::shared_ptr<vk::Framebuffer>> worldPostColorToDepthFramebuffers;
    std::vector<std::shared_ptr<vk::Framebuffer>> worldPostFramebuffers;

    // output
    std::shared_ptr<vk::DeviceLocalImage> postRenderedImage;

    PostRenderModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                            std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                            std::shared_ptr<PostRenderModule> postRenderModule);

    void render() override;

  private:
    uint32_t eyeCount_;
};
