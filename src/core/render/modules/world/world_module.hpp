#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <map>

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldPipelineContext;

struct WorldModuleContext;

enum class StereoMode {
    SingleInstance3DDispatch,
    SingleInstanceMultiDispatch,
    DualInstance
};

class WorldModule {
  public:
    WorldModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    virtual bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) = 0;
    virtual bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) = 0;

    virtual void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) = 0;

    virtual void build() = 0;
    virtual std::vector<std::shared_ptr<WorldModuleContext>> &contexts() = 0;

    virtual void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) = 0;

    virtual void preClose() = 0;

    virtual StereoMode stereoMode() const { return StereoMode::SingleInstance3DDispatch; }
    virtual uint32_t eyeCount() const { return eyeCount_; }
    virtual void setEyeCount(uint32_t count) { eyeCount_ = count; }

  protected:
    uint32_t eyeCount_ = 1;
    std::weak_ptr<Framework> framework_;
    std::weak_ptr<WorldPipeline> worldPipeline_;
};

struct WorldModuleContext {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<WorldPipelineContext> worldPipelineContext;

    WorldModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                       std::shared_ptr<WorldPipelineContext> worldPipelineContext);

    virtual StereoMode stereoMode() const { return StereoMode::SingleInstance3DDispatch; }

    virtual void render() = 0;
    virtual void render3D(uint32_t eyeCount) { render(); }
    virtual void renderEye(uint32_t eyeIndex) { currentEyeIndex = eyeIndex; render(); }

    uint32_t currentEyeIndex = 0;
};