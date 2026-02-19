#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

namespace mcvr {

enum class XeSSQualityMode {
    NativeAA = 0,
    UltraQualityPlus = 1,
    UltraQuality = 2,
    Quality = 3,
    Balanced = 4,
    Performance = 5,
    UltraPerformance = 6
};

struct XeSSImage {
    VkImageView imageView = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkImageSubresourceRange subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct XeSSConfig {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t displayWidth = 0;
    uint32_t displayHeight = 0;

    XeSSQualityMode qualityMode = XeSSQualityMode::Quality;
    uint32_t initFlags = 0;

    // xessSetVelocityScale values. Use (1,1) for pixel-space motion vectors.
    float velocityScaleX = 1.0f;
    float velocityScaleY = 1.0f;
};

struct XeSSInput {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    XeSSImage colorTexture;
    XeSSImage velocityTexture;
    XeSSImage depthTexture;
    XeSSImage exposureTexture;
    XeSSImage responsiveMaskTexture;
    XeSSImage outputTexture;

    float jitterOffsetX = 0.0f;
    float jitterOffsetY = 0.0f;
    float exposureScale = 1.0f;
    bool resetHistory = false;

    uint32_t inputWidth = 0;
    uint32_t inputHeight = 0;
};

class XeSSWrapper {
  public:
    XeSSWrapper();
    ~XeSSWrapper();

    static bool getRequiredInstanceExtensions(std::vector<const char *> &extensions, uint32_t *minVkApiVersion);
    static bool getRequiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                                            std::vector<const char *> &extensions);
    static bool queryOptimalInputResolution(VkInstance instance,
                                            VkPhysicalDevice physicalDevice,
                                            VkDevice device,
                                            uint32_t outputWidth,
                                            uint32_t outputHeight,
                                            XeSSQualityMode qualityMode,
                                            uint32_t *outInputWidth,
                                            uint32_t *outInputHeight);

    bool initialize(const XeSSConfig &config);
    bool resize(uint32_t renderWidth, uint32_t renderHeight, uint32_t displayWidth, uint32_t displayHeight);
    bool dispatch(const XeSSInput &input);
    void destroy();

    bool isAvailable() const;
    bool isInitialized() const;
    const char *getName() const;

  private:
    bool initContextAndPipelines();
    bool initXeSS(uint32_t displayWidth, uint32_t displayHeight, XeSSQualityMode qualityMode, uint32_t initFlags);

    static bool isImageValid(const XeSSImage &image);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t displayWidth_ = 0;
    uint32_t displayHeight_ = 0;
    XeSSQualityMode qualityMode_ = XeSSQualityMode::Quality;
    uint32_t initFlags_ = 0;
    float velocityScaleX_ = 1.0f;
    float velocityScaleY_ = 1.0f;

    bool initialized_ = false;
    bool contextCreated_ = false;

#ifdef MCVR_ENABLE_XESS
    void *contextHandle_ = nullptr;
#endif
};

inline bool XeSSWrapper::isInitialized() const {
    return initialized_;
}

} // namespace mcvr
