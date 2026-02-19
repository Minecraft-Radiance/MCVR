#include "xess_wrapper.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef MCVR_ENABLE_XESS
#include <xess/xess.h>
#include <xess/xess_vk.h>
#endif

namespace mcvr {

XeSSWrapper::XeSSWrapper() = default;

XeSSWrapper::~XeSSWrapper() {
    destroy();
}

bool XeSSWrapper::getRequiredInstanceExtensions(std::vector<const char *> &extensions, uint32_t *minVkApiVersion) {
    extensions.clear();

#ifndef MCVR_ENABLE_XESS
    (void)minVkApiVersion;
    return false;
#else
    uint32_t count = 0;
    const char *const *rawExtensions = nullptr;
    uint32_t minVersion = 0;
    xess_result_t result = xessVKGetRequiredInstanceExtensions(&count, &rawExtensions, &minVersion);
    if (result != XESS_RESULT_SUCCESS || rawExtensions == nullptr) { return false; }

    extensions.reserve(count);
    for (uint32_t i = 0; i < count; i++) { extensions.push_back(rawExtensions[i]); }

    if (minVkApiVersion != nullptr) { *minVkApiVersion = minVersion; }
    return true;
#endif
}

bool XeSSWrapper::getRequiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                                              std::vector<const char *> &extensions) {
    extensions.clear();

#ifndef MCVR_ENABLE_XESS
    (void)instance;
    (void)physicalDevice;
    return false;
#else
    uint32_t count = 0;
    const char *const *rawExtensions = nullptr;
    xess_result_t result = xessVKGetRequiredDeviceExtensions(instance, physicalDevice, &count, &rawExtensions);
    if (result != XESS_RESULT_SUCCESS || rawExtensions == nullptr) { return false; }

    extensions.reserve(count);
    for (uint32_t i = 0; i < count; i++) { extensions.push_back(rawExtensions[i]); }
    return true;
#endif
}

bool XeSSWrapper::queryOptimalInputResolution(VkInstance instance,
                                              VkPhysicalDevice physicalDevice,
                                              VkDevice device,
                                              uint32_t outputWidth,
                                              uint32_t outputHeight,
                                              XeSSQualityMode qualityMode,
                                              uint32_t *outInputWidth,
                                              uint32_t *outInputHeight) {
#ifndef MCVR_ENABLE_XESS
    (void)instance;
    (void)physicalDevice;
    (void)device;
    (void)outputWidth;
    (void)outputHeight;
    (void)qualityMode;
    (void)outInputWidth;
    (void)outInputHeight;
    return false;
#else
    if (instance == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE || device == VK_NULL_HANDLE ||
        outputWidth == 0 || outputHeight == 0 || outInputWidth == nullptr || outInputHeight == nullptr) {
        return false;
    }

    auto toXeSSQuality = [](XeSSQualityMode mode) -> xess_quality_settings_t {
        switch (mode) {
            case XeSSQualityMode::NativeAA: return XESS_QUALITY_SETTING_AA;
            case XeSSQualityMode::Quality: return XESS_QUALITY_SETTING_QUALITY;
            case XeSSQualityMode::Balanced: return XESS_QUALITY_SETTING_BALANCED;
            case XeSSQualityMode::Performance: return XESS_QUALITY_SETTING_PERFORMANCE;
            case XeSSQualityMode::UltraPerformance: return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
            default: return XESS_QUALITY_SETTING_QUALITY;
        }
    };

    xess_context_handle_t ctx = nullptr;
    xess_result_t createResult = xessVKCreateContext(instance, physicalDevice, device, &ctx);
    if (createResult != XESS_RESULT_SUCCESS || ctx == nullptr) { return false; }

    xess_2d_t outRes = {outputWidth, outputHeight};
    xess_2d_t optimal = {};
    xess_2d_t minRes = {};
    xess_2d_t maxRes = {};

    xess_result_t result = xessGetOptimalInputResolution(ctx, &outRes, toXeSSQuality(qualityMode), &optimal, &minRes, &maxRes);
    if (result != XESS_RESULT_SUCCESS || optimal.x == 0 || optimal.y == 0) {
        xess_2d_t inputRes = {};
        result = xessGetInputResolution(ctx, &outRes, toXeSSQuality(qualityMode), &inputRes);
        if (result == XESS_RESULT_SUCCESS && inputRes.x > 0 && inputRes.y > 0) {
            optimal = inputRes;
        }
    }

    xessDestroyContext(ctx);

    if (optimal.x == 0 || optimal.y == 0) { return false; }
    *outInputWidth = optimal.x;
    *outInputHeight = optimal.y;
    return true;
#endif
}

bool XeSSWrapper::initialize(const XeSSConfig &config) {
    destroy();

    instance_ = config.instance;
    physicalDevice_ = config.physicalDevice;
    device_ = config.device;
    renderWidth_ = config.renderWidth;
    renderHeight_ = config.renderHeight;
    displayWidth_ = config.displayWidth;
    displayHeight_ = config.displayHeight;
    qualityMode_ = config.qualityMode;
    initFlags_ = config.initFlags;
    velocityScaleX_ = config.velocityScaleX;
    velocityScaleY_ = config.velocityScaleY;

    if (instance_ == VK_NULL_HANDLE || physicalDevice_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE ||
        renderWidth_ == 0 || renderHeight_ == 0 || displayWidth_ == 0 || displayHeight_ == 0) {
        std::cerr << "[XeSS] Invalid init config" << std::endl;
        return false;
    }

#ifndef MCVR_ENABLE_XESS
    std::cerr << "[XeSS] XeSS support is not enabled at compile time" << std::endl;
    return false;
#else
    if (!initContextAndPipelines()) { return false; }

    if (!initXeSS(displayWidth_, displayHeight_, qualityMode_, initFlags_)) {
        destroy();
        return false;
    }

    initialized_ = true;
    return true;
#endif
}

bool XeSSWrapper::resize(uint32_t renderWidth, uint32_t renderHeight, uint32_t displayWidth, uint32_t displayHeight) {
    if (!contextCreated_) { return false; }

    renderWidth_ = renderWidth;
    renderHeight_ = renderHeight;
    displayWidth_ = displayWidth;
    displayHeight_ = displayHeight;

#ifndef MCVR_ENABLE_XESS
    return false;
#else
    return initXeSS(displayWidth_, displayHeight_, qualityMode_, initFlags_);
#endif
}

bool XeSSWrapper::dispatch(const XeSSInput &input) {
    if (!initialized_ || !contextCreated_ || input.commandBuffer == VK_NULL_HANDLE) { return false; }

    if (!isImageValid(input.colorTexture) || !isImageValid(input.velocityTexture) || !isImageValid(input.outputTexture) ||
        input.inputWidth == 0 || input.inputHeight == 0) {
        std::cerr << "[XeSS] Invalid execute input" << std::endl;
        return false;
    }

#ifndef MCVR_ENABLE_XESS
    (void)input;
    return false;
#else
    auto toXeSSImage = [](const XeSSImage &src) -> xess_vk_image_view_info {
        xess_vk_image_view_info dst{};
        dst.imageView = src.imageView;
        dst.image = src.image;
        dst.subresourceRange = src.subresourceRange;
        dst.format = src.format;
        dst.width = src.width;
        dst.height = src.height;
        return dst;
    };

    xess_vk_execute_params_t exec{};
    exec.colorTexture = toXeSSImage(input.colorTexture);
    exec.velocityTexture = toXeSSImage(input.velocityTexture);
    exec.depthTexture = toXeSSImage(input.depthTexture);
    exec.exposureScaleTexture = toXeSSImage(input.exposureTexture);
    exec.responsivePixelMaskTexture = toXeSSImage(input.responsiveMaskTexture);
    exec.outputTexture = toXeSSImage(input.outputTexture);
    exec.jitterOffsetX = input.jitterOffsetX;
    exec.jitterOffsetY = input.jitterOffsetY;
    exec.exposureScale = input.exposureScale;
    exec.resetHistory = input.resetHistory ? 1u : 0u;
    exec.inputWidth = input.inputWidth;
    exec.inputHeight = input.inputHeight;

    xess_result_t result = xessVKExecute(reinterpret_cast<xess_context_handle_t>(contextHandle_), input.commandBuffer, &exec);
    if (result != XESS_RESULT_SUCCESS) {
        std::cerr << "[XeSS] xessVKExecute failed: " << static_cast<int>(result) << std::endl;
        return false;
    }
    return true;
#endif
}

void XeSSWrapper::destroy() {
#ifdef MCVR_ENABLE_XESS
    if (contextCreated_ && contextHandle_ != nullptr) {
        xessDestroyContext(reinterpret_cast<xess_context_handle_t>(contextHandle_));
        contextHandle_ = nullptr;
    }
#endif
    initialized_ = false;
    contextCreated_ = false;
}

bool XeSSWrapper::isAvailable() const {
#ifdef MCVR_ENABLE_XESS
    return true;
#else
    return false;
#endif
}

const char *XeSSWrapper::getName() const {
    return "Intel XeSS";
}

bool XeSSWrapper::initContextAndPipelines() {
#ifndef MCVR_ENABLE_XESS
    return false;
#else
    xess_result_t result = xessVKCreateContext(instance_, physicalDevice_, device_,
                                               reinterpret_cast<xess_context_handle_t *>(&contextHandle_));
    if (result != XESS_RESULT_SUCCESS || contextHandle_ == nullptr) {
        std::cerr << "[XeSS] xessVKCreateContext failed: " << static_cast<int>(result) << std::endl;
        return false;
    }
    contextCreated_ = true;
    return true;
#endif
}

bool XeSSWrapper::initXeSS(uint32_t displayWidth, uint32_t displayHeight, XeSSQualityMode qualityMode, uint32_t initFlags) {
#ifndef MCVR_ENABLE_XESS
    (void)displayWidth;
    (void)displayHeight;
    (void)qualityMode;
    (void)initFlags;
    return false;
#else
    auto toXeSSQuality = [](XeSSQualityMode mode) -> xess_quality_settings_t {
        switch (mode) {
            case XeSSQualityMode::NativeAA: return XESS_QUALITY_SETTING_AA;
            case XeSSQualityMode::Quality: return XESS_QUALITY_SETTING_QUALITY;
            case XeSSQualityMode::Balanced: return XESS_QUALITY_SETTING_BALANCED;
            case XeSSQualityMode::Performance: return XESS_QUALITY_SETTING_PERFORMANCE;
            case XeSSQualityMode::UltraPerformance: return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
            default: return XESS_QUALITY_SETTING_QUALITY;
        }
    };

    xess_vk_init_params_t initParams{};
    initParams.outputResolution = {displayWidth, displayHeight};
    initParams.qualitySetting = toXeSSQuality(qualityMode);
    initParams.initFlags = initFlags;
    initParams.creationNodeMask = 0;
    initParams.visibleNodeMask = 0;
    initParams.tempBufferHeap = VK_NULL_HANDLE;
    initParams.bufferHeapOffset = 0;
    initParams.tempTextureHeap = VK_NULL_HANDLE;
    initParams.textureHeapOffset = 0;
    initParams.pipelineCache = VK_NULL_HANDLE;

    xess_result_t initResult = xessVKInit(reinterpret_cast<xess_context_handle_t>(contextHandle_), &initParams);
    if (initResult != XESS_RESULT_SUCCESS) {
        std::cerr << "[XeSS] xessVKInit failed: " << static_cast<int>(initResult) << std::endl;
        return false;
    }

    xess_result_t velocityResult =
        xessSetVelocityScale(reinterpret_cast<xess_context_handle_t>(contextHandle_), velocityScaleX_, velocityScaleY_);
    if (velocityResult != XESS_RESULT_SUCCESS) {
        std::cerr << "[XeSS] xessSetVelocityScale failed: " << static_cast<int>(velocityResult) << std::endl;
    }

    return true;
#endif
}

bool XeSSWrapper::isImageValid(const XeSSImage &image) {
    return image.image != VK_NULL_HANDLE && image.imageView != VK_NULL_HANDLE && image.format != VK_FORMAT_UNDEFINED &&
           image.width > 0 && image.height > 0;
}

} // namespace mcvr
