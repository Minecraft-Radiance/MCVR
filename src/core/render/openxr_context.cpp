#ifdef MCVR_ENABLE_OPENXR

#include "core/render/openxr_context.hpp"
#include "core/render/renderer.hpp"

#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

// Tee streambuf: writes to both a file and another streambuf
class TeeStreambuf : public std::streambuf {
public:
    TeeStreambuf(std::streambuf *a, std::streambuf *b) : a_(a), b_(b) {}
protected:
    int overflow(int c) override {
        if (c == EOF) return !EOF;
        int r1 = a_->sputc(c);
        int r2 = b_->sputc(c);
        return (r1 == EOF || r2 == EOF) ? EOF : c;
    }
    int sync() override { a_->pubsync(); b_->pubsync(); return 0; }
private:
    std::streambuf *a_;
    std::streambuf *b_;
};

static std::ofstream &xrLogFile() {
    static std::ofstream file("openxr_debug.log", std::ios::out | std::ios::trunc);
    return file;
}

static std::ostream &xrCout() {
    static TeeStreambuf tee(std::cout.rdbuf(), xrLogFile().rdbuf());
    static std::ostream teeStream(&tee);
    return teeStream << "[OpenXR] ";
}

static std::ostream &xrCerr() {
    static TeeStreambuf tee(std::cerr.rdbuf(), xrLogFile().rdbuf());
    static std::ostream teeStream(&tee);
    return teeStream << "[OpenXR][ERROR] ";
}

static const char *xrResultStr(XrResult result) {
    switch (result) {
        case XR_SUCCESS: return "XR_SUCCESS";
        case XR_ERROR_FORM_FACTOR_UNAVAILABLE: return "XR_ERROR_FORM_FACTOR_UNAVAILABLE";
        case XR_ERROR_RUNTIME_UNAVAILABLE: return "XR_ERROR_RUNTIME_UNAVAILABLE";
        case XR_ERROR_INSTANCE_LOST: return "XR_ERROR_INSTANCE_LOST";
        case XR_ERROR_SESSION_LOST: return "XR_ERROR_SESSION_LOST";
        default: return "XR_UNKNOWN";
    }
}

#define XR_CHECK(call, msg)                                                         \
    do {                                                                            \
        XrResult _r = (call);                                                       \
        if (XR_FAILED(_r)) {                                                        \
            xrCerr() << (msg) << " failed: " << xrResultStr(_r) << std::endl;       \
            return false;                                                            \
        }                                                                           \
    } while (0)

#define XR_CHECK_VOID(call, msg)                                                    \
    do {                                                                            \
        XrResult _r = (call);                                                       \
        if (XR_FAILED(_r)) {                                                        \
            xrCerr() << (msg) << " failed: " << xrResultStr(_r) << std::endl;       \
            return;                                                                  \
        }                                                                           \
    } while (0)

// Split a space-delimited string of extensions into a vector of individual names.
static std::vector<std::string> splitExtensions(const char *str) {
    std::vector<std::string> result;
    std::istringstream ss(str);
    std::string token;
    while (ss >> token) { result.push_back(token); }
    return result;
}

// ---- Stage A: preVulkanInit ----

bool OpenXRContext::preVulkanInit() {
    // 1. Create XrInstance
    XrInstanceCreateInfo instanceCI{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strncpy(instanceCI.applicationInfo.applicationName, "MCVR", XR_MAX_APPLICATION_NAME_SIZE);
    instanceCI.applicationInfo.applicationVersion = 1;
    std::strncpy(instanceCI.applicationInfo.engineName, "Radiance", XR_MAX_ENGINE_NAME_SIZE);
    instanceCI.applicationInfo.engineVersion = 1;
    instanceCI.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 1, 0);

    // Request Vulkan graphics binding extension (v1 — matches our use of v1 query functions)
    // Also request optional extensions: visibility mask, eye gaze
    std::vector<const char *> requestedExts = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME};

    // Check available extensions and add optional ones
    uint32_t availExtCount = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &availExtCount, nullptr);
    std::vector<XrExtensionProperties> availExts(availExtCount, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, availExtCount, &availExtCount, availExts.data());
    for (auto &ext : availExts) {
        if (std::strcmp(ext.extensionName, "XR_KHR_visibility_mask") == 0) {
            requestedExts.push_back("XR_KHR_visibility_mask");
            visMaskAvailable_ = true;
            xrCout() << "Visibility mask extension available" << std::endl;
        }
        if (std::strcmp(ext.extensionName, "XR_EXT_eye_gaze_interaction") == 0) {
            requestedExts.push_back("XR_EXT_eye_gaze_interaction");
            xrCout() << "Eye gaze interaction extension available" << std::endl;
        }
    }

    instanceCI.enabledExtensionCount = static_cast<uint32_t>(requestedExts.size());
    instanceCI.enabledExtensionNames = requestedExts.data();

    XR_CHECK(xrCreateInstance(&instanceCI, &xrInstance_), "xrCreateInstance");
    xrCout() << "XrInstance created" << std::endl;

    // 2. Get system (HMD)
    XrSystemGetInfo systemGI{XR_TYPE_SYSTEM_GET_INFO};
    systemGI.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XR_CHECK(xrGetSystem(xrInstance_, &systemGI, &systemId_), "xrGetSystem");
    xrCout() << "XrSystem acquired (id=" << systemId_ << ")" << std::endl;

    // Query system properties (device name)
    XrSystemProperties sysProps{XR_TYPE_SYSTEM_PROPERTIES};
    if (xrGetSystemProperties(xrInstance_, systemId_, &sysProps) == XR_SUCCESS) {
        systemName_ = sysProps.systemName;
        xrCout() << "System name: " << systemName_ << std::endl;
    }

    // 3. Query Vulkan instance extensions required by OpenXR
    PFN_xrGetVulkanInstanceExtensionsKHR xrGetVulkanInstanceExtensionsKHR = nullptr;
    XR_CHECK(xrGetInstanceProcAddr(xrInstance_, "xrGetVulkanInstanceExtensionsKHR",
                                   reinterpret_cast<PFN_xrVoidFunction *>(&xrGetVulkanInstanceExtensionsKHR)),
             "get xrGetVulkanInstanceExtensionsKHR");

    uint32_t bufSize = 0;
    XR_CHECK(xrGetVulkanInstanceExtensionsKHR(xrInstance_, systemId_, 0, &bufSize, nullptr),
             "xrGetVulkanInstanceExtensionsKHR(size)");
    std::vector<char> instExtBuf(bufSize);
    XR_CHECK(xrGetVulkanInstanceExtensionsKHR(xrInstance_, systemId_, bufSize, &bufSize, instExtBuf.data()),
             "xrGetVulkanInstanceExtensionsKHR(data)");
    requiredInstanceExts_ = splitExtensions(instExtBuf.data());

    xrCout() << "Required Vulkan instance extensions:" << std::endl;
    for (auto &e : requiredInstanceExts_) { xrCout() << "  " << e << std::endl; }

    // 4. Query Vulkan device extensions required by OpenXR
    PFN_xrGetVulkanDeviceExtensionsKHR xrGetVulkanDeviceExtensionsKHR = nullptr;
    XR_CHECK(xrGetInstanceProcAddr(xrInstance_, "xrGetVulkanDeviceExtensionsKHR",
                                   reinterpret_cast<PFN_xrVoidFunction *>(&xrGetVulkanDeviceExtensionsKHR)),
             "get xrGetVulkanDeviceExtensionsKHR");

    bufSize = 0;
    XR_CHECK(xrGetVulkanDeviceExtensionsKHR(xrInstance_, systemId_, 0, &bufSize, nullptr),
             "xrGetVulkanDeviceExtensionsKHR(size)");
    std::vector<char> devExtBuf(bufSize);
    XR_CHECK(xrGetVulkanDeviceExtensionsKHR(xrInstance_, systemId_, bufSize, &bufSize, devExtBuf.data()),
             "xrGetVulkanDeviceExtensionsKHR(data)");
    requiredDeviceExts_ = splitExtensions(devExtBuf.data());

    xrCout() << "Required Vulkan device extensions:" << std::endl;
    for (auto &e : requiredDeviceExts_) { xrCout() << "  " << e << std::endl; }

    // 5. Query view configuration (stereo) to get recommended resolution
    uint32_t viewCount = 0;
    XR_CHECK(xrEnumerateViewConfigurationViews(xrInstance_, systemId_, viewConfigType_, 0, &viewCount, nullptr),
             "xrEnumerateViewConfigurationViews(count)");
    if (viewCount != 2) {
        xrCerr() << "Expected 2 stereo views, got " << viewCount << std::endl;
        return false;
    }
    configViews_[0] = {XR_TYPE_VIEW_CONFIGURATION_VIEW};
    configViews_[1] = {XR_TYPE_VIEW_CONFIGURATION_VIEW};
    XR_CHECK(xrEnumerateViewConfigurationViews(xrInstance_, systemId_, viewConfigType_, 2, &viewCount,
                                               configViews_.data()),
             "xrEnumerateViewConfigurationViews(data)");

    xrCout() << "Recommended per-eye resolution: "
             << configViews_[0].recommendedImageRectWidth << "x"
             << configViews_[0].recommendedImageRectHeight << std::endl;

    return true;
}

VkPhysicalDevice OpenXRContext::getXRPhysicalDevice(VkInstance vkInstance) const {
    PFN_xrGetVulkanGraphicsDeviceKHR xrGetVulkanGraphicsDeviceKHR = nullptr;
    xrGetInstanceProcAddr(xrInstance_, "xrGetVulkanGraphicsDeviceKHR",
                          reinterpret_cast<PFN_xrVoidFunction *>(&xrGetVulkanGraphicsDeviceKHR));
    VkPhysicalDevice xrDevice = VK_NULL_HANDLE;
    XrResult r = xrGetVulkanGraphicsDeviceKHR(xrInstance_, systemId_, vkInstance, &xrDevice);
    if (XR_FAILED(r)) {
        xrCerr() << "xrGetVulkanGraphicsDeviceKHR failed" << std::endl;
        return VK_NULL_HANDLE;
    }
    return xrDevice;
}

// ---- Stage B: postVulkanInit ----

bool OpenXRContext::postVulkanInit(VkInstance vkInstance, VkPhysicalDevice vkPhysicalDevice,
                                   VkDevice vkDevice, uint32_t queueFamilyIndex,
                                   uint32_t queueIndex) {
    vkInstance_ = vkInstance;
    vkPhysicalDevice_ = vkPhysicalDevice;
    vkDevice_ = vkDevice;
    queueFamilyIndex_ = queueFamilyIndex;
    queueIndex_ = queueIndex;
    vulkanReady_ = true;
    xrCout() << "OpenXR Vulkan bridge ready (session deferred)" << std::endl;
    return true;
}

bool OpenXRContext::createSession(VkInstance vkInstance, VkPhysicalDevice vkPhysicalDevice,
                                  VkDevice vkDevice, uint32_t queueFamilyIndex,
                                  uint32_t queueIndex) {
    // Vulkan graphics binding
    XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    binding.instance = vkInstance;
    binding.physicalDevice = vkPhysicalDevice;
    binding.device = vkDevice;
    binding.queueFamilyIndex = queueFamilyIndex;
    binding.queueIndex = queueIndex;

    // Spec requires calling xrGetVulkanGraphicsRequirementsKHR before xrCreateSession
    PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR = nullptr;
    XR_CHECK(xrGetInstanceProcAddr(xrInstance_, "xrGetVulkanGraphicsRequirementsKHR",
                                   reinterpret_cast<PFN_xrVoidFunction *>(&xrGetVulkanGraphicsRequirementsKHR)),
             "get xrGetVulkanGraphicsRequirementsKHR");
    XrGraphicsRequirementsVulkanKHR graphicsReqs{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
    XR_CHECK(xrGetVulkanGraphicsRequirementsKHR(xrInstance_, systemId_, &graphicsReqs),
             "xrGetVulkanGraphicsRequirementsKHR");
    xrCout() << "Vulkan requirements: min=" << XR_VERSION_MAJOR(graphicsReqs.minApiVersionSupported)
             << "." << XR_VERSION_MINOR(graphicsReqs.minApiVersionSupported)
             << " max=" << XR_VERSION_MAJOR(graphicsReqs.maxApiVersionSupported)
             << "." << XR_VERSION_MINOR(graphicsReqs.maxApiVersionSupported) << std::endl;

    XrSessionCreateInfo sessionCI{XR_TYPE_SESSION_CREATE_INFO};
    sessionCI.next = &binding;
    sessionCI.systemId = systemId_;
    XR_CHECK(xrCreateSession(xrInstance_, &sessionCI, &session_), "xrCreateSession");
    xrCout() << "XrSession created" << std::endl;

    // Create reference space (LOCAL = seated, STAGE = standing/room-scale)
    XrReferenceSpaceCreateInfo spaceCI{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    spaceCI.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    spaceCI.poseInReferenceSpace = {{0, 0, 0, 1}, {0, 0, 0}};
    XR_CHECK(xrCreateReferenceSpace(session_, &spaceCI, &appSpace_), "xrCreateReferenceSpace");
    xrCout() << "Reference space created (LOCAL)" << std::endl;

    return true;
}

bool OpenXRContext::createSwapchains() {
    // Enumerate supported swapchain formats — prefer SRGB, fall back to UNORM
    uint32_t formatCount = 0;
    XR_CHECK(xrEnumerateSwapchainFormats(session_, 0, &formatCount, nullptr), "xrEnumerateSwapchainFormats(count)");
    std::vector<int64_t> formats(formatCount);
    XR_CHECK(xrEnumerateSwapchainFormats(session_, formatCount, &formatCount, formats.data()),
             "xrEnumerateSwapchainFormats(data)");

    // Prefer R8G8B8A8_SRGB, then R8G8B8A8_UNORM, then B8G8R8A8_SRGB
    int64_t selectedFormat = formats[0];
    const int64_t preferred[] = {
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_B8G8R8A8_UNORM,
    };
    for (int64_t pref : preferred) {
        for (int64_t fmt : formats) {
            if (fmt == pref) { selectedFormat = pref; goto found; }
        }
    }
    found:
    xrCout() << "Swapchain format: " << selectedFormat << std::endl;

    // Create per-eye swapchains
    for (uint32_t eye = 0; eye < 2; eye++) {
        auto &es = eyeSwapchains_[eye];
        es.width = configViews_[eye].recommendedImageRectWidth;
        es.height = configViews_[eye].recommendedImageRectHeight;

        XrSwapchainCreateInfo swapCI{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        swapCI.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        swapCI.format = selectedFormat;
        swapCI.sampleCount = 1;
        swapCI.width = es.width;
        swapCI.height = es.height;
        swapCI.faceCount = 1;
        swapCI.arraySize = 1;
        swapCI.mipCount = 1;
        XR_CHECK(xrCreateSwapchain(session_, &swapCI, &es.handle), "xrCreateSwapchain");

        // Enumerate VkImages owned by the swapchain
        uint32_t imgCount = 0;
        XR_CHECK(xrEnumerateSwapchainImages(es.handle, 0, &imgCount, nullptr), "xrEnumerateSwapchainImages(count)");
        std::vector<XrSwapchainImageVulkanKHR> xrImages(imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        XR_CHECK(xrEnumerateSwapchainImages(es.handle, imgCount, &imgCount,
                                            reinterpret_cast<XrSwapchainImageBaseHeader *>(xrImages.data())),
                 "xrEnumerateSwapchainImages(data)");
        es.images.resize(imgCount);
        for (uint32_t i = 0; i < imgCount; i++) { es.images[i] = xrImages[i].image; }

        xrCout() << "Eye " << eye << " swapchain: " << es.width << "x" << es.height
                 << ", " << imgCount << " images" << std::endl;
    }
    return true;
}

// ---- Per-frame operations ----

void OpenXRContext::pollEvents() {
    XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(xrInstance_, &event) == XR_SUCCESS) {
        switch (event.type) {
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                auto *stateEvent = reinterpret_cast<XrEventDataSessionStateChanged *>(&event);
                handleSessionStateChange(stateEvent->state);
                break;
            }
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                xrCerr() << "XR instance loss pending!" << std::endl;
                break;
            default:
                break;
        }
        event = {XR_TYPE_EVENT_DATA_BUFFER};
    }
}

void OpenXRContext::handleSessionStateChange(XrSessionState newState) {
    sessionState_ = newState;
    xrCout() << "Session state -> " << static_cast<int>(newState) << std::endl;

    switch (newState) {
        case XR_SESSION_STATE_READY: {
            if (sessionRequested_) {
                XrSessionBeginInfo beginInfo{XR_TYPE_SESSION_BEGIN_INFO};
                beginInfo.primaryViewConfigurationType = viewConfigType_;
                XrResult r = xrBeginSession(session_, &beginInfo);
                if (XR_SUCCEEDED(r)) {
                    sessionRunning_ = true;
                    xrCout() << "Session started" << std::endl;
                }
            } else {
                xrCout() << "Session READY but not requested, waiting..." << std::endl;
            }
            break;
        }
        case XR_SESSION_STATE_STOPPING:
            if (session_ != XR_NULL_HANDLE) {
                xrEndSession(session_);
            }
            sessionRunning_ = false;
            xrCout() << "Session stopped" << std::endl;
            if (destroyPending_ && !sessionRequested_) {
                destroySessionResources();
            }
            break;
        case XR_SESSION_STATE_EXITING:
        case XR_SESSION_STATE_LOSS_PENDING:
            sessionRunning_ = false;
            if (destroyPending_) {
                destroySessionResources();
            }
            break;
        default:
            break;
    }
}

bool OpenXRContext::requestSessionStart() {
    if (!vulkanReady_) {
        xrCerr() << "Cannot start XR session: Vulkan bridge is not ready" << std::endl;
        return false;
    }

    sessionRequested_ = true;
    destroyPending_ = false;

    if (session_ == XR_NULL_HANDLE) {
        if (!createSession(vkInstance_, vkPhysicalDevice_, vkDevice_, queueFamilyIndex_, queueIndex_)) {
            xrCerr() << "Failed to create XR session on demand" << std::endl;
            sessionRequested_ = false;
            return false;
        }
        if (!createSwapchains()) {
            xrCerr() << "Failed to create XR swapchains on demand" << std::endl;
            destroySessionResources();
            sessionRequested_ = false;
            return false;
        }

        if (input_.createActions(xrInstance_, session_) && input_.attachToSession(session_)) {
            inputInitialized_ = true;
        }

        if (visMaskAvailable_) {
            queryVisibilityMask();
        }
    }

    if (sessionRunning_) return true;

    // If runtime is already READY, begin immediately. Otherwise pollEvents()
    // will begin when READY arrives.
    if (sessionState_ == XR_SESSION_STATE_READY) {
        XrSessionBeginInfo beginInfo{XR_TYPE_SESSION_BEGIN_INFO};
        beginInfo.primaryViewConfigurationType = viewConfigType_;
        XrResult r = xrBeginSession(session_, &beginInfo);
        if (XR_SUCCEEDED(r)) {
            sessionRunning_ = true;
            xrCout() << "Session started (requested immediately)" << std::endl;
            return true;
        }
        xrCerr() << "xrBeginSession failed on request" << std::endl;
        sessionRequested_ = false;
        return false;
    }

    return true;
}

void OpenXRContext::requestSessionStop() {
    sessionRequested_ = false;
    destroyPending_ = true;

    if (session_ == XR_NULL_HANDLE) {
        destroyPending_ = false;
        return;
    }

    if (frameState_ == FRAME_LATCHED) {
        endFrame();
    }

    for (auto &es : eyeSwapchains_) {
        if (es.handle != XR_NULL_HANDLE && es.imageAcquired) {
            XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            XrResult rr = xrReleaseSwapchainImage(es.handle, &releaseInfo);
            if (XR_FAILED(rr)) {
                xrCerr() << "xrReleaseSwapchainImage during stop failed" << std::endl;
            }
            es.imageAcquired = false;
        }
    }

    if (session_ != XR_NULL_HANDLE && sessionRunning_) {
        // Ask runtime to transition to STOPPING first.
        XrResult r = xrRequestExitSession(session_);
        if (XR_FAILED(r)) {
            xrCerr() << "xrRequestExitSession failed, forcing xrEndSession" << std::endl;
            xrEndSession(session_);
            sessionRunning_ = false;
        }
    }

    if (!sessionRunning_) {
        destroySessionResources();
        destroyPending_ = false;
    }
}

void OpenXRContext::destroySessionResources() {
    if (inputInitialized_) {
        input_.shutdown();
        inputInitialized_ = false;
    }

    for (auto &es : eyeSwapchains_) {
        if (es.handle != XR_NULL_HANDLE && es.imageAcquired) {
            XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            xrReleaseSwapchainImage(es.handle, &releaseInfo);
            es.imageAcquired = false;
        }
        if (es.handle != XR_NULL_HANDLE) {
            xrDestroySwapchain(es.handle);
            es.handle = XR_NULL_HANDLE;
        }
        es.images.clear();
        es.width = 0;
        es.height = 0;
        es.acquiredIndex = 0;
        es.imageAcquired = false;
    }

    if (appSpace_ != XR_NULL_HANDLE) {
        xrDestroySpace(appSpace_);
        appSpace_ = XR_NULL_HANDLE;
    }

    if (session_ != XR_NULL_HANDLE) {
        xrDestroySession(session_);
        session_ = XR_NULL_HANDLE;
    }

    sessionRunning_ = false;
    sessionState_ = XR_SESSION_STATE_UNKNOWN;
    frameState_ = FRAME_IDLE;
    viewsValid_ = false;
    destroyPending_ = false;
    visMaskVertices_[0].clear();
    visMaskVertices_[1].clear();
    visMaskIndices_[0].clear();
    visMaskIndices_[1].clear();
}

void OpenXRContext::beginFrameRecording() {
    frameState_ = FRAME_RECORDING;
    viewsValid_ = false;  // Will be updated in latchPose
    lastWaitFrameMs_ = 0.0f;
    lastSwapchainWaitMs_ = 0.0f;

    // Initialize XR frame state for upcoming latchPose() call
    xrFrameState_ = {XR_TYPE_FRAME_STATE};
}

bool OpenXRContext::latchPose() {
    // Consolidate session state validation
    if (!sessionRunning_ || sessionState_ < XR_SESSION_STATE_READY || sessionState_ > XR_SESSION_STATE_FOCUSED) {
        return false;
    }

    if (frameState_ != FRAME_RECORDING) {
        xrCerr() << "latchPose called without beginFrameRecording()" << std::endl;
        return false;
    }

    // CRITICAL: This is where we call the blocking xrWaitFrame
    XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
    auto waitFrameStart = std::chrono::high_resolution_clock::now();
    XrResult r = xrWaitFrame(session_, &waitInfo, &xrFrameState_);
    lastWaitFrameMs_ = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - waitFrameStart).count();
    if (XR_FAILED(r)) {
        xrCerr() << "xrWaitFrame failed" << std::endl;
        return false;
    }

    XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    r = xrBeginFrame(session_, &beginInfo);
    if (XR_FAILED(r)) {
        xrCerr() << "xrBeginFrame failed" << std::endl;
        return false;
    }

    frameState_ = FRAME_LATCHED;

    // Locate views (head pose + per-eye FOV) with latest timing
    viewsValid_ = false;
    XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
    locateInfo.viewConfigurationType = viewConfigType_;
    locateInfo.displayTime = xrFrameState_.predictedDisplayTime;
    locateInfo.space = appSpace_;

    XrViewState viewState{XR_TYPE_VIEW_STATE};
    uint32_t viewCount = 0;
    views_[0] = {XR_TYPE_VIEW};
    views_[1] = {XR_TYPE_VIEW};
    r = xrLocateViews(session_, &locateInfo, &viewState, 2, &viewCount, views_.data());
    if (XR_SUCCEEDED(r) && (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
        viewsValid_ = true;
    }

    // Sync controller input with latest timing
    if (inputInitialized_) {
        input_.syncAndUpdate(session_, appSpace_, xrFrameState_.predictedDisplayTime);
    }

    return true;
}

VkImage OpenXRContext::acquireSwapchainImage(uint32_t eye) {
    auto &es = eyeSwapchains_[eye];
    if (es.handle == XR_NULL_HANDLE || es.images.empty()) {
        xrCerr() << "acquireSwapchainImage called without a valid swapchain" << std::endl;
        return VK_NULL_HANDLE;
    }

    XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    XrResult r = xrAcquireSwapchainImage(es.handle, &acquireInfo, &es.acquiredIndex);
    if (XR_FAILED(r)) {
        xrCerr() << "xrAcquireSwapchainImage failed" << std::endl;
        return VK_NULL_HANDLE;
    }

    XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    waitInfo.timeout = XR_INFINITE_DURATION;
    auto waitSwapchainStart = std::chrono::high_resolution_clock::now();
    r = xrWaitSwapchainImage(es.handle, &waitInfo);
    lastSwapchainWaitMs_ += std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - waitSwapchainStart).count();
    if (XR_FAILED(r)) {
        xrCerr() << "xrWaitSwapchainImage failed" << std::endl;
        return VK_NULL_HANDLE;
    }

    if (es.acquiredIndex >= es.images.size()) {
        xrCerr() << "acquired swapchain image index out of range" << std::endl;
        return VK_NULL_HANDLE;
    }

    es.imageAcquired = true;

    return es.images[es.acquiredIndex];
}

void OpenXRContext::releaseSwapchainImage(uint32_t eye) {
    auto &es = eyeSwapchains_[eye];
    if (es.handle == XR_NULL_HANDLE || !es.imageAcquired) {
        return;
    }

    XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    XrResult r = xrReleaseSwapchainImage(es.handle, &releaseInfo);
    if (XR_FAILED(r)) {
        xrCerr() << "xrReleaseSwapchainImage failed" << std::endl;
        return;
    }
    es.imageAcquired = false;
}

void OpenXRContext::endFrame() {
    if (frameState_ != FRAME_LATCHED) return;
    frameState_ = FRAME_IDLE;

    // Build projection views referencing each eye's swapchain
    for (uint32_t eye = 0; eye < 2; eye++) {
        projViews_[eye] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        projViews_[eye].pose = views_[eye].pose;
        projViews_[eye].fov = views_[eye].fov;
        projViews_[eye].subImage.swapchain = eyeSwapchains_[eye].handle;
        projViews_[eye].subImage.imageRect.offset = {0, 0};
        projViews_[eye].subImage.imageRect.extent = {
            static_cast<int32_t>(eyeSwapchains_[eye].width),
            static_cast<int32_t>(eyeSwapchains_[eye].height)};
        projViews_[eye].subImage.imageArrayIndex = 0;
    }

    XrCompositionLayerProjection projLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    projLayer.space = appSpace_;
    projLayer.viewCount = 2;
    projLayer.views = projViews_.data();

    const XrCompositionLayerBaseHeader *layers[] = {
        reinterpret_cast<const XrCompositionLayerBaseHeader *>(&projLayer)};

    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime = xrFrameState_.predictedDisplayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    if (xrFrameState_.shouldRender == XR_TRUE && viewsValid_) {
        endInfo.layerCount = 1;
        endInfo.layers = layers;
    } else {
        endInfo.layerCount = 0;
        endInfo.layers = nullptr;
    }

    XrResult r = xrEndFrame(session_, &endInfo);
    if (XR_FAILED(r)) {
        xrCerr() << "xrEndFrame failed: " << xrResultStr(r) << std::endl;
    }
}

// ---- Query helpers ----

void OpenXRContext::getEyeParams(VREyeParams eyes[2]) const {
    // Compute head pose to derive per-eye RELATIVE offsets.
    // headView is applied separately (in buffers.cpp), so eyeViewOffset must only
    // contain the small IPD translation, not the full world-space eye pose.
    glm::vec3 headPos{0.0f};
    glm::quat headOri{1.0f, 0.0f, 0.0f, 0.0f};
    if (viewsValid_) {
        const XrPosef &left = views_[0].pose;
        const XrPosef &right = views_[1].pose;
        headPos = {(left.position.x + right.position.x) * 0.5f,
                   (left.position.y + right.position.y) * 0.5f,
                   (left.position.z + right.position.z) * 0.5f};
        headOri = {left.orientation.w, left.orientation.x,
                   left.orientation.y, left.orientation.z};
    }
    glm::quat headOriInv = glm::inverse(headOri);

    for (uint32_t eye = 0; eye < 2; eye++) {
        auto &cv = configViews_[eye];
        eyes[eye].recommendedWidth = cv.recommendedImageRectWidth;
        eyes[eye].recommendedHeight = cv.recommendedImageRectHeight;

        if (viewsValid_) {
            const XrFovf &fov = views_[eye].fov;
            eyes[eye].tanLeft = std::tan(fov.angleLeft);
            eyes[eye].tanRight = std::tan(fov.angleRight);
            eyes[eye].tanUp = std::tan(fov.angleUp);
            eyes[eye].tanDown = std::tan(fov.angleDown);

            const XrPosef &pose = views_[eye].pose;
            glm::vec3 eyePos = {pose.position.x, pose.position.y, pose.position.z};
            glm::quat eyeOri = {pose.orientation.w, pose.orientation.x,
                                pose.orientation.y, pose.orientation.z};

            // Store offsets RELATIVE to head pose (typically just ±IPD/2 along X)
            eyes[eye].positionOffset = headOriInv * (eyePos - headPos);
            eyes[eye].orientationOffset = headOriInv * eyeOri;
        }
    }
}

void OpenXRContext::getHeadPose(VRHeadPose &pose) const {
    if (!viewsValid_ || !sessionRunning_) {
        pose = VRHeadPose{};
        return;
    }
    // Head pose is the average of left and right eye poses (approximation).
    // For most runtimes, views[0].pose and views[1].pose share orientation
    // and differ only in the IPD translation. We use the left eye's orientation
    // and the midpoint position.
    const XrPosef &left = views_[0].pose;
    const XrPosef &right = views_[1].pose;
    pose.position = {(left.position.x + right.position.x) * 0.5f,
                     (left.position.y + right.position.y) * 0.5f,
                     (left.position.z + right.position.z) * 0.5f};
    pose.orientation = {left.orientation.w, left.orientation.x,
                        left.orientation.y, left.orientation.z};
    pose.valid = true;
}

// ---- Visibility Mask ----

void OpenXRContext::queryVisibilityMask() {
    if (!visMaskAvailable_ || session_ == XR_NULL_HANDLE) return;

    PFN_xrGetVisibilityMaskKHR xrGetVisibilityMaskKHR = nullptr;
    XrResult r = xrGetInstanceProcAddr(xrInstance_, "xrGetVisibilityMaskKHR",
                                       reinterpret_cast<PFN_xrVoidFunction *>(&xrGetVisibilityMaskKHR));
    if (XR_FAILED(r) || !xrGetVisibilityMaskKHR) {
        xrCerr() << "Could not get xrGetVisibilityMaskKHR" << std::endl;
        visMaskAvailable_ = false;
        return;
    }

    for (uint32_t eye = 0; eye < 2; eye++) {
        XrVisibilityMaskKHR mask{XR_TYPE_VISIBILITY_MASK_KHR};
        // Query sizes first (HIDDEN_TRIANGLE_MESH gives the non-visible area)
        r = xrGetVisibilityMaskKHR(session_, viewConfigType_, eye,
                                   XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR, &mask);
        if (XR_FAILED(r) || mask.vertexCountOutput == 0) {
            xrCout() << "No visibility mask for eye " << eye << std::endl;
            continue;
        }

        std::vector<XrVector2f> verts(mask.vertexCountOutput);
        std::vector<uint32_t> indices(mask.indexCountOutput);
        mask.vertexCapacityInput = static_cast<uint32_t>(verts.size());
        mask.vertices = verts.data();
        mask.indexCapacityInput = static_cast<uint32_t>(indices.size());
        mask.indices = indices.data();

        r = xrGetVisibilityMaskKHR(session_, viewConfigType_, eye,
                                   XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR, &mask);
        if (XR_FAILED(r)) {
            xrCerr() << "Failed to get visibility mask data for eye " << eye << std::endl;
            continue;
        }

        // Convert to glm::vec2
        visMaskVertices_[eye].resize(mask.vertexCountOutput);
        for (uint32_t i = 0; i < mask.vertexCountOutput; i++) {
            visMaskVertices_[eye][i] = {verts[i].x, verts[i].y};
        }
        visMaskIndices_[eye] = std::move(indices);

        xrCout() << "Eye " << eye << " visibility mask: "
                 << mask.vertexCountOutput << " verts, "
                 << mask.indexCountOutput << " indices" << std::endl;
    }
}

// ---- Query ----

float OpenXRContext::floorHeight() const {
    // In STAGE reference space, the Y coordinate of the head pose IS the height
    // above the floor. Return the last known value from the renderer's VR system.
    const auto &vr = Renderer::instance().vrSystem();
    return vr.headPose.valid ? vr.headPose.position.y : 0.0f;
}

// ---- Lifecycle ----

void OpenXRContext::shutdown() {
    destroySessionResources();
    if (xrInstance_ != XR_NULL_HANDLE) {
        xrDestroyInstance(xrInstance_);
        xrInstance_ = XR_NULL_HANDLE;
    }
    xrCout() << "OpenXR shutdown complete" << std::endl;
}

OpenXRContext::~OpenXRContext() {
    shutdown();
}

#endif // MCVR_ENABLE_OPENXR
