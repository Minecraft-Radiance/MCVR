#pragma once

#ifdef MCVR_ENABLE_OPENXR

#include "core/render/vr_system.hpp"
#include "core/render/openxr_input.hpp"

#include <vulkan/vulkan.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <array>
#include <string>
#include <vector>

// Manages the full OpenXR lifecycle: instance, session, swapchains, and per-frame operations.
// Initialization is split into two stages because OpenXR needs to provide required Vulkan
// extensions BEFORE VkInstance/VkDevice creation:
//   Stage A  preVulkanInit()   — creates XrInstance, queries Vulkan extension requirements
//   Stage B  postVulkanInit()  — creates XrSession, reference space, and per-eye swapchains
struct OpenXRContext {
    // ---- Stage A: before Vulkan init ----

    // Creates XrInstance and XrSystemId; queries required Vulkan instance/device extensions.
    // Returns false if no HMD is connected or the runtime is unavailable.
    bool preVulkanInit();

    // Extension lists populated by preVulkanInit(), to be merged into Vulkan creation.
    const std::vector<std::string> &requiredInstanceExtensions() const { return requiredInstanceExts_; }
    const std::vector<std::string> &requiredDeviceExtensions() const { return requiredDeviceExts_; }

    // After VkInstance is created, let OpenXR verify/select the physical device.
    // Returns the XR-preferred VkPhysicalDevice (may differ from app's choice).
    VkPhysicalDevice getXRPhysicalDevice(VkInstance vkInstance) const;

    // ---- Stage B: after Vulkan init ----

    // Creates XrSession, reference space, and per-eye swapchains.
    bool postVulkanInit(VkInstance vkInstance,
                        VkPhysicalDevice vkPhysicalDevice,
                        VkDevice vkDevice,
                        uint32_t queueFamilyIndex,
                        uint32_t queueIndex);

    // ---- Per-frame operations ----

    // Poll and process OpenXR events (session state transitions, etc.).
    void pollEvents();

    // --- PIPELINED FRAMING API ---
    // Split frame operations into two phases for late-latch optimization:

    // Phase 1: Begin frame recording without waiting for compositor.
    // Allows CPU to start recording commands while GPU executes previous frame.
    // This is called early in acquireContext().
    void beginFrameRecording();

    // Phase 2: Late-latch pose data just before submission.
    // Calls xrWaitFrame + xrBeginFrame + xrLocateViews to get latest pose.
    // Returns true if compositor wants us to render this frame.
    // This is called in submitCommand() just before vkQueueSubmit.
    bool latchPose();

    // Acquire the current XR swapchain image for the given eye.
    // Returns the VkImage to blit into. Must call releaseSwapchainImage() after blit.
    VkImage acquireSwapchainImage(uint32_t eye);

    // Release the swapchain image after rendering/blitting.
    void releaseSwapchainImage(uint32_t eye);

    // End the XR frame: constructs projection layers and calls xrEndFrame.
    void endFrame();

    // ---- Query ----

    // Fill VREyeParams from the latest xrLocateViews result.
    void getEyeParams(VREyeParams eyes[2]) const;

    // Fill VRHeadPose from the latest xrLocateViews result.
    void getHeadPose(VRHeadPose &pose) const;

    // Per-eye swapchain dimensions (recommended by the runtime).
    uint32_t swapchainWidth() const { return eyeSwapchains_[0].width; }
    uint32_t swapchainHeight() const { return eyeSwapchains_[0].height; }

    bool isSessionRunning() const { return sessionRunning_; }
    bool shouldRender() const { return xrFrameState_.shouldRender == XR_TRUE; }
    XrSession session() const { return session_; }

    // Session control: keep runtime prepared but do not enter XR session
    // until the game requests it (e.g. on world enter).
    bool requestSessionStart();
    void requestSessionStop();
    bool sessionRequested() const { return sessionRequested_; }

    // Session state for Java-side pause/resume logic
    XrSessionState sessionState() const { return sessionState_; }

    // HMD system name (e.g. "Oculus Quest 3", "Valve Index")
    const std::string &systemName() const { return systemName_; }

    // Floor height: in STAGE reference space, headPose.position.y IS the height
    // above the floor. This returns the last known head Y (or 0 if not valid).
    float floorHeight() const;

    // Input subsystem (controllers, haptics, eye gaze)
    OpenXRInput &input() { return input_; }

    // Predicted display period in nanoseconds (for performance stats)
    int64_t predictedDisplayPeriodNs() const { return xrFrameState_.predictedDisplayPeriod; }
    float lastWaitFrameMs() const { return lastWaitFrameMs_; }
    float lastSwapchainWaitMs() const { return lastSwapchainWaitMs_; }

    // Visibility mask vertices for an eye (empty if extension not supported)
    const std::vector<glm::vec2> &visibilityMaskVertices(uint32_t eye) const { return visMaskVertices_[eye]; }
    const std::vector<uint32_t> &visibilityMaskIndices(uint32_t eye) const { return visMaskIndices_[eye]; }
    bool hasVisibilityMask() const { return visMaskAvailable_; }

    // ---- Lifecycle ----

    void shutdown();
    ~OpenXRContext();

private:
    // XR core handles
    XrInstance xrInstance_ = XR_NULL_HANDLE;
    XrSystemId systemId_ = XR_NULL_SYSTEM_ID;
    XrSession session_ = XR_NULL_HANDLE;
    XrSpace appSpace_ = XR_NULL_HANDLE;

    // Per-eye swapchain
    struct EyeSwapchain {
        XrSwapchain handle = XR_NULL_HANDLE;
        std::vector<VkImage> images;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t acquiredIndex = 0;
        bool imageAcquired = false;
    };
    std::array<EyeSwapchain, 2> eyeSwapchains_{};

    // Per-frame state
    XrFrameState xrFrameState_{XR_TYPE_FRAME_STATE};
    std::array<XrView, 2> views_{};
    std::array<XrCompositionLayerProjectionView, 2> projViews_{};
    // Pipelined frame state
    enum FrameState {
        FRAME_IDLE,              // No frame operations
        FRAME_RECORDING,         // Recording started, waiting for pose latch
        FRAME_LATCHED,          // Pose latched, ready for submit
        FRAME_STARTED           // XR frame started (legacy path)
    };
    FrameState frameState_ = FRAME_IDLE;
    bool viewsValid_ = false;

    // Session state
    XrSessionState sessionState_ = XR_SESSION_STATE_UNKNOWN;
    bool sessionRunning_ = false;
    bool sessionRequested_ = false;
    bool destroyPending_ = false;

    // System info (queried once in preVulkanInit)
    std::string systemName_;

    // Extension requirements discovered in preVulkanInit
    std::vector<std::string> requiredInstanceExts_;
    std::vector<std::string> requiredDeviceExts_;

    // View configuration
    XrViewConfigurationType viewConfigType_ = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    std::array<XrViewConfigurationView, 2> configViews_{};

    // Helpers
    bool createSession(VkInstance vkInstance, VkPhysicalDevice vkPhysicalDevice,
                       VkDevice vkDevice, uint32_t queueFamilyIndex, uint32_t queueIndex);
    bool createSwapchains();
    void destroySessionResources();
    void handleSessionStateChange(XrSessionState newState);
    void queryVisibilityMask();

    // Cached Vulkan handles for deferred session creation.
    VkInstance vkInstance_ = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysicalDevice_ = VK_NULL_HANDLE;
    VkDevice vkDevice_ = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex_ = 0;
    uint32_t queueIndex_ = 0;
    bool vulkanReady_ = false;

    // Input subsystem
    OpenXRInput input_;
    bool inputInitialized_ = false;

    // Visibility mask (XR_KHR_visibility_mask)
    bool visMaskAvailable_ = false;
    std::array<std::vector<glm::vec2>, 2> visMaskVertices_;
    std::array<std::vector<uint32_t>, 2> visMaskIndices_;

    // Timing diagnostics for CPU wait analysis.
    float lastWaitFrameMs_ = 0.0f;
    float lastSwapchainWaitMs_ = 0.0f;
};

#endif // MCVR_ENABLE_OPENXR
