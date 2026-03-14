#pragma once

#ifdef MCVR_ENABLE_OPENXR

#include <openxr/openxr.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <string>
#include <vector>

#include "core/render/vr_system.hpp"

// Manages OpenXR action sets, action bindings, and per-frame input sync.
// Also handles haptic feedback and eye gaze tracking.
struct OpenXRInput {
    // Initialise action sets, actions, and suggest interaction profile bindings.
    // Must be called BEFORE xrAttachSessionActionSets.
    bool createActions(XrInstance instance, XrSession session);

    // Attach action sets to the session. Must be called once after createActions.
    bool attachToSession(XrSession session);

    // Sync actions and locate hand spaces. Call once per frame after xrSyncActions.
    void syncAndUpdate(XrSession session, XrSpace appSpace, XrTime predictedTime);

    // Trigger haptic vibration on a hand (0=left, 1=right).
    // duration in nanoseconds, amplitude in [0,1], frequency in Hz (0 = runtime default).
    void vibrate(XrSession session, uint32_t hand, float amplitude, int64_t durationNs, float frequency);
    void stopVibration(XrSession session, uint32_t hand);

    // Eye gaze point in normalised per-eye coordinates (0.5, 0.5 = centre).
    // Falls back to (0.5, 0.5) when eye tracking is unavailable.
    glm::vec2 gazePoint{0.5f, 0.5f};
    bool gazeValid = false;

    // Latest controller states
    std::array<VRControllerState, 2> controllers{};

    void shutdown();

    // Extension support flags (set during createActions based on instance extensions)
    bool hasEyeGaze = false;

private:
    // Action set
    XrActionSet actionSet_ = XR_NULL_HANDLE;

    // Hand paths (/user/hand/left, /user/hand/right)
    XrPath handPaths_[2]{};

    // Per-hand actions
    XrAction aimPoseAction_ = XR_NULL_HANDLE;
    XrAction gripPoseAction_ = XR_NULL_HANDLE;
    XrAction triggerAction_ = XR_NULL_HANDLE;
    XrAction gripAction_ = XR_NULL_HANDLE;
    XrAction thumbstickAction_ = XR_NULL_HANDLE;
    XrAction primaryAction_ = XR_NULL_HANDLE;
    XrAction secondaryAction_ = XR_NULL_HANDLE;
    XrAction thumbstickClickAction_ = XR_NULL_HANDLE;
    XrAction menuAction_ = XR_NULL_HANDLE;
    XrAction hapticAction_ = XR_NULL_HANDLE;

    // Eye gaze action
    XrAction gazeAction_ = XR_NULL_HANDLE;

    // Per-hand action spaces (for locating poses)
    XrSpace aimSpaces_[2]{};
    XrSpace gripSpaces_[2]{};

    // Eye gaze space
    XrSpace gazeSpace_ = XR_NULL_HANDLE;

    // Helpers
    XrAction createAction(XrActionSet set, const char *name, const char *localizedName,
                          XrActionType type, uint32_t subactionPathCount, const XrPath *subactionPaths);
    bool suggestBindings(XrInstance instance, const char *profilePath,
                         const std::vector<XrActionSuggestedBinding> &bindings);
    XrPath toPath(XrInstance instance, const char *str);
};

#endif // MCVR_ENABLE_OPENXR
