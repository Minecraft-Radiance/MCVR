#ifdef MCVR_ENABLE_OPENXR

#include "core/render/openxr_input.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

static std::ostream &inputLog() {
    static std::ofstream file("openxr_debug.log", std::ios::out | std::ios::app);
    // Write to both cout and the log file
    std::cout << "[OpenXR Input] ";
    file << "[OpenXR Input] ";
    // We return cout; the file gets a duplicate line via the caller flushing.
    return std::cout;
}

// ---- Helpers ----

XrPath OpenXRInput::toPath(XrInstance instance, const char *str) {
    XrPath path = XR_NULL_PATH;
    xrStringToPath(instance, str, &path);
    return path;
}

XrAction OpenXRInput::createAction(XrActionSet set, const char *name, const char *localizedName,
                                   XrActionType type, uint32_t subactionPathCount,
                                   const XrPath *subactionPaths) {
    XrActionCreateInfo ci{XR_TYPE_ACTION_CREATE_INFO};
    std::strncpy(ci.actionName, name, XR_MAX_ACTION_NAME_SIZE);
    std::strncpy(ci.localizedActionName, localizedName, XR_MAX_LOCALIZED_ACTION_NAME_SIZE);
    ci.actionType = type;
    ci.countSubactionPaths = subactionPathCount;
    ci.subactionPaths = subactionPaths;
    XrAction action = XR_NULL_HANDLE;
    XrResult r = xrCreateAction(set, &ci, &action);
    if (XR_FAILED(r)) {
        inputLog() << "Failed to create action: " << name << std::endl;
    }
    return action;
}

bool OpenXRInput::suggestBindings(XrInstance instance, const char *profilePath,
                                  const std::vector<XrActionSuggestedBinding> &bindings) {
    XrPath profile = toPath(instance, profilePath);
    XrInteractionProfileSuggestedBinding suggestion{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggestion.interactionProfile = profile;
    suggestion.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    suggestion.suggestedBindings = bindings.data();
    XrResult r = xrSuggestInteractionProfileBindings(instance, &suggestion);
    if (XR_FAILED(r)) {
        inputLog() << "Failed to suggest bindings for " << profilePath << std::endl;
        return false;
    }
    return true;
}

// ---- createActions ----

bool OpenXRInput::createActions(XrInstance instance, XrSession session) {
    // Check if eye gaze extension is available
    uint32_t extCount = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr);
    std::vector<XrExtensionProperties> exts(extCount, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, exts.data());
    for (auto &ext : exts) {
        if (std::strcmp(ext.extensionName, "XR_EXT_eye_gaze_interaction") == 0) {
            hasEyeGaze = true;
        }
    }

    // Hand sub-action paths
    handPaths_[0] = toPath(instance, "/user/hand/left");
    handPaths_[1] = toPath(instance, "/user/hand/right");

    // Create action set
    XrActionSetCreateInfo asCI{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::strncpy(asCI.actionSetName, "gameplay", XR_MAX_ACTION_SET_NAME_SIZE);
    std::strncpy(asCI.localizedActionSetName, "Gameplay", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE);
    asCI.priority = 0;
    XrResult r = xrCreateActionSet(instance, &asCI, &actionSet_);
    if (XR_FAILED(r)) { inputLog() << "Failed to create action set" << std::endl; return false; }

    // Create actions
    aimPoseAction_        = createAction(actionSet_, "aim_pose",         "Aim Pose",          XR_ACTION_TYPE_POSE_INPUT,    2, handPaths_);
    gripPoseAction_       = createAction(actionSet_, "grip_pose",        "Grip Pose",         XR_ACTION_TYPE_POSE_INPUT,    2, handPaths_);
    triggerAction_        = createAction(actionSet_, "trigger",          "Trigger",           XR_ACTION_TYPE_FLOAT_INPUT,   2, handPaths_);
    gripAction_           = createAction(actionSet_, "grip",             "Grip",              XR_ACTION_TYPE_FLOAT_INPUT,   2, handPaths_);
    thumbstickAction_     = createAction(actionSet_, "thumbstick",       "Thumbstick",        XR_ACTION_TYPE_VECTOR2F_INPUT,2, handPaths_);
    primaryAction_        = createAction(actionSet_, "primary_button",   "Primary Button",    XR_ACTION_TYPE_BOOLEAN_INPUT, 2, handPaths_);
    secondaryAction_      = createAction(actionSet_, "secondary_button", "Secondary Button",  XR_ACTION_TYPE_BOOLEAN_INPUT, 2, handPaths_);
    thumbstickClickAction_= createAction(actionSet_, "thumbstick_click", "Thumbstick Click",  XR_ACTION_TYPE_BOOLEAN_INPUT, 2, handPaths_);
    menuAction_           = createAction(actionSet_, "menu",             "Menu",              XR_ACTION_TYPE_BOOLEAN_INPUT, 2, handPaths_);
    hapticAction_         = createAction(actionSet_, "haptic",           "Haptic",            XR_ACTION_TYPE_VIBRATION_OUTPUT, 2, handPaths_);

    // Create action spaces for pose actions
    for (uint32_t hand = 0; hand < 2; hand++) {
        XrActionSpaceCreateInfo spaceCI{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        spaceCI.poseInActionSpace = {{0, 0, 0, 1}, {0, 0, 0}};
        spaceCI.subactionPath = handPaths_[hand];

        spaceCI.action = aimPoseAction_;
        xrCreateActionSpace(session, &spaceCI, &aimSpaces_[hand]);

        spaceCI.action = gripPoseAction_;
        xrCreateActionSpace(session, &spaceCI, &gripSpaces_[hand]);
    }

    // ---- Suggest interaction profile bindings ----
    // PLACEHOLDER_BINDINGS_START
    auto p = [&](const char *s) { return toPath(instance, s); };

    // Oculus Touch (Quest 2/3/Pro)
    {
        std::vector<XrActionSuggestedBinding> bindings = {
            {aimPoseAction_,         p("/user/hand/left/input/aim/pose")},
            {aimPoseAction_,         p("/user/hand/right/input/aim/pose")},
            {gripPoseAction_,        p("/user/hand/left/input/grip/pose")},
            {gripPoseAction_,        p("/user/hand/right/input/grip/pose")},
            {triggerAction_,         p("/user/hand/left/input/trigger/value")},
            {triggerAction_,         p("/user/hand/right/input/trigger/value")},
            {gripAction_,            p("/user/hand/left/input/squeeze/value")},
            {gripAction_,            p("/user/hand/right/input/squeeze/value")},
            {thumbstickAction_,      p("/user/hand/left/input/thumbstick")},
            {thumbstickAction_,      p("/user/hand/right/input/thumbstick")},
            {primaryAction_,         p("/user/hand/left/input/x/click")},
            {primaryAction_,         p("/user/hand/right/input/a/click")},
            {secondaryAction_,       p("/user/hand/left/input/y/click")},
            {secondaryAction_,       p("/user/hand/right/input/b/click")},
            {thumbstickClickAction_, p("/user/hand/left/input/thumbstick/click")},
            {thumbstickClickAction_, p("/user/hand/right/input/thumbstick/click")},
            {menuAction_,            p("/user/hand/left/input/menu/click")},
            {hapticAction_,          p("/user/hand/left/output/haptic")},
            {hapticAction_,          p("/user/hand/right/output/haptic")},
        };
        suggestBindings(instance, "/interaction_profiles/oculus/touch_controller", bindings);
    }
    // PLACEHOLDER_BINDINGS_MID

    // Valve Index Controller
    {
        std::vector<XrActionSuggestedBinding> bindings = {
            {aimPoseAction_,         p("/user/hand/left/input/aim/pose")},
            {aimPoseAction_,         p("/user/hand/right/input/aim/pose")},
            {gripPoseAction_,        p("/user/hand/left/input/grip/pose")},
            {gripPoseAction_,        p("/user/hand/right/input/grip/pose")},
            {triggerAction_,         p("/user/hand/left/input/trigger/value")},
            {triggerAction_,         p("/user/hand/right/input/trigger/value")},
            {gripAction_,            p("/user/hand/left/input/squeeze/value")},
            {gripAction_,            p("/user/hand/right/input/squeeze/value")},
            {thumbstickAction_,      p("/user/hand/left/input/thumbstick")},
            {thumbstickAction_,      p("/user/hand/right/input/thumbstick")},
            {primaryAction_,         p("/user/hand/left/input/a/click")},
            {primaryAction_,         p("/user/hand/right/input/a/click")},
            {secondaryAction_,       p("/user/hand/left/input/b/click")},
            {secondaryAction_,       p("/user/hand/right/input/b/click")},
            {thumbstickClickAction_, p("/user/hand/left/input/thumbstick/click")},
            {thumbstickClickAction_, p("/user/hand/right/input/thumbstick/click")},
            {hapticAction_,          p("/user/hand/left/output/haptic")},
            {hapticAction_,          p("/user/hand/right/output/haptic")},
        };
        suggestBindings(instance, "/interaction_profiles/valve/index_controller", bindings);
    }

    // HTC Vive Controller
    {
        std::vector<XrActionSuggestedBinding> bindings = {
            {aimPoseAction_,         p("/user/hand/left/input/aim/pose")},
            {aimPoseAction_,         p("/user/hand/right/input/aim/pose")},
            {gripPoseAction_,        p("/user/hand/left/input/grip/pose")},
            {gripPoseAction_,        p("/user/hand/right/input/grip/pose")},
            {triggerAction_,         p("/user/hand/left/input/trigger/value")},
            {triggerAction_,         p("/user/hand/right/input/trigger/value")},
            {gripAction_,            p("/user/hand/left/input/squeeze/click")},
            {gripAction_,            p("/user/hand/right/input/squeeze/click")},
            {thumbstickAction_,      p("/user/hand/left/input/trackpad")},
            {thumbstickAction_,      p("/user/hand/right/input/trackpad")},
            {thumbstickClickAction_, p("/user/hand/left/input/trackpad/click")},
            {thumbstickClickAction_, p("/user/hand/right/input/trackpad/click")},
            {menuAction_,            p("/user/hand/left/input/menu/click")},
            {menuAction_,            p("/user/hand/right/input/menu/click")},
            {hapticAction_,          p("/user/hand/left/output/haptic")},
            {hapticAction_,          p("/user/hand/right/output/haptic")},
        };
        suggestBindings(instance, "/interaction_profiles/htc/vive_controller", bindings);
    }

    // Khronos Simple Controller (minimal fallback)
    {
        std::vector<XrActionSuggestedBinding> bindings = {
            {aimPoseAction_,  p("/user/hand/left/input/aim/pose")},
            {aimPoseAction_,  p("/user/hand/right/input/aim/pose")},
            {gripPoseAction_, p("/user/hand/left/input/grip/pose")},
            {gripPoseAction_, p("/user/hand/right/input/grip/pose")},
            {triggerAction_,  p("/user/hand/left/input/select/click")},
            {triggerAction_,  p("/user/hand/right/input/select/click")},
            {menuAction_,     p("/user/hand/left/input/menu/click")},
            {menuAction_,     p("/user/hand/right/input/menu/click")},
            {hapticAction_,   p("/user/hand/left/output/haptic")},
            {hapticAction_,   p("/user/hand/right/output/haptic")},
        };
        suggestBindings(instance, "/interaction_profiles/khr/simple_controller", bindings);
    }

    inputLog() << "Actions and bindings created" << std::endl;
    return true;
}

// ---- attachToSession ----

bool OpenXRInput::attachToSession(XrSession session) {
    XrSessionActionSetsAttachInfo attachInfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attachInfo.countActionSets = 1;
    attachInfo.actionSets = &actionSet_;
    XrResult r = xrAttachSessionActionSets(session, &attachInfo);
    if (XR_FAILED(r)) {
        inputLog() << "Failed to attach action sets to session" << std::endl;
        return false;
    }
    inputLog() << "Action sets attached to session" << std::endl;
    return true;
}

// ---- syncAndUpdate ----

void OpenXRInput::syncAndUpdate(XrSession session, XrSpace appSpace, XrTime predictedTime) {
    // Sync actions
    XrActiveActionSet activeSet{};
    activeSet.actionSet = actionSet_;
    activeSet.subactionPath = XR_NULL_PATH;
    XrActionsSyncInfo syncInfo{XR_TYPE_ACTIONS_SYNC_INFO};
    syncInfo.countActiveActionSets = 1;
    syncInfo.activeActionSets = &activeSet;
    XrResult r = xrSyncActions(session, &syncInfo);
    if (XR_FAILED(r)) return;

    // Update each hand
    for (uint32_t hand = 0; hand < 2; hand++) {
        auto &ctrl = controllers[hand];
        ctrl = VRControllerState{};

        // Aim pose
        XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
        XrSpaceVelocity vel{XR_TYPE_SPACE_VELOCITY};
        loc.next = &vel;
        r = xrLocateSpace(aimSpaces_[hand], appSpace, predictedTime, &loc);
        if (XR_SUCCEEDED(r) && (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
            (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
            ctrl.valid = true;
            ctrl.position = {loc.pose.position.x, loc.pose.position.y, loc.pose.position.z};
            ctrl.orientation = {loc.pose.orientation.w, loc.pose.orientation.x,
                                loc.pose.orientation.y, loc.pose.orientation.z};
            if (vel.velocityFlags & XR_SPACE_VELOCITY_LINEAR_VALID_BIT)
                ctrl.linearVelocity = {vel.linearVelocity.x, vel.linearVelocity.y, vel.linearVelocity.z};
            if (vel.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT)
                ctrl.angularVelocity = {vel.angularVelocity.x, vel.angularVelocity.y, vel.angularVelocity.z};
        }
        // PLACEHOLDER_SYNC_CONTINUE

        // Trigger (float)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = triggerAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateFloat state{XR_TYPE_ACTION_STATE_FLOAT};
            if (XR_SUCCEEDED(xrGetActionStateFloat(session, &gi, &state)) && state.isActive) {
                ctrl.triggerValue = state.currentState;
                ctrl.triggerPressed = state.currentState > 0.8f;
            }
        }

        // Grip (float)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = gripAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateFloat state{XR_TYPE_ACTION_STATE_FLOAT};
            if (XR_SUCCEEDED(xrGetActionStateFloat(session, &gi, &state)) && state.isActive) {
                ctrl.gripValue = state.currentState;
                ctrl.gripPressed = state.currentState > 0.8f;
            }
        }

        // Thumbstick (vec2)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = thumbstickAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateVector2f state{XR_TYPE_ACTION_STATE_VECTOR2F};
            if (XR_SUCCEEDED(xrGetActionStateVector2f(session, &gi, &state)) && state.isActive) {
                ctrl.thumbstick = {state.currentState.x, state.currentState.y};
            }
        }

        // Primary button (boolean)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = primaryAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            if (XR_SUCCEEDED(xrGetActionStateBoolean(session, &gi, &state)) && state.isActive) {
                ctrl.primaryButton = state.currentState == XR_TRUE;
            }
        }

        // Secondary button (boolean)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = secondaryAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            if (XR_SUCCEEDED(xrGetActionStateBoolean(session, &gi, &state)) && state.isActive) {
                ctrl.secondaryButton = state.currentState == XR_TRUE;
            }
        }

        // Thumbstick click (boolean)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = thumbstickClickAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            if (XR_SUCCEEDED(xrGetActionStateBoolean(session, &gi, &state)) && state.isActive) {
                ctrl.thumbstickClick = state.currentState == XR_TRUE;
            }
        }

        // Menu button (boolean)
        {
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
            gi.action = menuAction_;
            gi.subactionPath = handPaths_[hand];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            if (XR_SUCCEEDED(xrGetActionStateBoolean(session, &gi, &state)) && state.isActive) {
                ctrl.menuButton = state.currentState == XR_TRUE;
            }
        }
    }

    // Eye gaze (if available — requires extension enabled at instance level)
    gazeValid = false;
    gazePoint = {0.5f, 0.5f};
    if (hasEyeGaze && gazeAction_ != XR_NULL_HANDLE && gazeSpace_ != XR_NULL_HANDLE) {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = gazeAction_;
        XrActionStatePose state{XR_TYPE_ACTION_STATE_POSE};
        if (XR_SUCCEEDED(xrGetActionStatePose(session, &gi, &state)) && state.isActive) {
            XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
            if (XR_SUCCEEDED(xrLocateSpace(gazeSpace_, appSpace, predictedTime, &loc)) &&
                (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
                // Extract gaze direction from orientation, project onto screen plane.
                // gazeSpace orientation's -Z is the gaze direction.
                float qx = loc.pose.orientation.x, qy = loc.pose.orientation.y;
                float qz = loc.pose.orientation.z, qw = loc.pose.orientation.w;
                // Forward vector from quaternion
                float fx = 2.0f * (qx * qz + qw * qy);
                float fy = 2.0f * (qy * qz - qw * qx);
                float fz = 1.0f - 2.0f * (qx * qx + qy * qy);
                // Project: gaze point is where the gaze ray hits Z=-1 plane
                if (std::abs(fz) > 1e-4f) {
                    gazePoint.x = 0.5f + 0.5f * (fx / (-fz));
                    gazePoint.y = 0.5f - 0.5f * (fy / (-fz));
                    gazePoint.x = std::clamp(gazePoint.x, 0.0f, 1.0f);
                    gazePoint.y = std::clamp(gazePoint.y, 0.0f, 1.0f);
                    gazeValid = true;
                }
            }
        }
    }
}

// ---- Haptics ----

void OpenXRInput::vibrate(XrSession session, uint32_t hand, float amplitude,
                          int64_t durationNs, float frequency) {
    if (hand > 1 || hapticAction_ == XR_NULL_HANDLE) return;
    XrHapticActionInfo info{XR_TYPE_HAPTIC_ACTION_INFO};
    info.action = hapticAction_;
    info.subactionPath = handPaths_[hand];
    XrHapticVibration vibration{XR_TYPE_HAPTIC_VIBRATION};
    vibration.amplitude = std::clamp(amplitude, 0.0f, 1.0f);
    vibration.duration = durationNs;
    vibration.frequency = frequency;
    xrApplyHapticFeedback(session, &info, reinterpret_cast<const XrHapticBaseHeader *>(&vibration));
}

void OpenXRInput::stopVibration(XrSession session, uint32_t hand) {
    if (hand > 1 || hapticAction_ == XR_NULL_HANDLE) return;
    XrHapticActionInfo info{XR_TYPE_HAPTIC_ACTION_INFO};
    info.action = hapticAction_;
    info.subactionPath = handPaths_[hand];
    xrStopHapticFeedback(session, &info);
}

// ---- Shutdown ----

void OpenXRInput::shutdown() {
    for (uint32_t h = 0; h < 2; h++) {
        if (aimSpaces_[h] != XR_NULL_HANDLE)  { xrDestroySpace(aimSpaces_[h]);  aimSpaces_[h] = XR_NULL_HANDLE; }
        if (gripSpaces_[h] != XR_NULL_HANDLE) { xrDestroySpace(gripSpaces_[h]); gripSpaces_[h] = XR_NULL_HANDLE; }
    }
    if (gazeSpace_ != XR_NULL_HANDLE) { xrDestroySpace(gazeSpace_); gazeSpace_ = XR_NULL_HANDLE; }
    // Actions and action sets are destroyed when the XrInstance is destroyed
    actionSet_ = XR_NULL_HANDLE;
}

#endif // MCVR_ENABLE_OPENXR

