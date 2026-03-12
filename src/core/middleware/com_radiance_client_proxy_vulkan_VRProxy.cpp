#include "com_radiance_client_proxy_vulkan_VRProxy.h"

#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeSetEnabled(JNIEnv *,
                                                                                       jclass,
                                                                                       jboolean enabled) {
    Renderer::options.vrEnabled = enabled;

#ifdef MCVR_ENABLE_OPENXR
    if (Renderer::is_initialized()) {
        auto framework = Renderer::instance().framework();
        if (framework != nullptr && !enabled) {
            framework->stopXRSession();
        }
    }
#endif
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeStartXRSession(JNIEnv *,
                                                                                               jclass) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized()) return JNI_FALSE;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return JNI_FALSE;
    return framework->startXRSession() ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeStopXRSession(JNIEnv *,
                                                                                          jclass) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->stopXRSession();
#endif
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeSetRenderScale(JNIEnv *,
                                                                                           jclass,
                                                                                           jfloat renderScale) {
    Renderer::options.vrRenderScale = renderScale;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeSetIPD(JNIEnv *,
                                                                                   jclass,
                                                                                   jfloat ipd) {
    Renderer::options.vrIPD = ipd;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeSetWorldScale(JNIEnv *,
                                                                                          jclass,
                                                                                          jfloat worldScale) {
    Renderer::options.vrWorldScale = worldScale;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeSetWorldOrientation(
        JNIEnv *, jclass, jfloat qx, jfloat qy, jfloat qz, jfloat qw) {
    if (!Renderer::is_initialized()) return;
    Renderer::instance().vrSystem().worldOrientation = glm::normalize(glm::quat(qw, qx, qy, qz));
}

JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeIsEnabled(JNIEnv *,
                                                                                          jclass) {
    return Renderer::options.vrEnabled;
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetEyeCount(JNIEnv *,
                                                                                        jclass) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized()) return Renderer::options.vrEnabled ? 2 : 1;
    auto framework = Renderer::instance().framework();
    if (framework != nullptr && framework->isXRSessionRunning()) return 2;
    return 1;
#else
    return Renderer::options.vrEnabled ? 2 : 1;
#endif
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetEyeRenderWidth(JNIEnv *,
                                                                                              jclass) {
    if (!Renderer::is_initialized()) return 0;
    return static_cast<jint>(Renderer::instance().vrSystem().eyeRenderWidth());
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetEyeRenderHeight(JNIEnv *,
                                                                                               jclass) {
    if (!Renderer::is_initialized()) return 0;
    return static_cast<jint>(Renderer::instance().vrSystem().eyeRenderHeight());
}

JNIEXPORT jfloat JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetRefreshRate(JNIEnv *,
                                                                                             jclass) {
    if (!Renderer::is_initialized()) return 0.0f;
    return Renderer::instance().vrSystem().config.refreshRate;
}

JNIEXPORT jfloatArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetHeadPose(JNIEnv *env,
                                                                                               jclass) {
    jfloatArray result = env->NewFloatArray(7);
    if (!Renderer::is_initialized()) return result;
    const auto &vr = Renderer::instance().vrSystem();
    const auto &pose = vr.headPose;
    if (!pose.valid) return result;

    // Return raw tracking-space pose — worldOrientation is applied only
    // in the C++ rendering path (buffers.cpp).  Java-side VRData handles
    // the worldRotation transform independently.
    const glm::vec3 &pos = pose.position;
    const glm::quat &ori = pose.orientation;

    float data[7] = {pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w};
    env->SetFloatArrayRegion(result, 0, 7, data);
    return result;
}

JNIEXPORT jfloatArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetEyeFov(JNIEnv *env,
                                                                                             jclass,
                                                                                             jint eye) {
    jfloatArray result = env->NewFloatArray(4);
    if (!Renderer::is_initialized() || eye < 0 || eye > 1) return result;
    const auto &ep = Renderer::instance().vrSystem().eyes[eye];
    float data[4] = {ep.tanLeft, ep.tanRight, ep.tanUp, ep.tanDown};
    env->SetFloatArrayRegion(result, 0, 4, data);
    return result;
}

JNIEXPORT jintArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetRecommendedResolution(JNIEnv *env,
                                                                                                          jclass) {
    jintArray result = env->NewIntArray(2);
    if (!Renderer::is_initialized()) return result;
    const auto &eye0 = Renderer::instance().vrSystem().eyes[0];
    jint data[2] = {static_cast<jint>(eye0.recommendedWidth), static_cast<jint>(eye0.recommendedHeight)};
    env->SetIntArrayRegion(result, 0, 2, data);
    return result;
}

// ---- Controller input ----

JNIEXPORT jfloatArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetControllerPose(
    JNIEnv *env, jclass, jint hand) {
    // Returns [px, py, pz, qx, qy, qz, qw, vx, vy, vz, valid] = 11 floats
    jfloatArray result = env->NewFloatArray(11);
    if (!Renderer::is_initialized() || hand < 0 || hand > 1) return result;
    const auto &vr = Renderer::instance().vrSystem();
    const auto &ctrl = vr.controllers[hand];

    // Return raw tracking-space pose — worldOrientation is applied only
    // in the C++ rendering path (buffers.cpp).  Java-side VRData handles
    // the worldRotation transform independently.
    const glm::vec3 &pos = ctrl.position;
    const glm::quat &ori = ctrl.orientation;
    const glm::vec3 &vel = ctrl.linearVelocity;

    float data[11] = {
        pos.x, pos.y, pos.z,
        ori.x, ori.y, ori.z, ori.w,
        vel.x, vel.y, vel.z,
        ctrl.valid ? 1.0f : 0.0f
    };
    env->SetFloatArrayRegion(result, 0, 11, data);
    return result;
}

JNIEXPORT jfloatArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetControllerButtons(
    JNIEnv *env, jclass, jint hand) {
    // Returns [triggerValue, gripValue, thumbstickX, thumbstickY,
    //          triggerPressed, gripPressed, primaryButton, secondaryButton,
    //          thumbstickClick, menuButton] = 10 floats
    jfloatArray result = env->NewFloatArray(10);
    if (!Renderer::is_initialized() || hand < 0 || hand > 1) return result;
    const auto &ctrl = Renderer::instance().vrSystem().controllers[hand];
    float data[10] = {
        ctrl.triggerValue, ctrl.gripValue,
        ctrl.thumbstick.x, ctrl.thumbstick.y,
        ctrl.triggerPressed ? 1.0f : 0.0f,
        ctrl.gripPressed ? 1.0f : 0.0f,
        ctrl.primaryButton ? 1.0f : 0.0f,
        ctrl.secondaryButton ? 1.0f : 0.0f,
        ctrl.thumbstickClick ? 1.0f : 0.0f,
        ctrl.menuButton ? 1.0f : 0.0f
    };
    env->SetFloatArrayRegion(result, 0, 10, data);
    return result;
}

// ---- Haptics ----

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeVibrate(
    JNIEnv *, jclass, jint hand, jfloat amplitude, jlong durationNs, jfloat frequency) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized() || hand < 0 || hand > 1) return;
    auto *xrCtx = Renderer::instance().framework()->xrContext();
    if (!xrCtx || !xrCtx->isSessionRunning()) return;
    xrCtx->input().vibrate(
        xrCtx->session(), static_cast<uint32_t>(hand), amplitude, durationNs, frequency);
#endif
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeStopVibration(
    JNIEnv *, jclass, jint hand) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized() || hand < 0 || hand > 1) return;
    auto *xrCtx = Renderer::instance().framework()->xrContext();
    if (!xrCtx || !xrCtx->isSessionRunning()) return;
    xrCtx->input().stopVibration(xrCtx->session(), static_cast<uint32_t>(hand));
#endif
}

// ---- Performance stats ----

JNIEXPORT jfloatArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetPerformanceStats(
    JNIEnv *env, jclass) {
    // Returns [gpuFrameTimeMs, cpuFrameTimeMs, compositorTargetMs, fps, droppedFrames, headroom] = 6 floats
    jfloatArray result = env->NewFloatArray(6);
    if (!Renderer::is_initialized()) return result;
    const auto &stats = Renderer::instance().vrSystem().perfStats;
    float data[6] = {
        stats.gpuFrameTimeMs, stats.cpuFrameTimeMs,
        stats.compositorTargetMs, stats.fps,
        static_cast<float>(stats.droppedFrames), stats.headroom
    };
    env->SetFloatArrayRegion(result, 0, 6, data);
    return result;
}

// ---- World position offset ----

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeSetWorldPosition(
        JNIEnv *, jclass, jfloat x, jfloat y, jfloat z) {
    if (!Renderer::is_initialized()) return;
    Renderer::instance().vrSystem().worldPosition = glm::vec3(x, y, z);
}

JNIEXPORT jfloatArray JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetWorldPosition(
        JNIEnv *env, jclass) {
    jfloatArray result = env->NewFloatArray(3);
    if (!Renderer::is_initialized()) return result;
    const auto &pos = Renderer::instance().vrSystem().worldPosition;
    float data[3] = {pos.x, pos.y, pos.z};
    env->SetFloatArrayRegion(result, 0, 3, data);
    return result;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeRecenter(
        JNIEnv *, jclass) {
    if (!Renderer::is_initialized()) return;
    Renderer::instance().vrSystem().recenter();
}

// ---- Session state / device info ----

JNIEXPORT jfloat JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetFloorHeight(
        JNIEnv *, jclass) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized()) return 0.0f;
    auto *xrCtx = Renderer::instance().framework()->xrContext();
    if (!xrCtx) return 0.0f;
    return xrCtx->floorHeight();
#else
    return 0.0f;
#endif
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetSessionState(
        JNIEnv *, jclass) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized()) return 0;
    auto *xrCtx = Renderer::instance().framework()->xrContext();
    if (!xrCtx) return 0;
    return static_cast<jint>(xrCtx->sessionState());
#else
    return 0;
#endif
}

JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_VRProxy_nativeGetSystemName(
        JNIEnv *env, jclass) {
#ifdef MCVR_ENABLE_OPENXR
    if (!Renderer::is_initialized()) return env->NewStringUTF("unknown");
    auto *xrCtx = Renderer::instance().framework()->xrContext();
    if (!xrCtx) return env->NewStringUTF("unknown");
    return env->NewStringUTF(xrCtx->systemName().c_str());
#else
    return env->NewStringUTF("simulation");
#endif
}
