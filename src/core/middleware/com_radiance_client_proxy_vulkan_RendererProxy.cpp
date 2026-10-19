#include "com_radiance_client_proxy_vulkan_RendererProxy.h"
#include "core/middleware/jni_export.hpp"
#include "core/middleware/jni_exceptions.hpp"

#include "core/all_extern.hpp"
#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <unordered_map>

#if defined(_WIN32)
#    include <windows.h>
using DYNLIB_HANDLE = HMODULE;

static DYNLIB_HANDLE try_get_loaded_handle(const wchar_t *wname) {
    return GetModuleHandleW(wname);
}

static FARPROC getproc(DYNLIB_HANDLE h, const char *sym) {
    FARPROC p = GetProcAddress(h, sym);
    if (!p) {
        std::cerr << "GetProcAddress failed: " << sym << std::endl;
        std::abort();
    }
    return p;
}

#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#    include <dlfcn.h>
using DYNLIB_HANDLE = void *;

static DYNLIB_HANDLE try_get_loaded_handle(const char *name) {
    return dlopen(name, RTLD_NOW | RTLD_NOLOAD);
}

static void *getproc(DYNLIB_HANDLE h, const char *sym) {
    void *p = dlsym(h, sym);
    if (!p) {
        std::cerr << "dlsym failed: " << sym << " — " << dlerror() << std::endl;
        std::abort();
    }
    return p;
}

#else
#    error "Unsupported platform"
#endif

static DYNLIB_HANDLE bind_handle_from_candidates(JNIEnv *env, jobjectArray jnames) {
    jsize n = env->GetArrayLength(jnames);
    if (n == 0) return nullptr;
#if defined(_WIN32)
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jnames, i);
        const jchar *w = env->GetStringChars(s, nullptr);
        DYNLIB_HANDLE h = try_get_loaded_handle(reinterpret_cast<const wchar_t *>(w));
        env->ReleaseStringChars(s, w);
        env->DeleteLocalRef(s);
        if (h) return h;
    }
#else
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jnames, i);
        const char *c = env->GetStringUTFChars(s, nullptr);
        DYNLIB_HANDLE h = try_get_loaded_handle(c);
        env->ReleaseStringUTFChars(s, c);
        env->DeleteLocalRef(s);
        if (h) return h;
    }
#endif
    return nullptr;
}

static void bind_symbols(DYNLIB_HANDLE h) {
#if defined(_WIN32)
    auto gp = [&](const char *sym) { return getproc(h, sym); };
#else
    auto gp = [&](const char *sym) { return getproc(h, sym); };
#endif
    p_glfwInit = reinterpret_cast<PFN_glfwInit>(gp("glfwInit"));
    p_glfwTerminate = reinterpret_cast<PFN_glfwTerminate>(gp("glfwTerminate"));
    p_glfwGetWindowSize = reinterpret_cast<PFN_glfwGetWindowSize>(gp("glfwGetWindowSize"));
    p_glfwCreateWindowSurface = reinterpret_cast<PFN_glfwCreateWindowSurface>(gp("glfwCreateWindowSurface"));
    p_glfwGetRequiredInstanceExtensions =
        reinterpret_cast<PFN_glfwGetRequiredInstanceExtensions>(gp("glfwGetRequiredInstanceExtensions"));
    p_glfwSetWindowTitle = reinterpret_cast<PFN_glfwSetWindowTitle>(gp("glfwSetWindowTitle"));
    p_glfwSetFramebufferSizeCallback =
        reinterpret_cast<PFN_glfwSetFramebufferSizeCallback>(gp("glfwSetFramebufferSizeCallback"));
    p_glfwGetFramebufferSize = reinterpret_cast<PFN_glfwGetFramebufferSize>(gp("glfwGetFramebufferSize"));
    p_glfwWaitEvents = reinterpret_cast<PFN_glfwWaitEvents>(gp("glfwWaitEvents"));
#ifdef MCVR_UI_BACKEND_OPENGL
    p_glfwGetPlatform = reinterpret_cast<PFN_glfwGetPlatform>(gp("glfwGetPlatform"));
#    if defined(_WIN32)
    p_glfwGetWin32Window = reinterpret_cast<PFN_glfwGetWin32Window>(gp("glfwGetWin32Window"));
#    elif defined(__linux__) || defined(__unix__)
    const int platform = p_glfwGetPlatform();
    if (platform == GLFW_PLATFORM_X11) {
        p_glfwGetX11Display = reinterpret_cast<PFN_glfwGetX11Display>(gp("glfwGetX11Display"));
        p_glfwGetX11Window = reinterpret_cast<PFN_glfwGetX11Window>(gp("glfwGetX11Window"));
    } else if (platform == GLFW_PLATFORM_WAYLAND) {
        p_glfwGetWaylandDisplay = reinterpret_cast<PFN_glfwGetWaylandDisplay>(gp("glfwGetWaylandDisplay"));
        p_glfwGetWaylandWindow = reinterpret_cast<PFN_glfwGetWaylandWindow>(gp("glfwGetWaylandWindow"));
    }
#    endif
#endif
}

static GLFWwindow *bind_glfw(JNIEnv *env, jobjectArray candidates, jlong windowHandle) {
    DYNLIB_HANDLE h = bind_handle_from_candidates(env, candidates);
    if (!h) {
        std::cerr << "[GLFW-Bind] Could not find already-loaded GLFW via NOLOAD/GetModuleHandle."
                     " Ensure Java(LWJGL) loads GLFW before JNI and pass correct names/paths."
                  << std::endl;
        std::abort();
    }
    bind_symbols(h);
    return reinterpret_cast<GLFWwindow *>(static_cast<intptr_t>(windowHandle));
}

static void throwJavaException(JNIEnv *env, const char *className, const char *message) {
    jclass exceptionClass = env->FindClass(className);
    if (exceptionClass == nullptr) return;
    env->ThrowNew(exceptionClass, message);
    env->DeleteLocalRef(exceptionClass);
}

#ifdef MCVR_UI_BACKEND_OPENGL
using PresentationTargetHandle = std::shared_ptr<vk::PresentationTarget>;

static bool readOpenGLDeviceIdentity(JNIEnv *env,
                                     jbyteArray deviceUuids,
                                     jbyteArray driverUuid,
                                     vk::OpenGLDeviceIdentity &identity) {
    if (deviceUuids == nullptr || driverUuid == nullptr) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "OpenGL device and driver UUIDs are required");
        return false;
    }

    const jsize deviceUuidBytes = env->GetArrayLength(deviceUuids);
    const jsize driverUuidBytes = env->GetArrayLength(driverUuid);
    if (deviceUuidBytes <= 0 || deviceUuidBytes % VK_UUID_SIZE != 0) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
                           "OpenGL device UUID data must contain one or more 16-byte UUIDs");
        return false;
    }
    if (driverUuidBytes != VK_UUID_SIZE) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
                           "OpenGL driver UUID data must contain one 16-byte UUID");
        return false;
    }

    identity.deviceUuids.resize(static_cast<size_t>(deviceUuidBytes / VK_UUID_SIZE));
    for (jsize index = 0; index < static_cast<jsize>(identity.deviceUuids.size()); index++) {
        env->GetByteArrayRegion(deviceUuids, index * VK_UUID_SIZE, VK_UUID_SIZE,
                                reinterpret_cast<jbyte *>(identity.deviceUuids[index].data()));
        if (env->ExceptionCheck()) return false;
    }
    env->GetByteArrayRegion(driverUuid, 0, VK_UUID_SIZE, reinterpret_cast<jbyte *>(identity.driverUuid.data()));
    return !env->ExceptionCheck();
}
#endif

static std::u16string JStringToU16(JNIEnv *env, jstring jstr) {
    if (!jstr) return {};
    const jchar *chars = env->GetStringChars(jstr, nullptr);
    jsize len = env->GetStringLength(jstr);
    std::u16string u16(reinterpret_cast<const char16_t *>(chars), reinterpret_cast<const char16_t *>(chars) + len);
    env->ReleaseStringChars(jstr, chars);
    return u16;
}

static std::filesystem::path JStringToPath(JNIEnv *env, jstring jstr) {
    std::u16string u16 = JStringToU16(env, jstr);
    return std::filesystem::path(u16);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initFolderPath(JNIEnv *env,
                                                                                          jclass,
                                                                                          jstring folderPath) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.initFolderPath");
    if (folderPath == NULL) { return; }

    Renderer::folderPath = JStringToU16(env, folderPath);
    MCVR_PROFILE_CONFIGURE_OUTPUT(Renderer::folderPath);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initRenderer(JNIEnv *env,
                                                                                        jclass,
                                                                                        jobjectArray candidates,
                                                                                        jlong windowHandle) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.initRenderer");
    GLFWwindow *window = bind_glfw(env, candidates, windowHandle);
    Renderer::init(window);
#ifdef MCVR_UI_BACKEND_VULKAN
    Renderer::instance().framework()->acquireContext();
#endif
}

#ifdef MCVR_UI_BACKEND_OPENGL
JNIEXPORT jlong JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_prepareOpenGLPresentationTargetNative(
    JNIEnv *env, jclass, jobjectArray candidates, jlong windowHandle) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.prepareOpenGLPresentationTargetNative");
    try {
        GLFWwindow *window = bind_glfw(env, candidates, windowHandle);
        auto target = vk::createOpenGLUiPresentationTarget(window);
        return reinterpret_cast<jlong>(new PresentationTargetHandle(std::move(target)));
    } catch (const std::exception &error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return 0;
    } catch (...) {
        throwJavaException(env, "java/lang/IllegalStateException",
                           "Unknown native failure while creating the OpenGL UI presentation target");
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initRendererWithOpenGLDevice(
    JNIEnv *env, jclass, jlong presentationTargetHandle, jbyteArray deviceUuids, jbyteArray driverUuid) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.initRendererWithOpenGLDevice");
    vk::OpenGLDeviceIdentity identity;
    if (!readOpenGLDeviceIdentity(env, deviceUuids, driverUuid, identity)) return;
    auto *presentationTarget = reinterpret_cast<PresentationTargetHandle *>(presentationTargetHandle);
    if (presentationTarget == nullptr || *presentationTarget == nullptr) {
        throwJavaException(env, "java/lang/IllegalStateException", "OpenGL UI presentation target is invalid");
        return;
    }

    try {
        Renderer::init(*presentationTarget, std::move(identity));
    } catch (const std::exception &error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    } catch (...) {
        throwJavaException(env, "java/lang/IllegalStateException",
                           "Unknown native failure while matching the OpenGL and Vulkan devices");
    }
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_releaseOpenGLPresentationTargetNative(
    JNIEnv *, jclass, jlong presentationTargetHandle) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.releaseOpenGLPresentationTargetNative");
    delete reinterpret_cast<PresentationTargetHandle *>(presentationTargetHandle);
}
#endif

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_maxSupportedTextureSize(JNIEnv *, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.maxSupportedTextureSize");
    auto maxImageSize = Renderer::instance().framework()->physicalDevice()->properties().limits.maxImageDimension2D;
    return maxImageSize;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_acquireContext(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.acquireContext");
    mcvr::jni::invoke(env, "acquireContext", [&] {
        auto framework = Renderer::instance().framework();
        if (framework == nullptr) return;
        framework->acquireContext();
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_submitCommand(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.submitCommand");
    mcvr::jni::invoke(env, "submitCommand", [&] {
        auto framework = Renderer::instance().framework();
        if (framework == nullptr) return;
        framework->submitCommand();
    });
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_present(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.present");
    mcvr::jni::invoke(env, "present", [&] {
        auto framework = Renderer::instance().framework();
        if (framework == nullptr) return;
        framework->present();
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_getPresentedFrameRate(JNIEnv *, jclass) {
    auto renderer = Renderer::try_instance();
    if (renderer == nullptr) return 0;
    auto framework = renderer->framework();
    return framework == nullptr ? 0 : static_cast<jint>(framework->presentedFrameRate());
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_close(JNIEnv *env, jclass) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.close");
    mcvr::jni::invoke(env, "close", [&] {
        auto framework = Renderer::instance().framework();
        if (framework == nullptr) return;
        Renderer::instance().close();
    });
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_shouldRenderWorld(JNIEnv *, jclass, jboolean shouldRenderWorld) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.shouldRenderWorld");
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->shouldRender() = shouldRenderWorld;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_takeScreenshot(
    JNIEnv *, jclass, jboolean withUI, jint width, jint height, jint channel, jlong pointer) {
    MCVR_PROFILE_SCOPE("jni.RendererProxy.takeScreenshot");
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->takeScreenshot(withUI, width, height, channel, reinterpret_cast<void *>(pointer));
}
