#pragma once

#include <jni.h>
#include <cstdio>
#include <exception>
#include <utility>

namespace mcvr::jni {
inline void throwRuntimeException(JNIEnv *env, const char *message) noexcept {
    if (env->ExceptionCheck()) return;
    jclass type = env->FindClass("java/lang/RuntimeException");
    if (type == nullptr) return;
    env->ThrowNew(type, message);
    env->DeleteLocalRef(type);
}

// Native exceptions must not unwind through JVM-generated frames. Preserve an
// existing Java exception, and let Minecraft handle a renderer failure itself.
template <class Function>
void invoke(JNIEnv *env, const char *operation, Function &&function) noexcept {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[RADIANCE-E15-JNI] %s: %s\n", operation, error.what());
        throwRuntimeException(env, error.what());
    } catch (...) {
        std::fprintf(stderr, "[RADIANCE-E15-JNI] %s: unknown native exception\n", operation);
        throwRuntimeException(env, "Unknown native renderer exception");
    }
}
}
