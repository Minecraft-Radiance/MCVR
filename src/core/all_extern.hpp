#pragma once

#if defined(_WIN32)
#    define VK_USE_PLATFORM_WIN32_KHR
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#elif defined(__linux__) || defined(__unix__)
#    define VK_USE_PLATFORM_XLIB_KHR
#elif defined(__APPLE__)
#    define VK_USE_PLATFORM_MACOS_MVK
#else
#endif
#include "volk.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/string_cast.hpp>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(CORE_LIB)
typedef int (*PFN_glfwInit)(void);
typedef void (*PFN_glfwTerminate)(void);
typedef void (*PFN_glfwGetWindowSize)(GLFWwindow *, int *, int *);
typedef VkResult (*PFN_glfwCreateWindowSurface)(VkInstance,
                                                GLFWwindow *,
                                                const VkAllocationCallbacks *,
                                                VkSurfaceKHR *);
typedef const char **(*PFN_glfwGetRequiredInstanceExtensions)(uint32_t *);
typedef void (*PFN_glfwSetWindowTitle)(GLFWwindow *, const char *);
typedef void (*PFN_glfwSetFramebufferSizeCallback)(GLFWwindow *, GLFWframebuffersizefun);
typedef void (*PFN_glfwGetFramebufferSize)(GLFWwindow *, int *, int *);
typedef void (*PFN_glfwWaitEvents)(void);
typedef GLFWmonitor *(*PFN_glfwGetPrimaryMonitor)(void);
typedef const GLFWvidmode *(*PFN_glfwGetVideoMode)(GLFWmonitor *);

extern PFN_glfwInit p_glfwInit;
extern PFN_glfwTerminate p_glfwTerminate;
extern PFN_glfwGetWindowSize p_glfwGetWindowSize;
extern PFN_glfwCreateWindowSurface p_glfwCreateWindowSurface;
extern PFN_glfwGetRequiredInstanceExtensions p_glfwGetRequiredInstanceExtensions;
extern PFN_glfwSetWindowTitle p_glfwSetWindowTitle;
extern PFN_glfwSetFramebufferSizeCallback p_glfwSetFramebufferSizeCallback;
extern PFN_glfwGetFramebufferSize p_glfwGetFramebufferSize;
extern PFN_glfwWaitEvents p_glfwWaitEvents;
extern PFN_glfwGetPrimaryMonitor p_glfwGetPrimaryMonitor;
extern PFN_glfwGetVideoMode p_glfwGetVideoMode;

#    define GLFW_Init p_glfwInit
#    define GLFW_Terminate p_glfwTerminate
#    define GLFW_GetWindowSize p_glfwGetWindowSize
#    define GLFW_CreateWindowSurface p_glfwCreateWindowSurface
#    define GLFW_GetRequiredInstanceExtensions p_glfwGetRequiredInstanceExtensions
#    define GLFW_SetWindowTitle p_glfwSetWindowTitle
#    define GLFW_SetFramebufferSizeCallback p_glfwSetFramebufferSizeCallback
#    define GLFW_GetFramebufferSize p_glfwGetFramebufferSize
#    define GLFW_WaitEvents p_glfwWaitEvents
#    define GLFW_GetPrimaryMonitor p_glfwGetPrimaryMonitor
#    define GLFW_GetVideoMode p_glfwGetVideoMode
#else
#    define GLFW_Init glfwInit
#    define GLFW_Terminate glfwTerminate
#    define GLFW_GetWindowSize glfwGetWindowSize
#    define GLFW_CreateWindowSurface glfwCreateWindowSurface
#    define GLFW_GetRequiredInstanceExtensions glfwGetRequiredInstanceExtensions
#    define GLFW_SetWindowTitle glfwSetWindowTitle
#    define GLFW_SetFramebufferSizeCallback glfwSetFramebufferSizeCallback
#    define GLFW_GetFramebufferSize glfwGetFramebufferSize
#    define GLFW_WaitEvents glfwWaitEvents
#    define GLFW_GetPrimaryMonitor glfwGetPrimaryMonitor
#    define GLFW_GetVideoMode glfwGetVideoMode
#endif

#include <memory>
template <typename T, typename... Args>
concept TwoStepInit =
    std::is_default_constructible_v<T> && requires(T t, Args &&...args) { t.init(std::forward<Args>(args)...); };

template <typename Derived>
class SharedObject : public std::enable_shared_from_this<Derived> {
    friend Derived;

  public:
    template <typename... Args>
    static std::shared_ptr<Derived> create(Args &&...args) {
        if constexpr (TwoStepInit<Derived, Args...>) {
            auto ptr = std::make_shared<Derived>();
            ptr->init(std::forward<Args>(args)...);
            return ptr;
        } else {
            return std::make_shared<Derived>(std::forward<Args>(args)...);
        }
    }
};
#define sharedDecl(funName, retName) std::shared_ptr<retName> funName();
#define sharedImpl(namespace, funName, varName, retName)                                                               \
    std::shared_ptr<retName> namespace ::funName() {                                                                   \
        return std::shared_ptr<retName>(shared_from_this(), varName##_);                                               \
    }

// #include <stacktrace>
// #define printStackTrace(os) os << std::stacktrace::current() << std::endl

#include "stb_image.h"
