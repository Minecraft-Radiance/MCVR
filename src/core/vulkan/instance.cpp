#include "core/vulkan/instance.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"

#include <iostream>
#include <set>
#include <vector>

std::vector<std::string> vk::Instance::extraExtensions{};

#ifndef NDEBUG
const bool ENABLE_DEBUGGING = true;
#else
const bool ENABLE_DEBUGGING = false;
#endif

const char *DEBUG_LAYER = "VK_LAYER_KHRONOS_validation";

std::ostream &instanceCout() {
    return std::cout << "[Instance] ";
}

std::ostream &instanceCerr() {
    return std::cerr << "[Instance] ";
}

// Debug callback
VkBool32 debugCallback(VkDebugReportFlagsEXT flags,
                       VkDebugReportObjectTypeEXT objType,
                       uint64_t srcObject,
                       size_t location,
                       int32_t msgCode,
                       const char *pLayerPrefix,
                       const char *pMsg,
                       void *pUserData) {
    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        instanceCerr() << "ERROR: [" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg << std::endl;
    } else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        instanceCerr() << "WARNING: [" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg << std::endl;
    }

    return VK_FALSE;
}

vk::Instance::Instance() {
    GLFW_Init();

    if (volkInitialize() != VK_SUCCESS) {
        printf("volkInitialize failed!\n");
        exit(EXIT_SUCCESS);
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanClear";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "ClearScreenEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::set<std::string> extStorage;

    // Get instance extensions required by GLFW to draw to window
    unsigned int glfwExtensionCount;
    const char **glfwExtensions;
    glfwExtensions = GLFW_GetRequiredInstanceExtensions(&glfwExtensionCount);
#ifdef DEBUG
    instanceCout() << "glfw extensions:" << std::endl;
#endif
    for (int i = 0; i < glfwExtensionCount; i++) {
#ifdef DEBUG
        instanceCout() << "\t" << glfwExtensions[i] << std::endl;
#endif
        extStorage.insert(glfwExtensions[i]);
    }

    // dlss extensions
    std::vector<VkExtensionProperties> dlssExtensions;
    NgxContext::getDlssRRRequiredInstanceExtensions(dlssExtensions);
#ifdef DEBUG
    instanceCout() << "dlss extensions:" << std::endl;
#endif
    for (int i = 0; i < dlssExtensions.size(); i++) {
#ifdef DEBUG
        instanceCout() << "\t" << dlssExtensions[i].extensionName << std::endl;
#endif
        extStorage.insert(dlssExtensions[i].extensionName);
    }

    // dynamic vertex input state ext
    // repeated for dlss, but make sure
    extStorage.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    // Extra extensions (e.g. from OpenXR)
    for (const auto &ext : extraExtensions) { extStorage.insert(ext); }

    // if (ENABLE_DEBUGGING) { push_ext(VK_EXT_DEBUG_REPORT_EXTENSION_NAME); }

    // Check for extensions
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    if (extensionCount == 0) {
        instanceCerr() << "no extensions supported!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

#ifdef DEBUG
    instanceCout() << "supported extensions:" << std::endl;
    for (const auto &extension : availableExtensions) {
        instanceCout() << "\t" << extension.extensionName << std::endl;
    }
#endif

    std::vector<const char *> extensions;
    for (const auto &extension : extStorage) { extensions.push_back(extension.c_str()); }

#ifdef DEBUG
    instanceCout() << "selected extensions:" << std::endl;
    for (const auto &extension : extensions) { instanceCout() << "\t" << extension << std::endl; }
#endif

    // TODO: chek selected extensions in available ones

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (ENABLE_DEBUGGING) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &DEBUG_LAYER;
    }

    // Initialize Vulkan instance
    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        instanceCerr() << "failed to create instance!" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        instanceCout() << "created vulkan instance" << std::endl;
#endif
    }

    volkLoadInstance(instance_);

    // if (ENABLE_DEBUGGING) {
    //     VkDebugReportCallbackCreateInfoEXT createInfo = {};
    //     createInfo.pfnCallback = debugCallback;

    //     if (vkCreateDebugReportCallbackEXT(instance_, &createInfo, nullptr, &callback_) != VK_SUCCESS) {
    //         instanceCerr() << "failed to create debug callback" << std::endl;
    //         exit(EXIT_FAILURE);
    //     } else {
    //         instanceCout() << "created debug callback" << std::endl;
    //     }
    // } else {
    //     instanceCout() << "skipped creating debug callback" << std::endl;
    // }
}

vk::Instance::~Instance() {
    vkDestroyInstance(instance_, nullptr);
    // if (ENABLE_DEBUGGING) { vkDestroyDebugReportCallbackEXT(instance_, callback_, nullptr); }

#ifdef DEBUG
    instanceCout() << "instance deconstructed" << std::endl;
#endif
}

VkInstance &vk::Instance::vkInstance() {
    return instance_;
}