#include "core/vulkan/instance.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/xess_upscaler/xess_wrapper.hpp"

#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

const char *DEBUG_LAYER = "VK_LAYER_KHRONOS_validation";

std::ostream &instanceCerr() {
    return std::cerr << "[Instance] ";
}

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

vk::Instance::Instance(WindowSurfaceBackend surfaceBackend) {
    GLFW_Init();

    if (volkInitialize() != VK_SUCCESS) {
        instanceCerr() << "volkInitialize failed!" << std::endl;
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
    std::vector<std::string> dlssRRRequiredExtensions;
    bool dlssRRRequirementQuerySuccess = false;
    std::vector<std::string> dlssSRRequiredExtensions;
    bool dlssSRRequirementQuerySuccess = false;
    std::vector<std::string> dlssFrameGenerationRequiredExtensions;
    bool dlssFrameGenerationRequirementQuerySuccess = false;
#ifdef MCVR_ENABLE_XESS
    std::vector<std::string> xessRequiredExtensions;
    bool xessRequirementQuerySuccess = false;
#endif

    unsigned int glfwExtensionCount;
    const char **glfwExtensions;
    glfwExtensions = GLFW_GetRequiredInstanceExtensions(&glfwExtensionCount);
    for (int i = 0; i < glfwExtensionCount; i++) {
        extStorage.insert(glfwExtensions[i]);
    }
    extStorage.insert(VK_KHR_SURFACE_EXTENSION_NAME);
    const char *nativeSurfaceExtension = requiredWindowSurfaceExtension(surfaceBackend);
    if (nativeSurfaceExtension != nullptr) { extStorage.insert(nativeSurfaceExtension); }

    std::vector<VkExtensionProperties> dlssRRExtensions;
    NVSDK_NGX_Result dlssRRExtensionQueryResult = NgxContext::getDlssRRRequiredInstanceExtensions(dlssRRExtensions);
    if (NVSDK_NGX_SUCCEED(dlssRRExtensionQueryResult)) {
        dlssRRRequirementQuerySuccess = true;
        for (const auto &dlssExtension : dlssRRExtensions) {
            extStorage.insert(dlssExtension.extensionName);
            dlssRRRequiredExtensions.emplace_back(dlssExtension.extensionName);
        }
    } else {
        instanceCerr() << "failed to query dlss ray reconstruction instance extensions; skipping." << std::endl;
    }

    std::vector<VkExtensionProperties> dlssSRExtensions;
    NVSDK_NGX_Result dlssSRExtensionQueryResult = NgxContext::getDlssSRRequiredInstanceExtensions(dlssSRExtensions);
    if (NVSDK_NGX_SUCCEED(dlssSRExtensionQueryResult)) {
        dlssSRRequirementQuerySuccess = true;
        for (const auto &dlssExtension : dlssSRExtensions) {
            extStorage.insert(dlssExtension.extensionName);
            dlssSRRequiredExtensions.emplace_back(dlssExtension.extensionName);
        }
    } else {
        instanceCerr() << "failed to query dlss super resolution instance extensions; skipping." << std::endl;
    }

    std::vector<VkExtensionProperties> dlssFrameGenerationExtensions;
    NVSDK_NGX_Result dlssFrameGenerationExtensionQueryResult =
        NgxContext::getDlssFrameGenerationRequiredInstanceExtensions(dlssFrameGenerationExtensions);
    if (NVSDK_NGX_SUCCEED(dlssFrameGenerationExtensionQueryResult)) {
        dlssFrameGenerationRequirementQuerySuccess = true;
        for (const auto &dlssExtension : dlssFrameGenerationExtensions) {
            extStorage.insert(dlssExtension.extensionName);
            dlssFrameGenerationRequiredExtensions.emplace_back(dlssExtension.extensionName);
        }
    } else {
        instanceCerr() << "failed to query dlss frame generation instance extensions; skipping." << std::endl;
    }

#ifdef MCVR_ENABLE_XESS
    std::vector<const char *> xessExtensions;
    uint32_t xessMinApiVersion = 0;
    if (mcvr::XeSSWrapper::getRequiredInstanceExtensions(xessExtensions, &xessMinApiVersion)) {
        xessRequirementQuerySuccess = true;
        for (const char *extension : xessExtensions) {
            extStorage.insert(extension);
            xessRequiredExtensions.emplace_back(extension);
        }

        if (xessMinApiVersion > appInfo.apiVersion) { appInfo.apiVersion = xessMinApiVersion; }
    } else {
        instanceCerr() << "xess instance extensions unavailable; skipping." << std::endl;
    }
#endif

    extStorage.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    extStorage.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    if (extensionCount == 0) {
        instanceCerr() << "no extensions supported!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());
    std::unordered_set<std::string> availableExtensionSet;
    availableExtensionSet.reserve(availableExtensions.size());
    for (const auto &availableExtension : availableExtensions) {
        availableExtensionSet.insert(availableExtension.extensionName);
    }

    if (availableExtensionSet.find(VK_KHR_SURFACE_EXTENSION_NAME) == availableExtensionSet.end()) {
        throw std::runtime_error("Vulkan implementation does not support VK_KHR_surface");
    }
    if (nativeSurfaceExtension != nullptr && availableExtensionSet.find(nativeSurfaceExtension) == availableExtensionSet.end()) {
        throw std::runtime_error(std::string("Vulkan implementation does not support required ") +
                                 nativeSurfaceExtension + " extension");
    }

    auto areRequiredExtensionsSupported = [&](const std::vector<std::string> &requiredExtensions) {
        for (const auto &requiredExtension : requiredExtensions) {
            if (availableExtensionSet.find(requiredExtension) == availableExtensionSet.end()) { return false; }
        }
        return true;
    };

    dlssRRInstanceExtensionsCompatible_ =
        dlssRRRequirementQuerySuccess && areRequiredExtensionsSupported(dlssRRRequiredExtensions);
    if (!dlssRRInstanceExtensionsCompatible_) {
        instanceCerr() << "dlss ray reconstruction instance extension requirements are not fully satisfied."
                       << std::endl;
    }

    dlssSRInstanceExtensionsCompatible_ =
        dlssSRRequirementQuerySuccess && areRequiredExtensionsSupported(dlssSRRequiredExtensions);
    if (!dlssSRInstanceExtensionsCompatible_) {
        instanceCerr() << "dlss super resolution instance extension requirements are not fully satisfied." << std::endl;
    }

    dlssFrameGenerationInstanceExtensionsCompatible_ =
        dlssFrameGenerationRequirementQuerySuccess &&
        areRequiredExtensionsSupported(dlssFrameGenerationRequiredExtensions);
    if (!dlssFrameGenerationInstanceExtensionsCompatible_) {
        instanceCerr() << "dlss frame generation instance extension requirements are not fully satisfied." << std::endl;
    }

#ifdef MCVR_ENABLE_XESS
    xessInstanceExtensionsCompatible_ =
        xessRequirementQuerySuccess && areRequiredExtensionsSupported(xessRequiredExtensions);
    if (!xessInstanceExtensionsCompatible_) {
        instanceCerr() << "xess instance extension requirements are not fully satisfied." << std::endl;
    }
#endif

#ifdef __APPLE__
    extStorage.insert("VK_KHR_portability_enumeration");
#endif
    std::vector<const char *> extensions;
    for (const auto &extension : extStorage) {
        if (availableExtensionSet.find(extension) == availableExtensionSet.end()) {
            instanceCerr() << "extension not supported, skipping: " << extension << std::endl;
            continue;
        }
        extensions.push_back(extension.c_str());
    }

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
#ifdef __APPLE__
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

#ifdef DEBUG
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = &DEBUG_LAYER;

#endif

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        instanceCerr() << "failed to create instance!" << std::endl;
        exit(EXIT_FAILURE);
    }

    volkLoadInstance(instance_);
}

vk::Instance::~Instance() {
    vkDestroyInstance(instance_, nullptr);
}

VkInstance &vk::Instance::vkInstance() {
    return instance_;
}

bool vk::Instance::isDlssRRInstanceExtensionsCompatible() const {
    return dlssRRInstanceExtensionsCompatible_;
}

bool vk::Instance::isDlssSRInstanceExtensionsCompatible() const {
    return dlssSRInstanceExtensionsCompatible_;
}

bool vk::Instance::isDlssFrameGenerationInstanceExtensionsCompatible() const {
    return dlssFrameGenerationInstanceExtensionsCompatible_;
}

bool vk::Instance::isXessInstanceExtensionsCompatible() const {
    return xessInstanceExtensionsCompatible_;
}
