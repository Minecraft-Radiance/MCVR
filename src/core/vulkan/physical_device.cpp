#include "core/vulkan/physical_device.hpp"

#include "core/vulkan/instance.hpp"
#include "core/vulkan/result.hpp"
#include "core/vulkan/window.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

std::ostream &physicalDeviceCerr() {
    return std::cerr << "[PhysicalDevice] ";
}

bool uuidMatches(const vk::DeviceUuid &expected, const vk::DeviceUuid &actual) {
    return std::ranges::equal(expected, actual);
}

uint32_t vk::PhysicalDevice::physicalDevicePreference(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        default: return 0;
    }
}

std::string vk::PhysicalDevice::resultDescription(VkResult result) {
    return std::string(vk::resultName(result)) + " (" + std::to_string(static_cast<int32_t>(result)) + ")";
}

bool vk::PhysicalDevice::findRequiredQueueFamilies(VkPhysicalDevice device,
                                                   VkSurfaceKHR surface,
                                                   vk::PhysicalDevice::QueueFamilySelection &selection,
                                                   std::string &rejectionReason) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    if (queueFamilyCount == 0) {
        rejectionReason = "no queue families are exposed";
        return false;
    }

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    std::vector<VkBool32> presentationSupport(queueFamilyCount, VK_FALSE);
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentationSupport[i]);
        if (result != VK_SUCCESS) {
            rejectionReason = "failed to query presentation support for queue family " + std::to_string(i) + ": " +
                              resultDescription(result);
            return false;
        }
    }

    constexpr VkQueueFlags mainQueueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueCount > 0 && presentationSupport[i] == VK_TRUE &&
            (queueFamilies[i].queueFlags & mainQueueFlags) == mainQueueFlags) {
            selection.mainIndex = i;
            break;
        }
    }

    if (selection.mainIndex == vk::PhysicalDevice::invalidQueueFamily) {
        rejectionReason = "no queue family supports presentation, graphics, compute, and transfer";
        return false;
    }

    constexpr VkQueueFlags secondaryQueueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (i != selection.mainIndex && queueFamilies[i].queueCount > 0 &&
            (queueFamilies[i].queueFlags & secondaryQueueFlags) == secondaryQueueFlags) {
            selection.secondaryIndex = i;
            selection.independentSecondaryQueue = true;
            return true;
        }
    }

    selection.secondaryIndex = selection.mainIndex;
    if (queueFamilies[selection.mainIndex].queueCount >= 2) {
        selection.secondaryOffset = 1;
        selection.independentSecondaryQueue = true;
    }
    return true;
}

void vk::PhysicalDevice::evaluateRequiredExtensions(vk::PhysicalDevice::PhysicalDeviceCandidate &candidate) {
    uint32_t extensionCount = 0;
    VkResult result = vkEnumerateDeviceExtensionProperties(candidate.device, nullptr, &extensionCount, nullptr);
    if (result != VK_SUCCESS) {
        candidate.rejectionReasons.push_back("failed to enumerate device extensions: " + resultDescription(result));
        return;
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    result =
        vkEnumerateDeviceExtensionProperties(candidate.device, nullptr, &extensionCount, availableExtensions.data());
    if (result != VK_SUCCESS) {
        candidate.rejectionReasons.push_back("failed to retrieve device extensions: " + resultDescription(result));
        return;
    }

    std::unordered_set<std::string> extensionNames;
    extensionNames.reserve(availableExtensions.size());
    for (const auto &extension : availableExtensions) { extensionNames.insert(extension.extensionName); }

    const std::vector<const char *> requiredExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
#ifdef MCVR_UI_BACKEND_OPENGL
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(_WIN32)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#if defined(_WIN32)
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
#endif
    };
    for (const char *requiredExtension : requiredExtensions) {
        if (!extensionNames.contains(requiredExtension)) {
            candidate.rejectionReasons.push_back(std::string("missing required extension ") + requiredExtension);
        }
    }
}

void vk::PhysicalDevice::evaluateRequiredFeatures(vk::PhysicalDevice::PhysicalDeviceCandidate &candidate) {
    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan11Features;

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &vulkan12Features;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelerationStructureFeatures.pNext = &vulkan13Features;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures{};
    rayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingFeatures.pNext = &accelerationStructureFeatures;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &rayTracingFeatures;
    vkGetPhysicalDeviceFeatures2(candidate.device, &features);

    if (rayTracingFeatures.rayTracingPipeline != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing feature rayTracingPipeline");
    }
    if (rayTracingFeatures.rayTracingPipelineTraceRaysIndirect != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing feature rayTracingPipelineTraceRaysIndirect");
    }
#ifndef __APPLE__
    if (rayTracingFeatures.rayTraversalPrimitiveCulling != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing feature rayTraversalPrimitiveCulling");
    }
#endif // macOS ships the triangle-only vanilla-pt pack without skip-AABB flags.
    if (accelerationStructureFeatures.accelerationStructure != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing feature accelerationStructure");
    }
    if (vulkan13Features.synchronization2 != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing Vulkan 1.3 feature synchronization2");
    }
    if (vulkan12Features.bufferDeviceAddress != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing Vulkan 1.2 feature bufferDeviceAddress");
    }
    if (vulkan11Features.shaderDrawParameters != VK_TRUE) {
        candidate.rejectionReasons.emplace_back("missing Vulkan 1.1 feature shaderDrawParameters");
    }
}

void vk::PhysicalDevice::evaluatePresentationSupport(vk::PhysicalDevice::PhysicalDeviceCandidate &candidate,
                                                     VkSurfaceKHR surface) {
    VkSurfaceCapabilitiesKHR surfaceCapabilities{};
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate.device, surface, &surfaceCapabilities);
    if (result != VK_SUCCESS) {
        candidate.rejectionReasons.push_back("failed to query surface capabilities: " + resultDescription(result));
        return;
    }

    constexpr VkImageUsageFlags requiredImageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if ((surfaceCapabilities.supportedUsageFlags & requiredImageUsage) != requiredImageUsage) {
        candidate.rejectionReasons.emplace_back("surface does not support required swapchain image usages");
    }
    if ((surfaceCapabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) == 0) {
        candidate.rejectionReasons.emplace_back("surface does not support opaque swapchain composition");
    }

    uint32_t surfaceFormatCount = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(candidate.device, surface, &surfaceFormatCount, nullptr);
    if (result != VK_SUCCESS) {
        candidate.rejectionReasons.push_back("failed to query surface formats: " + resultDescription(result));
    } else if (surfaceFormatCount == 0) {
        candidate.rejectionReasons.emplace_back("surface exposes no swapchain formats");
    }

    uint32_t presentModeCount = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(candidate.device, surface, &presentModeCount, nullptr);
    if (result != VK_SUCCESS) {
        candidate.rejectionReasons.push_back("failed to query presentation modes: " + resultDescription(result));
    } else if (presentModeCount == 0) {
        candidate.rejectionReasons.emplace_back("surface exposes no presentation modes");
    }
}

void vk::PhysicalDevice::evaluateOpenGLDeviceIdentity(PhysicalDeviceCandidate &candidate,
                                                      const OpenGLDeviceIdentity &openGLDeviceIdentity) {
    VkPhysicalDeviceIDProperties idProperties{};
    idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &idProperties;
    vkGetPhysicalDeviceProperties2(candidate.device, &properties);

    candidate.properties = properties.properties;
    std::copy_n(idProperties.deviceUUID, VK_UUID_SIZE, candidate.deviceUuid.begin());
    std::copy_n(idProperties.driverUUID, VK_UUID_SIZE, candidate.driverUuid.begin());
    candidate.matchesOpenGLDevice =
        std::ranges::any_of(openGLDeviceIdentity.deviceUuids, [&](const DeviceUuid &deviceUuid) {
            return uuidMatches(deviceUuid, candidate.deviceUuid);
        });
    candidate.matchesOpenGLDriver = uuidMatches(openGLDeviceIdentity.driverUuid, candidate.driverUuid);

    if (!candidate.matchesOpenGLDevice) {
        candidate.rejectionReasons.emplace_back("device UUID does not match the OpenGL UI context");
    }
    if (candidate.matchesOpenGLDevice && !candidate.matchesOpenGLDriver) {
        physicalDeviceCerr() << "driver UUID does not match the OpenGL UI context; continuing because the device UUID matches"
                             << std::endl;
    }
}

vk::PhysicalDevice::PhysicalDeviceCandidate vk::PhysicalDevice::evaluatePhysicalDevice(
    VkPhysicalDevice device, VkSurfaceKHR surface, const std::optional<OpenGLDeviceIdentity> &openGLDeviceIdentity) {
    vk::PhysicalDevice::PhysicalDeviceCandidate candidate{};
    candidate.device = device;
    vkGetPhysicalDeviceProperties(device, &candidate.properties);

    if (openGLDeviceIdentity.has_value()) {
        evaluateOpenGLDeviceIdentity(candidate, openGLDeviceIdentity.value());
        if (!candidate.matchesOpenGLDevice) { return candidate; }
    }

    if (candidate.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        candidate.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        candidate.rejectionReasons.emplace_back("device type is not a discrete or integrated GPU");
    }

    if (candidate.properties.apiVersion < VK_API_VERSION_1_3) {
        candidate.rejectionReasons.emplace_back("Vulkan 1.3 is required");
    }

    evaluateRequiredExtensions(candidate);
    evaluateRequiredFeatures(candidate);

    std::string queueRejectionReason;
    if (!findRequiredQueueFamilies(device, surface, candidate.queueFamilies, queueRejectionReason)) {
        candidate.rejectionReasons.push_back(std::move(queueRejectionReason));
    }
    evaluatePresentationSupport(candidate, surface);

    return candidate;
}

void vk::PhysicalDevice::findPhysicalDevice() {
    uint32_t deviceCount = 0;
    if (vkEnumeratePhysicalDevices(instance_->vkInstance(), &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0) {
        physicalDeviceCerr() << "failed to get number of physical devices" << std::endl;
        exit(1);
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (vkEnumeratePhysicalDevices(instance_->vkInstance(), &deviceCount, devices.data()) != VK_SUCCESS) {
        physicalDeviceCerr() << "failed to retrieve physical devices" << std::endl;
        exit(1);
    }

    vk::PhysicalDevice::PhysicalDeviceCandidate selectedCandidate{};
    bool foundSuitableDevice = false;
    bool foundMatchingOpenGLDevice = false;
    for (VkPhysicalDevice device : devices) {
        vk::PhysicalDevice::PhysicalDeviceCandidate candidate =
            evaluatePhysicalDevice(device, window_->vkSurface(), openGLDeviceIdentity_);
        foundMatchingOpenGLDevice = foundMatchingOpenGLDevice || candidate.matchesOpenGLDevice;

        if (!candidate.rejectionReasons.empty()) {
            for (const auto &reason : candidate.rejectionReasons) {
                physicalDeviceCerr() << "rejected " << candidate.properties.deviceName << ": " << reason << std::endl;
            }
            continue;
        }

        if (!foundSuitableDevice || physicalDevicePreference(candidate.properties.deviceType) >
                                        physicalDevicePreference(selectedCandidate.properties.deviceType)) {
            selectedCandidate = std::move(candidate);
            foundSuitableDevice = true;
        }
    }

    if (!foundSuitableDevice) {
        if (openGLDeviceIdentity_.has_value()) {
            std::string message;
            if (foundMatchingOpenGLDevice) {
                message = "The GPU selected by the OpenGL UI context does not satisfy Radiance Vulkan requirements";
            } else {
                message = "No Vulkan physical device matches the GPU selected by the OpenGL UI context";
            }
            physicalDeviceCerr() << message << "." << std::endl;
            throw std::runtime_error(message);
        }
        physicalDeviceCerr() << "No suitable discrete or integrated physical device found." << std::endl;
        exit(EXIT_FAILURE);
    }

    physicalDevice_ = selectedCandidate.device;
    mainQueueIndex_ = selectedCandidate.queueFamilies.mainIndex;
    secondaryQueueIndex_ = selectedCandidate.queueFamilies.secondaryIndex;
    secondaryQueueOffset_ = selectedCandidate.queueFamilies.secondaryOffset;
    independentSecondaryQueue_ = selectedCandidate.queueFamilies.independentSecondaryQueue;
}

vk::PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance, std::shared_ptr<Window> window)
    : PhysicalDevice(std::move(instance), std::move(window), std::nullopt) {}

vk::PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance,
                                   std::shared_ptr<Window> window,
                                   OpenGLDeviceIdentity openGLDeviceIdentity)
    : PhysicalDevice(std::move(instance), std::move(window), std::make_optional(std::move(openGLDeviceIdentity))) {}

vk::PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance,
                                   std::shared_ptr<Window> window,
                                   std::optional<OpenGLDeviceIdentity> openGLDeviceIdentity)
    : instance_(std::move(instance)),
      window_(std::move(window)),
      openGLDeviceIdentity_(std::move(openGLDeviceIdentity)) {
    if (openGLDeviceIdentity_.has_value() && openGLDeviceIdentity_->deviceUuids.empty()) {
        throw std::invalid_argument("OpenGL device identity contains no device UUIDs");
    }
    findPhysicalDevice();

    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProperties{};
    accelStructProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
    rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    rayTracingProperties.pNext = &accelStructProperties;

    VkPhysicalDeviceProperties2 deviceProps2{};
    deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProps2.pNext = &rayTracingProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &deviceProps2);

    properties_ = deviceProps2.properties;
    rayTracingProperties_ = rayTracingProperties;
    accelerationStructProperties_ = accelStructProperties;
}

vk::PhysicalDevice::~PhysicalDevice() {
}

VkPhysicalDevice &vk::PhysicalDevice::vkPhysicalDevice() {
    return physicalDevice_;
}

uint32_t vk::PhysicalDevice::mainQueueIndex() {
    return mainQueueIndex_;
}

uint32_t vk::PhysicalDevice::secondaryQueueIndex() {
    return secondaryQueueIndex_;
}

void vk::PhysicalDevice::findQueueFamilies() {
    vk::PhysicalDevice::QueueFamilySelection selection{};
    std::string rejectionReason;
    if (!findRequiredQueueFamilies(physicalDevice_, window_->vkSurface(), selection, rejectionReason)) {
        physicalDeviceCerr() << "Selected device no longer has usable queue families: " << rejectionReason << std::endl;
        exit(EXIT_FAILURE);
    }

    mainQueueIndex_ = selection.mainIndex;
    secondaryQueueIndex_ = selection.secondaryIndex;
    secondaryQueueOffset_ = selection.secondaryOffset;
    independentSecondaryQueue_ = selection.independentSecondaryQueue;
}

uint32_t vk::PhysicalDevice::secondaryQueueOffset() {
    return secondaryQueueOffset_;
}

bool vk::PhysicalDevice::hasIndependentSecondaryQueue() const {
    return independentSecondaryQueue_;
}

VkPhysicalDeviceProperties vk::PhysicalDevice::properties() {
    return properties_;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR vk::PhysicalDevice::rayTracingProperties() {
    return rayTracingProperties_;
}

VkPhysicalDeviceAccelerationStructurePropertiesKHR vk::PhysicalDevice::accelerationStructProperties() {
    return accelerationStructProperties_;
}
