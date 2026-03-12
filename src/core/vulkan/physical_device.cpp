#include "core/vulkan/physical_device.hpp"

#include "core/vulkan/instance.hpp"
#include "core/vulkan/window.hpp"

#include <iostream>
#include <vector>

VkPhysicalDevice vk::PhysicalDevice::overrideDevice = VK_NULL_HANDLE;

std::ostream &physicalDeviceCout() {
    return std::cout << "[PhysicalDevice] ";
}

std::ostream &physicalDeviceCerr() {
    return std::cerr << "[PhysicalDevice] ";
}

bool isDeviceSuitable(VkPhysicalDevice device) {
    // check extension
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    bool hasSwapchain = false;
    bool hasRayTracing = false;

    for (const auto &ext : availableExtensions) {
        if (std::string(ext.extensionName) == VK_KHR_SWAPCHAIN_EXTENSION_NAME) { hasSwapchain = true; }
        if (std::string(ext.extensionName) == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) { hasRayTracing = true; }
    }

    if (!hasSwapchain || !hasRayTracing) return false;

    // check features
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &vulkan12Features;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelerationStructureFeatures.pNext = &vulkan13Features;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures = {};
    rayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingFeatures.pNext = &accelerationStructureFeatures;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &rayTracingFeatures;

    vkGetPhysicalDeviceFeatures2(device, &features2);
    if (!rayTracingFeatures.rayTracingPipeline || !accelerationStructureFeatures.accelerationStructure ||
        !vulkan13Features.synchronization2 || !vulkan12Features.bufferDeviceAddress) {
        return false;
    } else {
        return true;
    }
}

void vk::PhysicalDevice::findPhysicalDevice() {
    // If an override device was set (e.g. by OpenXR), use it directly.
    if (overrideDevice != VK_NULL_HANDLE) {
        physicalDevice_ = overrideDevice;
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
#ifdef DEBUG
        physicalDeviceCout() << "Using override physical device: " << properties.deviceName << std::endl;
#endif
        return;
    }

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

    // find the first supported physical device
    for (const auto &device : devices) {
        if (isDeviceSuitable(device)) {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(device, &properties);

            if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;

            physicalDevice_ = device;
#ifdef DEBUG
            physicalDeviceCout() << "found suitable physical device" << std::endl;
#endif

            // output device info
            uint32_t supportedVersion[] = {VK_VERSION_MAJOR(properties.apiVersion),
                                           VK_VERSION_MINOR(properties.apiVersion),
                                           VK_VERSION_PATCH(properties.apiVersion)};

#ifdef DEBUG
            physicalDeviceCout() << "selected device name: " << properties.deviceName << std::endl;
            physicalDeviceCout() << "supports Vulkan version: " << supportedVersion[0] << "." << supportedVersion[1]
                                 << "." << supportedVersion[2] << std::endl;
#endif

            return;
        }
    }

    // if no supported physical device is found
    physicalDeviceCerr() << "No suitable physical device found!" << std::endl;
    exit(EXIT_FAILURE);
}

vk::PhysicalDevice::PhysicalDevice(std::shared_ptr<Instance> instance, std::shared_ptr<Window> window)
    : instance_(instance), window_(window) {
    findPhysicalDevice();
    findQueueFamilies();

    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProperties{};
    accelStructProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{};
    rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    rtProperties.pNext = &accelStructProperties;

    VkPhysicalDeviceProperties2 deviceProps2{};
    deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProps2.pNext = &rtProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &deviceProps2);

    properties_ = deviceProps2.properties;
    rayTracingProperties_ = rtProperties;
    accelerationStructProperties_ = accelStructProperties;
}

vk::PhysicalDevice::~PhysicalDevice() {
#ifdef DEBUG
    physicalDeviceCout() << "physical device deconstructed" << std::endl;
#endif
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
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, nullptr);

    if (queueFamilyCount == 0) {
        physicalDeviceCerr() << "Physical device has no queue families!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, queueFamilies.data());

#ifdef DEBUG
    physicalDeviceCout() << "physical device has " << queueFamilyCount << " queue families" << std::endl;
#endif

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, window_->vkSurface(), &presentSupport);

        VkBool32 graphicsSupport = false;
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            graphicsSupport = true;
        }

        VkBool32 computeSupport = false;
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            computeSupport = true;
        }

        VkBool32 transferSupport = false;
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
            transferSupport = true;
        }

        // Early exit if all needed queue families are found
        if (presentSupport && graphicsSupport && computeSupport && transferSupport) {
            mainQueueIndex_ = i;
            // TODO: add more condition
            secondaryQueueIndex_ = i;
            break;
        }
    }

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, window_->vkSurface(), &presentSupport);

        VkBool32 graphicsSupport = false;
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            graphicsSupport = true;
        }

        VkBool32 computeSupport = false;
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            computeSupport = true;
        }

        VkBool32 transferSupport = false;
        if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
            transferSupport = true;
        }

        // Early exit if all needed queue families are found
        if (computeSupport && transferSupport && i != mainQueueIndex_) {
            secondaryQueueIndex_ = i;
            break;
        }
    }

    if (mainQueueIndex_ == -1) {
        physicalDeviceCerr() << "No queue family that supports graphics, compute and transfer found." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (secondaryQueueIndex_ == -1) {
        physicalDeviceCerr() << "No queue family that supports graphics, compute and transfer found." << std::endl;
        exit(EXIT_FAILURE);
    }
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
