#include "core/vulkan/physical_device.hpp"

#include "core/vulkan/instance.hpp"
#include "core/vulkan/window.hpp"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

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

namespace {
struct CandidateDevice {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    uint32_t index = 0;
    uint64_t score = 0;
};

uint64_t scorePhysicalDevice(VkPhysicalDevice device, const VkPhysicalDeviceProperties &properties) {
    uint64_t score = 0;

    switch (properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score += 1'000'000'000ull;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score += 500'000'000ull;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score += 100'000'000ull;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        score += 50'000'000ull;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
    default:
        break;
    }

    // // Prefer devices with higher practical limits.
    score += static_cast<uint64_t>(properties.limits.maxImageDimension2D);
    score += static_cast<uint64_t>(properties.limits.maxPerStageDescriptorSampledImages);
    return score;
}

std::optional<uint32_t> parseGpuIndexOverride(const char *envValue) {
    if (envValue == nullptr || envValue[0] == '\0') return std::nullopt;

    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(envValue, &end, 10);
    if (errno != 0 || end == envValue || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(parsed);
}
} // namespace

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

    // Optional manual override: choose device by enumerate index.
    // Example: MCVR_GPU_INDEX=1
    const char *gpuIndexEnv = std::getenv("MCVR_GPU_INDEX");
    const auto overrideIndex = parseGpuIndexOverride(gpuIndexEnv);
    if (gpuIndexEnv != nullptr && !overrideIndex.has_value()) {
        physicalDeviceCerr() << "MCVR_GPU_INDEX is invalid: '" << gpuIndexEnv << "', fallback to auto selection."
                             << std::endl;
    }

    if (overrideIndex.has_value()) {
        if (overrideIndex.value() < devices.size()) {
            VkPhysicalDevice overrideDevice = devices[overrideIndex.value()];
            VkPhysicalDeviceProperties overrideProperties{};
            vkGetPhysicalDeviceProperties(overrideDevice, &overrideProperties);
            if (isDeviceSuitable(overrideDevice) && overrideProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
                physicalDevice_ = overrideDevice;
                physicalDeviceCout() << "selected device via MCVR_GPU_INDEX=" << overrideIndex.value() << ": "
                                     << overrideProperties.deviceName << std::endl;
                return;
            }
            physicalDeviceCerr() << "MCVR_GPU_INDEX=" << overrideIndex.value()
                                 << " points to an unsupported device, fallback to auto selection." << std::endl;
        } else {
            physicalDeviceCerr() << "MCVR_GPU_INDEX=" << overrideIndex.value() << " is out of range [0, "
                                 << (devices.size() - 1) << "], fallback to auto selection." << std::endl;
        }
    }

    std::optional<CandidateDevice> best;
    for (uint32_t i = 0; i < devices.size(); i++) {
        const auto &device = devices[i];
        if (!isDeviceSuitable(device)) continue;

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

        const uint64_t score = scorePhysicalDevice(device, properties);
        if (!best.has_value() || score > best->score) { best = CandidateDevice{device, properties, i, score}; }
    }

    if (best.has_value()) {
        physicalDevice_ = best->device;
#ifdef DEBUG
        physicalDeviceCout() << "selected device index: " << best->index << std::endl;
#endif
        physicalDeviceCout() << "selected device name: " << best->properties.deviceName << std::endl;
        physicalDeviceCout() << "selected device score: " << best->score << std::endl;
        return;
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
