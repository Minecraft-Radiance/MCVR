#include "core/vulkan/device.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"

#include <cstring>
#include <iostream>
#include <unordered_set>
#include <vector>

std::vector<std::string> vk::Device::extraExtensions{};

std::ostream &deviceCout() {
    return std::cout << "[Device] ";
}

std::ostream &deviceCerr() {
    return std::cerr << "[Device] ";
}

vk::Device::Device(std::shared_ptr<Instance> instance,
                   std::shared_ptr<Window> window,
                   std::shared_ptr<PhysicalDevice> physicalDevice)
    : instance_(instance), window_(window), physicalDevice_(physicalDevice) {
    // enabled device extensions
    std::vector<const char *> enabledExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                   VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                                   VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                                   VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                                                   VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                                   VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                                                   VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
                                                   VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                                   VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
                                                   VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
                                                   VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
                                                   VK_KHR_MAINTENANCE_5_EXTENSION_NAME};

    std::vector<VkExtensionProperties> dlssExtensions;
    NVSDK_NGX_Result dlssResult =
        NgxContext::getDlssRRRequiredDeviceExtensions(instance_, physicalDevice_, dlssExtensions);
    if (NVSDK_NGX_FAILED(dlssResult)) {
        deviceCerr() << "dlss device extensions unavailable; skipping." << std::endl;
    } else {
#ifdef DEBUG
        deviceCout() << "dlss instance extensions:" << std::endl;
#endif
        for (int i = 0; i < dlssExtensions.size(); i++) {
            if (std::strcmp(dlssExtensions[i].extensionName, "VK_EXT_buffer_device_address") == 0)
                continue; // already enabled using PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
#ifdef DEBUG
            deviceCout() << "\t" << dlssExtensions[i].extensionName << std::endl;
#endif
            enabledExtensions.push_back(dlssExtensions[i].extensionName);
        }
    }

    // Extra extensions (e.g. from OpenXR)
    // Store as const char* pointers — the static vector owns the strings.
    for (const auto &ext : extraExtensions) { enabledExtensions.push_back(ext.c_str()); }

    uint32_t deviceExtensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_->vkPhysicalDevice(), nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_->vkPhysicalDevice(), nullptr, &deviceExtensionCount,
                                         deviceExtensions.data());

    std::unordered_set<std::string> supportedExtensions;
    supportedExtensions.reserve(deviceExtensions.size());
    for (const auto &ext : deviceExtensions) { supportedExtensions.insert(ext.extensionName); }

    std::vector<const char *> filteredExtensions;
    filteredExtensions.reserve(enabledExtensions.size());
    std::unordered_set<std::string> seenExtensions;
    for (const auto *ext : enabledExtensions) {
        if (supportedExtensions.find(ext) == supportedExtensions.end()) {
            deviceCerr() << "extension not supported, skipping: " << ext << std::endl;
            continue;
        }
        if (!seenExtensions.insert(ext).second) { continue; }
        filteredExtensions.push_back(ext);
    }

#ifdef DEBUG
    deviceCout() << "selected instance extensions:" << std::endl;
    for (int i = 0; i < filteredExtensions.size(); i++) { deviceCout() << "\t" << filteredExtensions[i] << std::endl; }
#endif

    // query supported features
    VkPhysicalDeviceMaintenance5Features supportedMaintenance5{};
    supportedMaintenance5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;

    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT supportedVertexInputDynamicState{};
    supportedVertexInputDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    supportedVertexInputDynamicState.pNext = &supportedMaintenance5;

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT supportedExtendedDynamicState3{};
    supportedExtendedDynamicState3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    supportedExtendedDynamicState3.pNext = &supportedVertexInputDynamicState;

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT supportedExtendedDynamicState2{};
    supportedExtendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    supportedExtendedDynamicState2.pNext = &supportedExtendedDynamicState3;

    VkPhysicalDeviceVulkan13Features supportedVulkan13{};
    supportedVulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    supportedVulkan13.pNext = &supportedExtendedDynamicState2;

    VkPhysicalDeviceVulkan12Features supportedVulkan12{};
    supportedVulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supportedVulkan12.pNext = &supportedVulkan13;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR supportedAccelerationStructureFeatures{};
    supportedAccelerationStructureFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    supportedAccelerationStructureFeatures.pNext = &supportedVulkan12;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supportedRayTracingFeatures{};
    supportedRayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    supportedRayTracingFeatures.pNext = &supportedAccelerationStructureFeatures;

    VkPhysicalDeviceFeatures2 supportedFeatures2{};
    supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures2.pNext = &supportedRayTracingFeatures;

    vkGetPhysicalDeviceFeatures2(physicalDevice_->vkPhysicalDevice(), &supportedFeatures2);

    std::unordered_set<std::string> selectedExtensions;
    selectedExtensions.reserve(filteredExtensions.size());
    for (const auto *ext : filteredExtensions) { selectedExtensions.insert(ext); }
    auto hasExtension = [&](const char *name) { return selectedExtensions.find(name) != selectedExtensions.end(); };

    // enabling features
    VkPhysicalDeviceMaintenance5Features maintenance5Features{};
    maintenance5Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    maintenance5Features.maintenance5 =
        hasExtension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ? supportedMaintenance5.maintenance5 : VK_FALSE;

    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertexInputDynamicState{};
    vertexInputDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    vertexInputDynamicState.pNext = &maintenance5Features;
    vertexInputDynamicState.vertexInputDynamicState = hasExtension(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME) ?
                                                          supportedVertexInputDynamicState.vertexInputDynamicState :
                                                          VK_FALSE;

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicState3{};
    extendedDynamicState3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    extendedDynamicState3.pNext = &vertexInputDynamicState;
    if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME)) {
        extendedDynamicState3.extendedDynamicState3PolygonMode =
            supportedExtendedDynamicState3.extendedDynamicState3PolygonMode;
        extendedDynamicState3.extendedDynamicState3ColorBlendEnable =
            supportedExtendedDynamicState3.extendedDynamicState3ColorBlendEnable;
        extendedDynamicState3.extendedDynamicState3ColorBlendEquation =
            supportedExtendedDynamicState3.extendedDynamicState3ColorBlendEquation;
        extendedDynamicState3.extendedDynamicState3ColorWriteMask =
            supportedExtendedDynamicState3.extendedDynamicState3ColorWriteMask;
        extendedDynamicState3.extendedDynamicState3LogicOpEnable =
            supportedExtendedDynamicState3.extendedDynamicState3LogicOpEnable;
    }

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicState2{};
    extendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    extendedDynamicState2.pNext = &extendedDynamicState3;
    if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME)) {
        extendedDynamicState2.extendedDynamicState2 = supportedExtendedDynamicState2.extendedDynamicState2;
#if defined(USE_AMD)
        // AMD drivers have issues with extendedDynamicState2LogicOp on both Linux and Windows
        VkBool32 wantLogicOp = VK_FALSE;
#else
        VkBool32 wantLogicOp = VK_TRUE;
#endif
        extendedDynamicState2.extendedDynamicState2LogicOp =
            (supportedExtendedDynamicState2.extendedDynamicState2LogicOp && wantLogicOp) ? VK_TRUE : VK_FALSE;

        // Store the flag for runtime checks
        extendedDynamicState2LogicOp_ = (extendedDynamicState2.extendedDynamicState2LogicOp == VK_TRUE);

#ifdef DEBUG
        std::cout << "[Device-Debug] extendedDynamicState2: "
                  << (extendedDynamicState2.extendedDynamicState2 ? "YES" : "NO") << std::endl;
        std::cout << "[Device-Debug] extendedDynamicState2LogicOp: "
                  << (extendedDynamicState2.extendedDynamicState2LogicOp ? "YES" : "NO") << std::endl;
#endif

        extendedDynamicState2.extendedDynamicState2PatchControlPoints = VK_FALSE;
    } else {
        std::cerr << "[Device-Debug] VK_EXT_extended_dynamic_state2 NOT FOUND!" << std::endl;
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &extendedDynamicState2;
    vulkan13Features.shaderDemoteToHelperInvocation = supportedVulkan13.shaderDemoteToHelperInvocation;
    vulkan13Features.synchronization2 = supportedVulkan13.synchronization2;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan13Features;
    vulkan12Features.bufferDeviceAddress = supportedVulkan12.bufferDeviceAddress;
    vulkan12Features.descriptorBindingUpdateUnusedWhilePending =
        supportedVulkan12.descriptorBindingUpdateUnusedWhilePending;
    vulkan12Features.descriptorBindingPartiallyBound = supportedVulkan12.descriptorBindingPartiallyBound;
    vulkan12Features.descriptorIndexing = supportedVulkan12.descriptorIndexing;
    vulkan12Features.runtimeDescriptorArray = supportedVulkan12.runtimeDescriptorArray;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing =
        supportedVulkan12.shaderSampledImageArrayNonUniformIndexing;
    vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind =
        supportedVulkan12.descriptorBindingUniformBufferUpdateAfterBind;
    vulkan12Features.descriptorBindingSampledImageUpdateAfterBind =
        supportedVulkan12.descriptorBindingSampledImageUpdateAfterBind;
    vulkan12Features.descriptorBindingStorageImageUpdateAfterBind =
        supportedVulkan12.descriptorBindingStorageImageUpdateAfterBind;
    vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind =
        supportedVulkan12.descriptorBindingStorageBufferUpdateAfterBind;
    vulkan12Features.shaderFloat16 = supportedVulkan12.shaderFloat16;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {};
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelerationStructureFeatures.pNext = &vulkan12Features;
    if (hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
        accelerationStructureFeatures.accelerationStructure =
            supportedAccelerationStructureFeatures.accelerationStructure;
        accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind =
            supportedAccelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
    }

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures = {};
    rayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingFeatures.pNext = &accelerationStructureFeatures;
    if (hasExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
        rayTracingFeatures.rayTracingPipeline = supportedRayTracingFeatures.rayTracingPipeline;
    }

    VkPhysicalDeviceFeatures features = {};
    features.shaderClipDistance = supportedFeatures2.features.shaderClipDistance;
    features.shaderCullDistance = supportedFeatures2.features.shaderCullDistance;
    features.logicOp = supportedFeatures2.features.logicOp;
    features.fillModeNonSolid = supportedFeatures2.features.fillModeNonSolid;
    features.depthBiasClamp = supportedFeatures2.features.depthBiasClamp;
    features.shaderInt64 = supportedFeatures2.features.shaderInt64;
    features.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
    features.shaderInt16 = supportedFeatures2.features.shaderInt16;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &rayTracingFeatures;
    features2.features = features;

    // create logical device
    VkDeviceCreateInfo deviceCreateInfo = {};
    if (physicalDevice_->mainQueueIndex() == physicalDevice_->secondaryQueueIndex()) {
        std::vector<float> queuePriorities{{1.0, 0.0}};
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = physicalDevice_->mainQueueIndex();
        queueCreateInfo.queueCount = 2;
        queueCreateInfo.pQueuePriorities = queuePriorities.data();

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = filteredExtensions.data();
        deviceCreateInfo.pNext = &features2;
        deviceCreateInfo.pEnabledFeatures = nullptr;

        if (vkCreateDevice(physicalDevice_->vkPhysicalDevice(), &deviceCreateInfo, nullptr, &device_) != VK_SUCCESS) {
            deviceCerr() << "Failed to create logical device!" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        std::vector<float> queuePriorities{{1.0, 0.0}};
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(2);
        queueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[0].queueFamilyIndex = physicalDevice_->mainQueueIndex();
        queueCreateInfos[0].queueCount = 1;
        queueCreateInfos[0].pQueuePriorities = &queuePriorities[0];

        queueCreateInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[1].queueFamilyIndex = physicalDevice_->secondaryQueueIndex();
        queueCreateInfos[1].queueCount = 1;
        queueCreateInfos[1].pQueuePriorities = &queuePriorities[1];

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = filteredExtensions.data();
        deviceCreateInfo.pNext = &features2;
        deviceCreateInfo.pEnabledFeatures = nullptr;

        if (vkCreateDevice(physicalDevice_->vkPhysicalDevice(), &deviceCreateInfo, nullptr, &device_) != VK_SUCCESS) {
            deviceCerr() << "Failed to create logical device!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    volkLoadDevice(device_);

#ifdef DEBUG
    deviceCout() << "Logical device created successfully!" << std::endl;
#endif

    vkGetDeviceQueue(device_, physicalDevice_->mainQueueIndex(), 0, &mainQueue_);
    vkGetDeviceQueue(device_, physicalDevice_->secondaryQueueIndex(),
                     physicalDevice_->mainQueueIndex() == physicalDevice_->secondaryQueueIndex() ? 1 : 0,
                     &secondaryQueue_);
}

vk::Device::~Device() {
    vkDestroyDevice(device_, nullptr);

#ifdef DEBUG
    deviceCout() << "device deconstructed" << std::endl;
#endif
}

VkDevice &vk::Device::vkDevice() {
    return device_;
}

VkQueue &vk::Device::mainVkQueue() {
    return mainQueue_;
}

VkQueue &vk::Device::secondaryQueue() {
    return secondaryQueue_;
}
