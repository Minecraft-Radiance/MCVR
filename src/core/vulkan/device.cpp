#include "core/vulkan/device.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/xess_upscaler/xess_wrapper.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/result.hpp"
#include "core/render/renderer.hpp"

#include <cstring>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

std::ostream &deviceCerr() {
    return std::cerr << "[Device] ";
}

vk::Device::Device(std::shared_ptr<Instance> instance,
                   std::shared_ptr<Window> window,
                   std::shared_ptr<PhysicalDevice> physicalDevice)
    : instance_(instance), window_(window), physicalDevice_(physicalDevice) {
    std::vector<const char *> enabledExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
        VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
    };
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

    std::vector<VkExtensionProperties> dlssRRExtensions;
    NVSDK_NGX_Result dlssRRResult =
        NgxContext::getDlssRRRequiredDeviceExtensions(instance_, physicalDevice_, dlssRRExtensions);
    if (NVSDK_NGX_FAILED(dlssRRResult)) {
        deviceCerr() << "dlss ray reconstruction device extensions unavailable; skipping." << std::endl;
    } else {
        dlssRRRequirementQuerySuccess = true;
        for (const auto &dlssExtension : dlssRRExtensions) {
            dlssRRRequiredExtensions.emplace_back(dlssExtension.extensionName);
            if (std::strcmp(dlssExtension.extensionName, "VK_EXT_buffer_device_address") == 0) { continue; }
            enabledExtensions.push_back(dlssExtension.extensionName);
        }
    }

    std::vector<VkExtensionProperties> dlssSRExtensions;
    NVSDK_NGX_Result dlssSRResult =
        NgxContext::getDlssSRRequiredDeviceExtensions(instance_, physicalDevice_, dlssSRExtensions);
    if (NVSDK_NGX_FAILED(dlssSRResult)) {
        deviceCerr() << "dlss super resolution device extensions unavailable; skipping." << std::endl;
    } else {
        dlssSRRequirementQuerySuccess = true;
        for (const auto &dlssExtension : dlssSRExtensions) {
            dlssSRRequiredExtensions.emplace_back(dlssExtension.extensionName);
            if (std::strcmp(dlssExtension.extensionName, "VK_EXT_buffer_device_address") == 0) { continue; }
            enabledExtensions.push_back(dlssExtension.extensionName);
        }
    }

    std::vector<VkExtensionProperties> dlssFrameGenerationExtensions;
    NVSDK_NGX_Result dlssFrameGenerationResult = NgxContext::getDlssFrameGenerationRequiredDeviceExtensions(
        instance_, physicalDevice_, dlssFrameGenerationExtensions);
    if (NVSDK_NGX_FAILED(dlssFrameGenerationResult)) {
        deviceCerr() << "dlss frame generation device extensions unavailable; skipping." << std::endl;
    } else {
        dlssFrameGenerationRequirementQuerySuccess = true;
        for (const auto &dlssExtension : dlssFrameGenerationExtensions) {
            dlssFrameGenerationRequiredExtensions.emplace_back(dlssExtension.extensionName);
            if (std::strcmp(dlssExtension.extensionName, "VK_EXT_buffer_device_address") == 0) { continue; }
            enabledExtensions.push_back(dlssExtension.extensionName);
        }
    }

#ifdef MCVR_ENABLE_XESS
    std::vector<const char *> xessExtensions;
    if (mcvr::XeSSWrapper::getRequiredDeviceExtensions(instance_->vkInstance(), physicalDevice_->vkPhysicalDevice(),
                                                       xessExtensions)) {
        xessRequirementQuerySuccess = true;
        for (const char *extension : xessExtensions) {
            xessRequiredExtensions.emplace_back(extension);
            enabledExtensions.push_back(extension);
        }
    } else {
        deviceCerr() << "xess device extensions unavailable; skipping." << std::endl;
    }
#endif

    uint32_t deviceExtensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_->vkPhysicalDevice(), nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_->vkPhysicalDevice(), nullptr, &deviceExtensionCount,
                                         deviceExtensions.data());

    std::unordered_set<std::string> supportedExtensions;
    supportedExtensions.reserve(deviceExtensions.size());
    for (const auto &ext : deviceExtensions) { supportedExtensions.insert(ext.extensionName); }

    if (supportedExtensions.contains("VK_KHR_portability_subset")) {
        enabledExtensions.push_back("VK_KHR_portability_subset");
    }
    if (supportedExtensions.find(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) != supportedExtensions.end()) {
        enabledExtensions.push_back(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
    }
    if (supportedExtensions.find(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) != supportedExtensions.end()) {
        enabledExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        memoryBudgetEnabled_ = true;
    }
#ifdef MCVR_UI_BACKEND_OPENGL
    const std::vector<const char *> openglUiRequiredExtensions = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#    if defined(_WIN32)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#    else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#    endif
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#    if defined(_WIN32)
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#    else
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#    endif
    };
    for (const char *extension : openglUiRequiredExtensions) {
        if (supportedExtensions.find(extension) == supportedExtensions.end()) {
            deviceCerr() << "OpenGL UI backend requires unsupported Vulkan device extension: " << extension
                         << std::endl;
            exit(EXIT_FAILURE);
        }
        enabledExtensions.push_back(extension);
    }
#endif

    bool supportsInvocationReorderEXTExtension = false;
    bool supportsInvocationReorderNVExtension = false;
#if defined(VK_EXT_ray_tracing_invocation_reorder)
    supportsInvocationReorderEXTExtension =
        supportedExtensions.find(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) != supportedExtensions.end();
#endif
#if defined(VK_NV_ray_tracing_invocation_reorder)
    supportsInvocationReorderNVExtension =
        supportedExtensions.find(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) != supportedExtensions.end();
#endif

    auto areRequiredExtensionsSupported = [&](const std::vector<std::string> &requiredExtensions) {
        for (const auto &requiredExtension : requiredExtensions) {
            if (requiredExtension == "VK_EXT_buffer_device_address") { continue; }
            if (supportedExtensions.find(requiredExtension) == supportedExtensions.end()) { return false; }
        }
        return true;
    };

    dlssRRDeviceExtensionsCompatible_ = instance_->isDlssRRInstanceExtensionsCompatible() &&
                                        dlssRRRequirementQuerySuccess &&
                                        areRequiredExtensionsSupported(dlssRRRequiredExtensions);
    if (!dlssRRDeviceExtensionsCompatible_) {
        deviceCerr() << "dlss ray reconstruction device extension requirements are not fully satisfied." << std::endl;
    }

    dlssSRDeviceExtensionsCompatible_ = instance_->isDlssSRInstanceExtensionsCompatible() &&
                                        dlssSRRequirementQuerySuccess &&
                                        areRequiredExtensionsSupported(dlssSRRequiredExtensions);
    if (!dlssSRDeviceExtensionsCompatible_) {
        deviceCerr() << "dlss super resolution device extension requirements are not fully satisfied." << std::endl;
    }

    dlssFrameGenerationDeviceExtensionsCompatible_ =
        instance_->isDlssFrameGenerationInstanceExtensionsCompatible() && dlssFrameGenerationRequirementQuerySuccess &&
        areRequiredExtensionsSupported(dlssFrameGenerationRequiredExtensions);
    if (!dlssFrameGenerationDeviceExtensionsCompatible_) {
        deviceCerr() << "dlss frame generation device extension requirements are not fully satisfied." << std::endl;
    }

#ifdef MCVR_ENABLE_FFX_UPSCALER
    const std::vector<std::string> fsrFrameGenerationRequiredExtensions = {
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,          VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
    };
    fsrFrameGenerationDeviceExtensionsCompatible_ =
        areRequiredExtensionsSupported(fsrFrameGenerationRequiredExtensions);
    if (!fsrFrameGenerationDeviceExtensionsCompatible_) {
        deviceCerr() << "fsr frame generation device extension requirements are not fully satisfied." << std::endl;
    }
#endif

#ifdef MCVR_ENABLE_XESS
    xessDeviceExtensionsCompatible_ = instance_->isXessInstanceExtensionsCompatible() && xessRequirementQuerySuccess &&
                                      areRequiredExtensionsSupported(xessRequiredExtensions);
    if (!xessDeviceExtensionsCompatible_) {
        deviceCerr() << "xess device extension requirements are not fully satisfied." << std::endl;
    }
#endif

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

    VkPhysicalDeviceMaintenance5Features supportedMaintenance5{};
    supportedMaintenance5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;

    VkPhysicalDeviceCoherentMemoryFeaturesAMD supportedCoherentMemoryFeatures{};
    supportedCoherentMemoryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;
    supportedCoherentMemoryFeatures.pNext = &supportedMaintenance5;

    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT supportedVertexInputDynamicState{};
    supportedVertexInputDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    supportedVertexInputDynamicState.pNext = &supportedCoherentMemoryFeatures;

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT supportedExtendedDynamicState3{};
    supportedExtendedDynamicState3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    supportedExtendedDynamicState3.pNext = &supportedVertexInputDynamicState;

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT supportedExtendedDynamicState2{};
    supportedExtendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    supportedExtendedDynamicState2.pNext = &supportedExtendedDynamicState3;

    VkPhysicalDeviceVulkan13Features supportedVulkan13{};
    supportedVulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    supportedVulkan13.pNext = &supportedExtendedDynamicState2;

    VkPhysicalDeviceVulkan11Features supportedVulkan11{};
    supportedVulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    supportedVulkan11.pNext = &supportedVulkan13;

    VkPhysicalDeviceVulkan12Features supportedVulkan12{};
    supportedVulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supportedVulkan12.pNext = &supportedVulkan11;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR supportedAccelerationStructureFeatures{};
    supportedAccelerationStructureFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    supportedAccelerationStructureFeatures.pNext = &supportedVulkan12;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supportedRayTracingFeatures{};
    supportedRayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    supportedRayTracingFeatures.pNext = &supportedAccelerationStructureFeatures;

    void *supportedFeatureChain = &supportedRayTracingFeatures;
#if defined(VK_NV_ray_tracing_invocation_reorder)
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV supportedInvocationReorderNV{};
    if (supportsInvocationReorderNVExtension) {
        supportedInvocationReorderNV.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
        supportedInvocationReorderNV.pNext = supportedFeatureChain;
        supportedFeatureChain = &supportedInvocationReorderNV;
    }
#endif
#if defined(VK_EXT_ray_tracing_invocation_reorder)
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT supportedInvocationReorderEXT{};
    if (supportsInvocationReorderEXTExtension) {
        supportedInvocationReorderEXT.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT;
        supportedInvocationReorderEXT.pNext = supportedFeatureChain;
        supportedFeatureChain = &supportedInvocationReorderEXT;
    }
#endif

    VkPhysicalDeviceFeatures2 supportedFeatures2{};
    supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures2.pNext = supportedFeatureChain;

    vkGetPhysicalDeviceFeatures2(physicalDevice_->vkPhysicalDevice(), &supportedFeatures2);

#if defined(VK_EXT_ray_tracing_invocation_reorder)
    if (supportsInvocationReorderEXTExtension && supportedInvocationReorderEXT.rayTracingInvocationReorder) {
        filteredExtensions.push_back(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
    }
#endif
#if defined(VK_NV_ray_tracing_invocation_reorder)
    if (
#    if defined(VK_EXT_ray_tracing_invocation_reorder)
        !(supportsInvocationReorderEXTExtension && supportedInvocationReorderEXT.rayTracingInvocationReorder) &&
#    endif
        supportsInvocationReorderNVExtension && supportedInvocationReorderNV.rayTracingInvocationReorder) {
        filteredExtensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
    }
#endif

    if (supportedVulkan11.shaderDrawParameters != VK_TRUE) {
        deviceCerr() << "Physical device does not support VkPhysicalDeviceVulkan11Features::shaderDrawParameters. "
                     << "This renderer requires SPIR-V DrawParameters for shaders that use gl_BaseInstance."
                     << std::endl;
        exit(EXIT_FAILURE);
    }

    std::unordered_set<std::string> selectedExtensions;
    selectedExtensions.reserve(filteredExtensions.size());
    for (const auto *ext : filteredExtensions) { selectedExtensions.insert(ext); }
    auto hasExtension = [&](const char *name) { return selectedExtensions.find(name) != selectedExtensions.end(); };

    VkPhysicalDeviceMaintenance5Features maintenance5Features{};
    maintenance5Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    maintenance5Features.maintenance5 =
        hasExtension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ? supportedMaintenance5.maintenance5 : VK_FALSE;

    VkPhysicalDeviceCoherentMemoryFeaturesAMD coherentMemoryFeatures{};
    coherentMemoryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;
    coherentMemoryFeatures.pNext = &maintenance5Features;
    coherentMemoryFeatures.deviceCoherentMemory = hasExtension(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) ?
                                                      supportedCoherentMemoryFeatures.deviceCoherentMemory :
                                                      VK_FALSE;

    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertexInputDynamicState{};
    vertexInputDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    vertexInputDynamicState.pNext = &coherentMemoryFeatures;
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

    // Keep the three attachment states together: unsupported devices bake all
    // of them into cached graphics pipeline variants instead of issuing setters.
    dynamicColorBlendState_ = extendedDynamicState3.extendedDynamicState3ColorBlendEnable &&
                             extendedDynamicState3.extendedDynamicState3ColorBlendEquation &&
                             extendedDynamicState3.extendedDynamicState3ColorWriteMask;

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicState2{};
    extendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    extendedDynamicState2.pNext = &extendedDynamicState3;
    if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME)) {
        extendedDynamicState2.extendedDynamicState2 = supportedExtendedDynamicState2.extendedDynamicState2;
        extendedDynamicState2.extendedDynamicState2LogicOp =
            supportedExtendedDynamicState2.extendedDynamicState2LogicOp ? VK_TRUE : VK_FALSE;

        extendedDynamicState2LogicOp_ = (extendedDynamicState2.extendedDynamicState2LogicOp == VK_TRUE);

        extendedDynamicState2.extendedDynamicState2PatchControlPoints = VK_FALSE;
    } else {
        std::cerr << "[Device] VK_EXT_extended_dynamic_state2 is unavailable" << std::endl;
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &extendedDynamicState2;
    vulkan13Features.shaderDemoteToHelperInvocation = supportedVulkan13.shaderDemoteToHelperInvocation;
    vulkan13Features.synchronization2 = supportedVulkan13.synchronization2;
    vulkan13Features.maintenance4 = supportedVulkan13.maintenance4;

    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.pNext = &vulkan13Features;
    vulkan11Features.storageBuffer16BitAccess = supportedVulkan11.storageBuffer16BitAccess;
    vulkan11Features.shaderDrawParameters = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan11Features;
    vulkan12Features.bufferDeviceAddress = supportedVulkan12.bufferDeviceAddress;
    vulkan12Features.descriptorBindingUpdateUnusedWhilePending =
        supportedVulkan12.descriptorBindingUpdateUnusedWhilePending;
    vulkan12Features.descriptorBindingPartiallyBound = supportedVulkan12.descriptorBindingPartiallyBound;
    vulkan12Features.descriptorIndexing = supportedVulkan12.descriptorIndexing;
    vulkan12Features.runtimeDescriptorArray = supportedVulkan12.runtimeDescriptorArray;
    vulkan12Features.descriptorBindingVariableDescriptorCount =
        supportedVulkan12.descriptorBindingVariableDescriptorCount;
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
    vulkan12Features.shaderBufferInt64Atomics = supportedVulkan12.shaderBufferInt64Atomics;
    vulkan12Features.shaderStorageBufferArrayNonUniformIndexing =
        supportedVulkan12.shaderStorageBufferArrayNonUniformIndexing;
    vulkan12Features.shaderStorageImageArrayNonUniformIndexing =
        supportedVulkan12.shaderStorageImageArrayNonUniformIndexing;
    vulkan12Features.shaderUniformBufferArrayNonUniformIndexing =
        supportedVulkan12.shaderUniformBufferArrayNonUniformIndexing;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {};
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelerationStructureFeatures.pNext = &vulkan12Features;
    if (hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
        accelerationStructureFeatures.accelerationStructure =
            supportedAccelerationStructureFeatures.accelerationStructure;
        accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind =
            supportedAccelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
    }
    accelerationStructureUpdateAfterBind_ =
        accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind == VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures = {};
    rayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingFeatures.pNext = &accelerationStructureFeatures;
    if (hasExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
        rayTracingFeatures.rayTracingPipeline = supportedRayTracingFeatures.rayTracingPipeline;
        rayTracingFeatures.rayTracingPipelineTraceRaysIndirect =
            supportedRayTracingFeatures.rayTracingPipelineTraceRaysIndirect;
        rayTracingFeatures.rayTraversalPrimitiveCulling = supportedRayTracingFeatures.rayTraversalPrimitiveCulling;
    }

    void *enabledFeatureChain = &rayTracingFeatures;
#if defined(VK_EXT_ray_tracing_invocation_reorder)
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT invocationReorderEXT{};
    if (hasExtension(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) &&
        supportedInvocationReorderEXT.rayTracingInvocationReorder) {
        invocationReorderEXT.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT;
        invocationReorderEXT.pNext = enabledFeatureChain;
        invocationReorderEXT.rayTracingInvocationReorder = VK_TRUE;
        enabledFeatureChain = &invocationReorderEXT;
        shaderInvocationReorderEXT_ = true;
    }
#endif
#if defined(VK_NV_ray_tracing_invocation_reorder)
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV invocationReorderNV{};
    if (!shaderInvocationReorderEXT_ && hasExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) &&
        supportedInvocationReorderNV.rayTracingInvocationReorder) {
        invocationReorderNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
        invocationReorderNV.pNext = enabledFeatureChain;
        invocationReorderNV.rayTracingInvocationReorder = VK_TRUE;
        enabledFeatureChain = &invocationReorderNV;
        shaderInvocationReorderNV_ = true;
    }
#endif

    VkPhysicalDeviceFeatures features = {};
    features.independentBlend = supportedFeatures2.features.independentBlend;
    features.shaderClipDistance = supportedFeatures2.features.shaderClipDistance;
    features.shaderCullDistance = supportedFeatures2.features.shaderCullDistance;
    features.logicOp = supportedFeatures2.features.logicOp;
    features.fillModeNonSolid = supportedFeatures2.features.fillModeNonSolid;
    features.depthBiasClamp = supportedFeatures2.features.depthBiasClamp;
    features.multiDrawIndirect = supportedFeatures2.features.multiDrawIndirect;
    features.drawIndirectFirstInstance = supportedFeatures2.features.drawIndirectFirstInstance;
    multiDrawIndirect_ = supportedFeatures2.features.multiDrawIndirect == VK_TRUE;
    drawIndirectFirstInstance_ = supportedFeatures2.features.drawIndirectFirstInstance == VK_TRUE;
    features.shaderInt64 = supportedFeatures2.features.shaderInt64;
    features.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
    features.shaderInt16 = supportedFeatures2.features.shaderInt16;
    features.shaderStorageImageExtendedFormats = supportedFeatures2.features.shaderStorageImageExtendedFormats;
    features.shaderStorageImageReadWithoutFormat = supportedFeatures2.features.shaderStorageImageReadWithoutFormat;
    features.shaderStorageImageWriteWithoutFormat = supportedFeatures2.features.shaderStorageImageWriteWithoutFormat;
    features.fragmentStoresAndAtomics = supportedFeatures2.features.fragmentStoresAndAtomics;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = enabledFeatureChain;
    features2.features = features;

#ifdef MCVR_ENABLE_XESS
    if (xessDeviceExtensionsCompatible_) {
        void *featureChain = &features2;
        if (!mcvr::XeSSWrapper::getRequiredDeviceFeatures(instance_->vkInstance(), physicalDevice_->vkPhysicalDevice(),
                                                          &featureChain)) {
            xessDeviceExtensionsCompatible_ = false;
            deviceCerr() << "xess device feature requirements are not fully satisfied." << std::endl;
        }
    }
#endif

    independentSecondaryQueue_ = physicalDevice_->hasIndependentSecondaryQueue();
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_->vkPhysicalDevice(), &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_->vkPhysicalDevice(), &queueFamilyCount,
                                             queueFamilies.data());
    const uint32_t mainFamily = physicalDevice_->mainQueueIndex();
    const uint32_t secondaryFamily = physicalDevice_->secondaryQueueIndex();
    const uint32_t secondaryOffset = physicalDevice_->secondaryQueueOffset();
    uint32_t asyncOffset = secondaryOffset;
    if (independentSecondaryQueue_ && queueFamilies[secondaryFamily].queueCount > secondaryOffset + 1) {
        asyncOffset = secondaryOffset + 1;
    }
    std::vector<float> mainPriorities(mainFamily == secondaryFamily ? asyncOffset + 1 : 1, 1.0f);
    std::vector<float> secondaryPriorities(asyncOffset + 1, 1.0f);
    if (independentSecondaryQueue_) {
        (mainFamily == secondaryFamily ? mainPriorities : secondaryPriorities)[secondaryOffset] = 0.0f;
    }
    const uint32_t presentOffset = static_cast<uint32_t>(mainPriorities.size());
    const bool independentPresentQueue = queueFamilies[mainFamily].queueCount > presentOffset;
    if (independentPresentQueue) { mainPriorities.push_back(1.0f); }
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    VkDeviceQueueCreateInfo mainQueueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    mainQueueInfo.queueFamilyIndex = mainFamily;
    mainQueueInfo.queueCount = static_cast<uint32_t>(mainPriorities.size());
    mainQueueInfo.pQueuePriorities = mainPriorities.data();
    queueCreateInfos.push_back(mainQueueInfo);
    if (secondaryFamily != mainFamily) {
        VkDeviceQueueCreateInfo secondaryQueueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        secondaryQueueInfo.queueFamilyIndex = secondaryFamily;
        secondaryQueueInfo.queueCount = static_cast<uint32_t>(secondaryPriorities.size());
        secondaryQueueInfo.pQueuePriorities = secondaryPriorities.data();
        queueCreateInfos.push_back(secondaryQueueInfo);
    }
    VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = filteredExtensions.data();
    deviceCreateInfo.pNext = &features2;
    vk::requireSuccess(vkCreateDevice(physicalDevice_->vkPhysicalDevice(), &deviceCreateInfo, nullptr, &device_),
                       "vkCreateDevice");

    volkLoadDevice(device_);
    initTextureDescriptorCapacity();
    deviceCerr() << "[RADIANCE-E15] colorState=" << (dynamicColorBlendState_ ? "dynamic" : "static-pipelines")
                 << " textureCapacity=" << textureDescriptorCapacity_
                 << " accelerationStructureUpdateAfterBind=" << accelerationStructureUpdateAfterBind_ << std::endl;
    initPipelineCache();

    vkGetDeviceQueue(device_, physicalDevice_->mainQueueIndex(), 0, &mainQueue_);
    if (independentPresentQueue) {
        vkGetDeviceQueue(device_, mainFamily, presentOffset, &presentQueue_);
    } else {
        presentQueue_ = mainQueue_;
    }
    if (independentSecondaryQueue_) {
        vkGetDeviceQueue(device_, physicalDevice_->secondaryQueueIndex(), physicalDevice_->secondaryQueueOffset(),
                         &secondaryQueue_);
    } else {
        secondaryQueue_ = mainQueue_;
    }
    vkGetDeviceQueue(device_, secondaryFamily, asyncOffset, &asyncComputeQueue_);
}

vk::Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        VkResult waitResult = vkDeviceWaitIdle(device_);
        if (waitResult != VK_SUCCESS) {
            deviceCerr() << "vkDeviceWaitIdle during shutdown failed (" << vk::resultName(waitResult) << ", "
                         << static_cast<int32_t>(waitResult) << ")" << std::endl;
        }
        savePipelineCache();
        if (pipelineCache_ != VK_NULL_HANDLE) { vkDestroyPipelineCache(device_, pipelineCache_, nullptr); }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
}

VkDevice &vk::Device::vkDevice() {
    return device_;
}

void vk::Device::initPipelineCache() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_->vkPhysicalDevice(), &properties);
    std::ostringstream name;
    name << std::hex << properties.vendorID << '-' << properties.deviceID << '-' << properties.driverVersion << '-';
    for (uint8_t byte : properties.pipelineCacheUUID) {
        name << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    }
    if (!Renderer::folderPath.empty()) {
        pipelineCachePath_ = Renderer::folderPath / "cache/pipelines" / (name.str() + ".bin");
    }
    std::vector<char> data;
    std::ifstream input(pipelineCachePath_, std::ios::binary | std::ios::ate);
    if (input) {
        const auto size = input.tellg();
        if (size >= static_cast<std::streamoff>(sizeof(VkPipelineCacheHeaderVersionOne)) && size <= 256 * 1024 * 1024) {
            data.resize(static_cast<size_t>(size));
            input.seekg(0);
            if (!input.read(data.data(), data.size())) { data.clear(); }
        }
    }
    if (!data.empty()) {
        VkPipelineCacheHeaderVersionOne header{};
        std::memcpy(&header, data.data(), sizeof(header));
        if (header.headerSize != sizeof(header) || header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
            header.vendorID != properties.vendorID || header.deviceID != properties.deviceID ||
            std::memcmp(header.pipelineCacheUUID, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
            data.clear();
        }
    }
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.empty() ? nullptr : data.data();
    VkResult result = vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_);
    if (result != VK_SUCCESS && !data.empty()) {
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        result = vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_);
    }
    if (result != VK_SUCCESS) {
        pipelineCache_ = VK_NULL_HANDLE;
        deviceCerr() << "Pipeline cache unavailable: " << vk::resultName(result) << std::endl;
    }
}

VkPipelineCache vk::Device::vkPipelineCache() const {
    return pipelineCache_;
}

void vk::Device::savePipelineCache() {
    if (pipelineCache_ == VK_NULL_HANDLE || pipelineCachePath_.empty()) { return; }
    size_t size = 0;
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, nullptr) != VK_SUCCESS || size == 0 ||
        size > 256 * 1024 * 1024) {
        return;
    }
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, data.data()) != VK_SUCCESS) { return; }
    std::error_code error;
    std::filesystem::create_directories(pipelineCachePath_.parent_path(), error);
    if (error) { return; }
    auto temporaryPath = pipelineCachePath_;
    temporaryPath += "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
    {
        std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
        output.write(data.data(), size);
        output.close();
        if (!output) {
            std::filesystem::remove(temporaryPath, error);
            return;
        }
    }
#ifdef _WIN32
    if (MoveFileExW(temporaryPath.c_str(), pipelineCachePath_.c_str(), MOVEFILE_REPLACE_EXISTING) != 0) { return; }
#else
    std::filesystem::rename(temporaryPath, pipelineCachePath_, error);
    if (!error) { return; }
#endif
    std::filesystem::remove(temporaryPath, error);
}

VkQueue &vk::Device::mainVkQueue() {
    return mainQueue_;
}

VkQueue &vk::Device::presentQueue() {
    return presentQueue_;
}

VkQueue &vk::Device::secondaryQueue() {
    return secondaryQueue_;
}

VkQueue &vk::Device::asyncComputeQueue() {
    return asyncComputeQueue_;
}

uint32_t vk::Device::mainQueueFamilyIndex() const {
    return physicalDevice_->mainQueueIndex();
}

uint32_t vk::Device::secondaryQueueFamilyIndex() const {
    return physicalDevice_->secondaryQueueIndex();
}

bool vk::Device::hasIndependentSecondaryQueue() const {
    return independentSecondaryQueue_;
}

std::recursive_mutex &vk::Device::mainQueueMutex() {
    return mainQueueMutex_;
}

std::recursive_mutex &vk::Device::presentQueueMutex() {
    return presentQueue_ == mainQueue_ ? mainQueueMutex_ : presentQueueMutex_;
}

std::recursive_mutex &vk::Device::secondaryQueueMutex() {
    return independentSecondaryQueue_ ? secondaryQueueMutex_ : mainQueueMutex_;
}

std::recursive_mutex &vk::Device::asyncComputeQueueMutex() {
    if (asyncComputeQueue_ == mainQueue_) { return mainQueueMutex_; }
    if (asyncComputeQueue_ == secondaryQueue_) { return secondaryQueueMutex(); }
    return asyncComputeQueueMutex_;
}

bool vk::Device::hasExtendedDynamicState2LogicOp() const {
    return extendedDynamicState2LogicOp_;
}

bool vk::Device::hasDynamicColorBlendState() const {
    return dynamicColorBlendState_;
}

bool vk::Device::hasAccelerationStructureUpdateAfterBind() const {
    return accelerationStructureUpdateAfterBind_;
}

uint32_t vk::Device::textureDescriptorCapacity() const {
    return textureDescriptorCapacity_;
}

void vk::Device::initTextureDescriptorCapacity() {
    // Query the exact UI layout, including its extra post-process sampler and
    // binding flags. Both UI and world textures share the resulting ID range.
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    constexpr VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                               VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                                               VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorBindingFlags bindingFlags[] = {flags, flags};
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = 2;
    flagsInfo.pBindingFlags = bindingFlags;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    uint32_t low = 0, high = 4096;
    while (low < high) {
        uint32_t candidate = low + (high - low + 1) / 2;
        bindings[0].descriptorCount = candidate;
        VkDescriptorSetLayoutSupport support{};
        support.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
        vkGetDescriptorSetLayoutSupport(device_, &layoutInfo, &support);
        if (support.supported) low = candidate;
        else high = candidate - 1;
    }
    if (low == 0) { throw std::runtime_error("No supported texture descriptor layout"); }
    textureDescriptorCapacity_ = low;
}

bool vk::Device::hasMultiDrawIndirect() const {
    return multiDrawIndirect_;
}

bool vk::Device::hasDrawIndirectFirstInstance() const {
    return drawIndirectFirstInstance_;
}

bool vk::Device::hasShaderInvocationReorderEXT() const {
    return shaderInvocationReorderEXT_;
}

bool vk::Device::hasShaderInvocationReorderNV() const {
    return shaderInvocationReorderNV_;
}

bool vk::Device::isDlssRRDeviceExtensionsCompatible() const {
    return dlssRRDeviceExtensionsCompatible_;
}

bool vk::Device::isDlssSRDeviceExtensionsCompatible() const {
    return dlssSRDeviceExtensionsCompatible_;
}

bool vk::Device::isDlssFrameGenerationDeviceExtensionsCompatible() const {
    return dlssFrameGenerationDeviceExtensionsCompatible_;
}

bool vk::Device::isFsrFrameGenerationDeviceExtensionsCompatible() const {
    return fsrFrameGenerationDeviceExtensionsCompatible_;
}

bool vk::Device::isXessDeviceExtensionsCompatible() const {
    return xessDeviceExtensionsCompatible_;
}

bool vk::Device::hasMemoryBudget() const {
    return memoryBudgetEnabled_;
}
