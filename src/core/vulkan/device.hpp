#pragma once

#include "core/all_extern.hpp"

#include <mutex>
#include <filesystem>

namespace vk {
class Instance;
class Window;
class PhysicalDevice;

class Device : public SharedObject<Device> {
  public:
    Device(std::shared_ptr<Instance> instance,
           std::shared_ptr<Window> window,
           std::shared_ptr<PhysicalDevice> physicalDevice);
    ~Device();

    VkDevice &vkDevice();
    VkPipelineCache vkPipelineCache() const;
    void savePipelineCache();
    VkQueue &mainVkQueue();
    VkQueue &presentQueue();
    VkQueue &secondaryQueue();
    VkQueue &asyncComputeQueue();
    uint32_t mainQueueFamilyIndex() const;
    uint32_t secondaryQueueFamilyIndex() const;
    bool hasIndependentSecondaryQueue() const;
    std::recursive_mutex &mainQueueMutex();
    std::recursive_mutex &presentQueueMutex();
    std::recursive_mutex &secondaryQueueMutex();
    std::recursive_mutex &asyncComputeQueueMutex();

    bool hasExtendedDynamicState2LogicOp() const;
    bool hasDynamicColorBlendState() const;
    bool hasAccelerationStructureUpdateAfterBind() const;
    uint32_t textureDescriptorCapacity() const;
    bool hasMultiDrawIndirect() const;
    bool hasDrawIndirectFirstInstance() const;
    bool hasShaderInvocationReorderEXT() const;
    bool hasShaderInvocationReorderNV() const;
    bool isDlssRRDeviceExtensionsCompatible() const;
    bool isDlssSRDeviceExtensionsCompatible() const;
    bool isDlssFrameGenerationDeviceExtensionsCompatible() const;
    bool isFsrFrameGenerationDeviceExtensionsCompatible() const;
    bool isXessDeviceExtensionsCompatible() const;
    bool hasMemoryBudget() const;

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;
    std::shared_ptr<PhysicalDevice> physicalDevice_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::filesystem::path pipelineCachePath_;
    void initPipelineCache();
    VkQueue mainQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue secondaryQueue_ = VK_NULL_HANDLE;
    VkQueue asyncComputeQueue_ = VK_NULL_HANDLE;
    std::recursive_mutex mainQueueMutex_;
    std::recursive_mutex presentQueueMutex_;
    std::recursive_mutex secondaryQueueMutex_;
    std::recursive_mutex asyncComputeQueueMutex_;

    bool extendedDynamicState2LogicOp_ = false;
    bool dynamicColorBlendState_ = false;
    bool accelerationStructureUpdateAfterBind_ = false;
    uint32_t textureDescriptorCapacity_ = 4096;
    void initTextureDescriptorCapacity();
    bool multiDrawIndirect_ = false;
    bool drawIndirectFirstInstance_ = false;
    bool shaderInvocationReorderEXT_ = false;
    bool shaderInvocationReorderNV_ = false;
    bool dlssRRDeviceExtensionsCompatible_ = false;
    bool dlssSRDeviceExtensionsCompatible_ = false;
    bool dlssFrameGenerationDeviceExtensionsCompatible_ = false;
    bool fsrFrameGenerationDeviceExtensionsCompatible_ = false;
    bool xessDeviceExtensionsCompatible_ = false;
    bool memoryBudgetEnabled_ = false;
    bool independentSecondaryQueue_ = false;
};
};
