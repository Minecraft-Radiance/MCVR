#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance;
class Window;

class PhysicalDevice : public SharedObject<PhysicalDevice> {
  public:
    PhysicalDevice(std::shared_ptr<Instance> instance, std::shared_ptr<Window> window);
    ~PhysicalDevice();

    VkPhysicalDevice &vkPhysicalDevice();
    uint32_t mainQueueIndex();
    uint32_t secondaryQueueIndex();

    // If set, use this physical device instead of auto-selecting (e.g. OpenXR override).
    static VkPhysicalDevice overrideDevice;

    void findPhysicalDevice();
    void findQueueFamilies();

    VkPhysicalDeviceProperties properties();
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties();
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties();

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    uint32_t mainQueueIndex_ = -1;
    uint32_t secondaryQueueIndex_ = -1;

    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties_;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties_;
};
} // namespace vk