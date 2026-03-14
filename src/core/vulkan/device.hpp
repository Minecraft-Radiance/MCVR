#pragma once

#include "core/all_extern.hpp"

#include <string>
#include <vector>

namespace vk {
class Instance;
class Window;
class PhysicalDevice;

class Device : public SharedObject<Device> {
  public:
    // Extra device extensions to enable (set before create(), e.g. by OpenXR).
    static std::vector<std::string> extraExtensions;

    Device(std::shared_ptr<Instance> instance,
           std::shared_ptr<Window> window,
           std::shared_ptr<PhysicalDevice> physicalDevice);
    ~Device();

    VkDevice &vkDevice();
    VkQueue &mainVkQueue();
    VkQueue &secondaryQueue();

    bool hasExtendedDynamicState2LogicOp() const { return extendedDynamicState2LogicOp_; }

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;
    std::shared_ptr<PhysicalDevice> physicalDevice_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue mainQueue_ = VK_NULL_HANDLE;
    VkQueue secondaryQueue_ = VK_NULL_HANDLE;

    bool extendedDynamicState2LogicOp_ = false;
};
}; // namespace vk