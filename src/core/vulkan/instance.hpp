#pragma once

#include "core/all_extern.hpp"

#include <string>
#include <vector>

namespace vk {
class Instance : public SharedObject<Instance> {
  public:
    // Extra instance extensions to enable (set before create(), e.g. by OpenXR).
    static std::vector<std::string> extraExtensions;

    Instance();
    ~Instance();

    VkInstance &vkInstance();

  private:
    VkInstance instance_;
    // VkDebugReportCallbackEXT callback_ = VK_NULL_HANDLE;
};
} // namespace vk