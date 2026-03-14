#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/render/pipeline.hpp"
#include "core/vulkan/all_core_vulkan.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"

#ifdef MCVR_ENABLE_OPENXR
#include "core/render/openxr_context.hpp"
#include <memory>
#endif

#include <map>
#include <mutex>

class Framework;
class UIModule;
struct UIModuleContext;

class GarbageCollector : public SharedObject<GarbageCollector> {
  public:
    GarbageCollector(std::shared_ptr<Framework> framework);

    template <typename T>
    void collect(std::shared_ptr<T> garbage);

    void clear();

  private:
    std::weak_ptr<Framework> framework_;
    std::vector<std::vector<std::shared_ptr<void>>> collectors_;
    uint32_t index_ = 0;
};

struct FrameworkContext : public SharedObject<FrameworkContext> {
    std::weak_ptr<Framework> framework;

    uint32_t frameIndex;

    std::shared_ptr<vk::Instance> instance;
    std::shared_ptr<vk::Window> window;
    std::shared_ptr<vk::PhysicalDevice> physicalDevice;
    std::shared_ptr<vk::Device> device;
    std::shared_ptr<vk::VMA> vma;
    std::shared_ptr<vk::Swapchain> swapchain;
    std::shared_ptr<vk::SwapchainImage> swapchainImage;
    std::shared_ptr<vk::CommandPool> commandPool;
    std::shared_ptr<vk::Semaphore> imageAcquiredSemaphore = nullptr;
    std::shared_ptr<vk::Semaphore> commandProcessedSemaphore;
    std::shared_ptr<vk::Fence> commandFinishedFence;

    std::shared_ptr<vk::CommandBuffer> uploadCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> overlayCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer;
    std::shared_ptr<vk::CommandBuffer> fuseCommandBuffer;

    FrameworkContext(std::shared_ptr<Framework> framework, uint32_t frame_index);
    ~FrameworkContext();

    void fuseFinal();
};

class Framework : public SharedObject<Framework> {
    friend FrameworkContext;
    friend GarbageCollector;

  public:
    Framework();
    ~Framework();

    void init(GLFWwindow *window);
    void acquireContext();
    void submitCommand();
    void present();
    void recreate();
    void waitDeviceIdle();
    void waitRenderQueueIdle();
    void waitBackendQueueIdle();
    void close();
    bool isRunning();

    void takeScreenshot(bool withUI, int width, int height, int channel, void *dstPointer);

  #ifdef MCVR_ENABLE_OPENXR
    bool startXRSession();
    void stopXRSession();
    bool isXRSessionRunning() const;
  #endif

    std::recursive_mutex &recreateMtx();

    std::shared_ptr<vk::Instance> instance();
    std::shared_ptr<vk::Window> window();
    std::shared_ptr<vk::PhysicalDevice> physicalDevice();
    std::shared_ptr<vk::Device> device();
    std::shared_ptr<vk::VMA> vma();
    std::shared_ptr<vk::Swapchain> swapchain();
    std::shared_ptr<vk::CommandPool> mainCommandPool();
    std::shared_ptr<vk::CommandPool> asyncCommandPool();

    std::shared_ptr<vk::CommandBuffer> worldAsyncCommandBuffer();

    std::vector<std::shared_ptr<vk::Semaphore>> &commandProcessedSemaphores();
    std::vector<std::shared_ptr<vk::Fence>> &commandFinishedFences();
    std::vector<std::shared_ptr<FrameworkContext>> &contexts();
    std::shared_ptr<FrameworkContext> safeAcquireCurrentContext();

    std::shared_ptr<Pipeline> pipeline();

    GarbageCollector &gc();

  private:
    std::shared_ptr<vk::Semaphore> acquireSemaphore();
    void recycleSemaphore(std::shared_ptr<vk::Semaphore> semaphore);

  private:
    std::shared_ptr<vk::Instance> instance_;
    std::shared_ptr<vk::Window> window_;
    std::shared_ptr<vk::PhysicalDevice> physicalDevice_;
    std::shared_ptr<vk::Device> device_;
    std::shared_ptr<vk::VMA> vma_;
    std::shared_ptr<vk::Swapchain> swapchain_;
    std::shared_ptr<vk::CommandPool> mainCommandPool_;
    std::shared_ptr<vk::CommandPool> asyncCommandPool_;

    std::vector<std::shared_ptr<vk::CommandBuffer>> uploadCommandBuffers_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> overlayCommandBuffers_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> worldCommandBuffers_;
    std::shared_ptr<vk::CommandBuffer> worldAsyncCommandBuffer_;

    std::shared_ptr<Pipeline> pipeline_;

    std::vector<std::shared_ptr<vk::Semaphore>> commandProcessedSemaphores_;
    std::vector<std::shared_ptr<vk::Fence>> commandFinishedFences_;

    std::vector<std::shared_ptr<FrameworkContext>> contexts_;

    std::shared_ptr<FrameworkContext> currentContext_ = nullptr;
    uint32_t currentContextIndex_ = 0;
    std::queue<uint32_t> indexHistory_;

    std::queue<std::shared_ptr<vk::Semaphore>> recycledImageAcquiredSemaphores_;
    std::recursive_mutex recreateMtx_;

    bool running_ = true;

    std::shared_ptr<GarbageCollector> gc_;

#ifdef MCVR_ENABLE_OPENXR
    std::unique_ptr<OpenXRContext> xrContext_;
    bool xrVRPopulated_ = false;
    VkQueryPool gpuTimestampPool_ = VK_NULL_HANDLE;
    float timestampPeriodNs_ = 0.0f;
  bool xrLastSessionRunning_ = false;
  uint32_t xrLastSwapchainWidth_ = 0;
  uint32_t xrLastSwapchainHeight_ = 0;
public:
    OpenXRContext *xrContext() { return xrContext_.get(); }
private:
#endif
};

template <typename T>
void GarbageCollector::collect(std::shared_ptr<T> garbage) {
    auto framework = framework_.lock();
    if (garbage != nullptr) { collectors_[index_].push_back(garbage); }
}