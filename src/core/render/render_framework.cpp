#include "core/render/render_framework.hpp"

#include "common/shared.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <sstream>

std::ostream &renderFrameworkCout() {
    return std::cout << "[Render Framework] ";
}

std::ostream &renderFrameworkCerr() {
    return std::cerr << "[Render Framework] ";
}

#ifdef MCVR_ENABLE_OPENXR
static void xrInitLog(const std::string &msg) {
    std::cout << "[Render Framework] " << msg << std::endl;
    std::ofstream file("openxr_debug.log", std::ios::out | std::ios::app);
    file << "[Render Framework] " << msg << std::endl;
}
#endif

FrameworkContext::FrameworkContext(std::shared_ptr<Framework> framework, uint32_t frameIndex)
    : framework(framework),
      frameIndex(frameIndex),
      instance(framework->instance_),
      window(framework->window_),
      physicalDevice(framework->physicalDevice_),
      device(framework->device_),
      vma(framework->vma_),
      swapchain(framework->swapchain_),
      swapchainImage(framework->swapchain_->swapchainImages()[frameIndex]),
      commandPool(framework->mainCommandPool_),
      commandProcessedSemaphore(framework->commandProcessedSemaphores_[frameIndex]),
      commandFinishedFence(framework->commandFinishedFences_[frameIndex]),
      uploadCommandBuffer(framework->uploadCommandBuffers_[frameIndex]),
      overlayCommandBuffer(framework->overlayCommandBuffers_[frameIndex]),
      worldCommandBuffer(framework->worldCommandBuffers_[frameIndex]),
      fuseCommandBuffer(framework->fuseCommandBuffers_[frameIndex]) {}

FrameworkContext::~FrameworkContext() {
#ifdef DEBUG
    std::cout << "[Context] context deconstructed" << std::endl;
#endif
}

void FrameworkContext::fuseFinal() {
    auto f = framework.lock();

    if (!f->isRunning()) return;

    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto pipelineContext = f->pipeline_->acquirePipelineContext(shared_from_this());

    fuseCommandBuffer->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = pipelineContext->uiModuleContext->overlayDrawColorImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = pipelineContext->uiModuleContext->overlayDrawColorImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = swapchainImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = swapchainImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
            });

    pipelineContext->uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // TODO: add to command buffer
    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(pipelineContext->uiModuleContext->overlayDrawColorImage->width()),
                               static_cast<int>(pipelineContext->uiModuleContext->overlayDrawColorImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(swapchainImage->width()), static_cast<int>(swapchainImage->height()),
                               1};

    vkCmdBlitImage(fuseCommandBuffer->vkCommandBuffer(),
                   pipelineContext->uiModuleContext->overlayDrawColorImage->vkImage(),
                   pipelineContext->uiModuleContext->overlayDrawColorImage->imageLayout(), swapchainImage->vkImage(),
                   swapchainImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

    fuseCommandBuffer->barriersBufferImage(
        {}, {{
                 .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = pipelineContext->uiModuleContext->overlayDrawColorImage->imageLayout(),
#ifdef USE_AMD
                 .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                 .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = pipelineContext->uiModuleContext->overlayDrawColorImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = swapchainImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = swapchainImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             }});

#ifdef USE_AMD
    pipelineContext->uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    pipelineContext->uiModuleContext->overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
}

Framework::Framework() {}

void Framework::init(GLFWwindow *window) {
#ifdef MCVR_ENABLE_OPENXR
    // Stage A: pre-Vulkan OpenXR init — queries required Vulkan extensions
    if (Renderer::options.vrEnabled) {
        xrInitLog("VR enabled, attempting OpenXR init...");
        try {
            xrContext_ = std::make_unique<OpenXRContext>();
            if (xrContext_->preVulkanInit()) {
                // Inject OpenXR-required extensions into Vulkan creation
                vk::Instance::extraExtensions = xrContext_->requiredInstanceExtensions();
                vk::Device::extraExtensions = xrContext_->requiredDeviceExtensions();
            } else {
                xrInitLog("ERROR: OpenXR pre-init failed, falling back to non-VR");
                xrContext_.reset();
                Renderer::options.vrEnabled = false;
            }
        } catch (const std::exception &e) {
            xrInitLog(std::string("ERROR: OpenXR pre-init exception: ") + e.what());
            xrContext_.reset();
            Renderer::options.vrEnabled = false;
        } catch (...) {
            xrInitLog("ERROR: OpenXR pre-init unknown exception");
            xrContext_.reset();
            Renderer::options.vrEnabled = false;
        }
    }
#endif

    instance_ = vk::Instance::create();
    window_ = vk::Window::create(instance_, window);

#ifdef MCVR_ENABLE_OPENXR
    // Let OpenXR choose the physical device if available
    if (xrContext_) {
        try {
            VkPhysicalDevice xrDev = xrContext_->getXRPhysicalDevice(instance_->vkInstance());
            if (xrDev != VK_NULL_HANDLE) { vk::PhysicalDevice::overrideDevice = xrDev; }
        } catch (...) {
            xrInitLog("ERROR: xrGetVulkanGraphicsDeviceKHR exception, ignoring");
        }
    }
#endif

    physicalDevice_ = vk::PhysicalDevice::create(instance_, window_);
    device_ = vk::Device::create(instance_, window_, physicalDevice_);

#ifdef MCVR_ENABLE_OPENXR
    // Stage B: post-Vulkan OpenXR init — creates session, swapchains
    if (xrContext_) {
        try {
            if (!xrContext_->postVulkanInit(instance_->vkInstance(), physicalDevice_->vkPhysicalDevice(),
                                            device_->vkDevice(), physicalDevice_->mainQueueIndex(), 0)) {
                xrInitLog("ERROR: OpenXR post-init failed, falling back to non-VR");
                xrContext_.reset();
                Renderer::options.vrEnabled = false;
            } else {
                xrInitLog("OpenXR runtime prepared; session will start on explicit request");
            }
        } catch (const std::exception &e) {
            xrInitLog(std::string("ERROR: OpenXR post-init exception: ") + e.what());
            xrContext_.reset();
            Renderer::options.vrEnabled = false;
        } catch (...) {
            xrInitLog("ERROR: OpenXR post-init unknown exception");
            xrContext_.reset();
            Renderer::options.vrEnabled = false;
        }
    }
    // Clear statics after use
    vk::Instance::extraExtensions.clear();
    vk::Device::extraExtensions.clear();
    vk::PhysicalDevice::overrideDevice = VK_NULL_HANDLE;
#endif

    vma_ = vk::VMA::create(instance_, physicalDevice_, device_);
    swapchain_ = vk::Swapchain::create(physicalDevice_, device_, window_);
    mainCommandPool_ = vk::CommandPool::create(physicalDevice_, device_);
    asyncCommandPool_ = vk::CommandPool::create(physicalDevice_, device_, physicalDevice_->secondaryQueueIndex());
    gc_ = GarbageCollector::create(shared_from_this());

    uint32_t imageCount = swapchain_->imageCount();

    // create command buffer for each context
    for (int i = 0; i < imageCount; i++) {
        uploadCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        overlayCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        worldCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        fuseCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
    }
    worldAsyncCommandBuffer_ = vk::CommandBuffer::create(device_, asyncCommandPool_);

    for (int i = 0; i < imageCount; i++) { commandFinishedFences_.push_back(vk::Fence::create(device_, true)); }

    for (int i = 0; i < imageCount; i++) { commandProcessedSemaphores_.push_back(vk::Semaphore::create(device_)); }

    for (int i = 0; i < imageCount; i++) { contexts_.push_back(FrameworkContext::create(shared_from_this(), i)); }

    pipeline_ = Pipeline::create(shared_from_this());

#ifdef MCVR_ENABLE_OPENXR
    // Create GPU timestamp query pool for frame timing (2 queries per frame-in-flight)
    if (xrContext_) {
        timestampPeriodNs_ = physicalDevice_->properties().limits.timestampPeriod;
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = imageCount * 2;
        vkCreateQueryPool(device_->vkDevice(), &qpci, nullptr, &gpuTimestampPool_);
        // Reset all queries to "unavailable"
        vkResetQueryPool(device_->vkDevice(), gpuTimestampPool_, 0, imageCount * 2);
    }
#endif
}

Framework::~Framework() {
#ifdef DEBUG
    std::cout << "[Framework] framework deconstructed" << std::endl;
#endif
}

void Framework::acquireContext() {
    if (!running_) return;

#ifdef MCVR_ENABLE_OPENXR
    // Poll OpenXR events and begin XR frame (updates head pose + FOV)
    if (xrContext_) {
        // Keep VRSystem config synchronized with options regardless of session state.
        auto &vr = Renderer::instance().vrSystem();
        vr.config.ipd = Renderer::options.vrIPD;
        vr.config.renderScale = Renderer::options.vrRenderScale;
        vr.config.worldScale = Renderer::options.vrWorldScale;

        xrContext_->pollEvents();
        bool sessionRunning = xrContext_->isSessionRunning();

        // Runtime transition point: session on/off means eyeCount/resolution path changed.
        if (sessionRunning != xrLastSessionRunning_) {
            xrLastSessionRunning_ = sessionRunning;
            Renderer::options.needRecreate = true;
            if (pipeline_ != nullptr) pipeline_->needRecreate = true;

            vr.enabled = Renderer::options.vrEnabled && sessionRunning;
            vr.eyeCount = vr.enabled ? 2u : 1u;

            if (sessionRunning) {
                vr.worldOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                vr.worldPosition = glm::vec3(0.0f);
            } else {
                vr.headPose = VRHeadPose{};
                vr.worldOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                vr.worldPosition = glm::vec3(0.0f);
            }
        }

        if (sessionRunning) {
            uint32_t xrW = xrContext_->swapchainWidth();
            uint32_t xrH = xrContext_->swapchainHeight();
            if (xrW > 0 && xrH > 0 && (xrW != xrLastSwapchainWidth_ || xrH != xrLastSwapchainHeight_)) {
                xrLastSwapchainWidth_ = xrW;
                xrLastSwapchainHeight_ = xrH;
                Renderer::options.needRecreate = true;
                if (pipeline_ != nullptr) pipeline_->needRecreate = true;
            }
        }

        if (xrContext_->isSessionRunning()) {
            xrContext_->beginFrame();
            // Update VRSystem with latest head pose and eye params
            xrContext_->getHeadPose(vr.headPose);
            xrContext_->getEyeParams(vr.eyes.data());
            // Keep config.ipd in sync with runtime-measured eye offsets.
            vr.config.ipd = glm::distance(vr.eyes[0].positionOffset, vr.eyes[1].positionOffset);

            // Update controller states from input subsystem
            auto &input = xrContext_->input();
            for (uint32_t h = 0; h < 2; h++) {
                auto &src = input.controllers[h];
                auto &dst = vr.controllers[h];
                dst.valid = src.valid;
                dst.position = src.position;
                dst.orientation = src.orientation;
                dst.linearVelocity = src.linearVelocity;
                dst.angularVelocity = src.angularVelocity;
                dst.triggerValue = src.triggerValue;
                dst.gripValue = src.gripValue;
                dst.thumbstick = src.thumbstick;
                dst.triggerPressed = src.triggerPressed;
                dst.gripPressed = src.gripPressed;
                dst.primaryButton = src.primaryButton;
                dst.secondaryButton = src.secondaryButton;
                dst.thumbstickClick = src.thumbstickClick;
                dst.menuButton = src.menuButton;
            }

            // Update eye gaze point
            vr.gazeValid = input.gazeValid;
            vr.gazePoint = input.gazePoint;

            // Performance stats: compositor target
            int64_t periodNs = xrContext_->predictedDisplayPeriodNs();
            vr.perfStats.compositorTargetMs = static_cast<float>(periodNs) / 1e6f;
        } else {
            vr.enabled = false;
            vr.eyeCount = 1;
            vr.headPose = VRHeadPose{};
        }
    }
#endif

    std::shared_ptr<FrameworkContext> lastContext;
    if (currentContext_) lastContext = currentContext_;
    VkResult result;

    std::shared_ptr<vk::Semaphore> imageAcquiredSemaphore = acquireSemaphore();
    uint32_t imageIndex;
    result = vkAcquireNextImageKHR(device_->vkDevice(), swapchain_->vkSwapchain(),
                                   5'000'000'000ull,  // 5 second timeout (avoid TDR on alt-tab/minimize)
                                   imageAcquiredSemaphore->vkSemaphore(), VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_TIMEOUT || result == VK_NOT_READY) {
        recycleSemaphore(imageAcquiredSemaphore);
        recreate();
        return;
    } else if (result != VK_SUCCESS) {
        std::cerr << "Cannot acquire images from swapchain" << std::endl;
        recycleSemaphore(imageAcquiredSemaphore);
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::shared_ptr<vk::Fence> fence = contexts_[imageIndex]->commandFinishedFence;
    result = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true,
                             5'000'000'000ull);  // 5 second timeout
    if (result == VK_TIMEOUT) {
        std::cerr << "vkWaitForFences timed out (possible TDR recovery), skipping frame" << std::endl;
        return;
    }
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }
    currentContextIndex_ = imageIndex;
    currentContext_ = contexts_[imageIndex];
    indexHistory_.push(imageIndex);
    if (indexHistory_.size() > swapchain_->imageCount()) indexHistory_.pop();

#ifdef MCVR_ENABLE_OPENXR
    // Read back GPU timestamps from the previous use of this frame slot
    if (gpuTimestampPool_ != VK_NULL_HANDLE && timestampPeriodNs_ > 0.0f) {
        uint64_t ts[2] = {0, 0};
        VkResult tsResult = vkGetQueryPoolResults(
            device_->vkDevice(), gpuTimestampPool_,
            imageIndex * 2, 2,
            sizeof(ts), ts, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (tsResult == VK_SUCCESS && ts[1] > ts[0]) {
            auto &vr = Renderer::instance().vrSystem();
            vr.perfStats.gpuFrameTimeMs =
                static_cast<float>(static_cast<double>(ts[1] - ts[0]) * timestampPeriodNs_ / 1e6);
        }
    }
#endif

    if (currentContext_->imageAcquiredSemaphore != VK_NULL_HANDLE) {
        recycleSemaphore(currentContext_->imageAcquiredSemaphore);
        currentContext_->imageAcquiredSemaphore = VK_NULL_HANDLE;
    }
    currentContext_->imageAcquiredSemaphore = imageAcquiredSemaphore;

    currentContext_->uploadCommandBuffer->begin();
    currentContext_->worldCommandBuffer->begin();
    currentContext_->overlayCommandBuffer->begin();
    currentContext_->fuseCommandBuffer->begin();

#ifdef MCVR_ENABLE_OPENXR
    // Reset + write GPU timestamp BEGIN in the first command buffer
    if (gpuTimestampPool_ != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = currentContext_->uploadCommandBuffer->vkCommandBuffer();
        vkCmdResetQueryPool(cmd, gpuTimestampPool_, currentContextIndex_ * 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            gpuTimestampPool_, currentContextIndex_ * 2);
    }
#endif

    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    std::shared_ptr<UIModuleContext> lastUIContext =
        lastContext == nullptr ? nullptr : pipeline_->acquirePipelineContext(lastContext)->uiModuleContext;

    pipelineContext->uiModuleContext->begin(lastUIContext);

    gc_->clear();
    Renderer::instance().buffers()->resetFrame();
    Renderer::instance().textures()->resetFrame();
    Renderer::instance().world()->resetFrame();
    Renderer::instance().world()->chunks()->resetFrame();
    Renderer::instance().world()->entities()->resetFrame();

    static int frames = 0;
    static auto lastTime = std::chrono::high_resolution_clock::now();
    static auto lastFrameStart = std::chrono::high_resolution_clock::now();

    auto frameStart = std::chrono::high_resolution_clock::now();

#ifdef MCVR_ENABLE_OPENXR
    // Track CPU frame time (from previous frame start to this one)
    if (xrContext_ && Renderer::options.vrEnabled) {
        auto &vr = Renderer::instance().vrSystem();
        float cpuMs = std::chrono::duration<float, std::milli>(frameStart - lastFrameStart).count();
        vr.perfStats.cpuFrameTimeMs = cpuMs;
        if (cpuMs > 0.0f) {
            vr.perfStats.fps = 1000.0f / cpuMs;
        }
        // Headroom = remaining time relative to compositor target (use max of CPU/GPU)
        if (vr.perfStats.compositorTargetMs > 0.0f) {
            float frameMs = std::max(cpuMs, vr.perfStats.gpuFrameTimeMs);
            vr.perfStats.headroom =
                (vr.perfStats.compositorTargetMs - frameMs) / vr.perfStats.compositorTargetMs;
        }
        // Dropped frames: frame time exceeds target
        float frameMs = std::max(cpuMs, vr.perfStats.gpuFrameTimeMs);
        if (frameMs > vr.perfStats.compositorTargetMs && vr.perfStats.compositorTargetMs > 0.0f) {
            vr.perfStats.droppedFrames++;
        }
    }
    lastFrameStart = frameStart;
#endif

    frames++;
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = currentTime - lastTime;

    if (elapsed.count() >= 1.0) {
        std::stringstream ss;
        ss << "FPS: " << frames;

        // GLFW_SetWindowTitle(window_->window(), ss.str().c_str());

        frames = 0;
        lastTime = currentTime;
    }
}

void Framework::submitCommand() {
    if (!running_) return;

    Renderer::instance().framework()->safeAcquireCurrentContext(); // ensure context is non nullptr

    Renderer::instance().textures()->performQueuedUpload();
    Renderer::instance().buffers()->performQueuedUpload();
    Renderer::instance().buffers()->buildAndUploadOverlayUniformBuffer();

    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    if (Renderer::instance().world()->shouldRender()) pipelineContext->worldPipelineContext->render();
    pipelineContext->uiModuleContext->end();

#ifdef MCVR_ENABLE_OPENXR
    // Blit world render output (array image layers) to XR swapchain images
    bool xrImagesAcquired = false;
    if (xrContext_ && xrContext_->isSessionRunning() && xrContext_->shouldRender()) {
        auto outputImage = pipelineContext->worldPipelineContext->outputImage;
        if (outputImage) {
            auto cmdBuf = currentContext_->worldCommandBuffer->vkCommandBuffer();
            auto mainQueueIndex = physicalDevice_->mainQueueIndex();

            for (uint32_t eye = 0; eye < 2; eye++) {
                VkImage xrImage = xrContext_->acquireSwapchainImage(eye);
                if (xrImage == VK_NULL_HANDLE) {
                    xrImagesAcquired = xrImagesAcquired || (eye > 0);
                    break;
                }
                xrImagesAcquired = true;
                uint32_t xrW = xrContext_->swapchainWidth();
                uint32_t xrH = xrContext_->swapchainHeight();

                // Transition XR image to TRANSFER_DST
                VkImageMemoryBarrier toTransferDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                toTransferDst.srcAccessMask = 0;
                toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toTransferDst.srcQueueFamilyIndex = mainQueueIndex;
                toTransferDst.dstQueueFamilyIndex = mainQueueIndex;
                toTransferDst.image = xrImage;
                toTransferDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmdBuf,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &toTransferDst);

                // Transition source layer to TRANSFER_SRC
                VkImageMemoryBarrier srcToTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                srcToTransfer.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                srcToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                srcToTransfer.oldLayout = outputImage->imageLayout();
                srcToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                srcToTransfer.srcQueueFamilyIndex = mainQueueIndex;
                srcToTransfer.dstQueueFamilyIndex = mainQueueIndex;
                srcToTransfer.image = outputImage->vkImage();
                srcToTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, eye, 1};
                vkCmdPipelineBarrier(cmdBuf,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &srcToTransfer);

                // Blit from output array layer to XR swapchain image
                VkImageBlit blitRegion{};
                blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eye, 1};
                blitRegion.srcOffsets[0] = {0, 0, 0};
                blitRegion.srcOffsets[1] = {
                    static_cast<int32_t>(outputImage->width()),
                    static_cast<int32_t>(outputImage->height()), 1};
                blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                blitRegion.dstOffsets[0] = {0, 0, 0};
                blitRegion.dstOffsets[1] = {static_cast<int32_t>(xrW), static_cast<int32_t>(xrH), 1};
                vkCmdBlitImage(cmdBuf,
                    outputImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    xrImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &blitRegion, VK_FILTER_LINEAR);

                // Transition XR image to COLOR_ATTACHMENT for the compositor
                VkImageMemoryBarrier toColorAttach{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                toColorAttach.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                toColorAttach.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
                toColorAttach.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toColorAttach.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                toColorAttach.srcQueueFamilyIndex = mainQueueIndex;
                toColorAttach.dstQueueFamilyIndex = mainQueueIndex;
                toColorAttach.image = xrImage;
                toColorAttach.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmdBuf,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &toColorAttach);

                // Transition source layer back to original layout
                VkImageMemoryBarrier srcBack{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                srcBack.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                srcBack.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                srcBack.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                srcBack.newLayout = outputImage->imageLayout();
                srcBack.srcQueueFamilyIndex = mainQueueIndex;
                srcBack.dstQueueFamilyIndex = mainQueueIndex;
                srcBack.image = outputImage->vkImage();
                srcBack.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, eye, 1};
                vkCmdPipelineBarrier(cmdBuf,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &srcBack);
            }
        }
    }
#endif

    currentContext_->fuseFinal();

#ifdef MCVR_ENABLE_OPENXR
    // Write GPU timestamp END in the last command buffer
    if (gpuTimestampPool_ != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(currentContext_->fuseCommandBuffer->vkCommandBuffer(),
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            gpuTimestampPool_, currentContextIndex_ * 2 + 1);
    }
#endif

    currentContext_->uploadCommandBuffer->end();
    currentContext_->worldCommandBuffer->end();
    currentContext_->overlayCommandBuffer->end();
    currentContext_->fuseCommandBuffer->end();

    std::vector<VkSemaphore> waitSemaphores = {currentContext_->imageAcquiredSemaphore->vkSemaphore()};
    std::vector<VkPipelineStageFlags> waitStageMasks = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    std::vector<VkSemaphore> signalSemaphores = {currentContext_->commandProcessedSemaphore->vkSemaphore()};
    std::vector<VkCommandBuffer> commandbuffers = {
        currentContext_->uploadCommandBuffer->vkCommandBuffer(),
        currentContext_->worldCommandBuffer->vkCommandBuffer(),
        currentContext_->overlayCommandBuffer->vkCommandBuffer(),
        currentContext_->fuseCommandBuffer->vkCommandBuffer(),
    };

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = waitSemaphores.size();
    vkSubmitInfo.pWaitSemaphores = waitSemaphores.data();
    vkSubmitInfo.pWaitDstStageMask = waitStageMasks.data();
    vkSubmitInfo.commandBufferCount = commandbuffers.size();
    vkSubmitInfo.pCommandBuffers = commandbuffers.data();
    vkSubmitInfo.signalSemaphoreCount = signalSemaphores.size();
    vkSubmitInfo.pSignalSemaphores = signalSemaphores.data();

    std::shared_ptr<vk::Fence> fence = currentContext_->commandFinishedFence;
    vkResetFences(device_->vkDevice(), 1, &fence->vkFence());
    vkQueueSubmit(device_->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence());

#ifdef MCVR_ENABLE_OPENXR
    // Release XR swapchain images AFTER GPU commands are submitted
    if (xrImagesAcquired) {
        for (uint32_t eye = 0; eye < 2; eye++) {
            xrContext_->releaseSwapchainImage(eye);
        }
    }
#endif
}

void Framework::present() {
    if (!running_) return;

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentContext_->commandProcessedSemaphore->vkSemaphore();

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_->vkSwapchain();
    presentInfo.pImageIndices = &currentContext_->frameIndex;

    VkResult result = vkQueuePresentKHR(device_->mainVkQueue(), &presentInfo);

#ifdef MCVR_ENABLE_OPENXR
    // End the XR frame (submit layers to compositor)
    if (xrContext_ && xrContext_->isSessionRunning()) {
        xrContext_->endFrame();
    }
#endif

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || vk::Window::framebufferResized ||
        Renderer::options.needRecreate || pipeline_->needRecreate) {
        recreate();
        return;
    } else if (result != VK_SUCCESS) {
        std::cerr << "failed to submit present command buffer" << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }
}

void Framework::recreate() {
    if (!running_) return;

    std::unique_lock<std::recursive_mutex> lck(Renderer::instance().framework()->recreateMtx());

    Renderer::options.needRecreate = false;
    vk::Window::framebufferResized = false;
    pipeline_->needRecreate = false;

    waitRenderQueueIdle();

    int width = 0, height = 0;
    GLFW_GetFramebufferSize(window_->window(), &width, &height);
    // In VR mode, don't block on minimized window — VR rendering doesn't need the desktop window.
    // Without this, minimizing/alt-tabbing causes an infinite loop that triggers the watchdog.
    int waitAttempts = 0;
    while (width == 0 || height == 0) {
        if (waitAttempts++ > 50) {  // ~5 seconds with typical event wait timing
            std::cerr << "Window framebuffer still 0x0 after 50 attempts, using fallback size" << std::endl;
            width = 1;
            height = 1;
            break;
        }
        GLFW_GetFramebufferSize(window_->window(), &width, &height);
        GLFW_WaitEvents();
    }

    currentContextIndex_ = 0;
    currentContext_ = nullptr;
    contexts_.clear();

    uploadCommandBuffers_.clear();
    overlayCommandBuffers_.clear();
    worldCommandBuffers_.clear();
    fuseCommandBuffers_.clear();
    commandFinishedFences_.clear();
    commandProcessedSemaphores_.clear();

    swapchain_->reconstruct();

    uint32_t size = swapchain_->imageCount();

    // create command buffer for each context
    for (int i = 0; i < size; i++) {
        uploadCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        overlayCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        worldCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        fuseCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
    }

    // create fence for each context
    for (int i = 0; i < size; i++) { commandFinishedFences_.push_back(vk::Fence::create(device_, true)); }

    // create semaphore for each context for command procssed
    for (int i = 0; i < size; i++) { commandProcessedSemaphores_.push_back(vk::Semaphore::create(device_)); }

    for (int i = 0; i < size; i++) { contexts_.push_back(FrameworkContext::create(shared_from_this(), i)); }

    pipeline_->recreate(shared_from_this());

    Renderer::instance().textures()->bindAllTextures();
}

void Framework::waitDeviceIdle() {
    vkDeviceWaitIdle(device_->vkDevice());
}

void Framework::waitRenderQueueIdle() {
    vkQueueWaitIdle(device_->mainVkQueue());
}

void Framework::waitBackendQueueIdle() {
    vkQueueWaitIdle(device_->secondaryQueue());
}

void Framework::close() {
    if (running_) { pipeline_->close(); }
#ifdef MCVR_ENABLE_OPENXR
    if (gpuTimestampPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_->vkDevice(), gpuTimestampPool_, nullptr);
        gpuTimestampPool_ = VK_NULL_HANDLE;
    }
    if (xrContext_) { xrContext_->shutdown(); xrContext_.reset(); }
#endif
    running_ = false;
}

#ifdef MCVR_ENABLE_OPENXR
bool Framework::startXRSession() {
    if (!running_ || xrContext_ == nullptr) return false;
    if (!Renderer::options.vrEnabled) return false;

    auto &vr = Renderer::instance().vrSystem();
    vr.worldOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    vr.worldPosition = glm::vec3(0.0f);

    bool ok = xrContext_->requestSessionStart();
    Renderer::options.needRecreate = true;
    if (pipeline_ != nullptr) pipeline_->needRecreate = true;
    return ok;
}

void Framework::stopXRSession() {
    if (!running_ || xrContext_ == nullptr) return;

    xrContext_->requestSessionStop();
    auto &vr = Renderer::instance().vrSystem();
    vr.enabled = false;
    vr.eyeCount = 1;
    vr.headPose = VRHeadPose{};
    vr.worldOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    vr.worldPosition = glm::vec3(0.0f);

    Renderer::options.needRecreate = true;
    if (pipeline_ != nullptr) pipeline_->needRecreate = true;
}

bool Framework::isXRSessionRunning() const {
    return xrContext_ != nullptr && xrContext_->isSessionRunning();
}
#endif

bool Framework::isRunning() {
    return running_;
}

void Framework::takeScreenshot(bool withUI, int width, int height, int channel, void *dstPointer) {
    if (indexHistory_.empty()) return;

    uint32_t targetIndex = indexHistory_.front();
    auto context = contexts_[targetIndex];
    std::shared_ptr<vk::Fence> fence = context->commandFinishedFence;
    VkResult result = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error for screenshot: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::shared_ptr<vk::HostVisibleBuffer> dstBuffer;
    std::shared_ptr<vk::DeviceLocalImage> srcImage;

    if (withUI) {
        auto pipelineContext = pipeline_->acquirePipelineContext(context);
        srcImage = pipelineContext->uiModuleContext->overlayDrawColorImage;

        uint32_t finalImageBufferSize =
            srcImage->width() * srcImage->height() * srcImage->layer() * vk::formatToByte(srcImage->vkFormat());
        if (finalImageBufferSize != width * height * channel) return;

        dstBuffer =
            vk::HostVisibleBuffer::create(vma_, device_, finalImageBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    } else {
        auto pipelineContext = pipeline_->acquirePipelineContext(context);
        srcImage = pipelineContext->worldPipelineContext->outputImage;

        uint32_t worldImageBufferSize =
            srcImage->width() * srcImage->height() * srcImage->layer() * vk::formatToByte(srcImage->vkFormat());
        if (worldImageBufferSize != width * height * channel) return;

        dstBuffer =
            vk::HostVisibleBuffer::create(vma_, device_, worldImageBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    VkImageLayout initialLayout = srcImage->imageLayout();
    auto mainQueueIndex = physicalDevice_->mainQueueIndex();

    std::shared_ptr<vk::CommandBuffer> oneTimeBuffer = vk::CommandBuffer::create(device_, mainCommandPool_);
    oneTimeBuffer->begin();

    oneTimeBuffer->barriersBufferImage(
        {},
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = initialLayout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = srcImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        }});
    VkBufferImageCopy bufferImageCopy{};
    bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferImageCopy.imageSubresource.mipLevel = 0;
    bufferImageCopy.imageSubresource.baseArrayLayer = 0;
    bufferImageCopy.imageSubresource.layerCount = 1;
    bufferImageCopy.imageExtent.width = srcImage->width();
    bufferImageCopy.imageExtent.height = srcImage->height();
    bufferImageCopy.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(oneTimeBuffer->vkCommandBuffer(), srcImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dstBuffer->vkBuffer(), 1, &bufferImageCopy);
    oneTimeBuffer
        ->barriersBufferImage(
            {}, {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .newLayout = initialLayout,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = srcImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }})
        ->end();

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = 0;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &oneTimeBuffer->vkCommandBuffer();
    vkSubmitInfo.signalSemaphoreCount = 0;
    std::shared_ptr<vk::Fence> oneTimeFence = vk::Fence::create(device_);

    vkQueueSubmit(device_->mainVkQueue(), 1, &vkSubmitInfo, oneTimeFence->vkFence());
    result = vkWaitForFences(device_->vkDevice(), 1, &oneTimeFence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error for screenshot: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::memcpy(dstPointer, dstBuffer->mappedPtr(), dstBuffer->size());
}

std::recursive_mutex &Framework::recreateMtx() {
    return recreateMtx_;
}

std::shared_ptr<vk::Instance> Framework::instance() {
    return instance_;
}

std::shared_ptr<vk::Window> Framework::window() {
    return window_;
}

std::shared_ptr<vk::PhysicalDevice> Framework::physicalDevice() {
    return physicalDevice_;
}

std::shared_ptr<vk::Device> Framework::device() {
    return device_;
}

std::shared_ptr<vk::VMA> Framework::vma() {
    return vma_;
}

std::shared_ptr<vk::Swapchain> Framework::swapchain() {
    return swapchain_;
}

std::shared_ptr<vk::CommandPool> Framework::mainCommandPool() {
    return mainCommandPool_;
}

std::shared_ptr<vk::CommandPool> Framework::asyncCommandPool() {
    return asyncCommandPool_;
}

std::shared_ptr<vk::CommandBuffer> Framework::worldAsyncCommandBuffer() {
    return worldAsyncCommandBuffer_;
}

std::vector<std::shared_ptr<vk::Semaphore>> &Framework::commandProcessedSemaphores() {
    return commandProcessedSemaphores_;
}

std::vector<std::shared_ptr<vk::Fence>> &Framework::commandFinishedFences() {
    return commandFinishedFences_;
}

std::vector<std::shared_ptr<FrameworkContext>> &Framework::contexts() {
    return contexts_;
}

std::shared_ptr<FrameworkContext> Framework::safeAcquireCurrentContext() {
    std::unique_lock<std::recursive_mutex> lck(recreateMtx_);
    // for continous window operation, currentContext_ will always be reset, busy waiting
    while (currentContext_ == nullptr) {
        // ensure currentContext_ is not nullptr after seapchain recreation
        acquireContext();
    }
    return currentContext_;
}

std::shared_ptr<Pipeline> Framework::pipeline() {
    return pipeline_;
}

GarbageCollector &Framework::gc() {
    return *gc_;
}

std::shared_ptr<vk::Semaphore> Framework::acquireSemaphore() {
    std::shared_ptr<vk::Semaphore> semaphore;
    if (recycledImageAcquiredSemaphores_.empty()) {
        semaphore = vk::Semaphore::create(device_);
    } else {
        semaphore = recycledImageAcquiredSemaphores_.front();
        recycledImageAcquiredSemaphores_.pop();
    }
    return semaphore;
}

void Framework::recycleSemaphore(std::shared_ptr<vk::Semaphore> semaphore) {
    recycledImageAcquiredSemaphores_.push(semaphore);
}

GarbageCollector::GarbageCollector(std::shared_ptr<Framework> framework) : framework_(framework) {
    collectors_.resize(framework->swapchain_->imageCount());
}

void GarbageCollector::clear() {
    index_ = (index_ + 1) % collectors_.size();

    auto framework = framework_.lock();
    collectors_[index_].clear();
}
