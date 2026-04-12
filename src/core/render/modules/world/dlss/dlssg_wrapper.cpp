/*
 * Copyright (c) 2026 Radiance
 *
 * This file is licensed under the GNU General Public License
 * as published by the Free Software Foundation; either version 3 of the License,
 * or (at your option) any later version.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/render/modules/world/dlss/dlssg_wrapper.hpp"

#include <nvsdk_ngx_helpers_dlssg_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <glm/ext.hpp>
#include <iostream>

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"

#define LOGE_FG std::cout << "[DLSS-FG Error] "
#define LOGI_FG std::cout << "[DLSS-FG Info] "

#define NGX_RETURN_ON_FAIL_FG(x)                                                                                       \
    {                                                                                                                  \
        NVSDK_NGX_Result result = (x);                                                                                 \
        if (NVSDK_NGX_FAILED(result)) {                                                                                \
            LOGE_FG << "NGX call failed at " << __func__ << ":" << __LINE__ << std::endl;                              \
            return result;                                                                                             \
        }                                                                                                              \
    }

NVSDK_NGX_Result DlssFG::init(std::shared_ptr<vk::Device> device,
                               std::shared_ptr<vk::CommandPool> cmdPool,
                               NVSDK_NGX_Parameter *ngxParams,
                               uint32_t width,
                               uint32_t height,
                               VkFormat backbufferFormat) {
    LOGI_FG << "Initializing DLSS Frame Generation (" << width << "x" << height << ")" << std::endl;

    assert(!m_dlssgHandle && "Cannot call init twice");

    m_device = device;
    m_ngxParams = ngxParams;
    m_size = {width, height};

    m_resources.fill({.Resource = {.ImageViewInfo = {}}});

    NVSDK_NGX_DLSSG_Create_Params createParams{};
    createParams.Width = width;
    createParams.Height = height;
    createParams.NativeBackbufferFormat = static_cast<unsigned int>(backbufferFormat);
    createParams.RenderWidth = width;
    createParams.RenderHeight = height;
    createParams.DynamicResolutionScaling = false;

    const uint32_t creationNodeMask = 0x1;
    const uint32_t visibilityNodeMask = 0x1;

    {
        std::shared_ptr<vk::CommandBuffer> cmdBuffer = vk::CommandBuffer::create(device, cmdPool);
        cmdBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        NGX_RETURN_ON_FAIL_FG(NGX_VK_CREATE_DLSSG(cmdBuffer->vkCommandBuffer(), creationNodeMask, visibilityNodeMask,
                                                   &m_dlssgHandle, ngxParams, &createParams));
        cmdBuffer->end()->submitMainQueueIndividual(device);
    }

    LOGI_FG << "DLSS Frame Generation initialized successfully" << std::endl;
    return NVSDK_NGX_Result_Success;
}

void DlssFG::deinit() {
    LOGI_FG << "Deinitializing DLSS Frame Generation" << std::endl;
    if (m_dlssgHandle) { NVSDK_NGX_VULKAN_ReleaseFeature(m_dlssgHandle); }
    m_dlssgHandle = nullptr;
    m_device = nullptr;
}

DlssFG::~DlssFG() {
    assert(!m_dlssgHandle && "Must call deinit");
}

void DlssFG::setResource(FGResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image, bool readWrite) {
    assert(m_dlssgHandle);

    NVSDK_NGX_Resource_VK resource = NVSDK_NGX_Create_ImageView_Resource_VK(
        image->vkImageView(), image->vkImage(), vk::wholeColorSubresourceRange, image->vkFormat(),
        image->width(), image->height(), readWrite);

    m_resources[resourceId] = resource;
}

void DlssFG::setResource(FGResource resourceId, std::shared_ptr<vk::SwapchainImage> image, bool readWrite) {
    assert(m_dlssgHandle);

    NVSDK_NGX_Resource_VK resource = NVSDK_NGX_Create_ImageView_Resource_VK(
        image->vkImageView(), image->vkImage(), vk::wholeColorSubresourceRange, image->vkFormat(),
        image->width(), image->height(), readWrite);

    m_resources[resourceId] = resource;
}

void DlssFG::resetResource(FGResource resourceId) {
    m_resources[resourceId] = {};
}

NVSDK_NGX_Result DlssFG::evaluate(std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                                   const glm::mat4 &viewMatrix,
                                   const glm::mat4 &projMatrix,
                                   const glm::mat4 &prevViewMatrix,
                                   const glm::mat4 &prevProjMatrix,
                                   glm::vec2 jitter,
                                   glm::vec3 cameraPos,
                                   float cameraNear,
                                   float cameraFar,
                                   float cameraFOV,
                                   float cameraAspectRatio,
                                   uint32_t multiFrameCount,
                                   uint32_t multiFrameIndex,
                                   bool reset) {
    assert(m_dlssgHandle);

    auto getResource = [this](FGResource res) -> NVSDK_NGX_Resource_VK * {
        return m_resources[res].Resource.ImageViewInfo.ImageView ? &m_resources[res] : nullptr;
    };

    NVSDK_NGX_VK_DLSSG_Eval_Params evalParams{};
    evalParams.pBackbuffer = getResource(RESOURCE_BACKBUFFER);
    evalParams.pDepth = getResource(RESOURCE_DEPTH);
    evalParams.pMVecs = getResource(RESOURCE_MVECS);
    evalParams.pHudless = nullptr;
    evalParams.pUI = nullptr;
    evalParams.pNoPostProcessingColor = nullptr;
    evalParams.pBidirectionalDistortionField = nullptr;
    evalParams.pOutputInterpFrame = getResource(RESOURCE_OUTPUT_INTERP);
    evalParams.pOutputRealFrame = nullptr;
    evalParams.pOutputDisableInterpolation = nullptr;

    // Compute camera matrices for FG
    // viewToClip = projection matrix
    // clipToView = inverse projection
    // clipToPrevClip = prevViewProj * inv(currentViewProj)
    glm::mat4 viewToClip = projMatrix;
    glm::mat4 clipToView = glm::inverse(projMatrix);
    glm::mat4 currentViewProj = projMatrix * viewMatrix;
    glm::mat4 prevViewProj = prevProjMatrix * prevViewMatrix;
    glm::mat4 clipToPrevClip = prevViewProj * glm::inverse(currentViewProj);
    glm::mat4 prevClipToClip = currentViewProj * glm::inverse(prevViewProj);

    NVSDK_NGX_DLSSG_Opt_Eval_Params optParams{};
    optParams.multiFrameCount = multiFrameCount;
    optParams.multiFrameIndex = multiFrameIndex;

    // GLM is column-major, DLSS-G expects row-major with left-multiply.
    // Same trick as DLSS-RR: (M^T)^T = M, so supply original matrices directly.
    std::memcpy(optParams.cameraViewToClip, glm::value_ptr(viewToClip), sizeof(float) * 16);
    std::memcpy(optParams.clipToCameraView, glm::value_ptr(clipToView), sizeof(float) * 16);
    std::memset(optParams.clipToLensClip, 0, sizeof(float) * 16);
    // Identity for clipToLensClip (no lens distortion)
    optParams.clipToLensClip[0][0] = 1.0f;
    optParams.clipToLensClip[1][1] = 1.0f;
    optParams.clipToLensClip[2][2] = 1.0f;
    optParams.clipToLensClip[3][3] = 1.0f;
    std::memcpy(optParams.clipToPrevClip, glm::value_ptr(clipToPrevClip), sizeof(float) * 16);
    std::memcpy(optParams.prevClipToClip, glm::value_ptr(prevClipToClip), sizeof(float) * 16);

    optParams.jitterOffset[0] = -jitter.x;
    optParams.jitterOffset[1] = -jitter.y;
    optParams.mvecScale[0] = 1.0f;
    optParams.mvecScale[1] = 1.0f;
    optParams.cameraPinholeOffset[0] = 0.0f;
    optParams.cameraPinholeOffset[1] = 0.0f;

    optParams.cameraPos[0] = cameraPos.x;
    optParams.cameraPos[1] = cameraPos.y;
    optParams.cameraPos[2] = cameraPos.z;

    // Extract camera vectors from view matrix
    // View matrix rows (after transpose for column-major) give right, up, forward
    glm::mat4 viewInv = glm::inverse(viewMatrix);
    optParams.cameraRight[0] = viewInv[0][0];
    optParams.cameraRight[1] = viewInv[0][1];
    optParams.cameraRight[2] = viewInv[0][2];
    optParams.cameraUp[0] = viewInv[1][0];
    optParams.cameraUp[1] = viewInv[1][1];
    optParams.cameraUp[2] = viewInv[1][2];
    optParams.cameraFwd[0] = -viewInv[2][0];
    optParams.cameraFwd[1] = -viewInv[2][1];
    optParams.cameraFwd[2] = -viewInv[2][2];

    optParams.cameraNear = cameraNear;
    optParams.cameraFar = cameraFar;
    optParams.cameraFOV = cameraFOV;
    optParams.cameraAspectRatio = cameraAspectRatio;

    optParams.colorBuffersHDR = false; // post-tonemapping backbuffer is LDR/SDR
    optParams.depthInverted = false;
    optParams.cameraMotionIncluded = true;
    optParams.reset = reset;
    optParams.automodeOverrideReset = false;
    optParams.notRenderingGameFrames = false;
    optParams.orthoProjection = false;
    optParams.motionVectorsInvalidValue = 0.0f;
    optParams.motionVectorsDilated = false;
    optParams.menuDetectionEnabled = false;

    NGX_RETURN_ON_FAIL_FG(
        NGX_VK_EVALUATE_DLSSG(cmdBuffer->vkCommandBuffer(), m_dlssgHandle, m_ngxParams, &evalParams, &optParams));

    return NVSDK_NGX_Result_Success;
}
