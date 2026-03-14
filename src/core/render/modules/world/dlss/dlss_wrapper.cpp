/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * MODIFICATIONS and INTEGRATION:
 *
 * Copyright (c) 2026 Radiance
 *
 * This file has been modified from its original version to be integrated into
 * Radiance Mod.
 *
 * Modifications include:
 * - Integration with Radiance Mod's vulkan rendering system.
 * - Refactoring of APIs, introducing RAII and supporting shared_ptr.
 *
 * These modifications are licensed under the GNU General Public License
 * as published by the Free Software Foundation; either version 3 of the License,
 * or (at your option) any later version.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"

// #include <nvh/nvprint.hpp>
// #include <nvp/nvpsystem.hpp>
// #include <nvvk/commands_vk.hpp>
// #include <nvvk/debug_util_vk.hpp>

#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <glm/ext.hpp>

#include <codecvt>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Static members  and globals
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Application ID assigned from NVIDIA, currently unused, but can't be 0
static const unsigned long long g_ApplicationID = 0xbaadf00dbaadcafe;

#define LOGE std::cout << "[DLSS Error] "
#define LOGW std::cout << "[DLSS Warning] "
#define LOGI std::cout << "[DLSS Info] "

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define NGX_RETURN_ON_FAIL(x)                                                                                          \
    {                                                                                                                  \
        NVSDK_NGX_Result result = checkNgxResult((x), __func__, __LINE__);                                             \
        if (NVSDK_NGX_FAILED(result)) return result;                                                                   \
    }

#define NGX_CHECK(x) checkNgxResult((x), __func__, __LINE__)


// Function taken from stack overflow https://stackoverflow.com/questions/4804298/how-to-convert-wstring-into-string
std::string ws2s(const std::wstring &wstr) {
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.to_bytes(wstr);
}

NVSDK_NGX_Result checkNgxResult(NVSDK_NGX_Result result, const char *func, int line) {
    if (NVSDK_NGX_FAILED(result)) {
        std::ostringstream stream;
        LOGE << ws2s(GetNGXResultAsString(result)) << " at " << func << ":" << line << std::endl;
    }

    return result;
}

void NVSDK_CONV NGX_AppLogCallback(const char *message,
                                   NVSDK_NGX_Logging_Level loggingLevel,
                                   NVSDK_NGX_Feature sourceComponent) {
    LOGE << message << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Class code
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NVSDK_NGX_Result NgxContext::init(const NgxInitInfo &initInfo) {
#ifdef DEBUG
    std::cout << "NgxContext::init" << std::endl;
#endif

    if (!initInfo.instance || !initInfo.physicalDevice || !initInfo.device) {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    assert(!device_ && !ngxParams_ && "Init already called");

    applicationPath_.assign(initInfo.applicationPath.begin(), initInfo.applicationPath.end());

    NVSDK_NGX_FeatureCommonInfo info = {};
    // info.LoggingInfo.LoggingCallback     = &NGX_AppLogCallback;
    info.LoggingInfo.MinimumLoggingLevel = initInfo.loggingLevel;

    // Init NGX API
    NGX_RETURN_ON_FAIL(NVSDK_NGX_VULKAN_Init(g_ApplicationID, applicationPath_.c_str(), initInfo.instance->vkInstance(),
                                             initInfo.physicalDevice->vkPhysicalDevice(), initInfo.device->vkDevice(),
                                             vkGetInstanceProcAddr, vkGetDeviceProcAddr, &info));

    device_ = initInfo.device;

    NVSDK_NGX_Result result = NGX_CHECK(NVSDK_NGX_VULKAN_GetCapabilityParameters(&ngxParams_));

    if (NVSDK_NGX_FAILED(result)) {
        deinit();
        return result;
    }

    return NVSDK_NGX_Result_Success;
}

NgxContext::~NgxContext() {
    assert(!ngxParams_ && !device_ && "Must call deinit");
}

void NgxContext::deinit() {
    if (ngxParams_) { NVSDK_NGX_VULKAN_DestroyParameters(ngxParams_); }
    if (device_) { NVSDK_NGX_VULKAN_Shutdown1(device_->vkDevice()); }

    ngxParams_ = nullptr;
    device_ = nullptr;
}

NVSDK_NGX_Result NgxContext::queryDlssRRAvailable() {
    assert(ngxParams_);

    int DLSS_Supported;
    int needsUpdatedDriver;
    unsigned minDriverVersionMajor;
    unsigned minDriverVersionMinor;

    // Check if DLSS_D (which is the DLSS_RR RayReconstruction Denoiser) is available.
    // Beware: Don't confuse this with DLSS, which is a different feature just providing upscaling
    NVSDK_NGX_Result resUpdatedDriver =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver));
    NVSDK_NGX_Result resVersionMajor = NGX_CHECK(
        ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &minDriverVersionMajor));
    NVSDK_NGX_Result resVersionMinor = NGX_CHECK(
        ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minDriverVersionMinor));

    if (NVSDK_NGX_SUCCEED(resUpdatedDriver)) {
        if (needsUpdatedDriver) {
            // NVIDIA DLSS cannot be loaded due to outdated driver.
            if (NVSDK_NGX_SUCCEED(resVersionMajor) && NVSDK_NGX_SUCCEED(resVersionMinor)) {
                // Min Driver Version required: minDriverVersionMajor.minDriverVersionMinor
                LOGW << "Minimum driver version required: " << minDriverVersionMajor << "." << minDriverVersionMinor
                     << std::endl;
                return NVSDK_NGX_Result_FAIL_OutOfDate;
            }
        }
    }

    NVSDK_NGX_Result resDlssSupported =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &DLSS_Supported));
    if (NVSDK_NGX_FAILED(resDlssSupported) || !DLSS_Supported) {
        LOGW << "not available on this hardware/platform" << std::endl;
        return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }
    resDlssSupported =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &DLSS_Supported));
    if (NVSDK_NGX_FAILED(resDlssSupported) || !DLSS_Supported) {
        LOGW << "denied for this application" << std::endl;
        return NVSDK_NGX_Result_FAIL_Denied;
    }

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::initDlssRR(const DlssRRInitInfo &initInfo,
                                        std::shared_ptr<vk::CommandPool> cmdPool,
                                        std::shared_ptr<DlssRR> dlssrr) {
    NGX_RETURN_ON_FAIL(dlssrr->init(device_, cmdPool, ngxParams_, initInfo));
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::getDlssRRRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions) {
    NVSDK_NGX_FeatureCommonInfo commonInfo = {};

    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    info.Identifier.v.ApplicationId = g_ApplicationID;
    // info.ApplicationDataPath        = m_applicationPath.c_str();
    info.FeatureInfo = &commonInfo;

    uint32_t numExtensions = 0;
    VkExtensionProperties *props;

    NGX_RETURN_ON_FAIL(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &numExtensions, &props));

    extensions.insert(extensions.end(), props, props + numExtensions);

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::getDlssRRRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                               std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                               std::vector<VkExtensionProperties> &extensions) {
    NVSDK_NGX_FeatureCommonInfo commonInfo = {};

    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    info.Identifier.v.ApplicationId = g_ApplicationID;
    info.ApplicationDataPath = L"";
    info.FeatureInfo = &commonInfo;

    uint32_t numExtensions = 0;
    VkExtensionProperties *props;

    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance->vkInstance(), physicalDevice->vkPhysicalDevice(), &info, &numExtensions, &props);
    if (NVSDK_NGX_FAILED(result)) {
        LOGW << ws2s(GetNGXResultAsString(result)) << " while querying DLSS device extensions; skipping." << std::endl;
        return result;
    }

    extensions.insert(extensions.end(), props, props + numExtensions);

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::querySupportedDlssInputSizes(const QuerySizeInfo &queryInfo, SupportedSizes &sizes) {
    assert(ngxParams_);
    assert(queryInfo.quality != NVSDK_NGX_PerfQuality_Value_UltraQuality); // Unsupported currently for DLSS_RR

    float sharpness; // unused in DLSS_RR?

    NGX_RETURN_ON_FAIL(NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        ngxParams_, queryInfo.outputSize.width, queryInfo.outputSize.height, queryInfo.quality,
        &sizes.optimalSize.width, &sizes.optimalSize.height, &sizes.maxSize.width, &sizes.maxSize.height,
        &sizes.minSize.width, &sizes.minSize.height, &sharpness));

    // NGX_DLSSD_GET_OPTIMAL_SETTINGS can return successfully yet still return garbage values.
    assert(sizes.optimalSize.width > 0 && sizes.optimalSize.height > 0 && sizes.maxSize.width > 0 &&
           sizes.maxSize.height > 0 && sizes.minSize.width > 0 && sizes.minSize.height > 0);

    return NVSDK_NGX_Result_Success;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NVSDK_NGX_Result DlssRR::init(std::shared_ptr<vk::Device> device,
                              std::shared_ptr<vk::CommandPool> cmdPool,
                              NVSDK_NGX_Parameter *ngxParams,
                              const NgxContext::DlssRRInitInfo &info) {
#ifdef DEBUG
    std::cout << "DlssRR::init" << std::endl;
#endif

    assert(!m_dlssdHandle && "Cannot call init twice");

    m_device = device;
    m_ngxParams = ngxParams;

    m_outputSize = info.outputSize;
    m_inputSize = info.inputSize;

    m_resources.fill({.Resource = {.ImageViewInfo = {}}});

    NVSDK_NGX_DLSSD_Create_Params dlssdParams{};

    dlssdParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    // We expose only packed normal/roughness here because of providing float16 normals
    dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed; // we pack roughness into the normal's w channel
    dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;        // we're providing linear depth

    dlssdParams.InWidth = m_inputSize.width;
    dlssdParams.InHeight = m_inputSize.height;
    dlssdParams.InTargetWidth = m_outputSize.width;
    dlssdParams.InTargetHeight = m_outputSize.height;

    // Though marked as 'optional', these are absolutely needed
    dlssdParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    dlssdParams.InPerfQualityValue = info.quality;

    const uint32_t creationNodeMask = 0x1;
    const uint32_t visibilityNodeMask = 0x1;

    // This allows you to switch "presets", i.e. different models for the denoiser
    m_ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, info.preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, info.preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, info.preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, info.preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, info.preset);

    {
        std::shared_ptr<vk::CommandBuffer> cmdBuffer = vk::CommandBuffer::create(device, cmdPool);
        cmdBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        NGX_RETURN_ON_FAIL(NGX_VULKAN_CREATE_DLSSD_EXT1(device->vkDevice(), cmdBuffer->vkCommandBuffer(),
                                                        creationNodeMask, visibilityNodeMask, &m_dlssdHandle, ngxParams,
                                                        &dlssdParams));
        cmdBuffer->end()->submitMainQueueIndividual(device);
    }

    return NVSDK_NGX_Result_Success;
}

void DlssRR::deinit() {
#ifdef DEBUG
    std::cout << "DlssRR::deinit" << std::endl;
#endif

    if (m_dlssdHandle) { NVSDK_NGX_VULKAN_ReleaseFeature(m_dlssdHandle); }
    m_dlssdHandle = nullptr;
    m_device = VK_NULL_HANDLE;
}

DlssRR::~DlssRR() {
#ifdef DEBUG
    std::cout << "DlssRR::~DlssRR" << std::endl;
#endif
    assert(!m_dlssdHandle && "Must call deinit");
}

void DlssRR::setResource(DlssResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image, uint32_t viewIndex) {
    assert(m_dlssdHandle);

    VkExtent2D size = resourceId == RESOURCE_COLOR_OUT ? m_outputSize : m_inputSize;

    VkImageSubresourceRange range = vk::wholeColorSubresourceRange;
    if (viewIndex > 0) {
        range = vk::colorSubresourceRangeForLayer(viewIndex - 1);
    }

    NVSDK_NGX_Resource_VK resource = NVSDK_NGX_Create_ImageView_Resource_VK(
        image->vkImageView(viewIndex), image->vkImage(), range, image->vkFormat(), size.width,
        size.height, resourceId == RESOURCE_COLOR_OUT /*readWrite*/);

    m_resources[resourceId] = resource;
}

void DlssRR::resetResource(DlssResource resourceId) {
    m_resources[resourceId] = {};
}

NVSDK_NGX_Result DlssRR::denoise(std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                                 glm::uvec2 renderSize,
                                 glm::vec2 jitter,
                                 const glm::mat4 &modelView,
                                 const glm::mat4 &projection,
                                 bool reset) {
    assert(m_dlssdHandle);

    auto getResource = [this](DlssResource res) -> NVSDK_NGX_Resource_VK * {
        return m_resources[res].Resource.ImageViewInfo.ImageView ? &m_resources[res] : nullptr;
    };

    NVSDK_NGX_VK_DLSSD_Eval_Params evalParams = {};

    evalParams.pInColor = getResource(RESOURCE_COLOR_IN);
    evalParams.pInOutput = getResource(RESOURCE_COLOR_OUT);
    evalParams.pInDiffuseAlbedo = getResource(RESOURCE_DIFFUSE_ALBEDO);
    evalParams.pInSpecularAlbedo = getResource(RESOURCE_SPECULAR_ALBEDO);
    evalParams.pInSpecularHitDistance = getResource(RESOURCE_SPECULAR_HITDISTANCE);
    evalParams.pInNormals = getResource(RESOURCE_NORMALROUGHNESS);
    evalParams.pInDepth = getResource(RESOURCE_LINEARDEPTH);
    evalParams.pInMotionVectors = getResource(RESOURCE_MOTIONVECTOR);
    // Is this needed with NVSDK_NGX_DLSS_Roughness_Mode_Packed?
    evalParams.pInRoughness = getResource(RESOURCE_NORMALROUGHNESS);

    evalParams.InJitterOffsetX = -jitter.x;
    evalParams.InJitterOffsetY = -jitter.y;
    evalParams.InMVScaleX = 1.0f;
    evalParams.InMVScaleY = 1.0f;

    evalParams.InRenderSubrectDimensions.Width = renderSize.x;
    evalParams.InRenderSubrectDimensions.Height = renderSize.y;

    // DLSS_RR expects row_major + 'left_multiply' matrices here.
    // Ours (glm) are column-major + right_multiply. To convert we'd have to
    // 1) transpose for row_major
    // 2) transpose again for left_multiply
    // Mdlss = (M^T)^T = M  ; thus, supply our original matrices and it magically works
    evalParams.pInWorldToViewMatrix = const_cast<float *>(glm::value_ptr(modelView));
    evalParams.pInViewToClipMatrix = const_cast<float *>(glm::value_ptr(projection));

    evalParams.InReset = reset;

    NGX_RETURN_ON_FAIL(
        NGX_VULKAN_EVALUATE_DLSSD_EXT(cmdBuffer->vkCommandBuffer(), m_dlssdHandle, m_ngxParams, &evalParams));

    return NVSDK_NGX_Result_Success;
}

std::string getNGXResultString(NVSDK_NGX_Result result) {
    return ws2s(GetNGXResultAsString(result));
}
