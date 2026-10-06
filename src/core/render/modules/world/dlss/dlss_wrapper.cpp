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

#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_defs_dlssg.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>
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
#include "core/vulkan/result.hpp"
#include "core/vulkan/sync.hpp"

static const unsigned long long g_ApplicationID = 0xbaadf00dbaadcafe;

#define LOGE std::cerr << "[DLSS Error] "
#define LOGW std::cerr << "[DLSS Warning] "

#define NGX_RETURN_ON_FAIL(x)                                                                                          \
    {                                                                                                                  \
        NVSDK_NGX_Result result = checkNgxResult((x), __func__, __LINE__);                                             \
        if (NVSDK_NGX_FAILED(result)) return result;                                                                   \
    }

#define NGX_CHECK(x) checkNgxResult((x), __func__, __LINE__)

std::shared_ptr<NgxContext> NgxContext::sharedContext_ = nullptr;

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

NVSDK_NGX_Result NgxContext::init(const NgxInitInfo &initInfo) {
    if (!initInfo.instance || !initInfo.physicalDevice || !initInfo.device) {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    assert(!device_ && !ngxParams_ && "Init already called");

    applicationPath_.assign(initInfo.applicationPath.begin(), initInfo.applicationPath.end());

    NVSDK_NGX_FeatureCommonInfo info = {};
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;

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

std::shared_ptr<NgxContext> NgxContext::acquireShared(const NgxInitInfo &initInfo) {
    if (sharedContext_ != nullptr && sharedContext_->parameters() != nullptr) { return sharedContext_; }

    auto context = NgxContext::create();
    NVSDK_NGX_Result result = context->init(initInfo);
    if (NVSDK_NGX_FAILED(result)) {
        context->deinit();
        return nullptr;
    }

    sharedContext_ = context;
    return sharedContext_;
}

void NgxContext::shutdownShared() {
    if (sharedContext_ != nullptr) {
        sharedContext_->deinit();
        sharedContext_ = nullptr;
    }
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

NVSDK_NGX_Parameter *NgxContext::parameters() {
    return ngxParams_;
}

NVSDK_NGX_Result NgxContext::queryDlssRRAvailable() {
    assert(ngxParams_);

    int DLSS_Supported;
    int needsUpdatedDriver;
    unsigned minDriverVersionMajor;
    unsigned minDriverVersionMinor;

    NVSDK_NGX_Result resUpdatedDriver =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver));
    NVSDK_NGX_Result resVersionMajor = NGX_CHECK(
        ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &minDriverVersionMajor));
    NVSDK_NGX_Result resVersionMinor = NGX_CHECK(
        ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minDriverVersionMinor));

    if (NVSDK_NGX_SUCCEED(resUpdatedDriver)) {
        if (needsUpdatedDriver) {
            if (NVSDK_NGX_SUCCEED(resVersionMajor) && NVSDK_NGX_SUCCEED(resVersionMinor)) {
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

NVSDK_NGX_Result NgxContext::queryDlssSRAvailable() {
    assert(ngxParams_);

    int featureSupported = 0;
    int needsUpdatedDriver = 0;
    unsigned minDriverVersionMajor = 0;
    unsigned minDriverVersionMinor = 0;

    NVSDK_NGX_Result resUpdatedDriver =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver));
    NVSDK_NGX_Result resVersionMajor =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minDriverVersionMajor));
    NVSDK_NGX_Result resVersionMinor =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minDriverVersionMinor));

    if (NVSDK_NGX_SUCCEED(resUpdatedDriver) && needsUpdatedDriver) {
        if (NVSDK_NGX_SUCCEED(resVersionMajor) && NVSDK_NGX_SUCCEED(resVersionMinor)) {
            LOGW << "DLSS super resolution minimum driver version required: " << minDriverVersionMajor << "."
                 << minDriverVersionMinor << std::endl;
        }
        return NVSDK_NGX_Result_FAIL_OutOfDate;
    }

    NVSDK_NGX_Result resSupported =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &featureSupported));
    if (NVSDK_NGX_FAILED(resSupported) || !featureSupported) {
        LOGW << "DLSS super resolution is not available on this hardware/platform" << std::endl;
        return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }

    int featureInitResult = 0;
    NVSDK_NGX_Result resInit =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &featureInitResult));
    if (NVSDK_NGX_FAILED(resInit)) { return resInit; }
    if (featureInitResult != NVSDK_NGX_Result_Success) {
        const auto result = static_cast<NVSDK_NGX_Result>(featureInitResult);
        LOGW << "DLSS super resolution initialization failed: " << getNGXResultString(result) << std::endl;
        return NVSDK_NGX_FAILED(result) ? result : NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::queryDlssFrameGenerationAvailable() {
    assert(ngxParams_);

    int featureSupported = 0;
    int needsUpdatedDriver = 0;
    unsigned minDriverVersionMajor = 0;
    unsigned minDriverVersionMinor = 0;

    NVSDK_NGX_Result resUpdatedDriver =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needsUpdatedDriver));
    NVSDK_NGX_Result resVersionMajor =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, &minDriverVersionMajor));
    NVSDK_NGX_Result resVersionMinor =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, &minDriverVersionMinor));

    if (NVSDK_NGX_SUCCEED(resUpdatedDriver) && needsUpdatedDriver) {
        if (NVSDK_NGX_SUCCEED(resVersionMajor) && NVSDK_NGX_SUCCEED(resVersionMinor)) {
            LOGW << "DLSS frame generation minimum driver version required: " << minDriverVersionMajor << "."
                 << minDriverVersionMinor << std::endl;
        }
        return NVSDK_NGX_Result_FAIL_OutOfDate;
    }

    NVSDK_NGX_Result resSupported =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &featureSupported));
    if (NVSDK_NGX_FAILED(resSupported) || !featureSupported) {
        LOGW << "DLSS frame generation is not available on this hardware/platform" << std::endl;
        return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }

    int initResult = 0;
    NVSDK_NGX_Result resInit =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &initResult));
    if (NVSDK_NGX_FAILED(resInit) || !initResult) {
        LOGW << "DLSS frame generation denied for this application" << std::endl;
        return NVSDK_NGX_Result_FAIL_Denied;
    }

    return NVSDK_NGX_Result_Success;
}

uint32_t NgxContext::queryDlssFrameGenerationMaxFrames() {
    assert(ngxParams_);
    unsigned int maxGeneratedFrames = 1;
    NVSDK_NGX_Result result =
        NGX_CHECK(ngxParams_->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &maxGeneratedFrames));
    if (NVSDK_NGX_FAILED(result) || maxGeneratedFrames == 0) { return 1; }
    return maxGeneratedFrames;
}

NVSDK_NGX_Result NgxContext::initDlssRR(const DlssRRInitInfo &initInfo,
                                        std::shared_ptr<vk::CommandPool> cmdPool,
                                        std::shared_ptr<DlssRR> dlssrr) {
    NGX_RETURN_ON_FAIL(dlssrr->init(device_, cmdPool, ngxParams_, initInfo));
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::initDlssSR(const DlssSRInitInfo &initInfo,
                                        std::shared_ptr<vk::CommandPool> cmdPool,
                                        std::shared_ptr<DlssSR> dlsssr) {
    NGX_RETURN_ON_FAIL(dlsssr->init(device_, cmdPool, ngxParams_, initInfo));
    return NVSDK_NGX_Result_Success;
}

class NgxFeatureExtensionQuery {
  public:
    static NVSDK_NGX_Result getRequiredInstanceExtensionsForFeature(NVSDK_NGX_Feature feature,
                                                                    std::vector<VkExtensionProperties> &extensions) {
        NVSDK_NGX_FeatureCommonInfo commonInfo = {};

        NVSDK_NGX_FeatureDiscoveryInfo info{};
        info.SDKVersion = NVSDK_NGX_Version_API;
        info.FeatureID = feature;
        info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
        info.Identifier.v.ApplicationId = g_ApplicationID;
        info.FeatureInfo = &commonInfo;

        uint32_t numExtensions = 0;
        VkExtensionProperties *props;

        NGX_RETURN_ON_FAIL(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &numExtensions, &props));

        extensions.insert(extensions.end(), props, props + numExtensions);

        return NVSDK_NGX_Result_Success;
    }

    static NVSDK_NGX_Result getRequiredDeviceExtensionsForFeature(std::shared_ptr<vk::Instance> instance,
                                                                  std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                                  NVSDK_NGX_Feature feature,
                                                                  const char *featureName,
                                                                  std::vector<VkExtensionProperties> &extensions) {
        NVSDK_NGX_FeatureCommonInfo commonInfo = {};

        NVSDK_NGX_FeatureDiscoveryInfo info{};
        info.SDKVersion = NVSDK_NGX_Version_API;
        info.FeatureID = feature;
        info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
        info.Identifier.v.ApplicationId = g_ApplicationID;
        info.ApplicationDataPath = L"";
        info.FeatureInfo = &commonInfo;

        uint32_t numExtensions = 0;
        VkExtensionProperties *props;

        NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            instance->vkInstance(), physicalDevice->vkPhysicalDevice(), &info, &numExtensions, &props);
        if (NVSDK_NGX_FAILED(result)) {
            LOGW << ws2s(GetNGXResultAsString(result)) << " while querying " << featureName
                 << " device extensions; skipping." << std::endl;
            return result;
        }

        extensions.insert(extensions.end(), props, props + numExtensions);

        return NVSDK_NGX_Result_Success;
    }
};

NVSDK_NGX_Result NgxContext::getDlssRRRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions) {
    return NgxFeatureExtensionQuery::getRequiredInstanceExtensionsForFeature(NVSDK_NGX_Feature_RayReconstruction,
                                                                             extensions);
}

NVSDK_NGX_Result NgxContext::getDlssSRRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions) {
    return NgxFeatureExtensionQuery::getRequiredInstanceExtensionsForFeature(NVSDK_NGX_Feature_SuperSampling,
                                                                             extensions);
}

NVSDK_NGX_Result
NgxContext::getDlssFrameGenerationRequiredInstanceExtensions(std::vector<VkExtensionProperties> &extensions) {
    return NgxFeatureExtensionQuery::getRequiredInstanceExtensionsForFeature(NVSDK_NGX_Feature_FrameGeneration,
                                                                             extensions);
}

NVSDK_NGX_Result NgxContext::getDlssRRRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                               std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                               std::vector<VkExtensionProperties> &extensions) {
    return NgxFeatureExtensionQuery::getRequiredDeviceExtensionsForFeature(
        instance, physicalDevice, NVSDK_NGX_Feature_RayReconstruction, "DLSS ray reconstruction", extensions);
}

NVSDK_NGX_Result NgxContext::getDlssSRRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                               std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                               std::vector<VkExtensionProperties> &extensions) {
    return NgxFeatureExtensionQuery::getRequiredDeviceExtensionsForFeature(
        instance, physicalDevice, NVSDK_NGX_Feature_SuperSampling, "DLSS super resolution", extensions);
}

NVSDK_NGX_Result
NgxContext::getDlssFrameGenerationRequiredDeviceExtensions(std::shared_ptr<vk::Instance> instance,
                                                           std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                                                           std::vector<VkExtensionProperties> &extensions) {
    return NgxFeatureExtensionQuery::getRequiredDeviceExtensionsForFeature(
        instance, physicalDevice, NVSDK_NGX_Feature_FrameGeneration, "DLSS frame generation", extensions);
}

NVSDK_NGX_Result NgxContext::querySupportedDlssRRInputSizes(const QuerySizeInfo &queryInfo, SupportedSizes &sizes) {
    assert(ngxParams_);
    assert(queryInfo.quality != NVSDK_NGX_PerfQuality_Value_UltraQuality);

    float sharpness;

    NGX_RETURN_ON_FAIL(NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        ngxParams_, queryInfo.outputSize.width, queryInfo.outputSize.height, queryInfo.quality,
        &sizes.optimalSize.width, &sizes.optimalSize.height, &sizes.maxSize.width, &sizes.maxSize.height,
        &sizes.minSize.width, &sizes.minSize.height, &sharpness));

    assert(sizes.optimalSize.width > 0 && sizes.optimalSize.height > 0 && sizes.maxSize.width > 0 &&
           sizes.maxSize.height > 0 && sizes.minSize.width > 0 && sizes.minSize.height > 0);

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::querySupportedDlssSRInputSizes(const QuerySizeInfo &queryInfo, SupportedSizes &sizes) {
    assert(ngxParams_);

    float sharpness = 0.0f;
    NGX_RETURN_ON_FAIL(NGX_DLSS_GET_OPTIMAL_SETTINGS(
        ngxParams_, queryInfo.outputSize.width, queryInfo.outputSize.height, queryInfo.quality,
        &sizes.optimalSize.width, &sizes.optimalSize.height, &sizes.maxSize.width, &sizes.maxSize.height,
        &sizes.minSize.width, &sizes.minSize.height, &sharpness));

    if (sizes.optimalSize.width == 0 || sizes.optimalSize.height == 0 || sizes.maxSize.width == 0 ||
        sizes.maxSize.height == 0 || sizes.minSize.width == 0 || sizes.minSize.height == 0) {
        LOGW << "DLSS super resolution mode has no valid input resolution for " << queryInfo.outputSize.width << "x"
             << queryInfo.outputSize.height << std::endl;
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssRR::init(std::shared_ptr<vk::Device> device,
                              std::shared_ptr<vk::CommandPool> cmdPool,
                              NVSDK_NGX_Parameter *ngxParams,
                              const NgxContext::DlssRRInitInfo &info) {
    assert(!m_dlssdHandle && "Cannot call init twice");

    m_device = device;
    m_ngxParams = ngxParams;

    m_outputSize = info.outputSize;
    m_inputSize = info.inputSize;

    m_resources.fill({.Resource = {.ImageViewInfo = {}}});

    NVSDK_NGX_DLSSD_Create_Params dlssdParams{};

    dlssdParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;

    dlssdParams.InWidth = m_inputSize.width;
    dlssdParams.InHeight = m_inputSize.height;
    dlssdParams.InTargetWidth = m_outputSize.width;
    dlssdParams.InTargetHeight = m_outputSize.height;

    dlssdParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    dlssdParams.InPerfQualityValue = info.quality;

    const uint32_t creationNodeMask = 0x1;
    const uint32_t visibilityNodeMask = 0x1;

    m_ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA, info.preset);
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
        auto initializationFence = vk::Fence::create(device);
        cmdBuffer->end()->submitMainQueueIndividual(device, initializationFence);
        vk::waitForFence(device->vkDevice(), initializationFence->vkFence(), "vkWaitForFences(DLSS RR initialization)");
    }

    return NVSDK_NGX_Result_Success;
}

void DlssRR::deinit() {
    if (m_dlssdHandle) { NVSDK_NGX_VULKAN_ReleaseFeature(m_dlssdHandle); }
    m_dlssdHandle = nullptr;
    m_device = VK_NULL_HANDLE;
}

DlssRR::~DlssRR() {
    assert(!m_dlssdHandle && "Must call deinit");
}

void DlssRR::setResource(DlssResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image) {
    assert(m_dlssdHandle);

    VkExtent2D size = resourceId == RESOURCE_COLOR_OUT ? m_outputSize : m_inputSize;

    NVSDK_NGX_Resource_VK resource = NVSDK_NGX_Create_ImageView_Resource_VK(
        image->vkImageView(), image->vkImage(), vk::wholeColorSubresourceRange, image->vkFormat(), size.width,
        size.height, resourceId == RESOURCE_COLOR_OUT);

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
    evalParams.pInRoughness = getResource(RESOURCE_NORMALROUGHNESS);

    evalParams.InJitterOffsetX = -jitter.x;
    evalParams.InJitterOffsetY = -jitter.y;
    evalParams.InMVScaleX = 1.0f;
    evalParams.InMVScaleY = 1.0f;

    evalParams.InRenderSubrectDimensions.Width = renderSize.x;
    evalParams.InRenderSubrectDimensions.Height = renderSize.y;

    evalParams.pInWorldToViewMatrix = const_cast<float *>(glm::value_ptr(modelView));
    evalParams.pInViewToClipMatrix = const_cast<float *>(glm::value_ptr(projection));

    evalParams.InReset = reset;

    NGX_RETURN_ON_FAIL(
        NGX_VULKAN_EVALUATE_DLSSD_EXT(cmdBuffer->vkCommandBuffer(), m_dlssdHandle, m_ngxParams, &evalParams));

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssSR::init(std::shared_ptr<vk::Device> device,
                              std::shared_ptr<vk::CommandPool> cmdPool,
                              NVSDK_NGX_Parameter *ngxParams,
                              const NgxContext::DlssSRInitInfo &info) {
    if (!device || !cmdPool || !ngxParams || info.inputSize.width == 0 || info.inputSize.height == 0 ||
        info.outputSize.width == 0 || info.outputSize.height == 0) {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    assert(!m_dlssHandle && "Cannot call init twice");

    m_device = device;
    m_ngxParams = ngxParams;
    m_inputSize = info.inputSize;
    m_outputSize = info.outputSize;
    m_resources.fill({.Resource = {.ImageViewInfo = {}}});

    const unsigned int preset = static_cast<unsigned int>(info.preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, preset);
    m_ngxParams->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, preset);

    NVSDK_NGX_DLSS_Create_Params dlssParams{};
    dlssParams.Feature.InWidth = m_inputSize.width;
    dlssParams.Feature.InHeight = m_inputSize.height;
    dlssParams.Feature.InTargetWidth = m_outputSize.width;
    dlssParams.Feature.InTargetHeight = m_outputSize.height;
    dlssParams.Feature.InPerfQualityValue = info.quality;
    dlssParams.InFeatureCreateFlags = info.featureCreateFlags;
    dlssParams.InEnableOutputSubrects = false;

    const uint32_t creationNodeMask = 0x1;
    const uint32_t visibilityNodeMask = 0x1;
    std::shared_ptr<vk::CommandBuffer> cmdBuffer = vk::CommandBuffer::create(device, cmdPool);
    cmdBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    NVSDK_NGX_Result result =
        NGX_CHECK(NGX_VULKAN_CREATE_DLSS_EXT1(device->vkDevice(), cmdBuffer->vkCommandBuffer(), creationNodeMask,
                                              visibilityNodeMask, &m_dlssHandle, ngxParams, &dlssParams));
    if (NVSDK_NGX_FAILED(result)) {
        m_dlssHandle = nullptr;
        m_ngxParams = nullptr;
        m_device = VK_NULL_HANDLE;
        return result;
    }
    auto initializationFence = vk::Fence::create(device);
    cmdBuffer->end()->submitMainQueueIndividual(device, initializationFence);
    vk::waitForFence(device->vkDevice(), initializationFence->vkFence(), "vkWaitForFences(DLSS SR initialization)");

    return NVSDK_NGX_Result_Success;
}

void DlssSR::deinit() {
    if (m_dlssHandle) { NVSDK_NGX_VULKAN_ReleaseFeature(m_dlssHandle); }
    m_dlssHandle = nullptr;
    m_ngxParams = nullptr;
    m_device = VK_NULL_HANDLE;
    m_resources.fill({.Resource = {.ImageViewInfo = {}}});
}

DlssSR::~DlssSR() {
    assert(!m_dlssHandle && "Must call deinit");
}

void DlssSR::setResource(DlssResource resourceId, std::shared_ptr<vk::DeviceLocalImage> image) {
    assert(m_dlssHandle);
    assert(image);

    const VkExtent2D size = resourceId == RESOURCE_COLOR_OUT ? m_outputSize : m_inputSize;
    m_resources[resourceId] = NVSDK_NGX_Create_ImageView_Resource_VK(
        image->vkImageView(), image->vkImage(), vk::wholeColorSubresourceRange, image->vkFormat(), size.width,
        size.height, resourceId == RESOURCE_COLOR_OUT);
}

void DlssSR::resetResource(DlssResource resourceId) {
    m_resources[resourceId] = {};
}

NVSDK_NGX_Result DlssSR::evaluate(std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                                  glm::uvec2 renderSize,
                                  glm::vec2 jitter,
                                  glm::vec2 motionVectorScale,
                                  bool reset,
                                  float frameTimeMilliseconds) {
    if (!cmdBuffer || !m_dlssHandle || !m_ngxParams || renderSize.x == 0 || renderSize.y == 0) {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }

    auto getResource = [this](DlssResource resource) -> NVSDK_NGX_Resource_VK * {
        return m_resources[resource].Resource.ImageViewInfo.ImageView ? &m_resources[resource] : nullptr;
    };

    NVSDK_NGX_VK_DLSS_Eval_Params evalParams{};
    evalParams.Feature.pInColor = getResource(RESOURCE_COLOR_IN);
    evalParams.Feature.pInOutput = getResource(RESOURCE_COLOR_OUT);
    evalParams.pInMotionVectors = getResource(RESOURCE_MOTIONVECTOR);
    evalParams.pInDepth = getResource(RESOURCE_DEPTH);
    if (!evalParams.Feature.pInColor || !evalParams.Feature.pInOutput || !evalParams.pInMotionVectors ||
        !evalParams.pInDepth) {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }

    evalParams.InJitterOffsetX = -jitter.x;
    evalParams.InJitterOffsetY = -jitter.y;
    evalParams.InRenderSubrectDimensions = {renderSize.x, renderSize.y};
    evalParams.InReset = reset ? 1 : 0;
    evalParams.InFrameTimeDeltaInMsec = frameTimeMilliseconds > 0.0f ? frameTimeMilliseconds : 1000.0f / 60.0f;
    evalParams.InMVScaleX = motionVectorScale.x == 0.0f ? 1.0f : motionVectorScale.x;
    evalParams.InMVScaleY = motionVectorScale.y == 0.0f ? 1.0f : motionVectorScale.y;

    NGX_RETURN_ON_FAIL(
        NGX_VULKAN_EVALUATE_DLSS_EXT(cmdBuffer->vkCommandBuffer(), m_dlssHandle, m_ngxParams, &evalParams));
    return NVSDK_NGX_Result_Success;
}

std::string getNGXResultString(NVSDK_NGX_Result result) {
    return ws2s(GetNGXResultAsString(result));
}
