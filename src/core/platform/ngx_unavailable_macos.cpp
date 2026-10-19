// Apple Silicon has no NVIDIA NGX runtime. Report the optional backend as
// unavailable so Radiance can use its Vulkan denoisers without Linux binaries.
#if defined(__APPLE__)
#include "core/all_extern.hpp"
#include "nvsdk_ngx_vk.h"
static constexpr auto unavailable = NVSDK_NGX_Result_FAIL_FeatureNotSupported;
NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version) { return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(const NVSDK_NGX_FeatureDiscoveryInfo*, uint32_t* n, VkExtensionProperties** p) { if(n) *n=0; if(p) *p=nullptr; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(VkInstance, VkPhysicalDevice, const NVSDK_NGX_FeatureDiscoveryInfo*, uint32_t* n, VkExtensionProperties** p) { if(n) *n=0; if(p) *p=nullptr; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetCapabilityParameters(NVSDK_NGX_Parameter** p) { if(p) *p=nullptr; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_DestroyParameters(NVSDK_NGX_Parameter*) { return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_Shutdown1(VkDevice) { return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_ReleaseFeature(NVSDK_NGX_Handle*) { return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_CreateFeature(VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle** p) { if(p) *p=nullptr; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_CreateFeature1(VkDevice, VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle** p) { if(p) *p=nullptr; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_VULKAN_EvaluateFeature_C(VkCommandBuffer, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback_C) { return unavailable; }
const wchar_t* GetNGXResultAsString(NVSDK_NGX_Result) { return L"NGX unavailable on macOS arm64"; }
void NVSDK_NGX_Parameter_SetF(NVSDK_NGX_Parameter*,const char*,float) {}
void NVSDK_NGX_Parameter_SetI(NVSDK_NGX_Parameter*,const char*,int) {}
void NVSDK_NGX_Parameter_SetUI(NVSDK_NGX_Parameter*,const char*,unsigned int) {}
void NVSDK_NGX_Parameter_SetVoidPointer(NVSDK_NGX_Parameter*,const char*,void*) {}
NVSDK_NGX_Result NVSDK_NGX_Parameter_GetF(NVSDK_NGX_Parameter*,const char*,float* p) { if(p) *p=0; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_Parameter_GetUI(NVSDK_NGX_Parameter*,const char*,unsigned int* p) { if(p) *p=0; return unavailable; }
NVSDK_NGX_Result NVSDK_NGX_Parameter_GetVoidPointer(NVSDK_NGX_Parameter*,const char*,void** p) { if(p) *p=nullptr; return unavailable; }
#endif
