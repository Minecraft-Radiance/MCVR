file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")
if(NOT MCVR_ENABLE_PROFILER)
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/profiler/.*\\.cpp$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/com_radiance_client_profiler_Profiler\\.cpp$")
endif()
if(MCVR_UI_BACKEND STREQUAL "OPENGL")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/render/modules/ui_module\\.cpp$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/com_radiance_client_proxy_vulkan_ShaderProxy\\.cpp$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/com_radiance_client_proxy_vulkan_VulkanUiProxy\\.cpp$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/com_radiance_client_proxy_vulkan_DrawCommandProxy_Overlay\\.cpp$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/com_radiance_client_proxy_vulkan_PipelineStateProxy_.*\\.cpp$")
else()
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/com_radiance_client_opengl_ui_OpenGLUiProxy\\.cpp$")
    list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/render/opengl_ui/.*\\.cpp$")
endif()

add_library(core SHARED ${SOURCE_FILES})
target_link_libraries(core PUBLIC ngx VulkanMemoryAllocator volk::volk glfw minizip tinyexpr
                                   nlohmann_json::nlohmann_json)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MCVR_UI_BACKEND STREQUAL "OPENGL")
    find_package(X11 REQUIRED)
    target_link_libraries(core PRIVATE X11::X11)
endif()

if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    set(MCVR_VULKAN_SDK "$ENV{VULKAN_SDK}")
else()
    set(MCVR_VULKAN_SDK "/home/jiong/VulkanSDK/1.4.328.1")
endif()

find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.hpp
    HINTS
    "${MCVR_VULKAN_SDK}/x86_64/include"
    "${MCVR_VULKAN_SDK}/include"
)
find_library(SHADERC_LIBRARY_RELEASE
    NAMES shaderc_combined shaderc_shared shaderc
    HINTS
    "${MCVR_VULKAN_SDK}/x86_64/lib"
    "${MCVR_VULKAN_SDK}/lib"
)
if(MSVC)
    find_library(SHADERC_LIBRARY_DEBUG
        NAMES shaderc_combinedd shaderc_sharedd shadercd
        HINTS
        "${MCVR_VULKAN_SDK}/lib"
    )
endif()
include(SelectLibraryConfigurations)
select_library_configurations(SHADERC)

if(NOT SHADERC_INCLUDE_DIR OR NOT SHADERC_LIBRARY)
    message(FATAL_ERROR "shaderc was not found. Set VULKAN_SDK or install shaderc.")
endif()

target_include_directories(core PUBLIC ${GENTOOL_OUTPUT_DIR} ${VOLK_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${VMA_INCLUDE_DIR} ${STB_INCLUDE_DIR} ${TINYEXPR_INCLUDE_DIR} ${JNI_INCLUDE_DIRS} ${GLM_INCLUDE_DIR} ${SHADERC_INCLUDE_DIR})
target_link_libraries(core PUBLIC ${SHADERC_LIBRARY})
target_compile_definitions(core PUBLIC VK_NO_PROTOTYPES CORE_LIB)
if(USE_AMD)
    target_compile_definitions(core PUBLIC USE_AMD)
endif()
if(MCVR_ENABLE_PROFILER)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_PROFILER)
endif()
if(MCVR_UI_BACKEND STREQUAL "OPENGL")
    target_compile_definitions(core PUBLIC MCVR_UI_BACKEND_OPENGL)
else()
    target_compile_definitions(core PUBLIC MCVR_UI_BACKEND_VULKAN)
endif()
if(MCVR_ENABLE_FFX_UPSCALER)
    target_link_libraries(core PUBLIC amd_fidelityfx_vk
                                      ffx_backend_vk_x64
                                      ffx_fsr3_x64
                                      ffx_fsr3upscaler_x64
                                      ffx_frameinterpolation_x64
                                      ffx_opticalflow_x64)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_FFX_UPSCALER)
endif()
if(MCVR_ENABLE_FSR1_SPATIAL)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_FSR1_SPATIAL)
endif()
if(MCVR_ENABLE_NRD)
    target_link_libraries(core PUBLIC NRD)
    target_include_directories(core PUBLIC ${NRD_INCLUDE_DIR})
    target_compile_definitions(core PUBLIC NRD_STATIC_LIBRARY=1)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_NRD)
endif()
if(MCVR_ENABLE_XESS)
    find_library(XESS_RUNTIME_LIB
        NAMES xess libxess
        PATHS
            "${PROJECT_SOURCE_DIR}/extern/xess/lib"
            "${PROJECT_SOURCE_DIR}/extern/xess/bin"
    )

    if(XESS_RUNTIME_LIB)
        target_link_libraries(core PUBLIC "${XESS_RUNTIME_LIB}")
        target_include_directories(core PUBLIC ${XESS_INCLUDE_DIR})
        target_compile_definitions(core PUBLIC MCVR_ENABLE_XESS)

        if(WIN32)
            file(GLOB XESS_RUNTIME_DLLS "${PROJECT_SOURCE_DIR}/extern/xess/bin/libxess*.dll")
            if(XESS_RUNTIME_DLLS)
                install(FILES ${XESS_RUNTIME_DLLS} DESTINATION ${MCVR_INSTALL_LIB_DIR})
                install(FILES ${XESS_RUNTIME_DLLS} DESTINATION ${MCVR_INSTALL_BIN_DIR})
            else()
                message(WARNING "XeSS enabled and linked, but no runtime DLLs were found in extern/xess/bin.")
            endif()
        endif()
    else()
        message(WARNING "XeSS headers were found but no XeSS runtime library was found; building without XeSS backend.")
    endif()
endif()
if(MSVC)
    target_compile_definitions(core PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(core PRIVATE /utf-8)
endif()

set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)

install(TARGETS core DESTINATION ${MCVR_INSTALL_LIB_DIR})
install(TARGETS core DESTINATION ${MCVR_INSTALL_BIN_DIR})
