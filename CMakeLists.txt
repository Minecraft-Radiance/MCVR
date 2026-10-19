cmake_minimum_required(VERSION 3.15)

project(minecraft-vulkan-renderer LANGUAGES C CXX)

if(NOT DEFINED JAVA_PROJECT_ROOT_DIR OR "${JAVA_PROJECT_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "JAVA_PROJECT_ROOT_DIR is required. Pass -DJAVA_PROJECT_ROOT_DIR=/path/to/root/of/the/java/project/")
else()
    file(TO_CMAKE_PATH "${JAVA_PROJECT_ROOT_DIR}" JAVA_PROJECT_ROOT_DIR)
    message(STATUS "JAVA_PROJECT_ROOT_DIR: ${JAVA_PROJECT_ROOT_DIR}")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 23)
add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if(NOT MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    endif()
endif()

set(MCVR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src ${JAVA_PROJECT_ROOT_DIR}/src/main/native/include/)
message(STATUS "Include dir: ${MCVR_INCLUDE_DIR}")

set(MCVR_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
message(STATUS "Install bin dir: ${MCVR_INSTALL_BIN_DIR}")

set(MCVR_INSTALL_LIB_DIR ${JAVA_PROJECT_ROOT_DIR}/src/main/resources/)
message(STATUS "Install lib dir: ${MCVR_INSTALL_LIB_DIR}")

include_directories(${MCVR_INCLUDE_DIR})

# GLFW
add_subdirectory(extern/glfw EXCLUDE_FROM_ALL)
set(GLFW_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glfw/include)

# Volk
set(VOLK_BUILD_TESTS OFF CACHE BOOL "Disable tests")
set(VOLK_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples")
add_subdirectory(extern/volk EXCLUDE_FROM_ALL)
set(VOLK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/volk)

# VMA
add_subdirectory(extern/vma EXCLUDE_FROM_ALL)
set(VMA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/vma/include)

# GLM
set(GLM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glm)

# VK.XML
set(VK_XML_PATH ${PROJECT_SOURCE_DIR}/extern/vulkan_headers/registry/vk.xml)

# STB
set(STB_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/stb)

# tinyexpr
add_library(tinyexpr STATIC extern/tinyexpr/tinyexpr.c)
set_target_properties(tinyexpr PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(TINYEXPR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/tinyexpr)

# DLSS
if(APPLE)
    add_library(ngx INTERFACE)
    target_include_directories(ngx INTERFACE "${PROJECT_SOURCE_DIR}/extern/DLSS/include")
else()
    add_subdirectory(extern/DLSS EXCLUDE_FROM_ALL)
endif()

# JNI
find_package(JNI REQUIRED)
message(STATUS "${JNI_INCLUDE_DIRS}")
message(STATUS "${JNI_LIBRARIES}")

option(USE_AMD "Enable AMD specific optimizations/features" ON)
if(USE_AMD)
    message(STATUS "AMD Support Enabled")
else()
    message(STATUS "AMD Support Disabled")
endif()

option(MCVR_ENABLE_PROFILE_CONFIG "Enable the native profiler and CPU-sampling-friendly build flags" OFF)
option(MCVR_ENABLE_PROFILER "Enable per-frame renderer profiler output" OFF)
option(MCVR_ENABLE_CPU_PROFILE_FLAGS "Build native code with symbols and frame pointers for CPU profilers" OFF)
set(MCVR_UI_BACKEND "OPENGL" CACHE STRING "UI backend to compile: OPENGL or VULKAN")
string(TOUPPER "${MCVR_UI_BACKEND}" MCVR_UI_BACKEND)
set_property(CACHE MCVR_UI_BACKEND PROPERTY STRINGS OPENGL VULKAN)
if(NOT MCVR_UI_BACKEND STREQUAL "VULKAN" AND NOT MCVR_UI_BACKEND STREQUAL "OPENGL")
    message(FATAL_ERROR "MCVR_UI_BACKEND must be VULKAN or OPENGL, got ${MCVR_UI_BACKEND}")
endif()
message(STATUS "MCVR UI backend: ${MCVR_UI_BACKEND}")
if(MCVR_ENABLE_PROFILE_CONFIG)
    set(MCVR_ENABLE_PROFILER ON CACHE BOOL "Enable per-frame renderer profiler output" FORCE)
    set(MCVR_ENABLE_CPU_PROFILE_FLAGS ON CACHE BOOL "Build native code with symbols and frame pointers for CPU profilers" FORCE)
endif()
if(MCVR_ENABLE_PROFILER)
    message(STATUS "MCVR Profiler Enabled")
else()
    message(STATUS "MCVR Profiler Disabled")
endif()
if(MCVR_ENABLE_CPU_PROFILE_FLAGS)
    if(MSVC)
        add_compile_options(/Zi)
        add_link_options(/DEBUG)
    else()
        add_compile_options(-g -fno-omit-frame-pointer -fno-optimize-sibling-calls)
        add_link_options(-g)
    endif()
    message(STATUS "MCVR CPU profile flags enabled")
endif()

set(MCVR_BUILT_IN_SHADER_PACKS vanilla-pt advanced)
if(APPLE)
    # The advanced pack requires primitive-culling features MageVK does not expose.
    set(MCVR_BUILT_IN_SHADER_PACKS vanilla-pt)
endif()
message(STATUS "Built-in shader packs: ${MCVR_BUILT_IN_SHADER_PACKS}")

option(MCVR_ENABLE_FFX_UPSCALER "Enable FidelityFX Upscaler support" ON)
option(MCVR_ENABLE_FSR1_SPATIAL "Enable standalone FidelityFX FSR1 spatial shaders" OFF)
if(MCVR_ENABLE_FFX_UPSCALER)
    message(STATUS "FidelityFX Upscaler Enabled")

    set(FFX_API_BACKEND VK_X64 CACHE STRING "FidelityFX API backend" FORCE)

    set(FFX_DENOISER OFF CACHE BOOL "Build FFX Denoiser" FORCE)
    set(FFX_ALL OFF CACHE BOOL "Build all FFX components" FORCE)
    set(FFX_SSSR OFF CACHE BOOL "Build FFX SSSR" FORCE)
    set(FFX_FSR3 ON CACHE BOOL "Build FFX FSR3" FORCE)
    set(FFX_FSR3UPSCALER ON CACHE BOOL "Build FFX FSR3 Upscaler" FORCE)
    set(FFX_FI ON CACHE BOOL "Build FFX Frame Interpolation" FORCE)
    set(FFX_OF ON CACHE BOOL "Build FFX Optical Flow" FORCE)

    set(FFX_AUTO_COMPILE_SHADERS ON CACHE BOOL "Compile shaders automatically as a prebuild step." FORCE)

    set(FFX_BIN_PATH ${CMAKE_BINARY_DIR}/ffx_sdk CACHE STRING "FFX binary output path" FORCE)
    add_subdirectory(extern/FidelityFX-SDK/ffx-api EXCLUDE_FROM_ALL)

    if(TARGET ffx_backend_vk_x64)
        target_include_directories(ffx_backend_vk_x64 PRIVATE ${VOLK_INCLUDE_DIR})
    endif()

else()
    message(STATUS "FidelityFX Upscaler Disabled")
endif()

# NRD Denoiser
option(MCVR_ENABLE_NRD "Enable NRD denoiser support" ON)
if(MCVR_ENABLE_NRD)
    message(STATUS "NRD Denoiser Enabled")

    set(NRD_STATIC_LIBRARY ON CACHE BOOL "Build NRD as a static library" FORCE)
    # MCVR consumes Vulkan shader bytecode. DirectX formats are optional build outputs.
    set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "NRD embeds SPIRV shaders" FORCE)
    option(NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" OFF)
    option(NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" OFF)
    set(REBLUR_PERFORMANCE_MODE ON CACHE BOOL "Use the NRD performance shader configuration" FORCE)
    set(NRD_SHADERS_PATH "${PROJECT_SOURCE_DIR}/extern/nrd/_Shaders" CACHE STRING "NRD shader headers path" FORCE)
    set(NRD_NORMAL_ENCODING "2" CACHE STRING "Normal encoding variant (0-4, matches nrd::NormalEncoding)" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "Roughness encoding variant (0-2, matches nrd::RoughnessEncoding)" FORCE)

    add_subdirectory(extern/nrd EXCLUDE_FROM_ALL)

    set(NRD_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/nrd/Include)

    if(TARGET NRD)
        target_include_directories(NRD PRIVATE "${NRD_SHADERS_PATH}")
    endif()
else()
    message(STATUS "NRD Denoiser Disabled")
endif()

# Intel XeSS
option(MCVR_ENABLE_XESS "Enable Intel XeSS support" ON)
if(MCVR_ENABLE_XESS)
    set(XESS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/xess/inc)
    if(EXISTS "${XESS_INCLUDE_DIR}/xess/xess.h")
        message(STATUS "Intel XeSS Enabled")
    else()
        message(WARNING "MCVR_ENABLE_XESS is ON but extern/xess is missing; building without XeSS backend.")
        set(MCVR_ENABLE_XESS OFF)
    endif()
else()
    message(STATUS "Intel XeSS Disabled")
endif()

# minizip
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(MZ_BUILD_TESTS OFF CACHE BOOL "Disable minizip tests" FORCE)
set(MZ_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable minizip unit tests" FORCE)
set(MZ_ZLIB ON CACHE BOOL "Enable Deflate support for shader pack archives" FORCE)
set(MZ_BZIP2 OFF CACHE BOOL "Disable unused bzip2 support" FORCE)
set(MZ_LZMA OFF CACHE BOOL "Disable unused LZMA support" FORCE)
set(MZ_OPENSSL OFF CACHE BOOL "Disable unused encrypted archive support" FORCE)
set(MZ_FETCH_LIBS ON CACHE BOOL "Fetch minizip dependencies automatically" FORCE)
set(MZ_FORCE_FETCH_LIBS ON CACHE BOOL "Build Minizip dependencies from source" FORCE)
set(MZ_ZSTD OFF CACHE BOOL "Disable zstd in minizip-ng due to upstream target mismatch" FORCE)
add_subdirectory(extern/minizip-ng EXCLUDE_FROM_ALL)
set_target_properties(minizip PROPERTIES POSITION_INDEPENDENT_CODE ON)

# nlohmann json
add_subdirectory(extern/json EXCLUDE_FROM_ALL)

add_subdirectory(src)
