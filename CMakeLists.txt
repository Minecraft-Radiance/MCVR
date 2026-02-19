cmake_minimum_required(VERSION 3.15)

project(minecraft-vulkan-renderer)

if(NOT DEFINED JAVA_PROJECT_ROOT_DIR OR "${JAVA_PROJECT_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "JAVA_PROJECT_ROOT_DIR is required. Pass -DJAVA_PROJECT_ROOT_DIR=/path/to/root/of/the/java/project/")
else()
    # Normalize path to use forward slashes
    file(TO_CMAKE_PATH "${JAVA_PROJECT_ROOT_DIR}" JAVA_PROJECT_ROOT_DIR)
    message(STATUS "JAVA_PROJECT_ROOT_DIR: ${JAVA_PROJECT_ROOT_DIR}")
endif()

set(CMAKE_CXX_STANDARD 23)
if (MSVC)
    add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
else()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -O0")
        add_compile_definitions(DEBUG)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2")
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

# DLSS
add_subdirectory(extern/DLSS EXCLUDE_FROM_ALL)

# JNI
find_package(JNI REQUIRED)
message(STATUS "${JNI_INCLUDE_DIRS}")
message(STATUS "${JNI_LIBRARIES}")

option(USE_AMD "Enable AMD specific optimizations/features" ON)
if (USE_AMD)
    message(STATUS "AMD Support Enabled")
else()
    message(STATUS "AMD Support Disabled")
endif()

# FidelityFX Upscaler (FSR3 Upscaler only)
option(MCVR_ENABLE_FFX_UPSCALER "Enable FidelityFX Upscaler support" ON)

if (MCVR_ENABLE_FFX_UPSCALER)
    message(STATUS "FidelityFX Upscaler Enabled")

    set(FFX_API_BACKEND VK_X64 CACHE STRING "FidelityFX API backend" FORCE)
    
    # Disable everything except FSR3 Upscaler
    set(FFX_DENOISER OFF CACHE BOOL "Build FFX Denoiser" FORCE)
    set(FFX_ALL OFF CACHE BOOL "Build all FFX components" FORCE)
    set(FFX_SSSR OFF CACHE BOOL "Build FFX SSSR" FORCE)
    set(FFX_FSR3 OFF CACHE BOOL "Build FFX FSR3" FORCE) # Full FSR3 (FrameGen) disabled
    set(FFX_FSR3UPSCALER ON CACHE BOOL "Build FFX FSR3 Upscaler" FORCE)
    set(FFX_FI OFF CACHE BOOL "Build FFX Frame Interpolation" FORCE)
    set(FFX_OF OFF CACHE BOOL "Build FFX Optical Flow" FORCE)
    
    set(FFX_AUTO_COMPILE_SHADERS ON CACHE BOOL "Compile shaders automatically as a prebuild step." FORCE)

    # Use ffx-api which includes sdk internally, providing the necessary API symbols (ffxQuery)
    set(FFX_BIN_PATH ${CMAKE_BINARY_DIR}/ffx_sdk CACHE STRING "FFX binary output path" FORCE)
    add_subdirectory(extern/FidelityFX-SDK/ffx-api EXCLUDE_FROM_ALL)

    if (TARGET ffx_backend_vk_x64)
        target_include_directories(ffx_backend_vk_x64 PRIVATE ${VOLK_INCLUDE_DIR})
    endif()
    
    # Note: ffx_denoiser_x64 target will NOT be created, so don't link against it.
else()
    message(STATUS "FidelityFX Upscaler Disabled")
endif()

# NRD Denoiser
option(MCVR_ENABLE_NRD "Enable NRD denoiser support" ON)

if (MCVR_ENABLE_NRD)
    message(STATUS "NRD Denoiser Enabled")

    set(NRD_STATIC_LIBRARY ON CACHE BOOL "Build NRD as a static library" FORCE)
    set(NRD_DISABLE_SHADER_COMPILATION ON CACHE BOOL "Disable NRD shader compilation" FORCE)
    set(NRD_SHADERS_PATH "${PROJECT_SOURCE_DIR}/extern/nrd/_Shaders" CACHE STRING "NRD shader headers path" FORCE)
    set(NRD_NORMAL_ENCODING "2" CACHE STRING "Normal encoding variant (0-4, matches nrd::NormalEncoding)" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "Roughness encoding variant (0-2, matches nrd::RoughnessEncoding)" FORCE)

    add_subdirectory(extern/nrd EXCLUDE_FROM_ALL)

    set(NRD_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/nrd/Include)

    if (TARGET NRD)
        target_include_directories(NRD PRIVATE "${NRD_SHADERS_PATH}")
    endif()
else()
    message(STATUS "NRD Denoiser Disabled")
endif()

# Intel XeSS Super Resolution
option(MCVR_ENABLE_XESS "Enable Intel XeSS Super Resolution support" ON)
if (MCVR_ENABLE_XESS)
    message(STATUS "Intel XeSS Enabled")
    set(XESS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/xess/inc)
else()
    message(STATUS "Intel XeSS Disabled")
endif()

add_subdirectory(src)
