file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")

add_library(core SHARED ${SOURCE_FILES})
target_link_libraries(core PUBLIC ngx ${JNI_LIBRARIES} VulkanMemoryAllocator volk::volk glfw)
target_include_directories(core PUBLIC ${GENTOOL_OUTPUT_DIR} ${VOLK_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${VMA_INCLUDE_DIR} ${STB_INCLUDE_DIR} ${JNI_INCLUDE_DIRS} ${GLM_INCLUDE_DIR})
target_compile_definitions(core PUBLIC VK_NO_PROTOTYPES CORE_LIB)
if (USE_AMD)
    target_compile_definitions(core PUBLIC USE_AMD)
endif ()
if (MCVR_ENABLE_FFX_UPSCALER)
    # amd_fidelityfx_vk provides the main FFX API (ffxCreateContext, ffxQuery, etc.)
    target_link_libraries(core PUBLIC amd_fidelityfx_vk ffx_backend_vk_x64 ffx_fsr3upscaler_x64)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_FFX_UPSCALER)
endif ()
if (MCVR_ENABLE_NRD)
    target_link_libraries(core PUBLIC NRD)
    target_include_directories(core PUBLIC ${NRD_INCLUDE_DIR})
    target_compile_definitions(core PUBLIC NRD_STATIC_LIBRARY=1)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_NRD)
endif ()
if (MCVR_ENABLE_XESS)
    target_include_directories(core PUBLIC ${XESS_INCLUDE_DIR})
    if (WIN32)
        target_link_libraries(core PUBLIC "${PROJECT_SOURCE_DIR}/extern/xess/lib/libxess.lib")
        target_compile_definitions(core PUBLIC MCVR_ENABLE_XESS)

        # Install XeSS runtime DLLs
        set(XESS_DLLS
            "${PROJECT_SOURCE_DIR}/extern/xess/bin/libxess.dll"
        )
        install(FILES ${XESS_DLLS} DESTINATION ${MCVR_INSTALL_LIB_DIR})
        install(FILES ${XESS_DLLS} DESTINATION ${MCVR_INSTALL_BIN_DIR})
    else ()
        # Linux builds require a native XeSS shared library (e.g. libxess.so/.dylib).
        # Missing right now. build without XeSS backend to avoid unresolved xess* symbols at runtime.
        find_library(XESS_RUNTIME_LIB
            NAMES xess libxess
            PATHS
                "${PROJECT_SOURCE_DIR}/extern/xess/lib"
                "${PROJECT_SOURCE_DIR}/extern/xess/bin"
        )

        if (XESS_RUNTIME_LIB)
            target_link_libraries(core PUBLIC "${XESS_RUNTIME_LIB}")
            target_compile_definitions(core PUBLIC MCVR_ENABLE_XESS)

            get_filename_component(XESS_RUNTIME_DIR "${XESS_RUNTIME_LIB}" DIRECTORY)
            set_property(TARGET core APPEND PROPERTY BUILD_RPATH "${XESS_RUNTIME_DIR}")
        else ()
            message(WARNING "MCVR_ENABLE_XESS=ON but no native XeSS runtime library found; building core without XeSS backend symbols.")
        endif ()
    endif ()
endif ()


if (MSVC)
    target_compile_definitions(core PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(core PRIVATE /utf-8)
endif ()

set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)

install(TARGETS core DESTINATION ${MCVR_INSTALL_LIB_DIR})
install(TARGETS core DESTINATION ${MCVR_INSTALL_BIN_DIR})
