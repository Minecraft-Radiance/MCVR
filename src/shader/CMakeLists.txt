if(WIN32)
    add_definitions(-DNOMINMAX)
    set(GLSLVALIDATOR glslangValidator.exe)
else()
    set(GLSLVALIDATOR glslangValidator)
endif(WIN32)

file(GLOB_RECURSE GLSL_SHADERS CONFIGURE_DEPENDS "*.vert" "*.frag" "*.rgen" "*.rmiss" "*.rchit" "*.rahit" "*.comp")
list(FILTER GLSL_SHADERS EXCLUDE REGEX "world/ray_tracing/internal")
file(GLOB_RECURSE GLSL_DEPENDS CONFIGURE_DEPENDS "../common/*.hpp" "../common/*.glsl" "util/*.glsl")
list(APPEND GLSL_DEPENDS "${PROJECT_SOURCE_DIR}/extern/nrd/Shaders/Include/NRD.hlsli")
file(GLOB_RECURSE FSR1_HEADER_DEPENDS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/extern/FidelityFX-SDK/sdk/include/FidelityFX/gpu/*.h")
list(APPEND GLSL_DEPENDS ${FSR1_HEADER_DEPENDS}
    "${CMAKE_CURRENT_SOURCE_DIR}/world/upscaler/fsr1_common.glsl")

set(SPIRV_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_PATH})

set(SPIRV_OUTPUT_FILES "")

set(SHADER_FLAGS "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Debug>:-Od>")

foreach(GLSL_SHADER ${GLSL_SHADERS})
    file(RELATIVE_PATH REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${GLSL_SHADER}")
    get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
    get_filename_component(FILE_NAME "${GLSL_SHADER}" NAME_WE)
    get_filename_component(FILE_EXT "${GLSL_SHADER}" LAST_EXT)
    string(REPLACE "." "" FILE_TYPE "${FILE_EXT}")

    set(OUTPUT_DIR "${SPIRV_OUTPUT_PATH}/${REL_DIR}")
    set(SPIRV_FILE "${OUTPUT_DIR}/${FILE_NAME}_${FILE_TYPE}.spv")

    add_custom_command(
        OUTPUT "${SPIRV_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
        COMMAND ${GLSLVALIDATOR} ${SHADER_FLAGS} -V "${GLSL_SHADER}" -o "${SPIRV_FILE}" --target-env vulkan1.4 -I${PROJECT_SOURCE_DIR}/src -I${CMAKE_CURRENT_SOURCE_DIR}/util -I${PROJECT_SOURCE_DIR}/extern/FidelityFX-SDK/sdk/include/FidelityFX/gpu
        DEPENDS "${GLSL_SHADER}" ${GLSL_DEPENDS}
        VERBATIM
        COMMAND_EXPAND_LISTS
    )

    list(APPEND SPIRV_OUTPUT_FILES "${SPIRV_FILE}")
endforeach()
set(NRD_HALF_RATE_COMPOSE_SPV "${SPIRV_OUTPUT_PATH}/world/nrd/compose_half_rate_comp.spv")
add_custom_command(
    OUTPUT "${NRD_HALF_RATE_COMPOSE_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OUTPUT_PATH}/world/nrd"
    COMMAND ${GLSLVALIDATOR} ${SHADER_FLAGS} -V "${CMAKE_CURRENT_SOURCE_DIR}/world/nrd/compose.comp"
            -o "${NRD_HALF_RATE_COMPOSE_SPV}" --target-env vulkan1.4 -DMCVR_NRD_HALF_RATE_RESOLVE=1
            -I${PROJECT_SOURCE_DIR}/src -I${CMAKE_CURRENT_SOURCE_DIR}/util
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/world/nrd/compose.comp" ${GLSL_DEPENDS}
    VERBATIM
    COMMAND_EXPAND_LISTS
)
list(APPEND SPIRV_OUTPUT_FILES "${NRD_HALF_RATE_COMPOSE_SPV}")

add_custom_target(shaders ALL
    DEPENDS ${SPIRV_OUTPUT_FILES} ${GLSL_DEPENDS}
)

install(
    DIRECTORY ${SPIRV_OUTPUT_PATH}/
    DESTINATION ${MCVR_INSTALL_BIN_DIR}/res
    FILES_MATCHING PATTERN "*.spv"
)

install(
    DIRECTORY ${SPIRV_OUTPUT_PATH}/
    DESTINATION ${MCVR_INSTALL_LIB_DIR}/shaders
    FILES_MATCHING PATTERN "*.spv"
)

set(ZIP_INSTALL_DEST "${MCVR_INSTALL_LIB_DIR}/shaders/world/ray_tracing")
set(RT_INTERNAL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/world/ray_tracing/internal")
set(RT_UTIL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/util")
set(RT_COMMON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")
include("${PROJECT_SOURCE_DIR}/cmake/PrepareSharc.cmake")
set(RT_SHARC_PATCHED_DIR "${CMAKE_CURRENT_BINARY_DIR}/patched_sharc")
mcvr_prepare_sharc("${PROJECT_SOURCE_DIR}/extern/sharc" "${RT_SHARC_PATCHED_DIR}")
set(RT_SHARC_SRC_DIR "${RT_SHARC_PATCHED_DIR}/include")
set(RT_ZIP_STAGING_DIR "${CMAKE_CURRENT_BINARY_DIR}/ray_tracing_internal_zip")
set(RT_LEGACY_SHADER_PACK_ZIPS restir-di.zip wavefront.zip)

foreach(RT_PACK_NAME ${MCVR_BUILT_IN_SHADER_PACKS})
    if(NOT EXISTS "${RT_INTERNAL_ROOT_DIR}/${RT_PACK_NAME}/configs.json")
        message(FATAL_ERROR "Built-in shader pack is missing configs.json: ${RT_INTERNAL_ROOT_DIR}/${RT_PACK_NAME}")
    endif()
endforeach()
message(STATUS "Installing ray tracing built-in shader packs: ${MCVR_BUILT_IN_SHADER_PACKS}")

install(CODE "
    file(MAKE_DIRECTORY \"${ZIP_INSTALL_DEST}\")
    file(GLOB RT_PACK_SETTINGS_FILES \"${ZIP_INSTALL_DEST}/*.zip.txt\")
    foreach(RT_PACK_SETTINGS_FILE \${RT_PACK_SETTINGS_FILES})
        file(REMOVE \"\${RT_PACK_SETTINGS_FILE}\")
    endforeach()
    file(REMOVE \"${ZIP_INSTALL_DEST}/internal.zip\")
    foreach(RT_LEGACY_SHADER_PACK_ZIP ${RT_LEGACY_SHADER_PACK_ZIPS})
        file(REMOVE \"${ZIP_INSTALL_DEST}/\${RT_LEGACY_SHADER_PACK_ZIP}\")
    endforeach()

    foreach(RT_PACK_NAME ${MCVR_BUILT_IN_SHADER_PACKS})
        set(RT_PACK_SRC_DIR \"${RT_INTERNAL_ROOT_DIR}/\${RT_PACK_NAME}\")
        set(RT_PACK_STAGING_DIR \"${RT_ZIP_STAGING_DIR}/\${RT_PACK_NAME}\")
        set(RT_PACK_OUTPUT \"${ZIP_INSTALL_DEST}/\${RT_PACK_NAME}.zip\")

        file(REMOVE_RECURSE \"\${RT_PACK_STAGING_DIR}\")
        file(MAKE_DIRECTORY \"\${RT_PACK_STAGING_DIR}\")
        file(COPY \"\${RT_PACK_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}\")
        file(COPY \"${RT_UTIL_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}/util\")
        file(COPY \"${RT_COMMON_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}/common\")
        file(READ \"\${RT_PACK_SRC_DIR}/configs.json\" RT_PACK_CONFIG_JSON)
        if(RT_PACK_CONFIG_JSON MATCHES \"\\\"sharc\\\"\")
            file(COPY \"${RT_SHARC_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}/extern/sharc/include\")
        endif()
        file(REMOVE \"\${RT_PACK_OUTPUT}\")

        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E tar \"cf\" \"\${RT_PACK_OUTPUT}\" --format=zip .
            WORKING_DIRECTORY \"\${RT_PACK_STAGING_DIR}\"
            RESULT_VARIABLE ZIP_RESULT
            OUTPUT_VARIABLE ZIP_STDOUT
            ERROR_VARIABLE ZIP_STDERR
        )

        if(NOT ZIP_RESULT EQUAL 0)
            message(FATAL_ERROR \"Failed to create \${RT_PACK_NAME}.zip (code=\${ZIP_RESULT})\\nstdout: \${ZIP_STDOUT}\\nstderr: \${ZIP_STDERR}\")
        endif()
    endforeach()
")
