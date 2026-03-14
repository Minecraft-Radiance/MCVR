#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 0, binding = 1) uniform sampler2D lightMap;

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(push_constant) uniform PushConstant {
    uint eyeIndex;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in uint inUseNorm;
layout(location = 2) in vec3 inNorm;
layout(location = 3) in uint inUseColorLayer;
layout(location = 4) in vec4 inColorLayer;
layout(location = 5) in uint inUseTexture;
layout(location = 6) in uint inUseOverlay;
layout(location = 7) in vec2 inTextureUV;
layout(location = 8) in ivec2 inOverlayUV;
layout(location = 9) in uint inUseGlint;
layout(location = 10) in uint inTextureID;
layout(location = 11) in vec2 inGlintUV;
layout(location = 12) in uint inGlintTexture;
layout(location = 13) in uint inUseLight;
layout(location = 14) in ivec2 inLightUV;
layout(location = 15) in uint inCoordinate;
layout(location = 16) in vec3 inPostBase;

layout(location = 0) out vec3 outPos;
layout(location = 1) flat out uint outUseNorm;
layout(location = 2) out vec3 outNorm;
layout(location = 3) flat out uint outUseColorLayer;
layout(location = 4) out vec4 outColorLayer;
layout(location = 5) flat out uint outUseTexture;
layout(location = 6) flat out uint outUseOverlay;
layout(location = 7) out vec2 outTextureUV;
layout(location = 8) flat out ivec2 outOverlayUV;
layout(location = 9) flat out uint outUseGlint;
layout(location = 10) flat out uint outTextureID;
layout(location = 11) out vec2 outGlintUV;
layout(location = 12) flat out uint outGlintTexture;
layout(location = 13) flat out uint outUseLight;
layout(location = 14) flat out ivec2 outLightUV;
layout(location = 15) out vec4 lightMapColor;
layout(location = 16) out vec4 overlayColor;

void main() {
    // Per-eye view offset
    mat4 eyeViewOffset = worldUBO.eyeViewOffsets[pc.eyeIndex];
    mat4 eyeView = eyeViewOffset * worldUBO.cameraEffectedViewMat;
    mat4 eyeViewOffsetInv = mat4(1.0);
    eyeViewOffsetInv[3] = vec4(-eyeViewOffset[3].xyz, 1.0);
    mat4 eyeViewInv = worldUBO.cameraViewMatInv * eyeViewOffsetInv;

    vec3 pos = inPos + inPostBase;
    if (inCoordinate == 0) {
        pos = pos - eyeViewInv[3].xyz;
    } else if (inCoordinate == 1) {
        pos = mat3(eyeViewInv) * pos;
    } else if (inCoordinate == 2) {
        pos = pos;
    }
    outPos = pos;
    outUseNorm = inUseNorm;
    if (inCoordinate == 0 || inCoordinate == 2) {
        outNorm = inNorm;
    } else if (inCoordinate == 1) {
        outNorm = normalize(mat3(eyeViewInv) * inNorm);
    }
    outUseColorLayer = inUseColorLayer;
    outColorLayer = inColorLayer;
    outUseTexture = inUseTexture;
    outUseOverlay = inUseOverlay;
    outTextureUV = inTextureUV;
    outOverlayUV = inOverlayUV;
    outUseGlint = inUseGlint;
    outTextureID = inTextureID;
    outGlintUV = inGlintUV;
    outGlintTexture = inGlintTexture;
    outUseLight = inUseLight;
    outLightUV = inLightUV;

    gl_Position = worldUBO.eyeProjOffsets[pc.eyeIndex] * worldUBO.cameraProjMat * eyeView * vec4(pos, 1.0);

    if (inUseLight > 0) {
        lightMapColor = texelFetch(lightMap, inLightUV / 16, 0);
    } else {
        lightMapColor = vec4(0.0);
    }
    if (inUseOverlay > 0) {
        overlayColor = texelFetch(textures[nonuniformEXT(worldUBO.overlayTextureID)], inOverlayUV, 0);
    } else {
        overlayColor = vec4(0.0);
    }
}
