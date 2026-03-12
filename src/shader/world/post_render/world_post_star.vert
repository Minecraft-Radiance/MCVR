#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(push_constant) uniform PushConstant {
    uint eyeIndex;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 4) in vec4 inColorLayer;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec4 outColorLayer;

mat3 rotationFromXToDir(vec3 sunDir) {
    vec3 a = vec3(1.0, 0.0, 0.0);
    vec3 b = normalize(sunDir);

    float c = clamp(dot(a, b), -1.0, 1.0);

    if (c > 0.9999) return mat3(1.0);

    if (c < -0.9999) { return mat3(-1, 0, 0, 0, 1, 0, 0, 0, -1); }

    vec3 v = normalize(cross(a, b));
    float s = length(cross(a, b));

    mat3 K = mat3(0.0, v.z, -v.y, -v.z, 0.0, v.x, v.y, -v.x, 0.0);

    mat3 I = mat3(1.0);
    return I + K * s + (K * K) * (1.0 - c);
}

float starVisibilityFromSunY(float sunY) {
    return smoothstep(0.0, -0.2, sunY);
}

void main() {
    mat3 R = rotationFromXToDir(skyUBO.sunDirection);
    vec3 pos = R * inPos;
    outPos = pos;
    gl_Position = worldUBO.eyeProjOffsets[pc.eyeIndex] * worldUBO.cameraProjMat * (worldUBO.eyeViewOffsets[pc.eyeIndex] * worldUBO.cameraEffectedViewMat) * vec4(pos, 1.0);

    float vis = starVisibilityFromSunY(skyUBO.sunDirection.y);
    outColorLayer = vec4(inColorLayer.rgb * vis, vis);
}
