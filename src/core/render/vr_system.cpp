#include "core/render/vr_system.hpp"

#include <algorithm>
#include <cmath>

// ---- VREyeParams ----

glm::mat4 VREyeParams::projectionMatrix(float nearZ, float farZ) const {
    // Build asymmetric frustum from tangent values
    float left = tanLeft * nearZ;
    float right = tanRight * nearZ;
    float top = tanUp * nearZ;
    float bottom = tanDown * nearZ;

    // Manual frustum construction (Vulkan clip space: Y down, Z [0,1])
    float width = right - left;
    float height = top - bottom;
    float depth = farZ - nearZ;

    glm::mat4 result(0.0f);
    result[0][0] = 2.0f * nearZ / width;
    result[1][1] = -2.0f * nearZ / height; // Vulkan Y flip
    result[2][0] = (right + left) / width;
    result[2][1] = -(top + bottom) / height; // Negated for Vulkan Y-flip (entire row 1 must be negated)
    result[2][2] = -farZ / depth;
    result[2][3] = -1.0f;
    result[3][2] = -(farZ * nearZ) / depth;

    return result;
}

glm::mat4 VREyeParams::viewOffset() const {
    glm::mat4 rotation = glm::mat4_cast(glm::inverse(orientationOffset));
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), -positionOffset);
    return rotation * translation;
}

uint32_t VREyeParams::renderWidth(float renderScale) const {
    return std::max(1u, static_cast<uint32_t>(recommendedWidth * renderScale));
}

uint32_t VREyeParams::renderHeight(float renderScale) const {
    return std::max(1u, static_cast<uint32_t>(recommendedHeight * renderScale));
}

// ---- VRHeadPose ----

glm::mat4 VRHeadPose::viewMatrix() const {
    if (!valid) return glm::mat4(1.0f);
    glm::mat4 rotation = glm::mat4_cast(glm::inverse(orientation));
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), -position);
    return rotation * translation;
}

// ---- VRSystem ----

void VRSystem::updateSimulation(uint32_t windowWidth, uint32_t windowHeight, float fovY) {
    if (!enabled || eyeCount <= 1) return;

    float halfIPD = config.ipd * 0.5f;
    float aspect = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);
    float tanHalfFovY = std::tan(fovY * 0.5f);
    float tanHalfFovX = tanHalfFovY * aspect;

    for (uint32_t eye = 0; eye < 2; eye++) {
        auto &e = eyes[eye];
        e.recommendedWidth = windowWidth;
        e.recommendedHeight = windowHeight;
        e.tanLeft = -tanHalfFovX;
        e.tanRight = tanHalfFovX;
        e.tanUp = tanHalfFovY;
        e.tanDown = -tanHalfFovY;

        float offsetX = (eye == 0) ? -halfIPD : halfIPD;
        e.positionOffset = glm::vec3(offsetX, 0.0f, 0.0f);
        e.orientationOffset = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }

    headPose = VRHeadPose{};
}

uint32_t VRSystem::eyeRenderWidth() const {
    return eyes[0].renderWidth(config.renderScale);
}

uint32_t VRSystem::eyeRenderHeight() const {
    return eyes[0].renderHeight(config.renderScale);
}

void VRSystem::updateFromOpenXR(const VRHeadPose &head, const VREyeParams inEyes[2]) {
    headPose = head;
    for (uint32_t eye = 0; eye < 2; eye++) {
        eyes[eye] = inEyes[eye];
    }
    // Derive IPD from the distance between the two eye positions
    float ipdMeasured = glm::distance(eyes[0].positionOffset, eyes[1].positionOffset);
    if (ipdMeasured > 0.0f) {
        config.ipd = ipdMeasured;
    }
}

void VRSystem::recenter() {
    if (headPose.valid) {
        worldPosition = headPose.position;
    }
}
