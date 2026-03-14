#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

// Per-hand controller state (populated from OpenXR input each frame)
struct VRControllerState {
    bool valid = false;
    glm::vec3 position{0.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};

    float triggerValue = 0.0f;
    float gripValue = 0.0f;
    glm::vec2 thumbstick{0.0f};

    bool triggerPressed = false;
    bool gripPressed = false;
    bool primaryButton = false;
    bool secondaryButton = false;
    bool thumbstickClick = false;
    bool menuButton = false;
};

// Performance statistics for monitoring/auto quality
struct VRPerformanceStats {
    float gpuFrameTimeMs = 0.0f;
    float cpuFrameTimeMs = 0.0f;  // full frame interval (cpuWorkMs + cpuWaitMs)
    float cpuWorkMs = 0.0f;       // CPU active work time (excludes xrWaitFrame compositor pacing)
    float cpuWaitMs = 0.0f;       // CPU time blocked in xrWaitFrame (compositor pacing / VSync wait)
    float compositorTargetMs = 0.0f;  // from XrFrameState.predictedDisplayPeriod
    float fps = 0.0f;
    uint32_t droppedFrames = 0;
    float headroom = 0.0f;  // (targetMs - max(cpuWorkMs, gpuFrameTimeMs)) / targetMs
};

struct VREyeParams {
    // Asymmetric FOV tangent values (negative left/down, positive right/up)
    float tanLeft = -1.0f;
    float tanRight = 1.0f;
    float tanUp = 1.0f;
    float tanDown = -1.0f;

    // Recommended render resolution from OpenXR (per-eye)
    uint32_t recommendedWidth = 1920;
    uint32_t recommendedHeight = 1080;

    // Eye position/orientation offset relative to head (head space)
    glm::vec3 positionOffset{0.0f};
    glm::quat orientationOffset{1.0f, 0.0f, 0.0f, 0.0f};

    // Build asymmetric projection matrix from tangent values
    glm::mat4 projectionMatrix(float nearZ, float farZ) const;

    // Build view offset matrix (head → eye transform)
    glm::mat4 viewOffset() const;

    // Actual render resolution = recommended × renderScale
    uint32_t renderWidth(float renderScale) const;
    uint32_t renderHeight(float renderScale) const;
};

struct VRHeadPose {
    glm::vec3 position{0.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    bool valid = false;

    glm::mat4 viewMatrix() const;
};

enum class TrackingOrigin : uint32_t {
    Seated = 0,
    Standing = 1,
};

struct VRConfig {
    float ipd = 0.063f;
    float renderScale = 0.5f;
    float worldScale = 1.0f;
    float refreshRate = 90.0f;
    TrackingOrigin trackingOrigin = TrackingOrigin::Standing;
};

struct VRSystem {
    bool enabled = false;
    uint32_t eyeCount = 1;

    VRConfig config;
    VRHeadPose headPose;
    std::array<VREyeParams, 2> eyes;

    // World orientation offset (tracking space → game world).
    // Set from Java each frame; encodes player body yaw, stick turn,
    // mouse yaw, and character pose (elytra roll, swimming, etc.).
    glm::quat worldOrientation{1.0f, 0.0f, 0.0f, 0.0f};

    // World position offset (tracking-space origin in game world).
    // When recenter is called, this is set to the current headPose.position
    // so subsequent physical movement is relative to that point.
    glm::vec3 worldPosition{0.0f};

    // Recenter: snapshot current head position/yaw as the new origin.
    void recenter();

    // Simulation mode: populate eyes[] from window size, FOV, and config.ipd
    void updateSimulation(uint32_t windowWidth, uint32_t windowHeight, float fovY);

    // Update from real OpenXR data (called each frame when XR is active)
    void updateFromOpenXR(const VRHeadPose &head, const VREyeParams inEyes[2]);

    // Actual per-eye render resolution (after renderScale)
    uint32_t eyeRenderWidth() const;
    uint32_t eyeRenderHeight() const;

    // Controller input (updated from OpenXR each frame)
    std::array<VRControllerState, 2> controllers;  // 0=left, 1=right

    // Performance stats (updated each frame)
    VRPerformanceStats perfStats;

    // Eye gaze foveated centre in normalised coords (0.5, 0.5 = screen centre)
    glm::vec2 gazePoint{0.5f, 0.5f};
    bool gazeValid = false;
};
