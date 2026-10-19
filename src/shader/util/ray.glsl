#ifndef RAY_GLSL
#define RAY_GLSL

#include "util/ray_payloads.glsl"

#if defined(MCVR_RAY_MATERIAL_PAYLOAD) && MCVR_RAY_MATERIAL_PAYLOAD
// No storage-image round trip is needed for the payload material path.
#elif defined(VPT_MATERIAL_STATE_BINDING)
layout(set = 5, binding = VPT_MATERIAL_STATE_BINDING, rgba16f) uniform image2DArray rayMaterialStateImage;
#elif defined(ADV_MATERIAL_STATE_BINDING)
layout(set = 5, binding = ADV_MATERIAL_STATE_BINDING, rgba16f) uniform image2DArray rayMaterialStateImage;
#else
layout(set = 5, binding = 7, rgba16f) uniform image2DArray rayMaterialStateImage;
#endif

const uint rayBounceMask = 0xFFu;
const uint rayInsideBoatBit = 1u << 8u;
const uint rayStopBit = 1u << 9u;
const uint rayContinueBit = 1u << 10u;
const uint rayNoisyBit = 1u << 11u;
const uint rayLobeShift = 12u;
const uint rayLobeMask = 0x3u << rayLobeShift;
const uint raySkipFogBit = 1u << 14u;
const uint rayIgnoreWaterSelfBit = 1u << 15u;
const uint rayCaptureSurfaceBit = 1u << 16u;
const uint raySurfaceCacheWrittenBit = 1u << 17u;
const uint raySurfaceCacheTargetSecondaryBit = 1u << 18u;
const uint rayIndirectVolumetricCloudBit = 1u << 19u;
const uint rayFirstContinuationHitBit = 1u << 20u;

ivec3 rayMaterialStateCoord(int layer) {
    return ivec3(ivec2(gl_LaunchIDEXT.xy), layer);
}

void raySetBounce(inout MainRay ray, uint bounce) {
    ray.stateBits = (ray.stateBits & ~rayBounceMask) | (bounce & rayBounceMask);
}

uint rayBounce(MainRay ray) {
    return ray.stateBits & rayBounceMask;
}

void raySetInsideBoat(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayInsideBoatBit) : (ray.stateBits & ~rayInsideBoatBit);
}

bool rayInsideBoat(MainRay ray) {
    return (ray.stateBits & rayInsideBoatBit) != 0u;
}

void raySetStop(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayStopBit) : (ray.stateBits & ~rayStopBit);
}

bool rayShouldStop(MainRay ray) {
    return (ray.stateBits & rayStopBit) != 0u;
}

void raySetContinue(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayContinueBit) : (ray.stateBits & ~rayContinueBit);
}

bool rayShouldContinue(MainRay ray) {
    return (ray.stateBits & rayContinueBit) != 0u;
}

void raySetNoisy(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayNoisyBit) : (ray.stateBits & ~rayNoisyBit);
}

bool rayIsNoisy(MainRay ray) {
    return (ray.stateBits & rayNoisyBit) != 0u;
}

void raySetSkipFog(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | raySkipFogBit) : (ray.stateBits & ~raySkipFogBit);
}

bool raySkipFog(MainRay ray) {
    return (ray.stateBits & raySkipFogBit) != 0u;
}

void raySetIgnoreWaterSelf(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayIgnoreWaterSelfBit) : (ray.stateBits & ~rayIgnoreWaterSelfBit);
}

bool rayIgnoreWaterSelf(MainRay ray) {
    return (ray.stateBits & rayIgnoreWaterSelfBit) != 0u;
}

void raySetCaptureSurface(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | rayCaptureSurfaceBit) : (ray.stateBits & ~rayCaptureSurfaceBit);
}

bool rayCaptureSurface(MainRay ray) {
    return (ray.stateBits & rayCaptureSurfaceBit) != 0u;
}

void raySetSurfaceCacheWritten(inout MainRay ray, bool enabled) {
    ray.stateBits =
        enabled ? (ray.stateBits | raySurfaceCacheWrittenBit) : (ray.stateBits & ~raySurfaceCacheWrittenBit);
}

bool raySurfaceCacheWritten(MainRay ray) {
    return (ray.stateBits & raySurfaceCacheWrittenBit) != 0u;
}

void raySetSurfaceCacheTargetSecondary(inout MainRay ray, bool enabled) {
    ray.stateBits = enabled ? (ray.stateBits | raySurfaceCacheTargetSecondaryBit) :
                              (ray.stateBits & ~raySurfaceCacheTargetSecondaryBit);
}

bool raySurfaceCacheTargetSecondary(MainRay ray) {
    return (ray.stateBits & raySurfaceCacheTargetSecondaryBit) != 0u;
}

void raySetIndirectVolumetricCloud(inout MainRay ray, bool enabled) {
    ray.stateBits =
        enabled ? (ray.stateBits | rayIndirectVolumetricCloudBit) : (ray.stateBits & ~rayIndirectVolumetricCloudBit);
}

bool rayUseIndirectVolumetricCloud(MainRay ray) {
    return (ray.stateBits & rayIndirectVolumetricCloudBit) != 0u;
}

void raySetFirstContinuationHit(inout MainRay ray, bool enabled) {
    ray.stateBits =
        enabled ? (ray.stateBits | rayFirstContinuationHitBit) : (ray.stateBits & ~rayFirstContinuationHitBit);
}

bool rayFirstContinuationHit(MainRay ray) {
    return (ray.stateBits & rayFirstContinuationHitBit) != 0u;
}

void raySetLobeType(inout MainRay ray, uint lobeType) {
    ray.stateBits = (ray.stateBits & ~rayLobeMask) | ((lobeType & 0x3u) << rayLobeShift);
}

uint rayLobeType(MainRay ray) {
    return (ray.stateBits & rayLobeMask) >> rayLobeShift;
}

void rayClearMaterial(inout MainRay ray) {
#if defined(MCVR_RAY_MATERIAL_PAYLOAD) && MCVR_RAY_MATERIAL_PAYLOAD
    ray.materialAlbedo = vec4(0.0);
    ray.materialF0Roughness = vec4(0.0);
    ray.materialProperties = vec4(0.0);
#else
    imageStore(rayMaterialStateImage, rayMaterialStateCoord(0), vec4(0.0));
    imageStore(rayMaterialStateImage, rayMaterialStateCoord(1), vec4(0.0));
    imageStore(rayMaterialStateImage, rayMaterialStateCoord(2), vec4(0.0));
#endif
}

void rayStoreMaterial(inout MainRay ray,
                      vec4 albedoValue,
                      vec3 f0,
                      float roughness,
                      float metallic,
                      float transmission,
                      float ior,
                      float emission) {
#if defined(MCVR_RAY_MATERIAL_PAYLOAD) && MCVR_RAY_MATERIAL_PAYLOAD
    ray.materialAlbedo = albedoValue;
    ray.materialF0Roughness = vec4(f0, roughness);
    ray.materialProperties = vec4(metallic, transmission, ior, emission);
#else
    imageStore(rayMaterialStateImage, rayMaterialStateCoord(0), albedoValue);
    imageStore(rayMaterialStateImage, rayMaterialStateCoord(1), vec4(f0, roughness));
    imageStore(rayMaterialStateImage, rayMaterialStateCoord(2), vec4(metallic, transmission, ior, emission));
#endif
}

void rayStoreAux(inout MainRay ray, vec2 aux) {
    ray.pad0 = packHalf2x16(aux);
}

vec2 rayLoadAux(MainRay ray) {
    return unpackHalf2x16(ray.pad0);
}

MaterialInfo rayLoadMaterial(MainRay ray) {
    MaterialInfo mat;
#if defined(MCVR_RAY_MATERIAL_PAYLOAD) && MCVR_RAY_MATERIAL_PAYLOAD
    vec4 albedoValue = ray.materialAlbedo;
    vec4 f0Roughness = ray.materialF0Roughness;
    vec4 materialValues = ray.materialProperties;
#else
    memoryBarrierImage();
    vec4 albedoValue = imageLoad(rayMaterialStateImage, rayMaterialStateCoord(0));
    vec4 f0Roughness = imageLoad(rayMaterialStateImage, rayMaterialStateCoord(1));
    vec4 materialValues = imageLoad(rayMaterialStateImage, rayMaterialStateCoord(2));
#endif

    mat.albedoValue = albedoValue;
    mat.f0 = f0Roughness.rgb;
    mat.roughness = f0Roughness.a;
    mat.metallic = materialValues.x;
    mat.transmission = materialValues.y;
    mat.ior = materialValues.z;
    mat.emission = materialValues.w;
    return mat;
}

#endif
