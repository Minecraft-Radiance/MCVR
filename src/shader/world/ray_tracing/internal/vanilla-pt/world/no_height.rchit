#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common/shared.hpp"
#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "common/fft_water.glsl"
#include "util/height_map.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 8) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

#include "common/celestial.glsl"

layout(set = 5, binding = 0) uniform sampler2D transLUT;
layout(set = 5, binding = 2) uniform samplerCube skyFull;

#include "common/volumetric_cloud.glsl"

layout(std430, buffer_reference, buffer_reference_align = 4) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"
#include "util/vista_material.glsl"
#include "common/constants.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

struct SampledSurface {
    vec2 uv;
    float depth;
    vec3 worldPos;
    vec3 geometricNormal;
    vec3 shadingNormal;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    vec3 tint;
    LabPBRMat mat;
};

vec3 applyNormalMapToBasis(vec3 matNormal, vec3 tangent, vec3 bitangent, vec3 geometricNormal, vec3 viewDir) {
    if (any(isnan(matNormal))) { return geometricNormal; }

    vec3 correctedLocalNormal = matNormal;
    correctedLocalNormal.y = -correctedLocalNormal.y;

    vec3 normal = normalizeF(tangent * correctedLocalNormal.x + bitangent * correctedLocalNormal.y +
                                 geometricNormal * correctedLocalNormal.z,
                             geometricNormal);
    float NdotV = dot(normal, viewDir);
    if (NdotV >= 0.99999) { return normal; }

    vec3 edgeNormal = normal - viewDir * NdotV;
    float edgeNormalLen2 = dot(edgeNormal, edgeNormal);
    if (edgeNormalLen2 <= 1e-10) { return geometricNormal; }

    float weight = 1.0 - NdotV;
    weight = sin(min(weight, PI * 0.5));
    weight = clamp(min(max(NdotV, dot(viewDir, geometricNormal)), 1.0 - weight), 0.0, 1.0);

    float tangentWeight2 = max(1.0 - weight * weight, 0.0);
    if (tangentWeight2 <= 1e-10) { return geometricNormal; }

    return viewDir * weight + edgeNormal * inversesqrt(edgeNormalLen2 / tangentWeight2);
}

void sampleSurfaceState(bool useTexture,
                        uint textureID,
                        TextureMapEntry textureMap,
                        vec2 atlasUvMin,
                        vec2 atlasUvMax,
                        vec2 uv,
                        float lod,
                        uint alphaMode,
                        vec4 colorLayerValue,
                        vec3 colorLayer,
                        vec3 glint,
                        bool useOverlay,
                        ivec2 overlayUV,
                        vec3 dPduWorld,
                        vec3 dPdvWorld,
                        vec3 baseGeoNormal,
                        bool hasHeightMap,
                        float maxDepthWorld,
                        HeightMapHit localHit,
                        vec3 worldPos,
                        vec3 viewDir,
                        bool isWaterMaterial,
                        bool isFftWaterSurface,
                        out SampledSurface surface) {
    vec4 albedoValue = vec4(1.0);
    vec4 specularValue = LABPBR_DEFAULT_SPECULAR;
    vec4 normalValue = LABPBR_DEFAULT_NORMAL;

    if (useTexture) {
        albedoValue = resolveSurfaceTextureColor(sampleTexture(textures[nonuniformEXT(textureID)], uv, lod, false), alphaMode);
        albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
        specularValue = textureMap.specular >= 0 ?
                            samplePBRSpecularTexture(textures[nonuniformEXT(textureMap.specular)], uv, atlasUvMin,
                                                     atlasUvMax, lod, VPT_PBR_SAMPLING_MODE) :
                            LABPBR_DEFAULT_SPECULAR;
        normalValue = textureMap.normal >= 0 ? samplePBRTexture(textures[nonuniformEXT(textureMap.normal)], uv,
                                                                atlasUvMin, atlasUvMax, lod, VPT_PBR_SAMPLING_MODE) :
                                               LABPBR_DEFAULT_NORMAL;
    }

    vec3 tint = albedoValue.rgb * colorLayer + glint;
    if (useOverlay) {
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUBO.overlayTextureID)], overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    vec3 geometricNormal = baseGeoNormal;
    vec3 shadingNormal = geometricNormal;
    bool useWaterMaterial = isWaterMaterial;
    if (useWaterMaterial) {
        vec3 waterTint = vec3(0.95, 0.98, 1.0);
        albedoValue.rgb = waterTint;
        tint = waterTint;
        mat.f0 = vec3(0.02);
        mat.albedo = waterTint;
        mat.roughness = 0.01;
        mat.metallic = 0.0;
        mat.transmission = 1.0;
        mat.ior = 1.333;

        if (isFftWaterSurface) {
            vec3 absWorldPos = worldPos + vec3(worldUBO.cameraPos.xyz);
            vec3 waterCoordNormal = baseGeoNormal.y >= 0.0 ? baseGeoNormal : -baseGeoNormal;
            vec2 waterCoord = fftWaterSurfaceCoord(absWorldPos, dPduWorld, dPdvWorld, waterCoordNormal);
            FftWaterSample waterSample = sampleFftWater(waterCoord, worldUBO.gameTime);
            vec3 tangent, bitangent;
            fftWaterStableBasis(baseGeoNormal, tangent, bitangent);
            vec3 localWaterNormal = normalize(vec3(-waterSample.slope.x, waterSample.slope.y, 1.0));
            mat.roughness = clamp(0.005 + 0.012 * min(length(waterSample.slope), 0.45), 0.005, 0.022);
            shadingNormal = applyNormalMapToBasis(localWaterNormal, tangent, bitangent, baseGeoNormal, viewDir);
        }
    }

    if (!isFftWaterSurface && VPT_PBR_SAMPLING_MODE != 0u && hasHeightMap && textureMap.normal >= 0) {
        geometricNormal =
            sampleNormal(textures[nonuniformEXT(textureMap.normal)], uv, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld,
                         baseGeoNormal, 0, VPT_PBR_SAMPLING_MODE, maxDepthWorld, viewDir);
    }

    if (!isFftWaterSurface) {
        vec3 tangent, bitangent;
        tangent = normalizeF(dPduWorld - geometricNormal * dot(geometricNormal, dPduWorld),
                             abs(geometricNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0));
        bitangent = normalizeF(cross(geometricNormal, tangent), dPdvWorld);
        tangent = normalizeF(cross(bitangent, geometricNormal), tangent);
        shadingNormal = applyNormalMapToBasis(mat.normal, tangent, bitangent, geometricNormal, viewDir);
    }

    surface.uv = uv;
    surface.depth = localHit.depth;
    surface.worldPos = worldPos;
    surface.geometricNormal = geometricNormal;
    surface.shadingNormal = shadingNormal;
    surface.albedoValue = albedoValue;
    surface.specularValue = specularValue;
    surface.normalValue = normalValue;
    surface.tint = tint;
    surface.mat = mat;
}

vec3 basePlaneWorldPosAtUv(vec2 uv, vec2 referenceUv, vec3 referenceWorldPos, vec3 dPduWorld, vec3 dPdvWorld) {
    vec2 uvOffset = uv - referenceUv;
    return referenceWorldPos + dPduWorld * uvOffset.x + dPdvWorld * uvOffset.y;
}

void initVanillaParallaxMiss(vec2 uv, float depth, vec3 baseNormal, out HeightMapHit hit) {
    hit.hit = false;
    hit.t = INF_DISTANCE;
    hit.uv = uv;
    hit.depth = depth;
}

bool traceNearestHeightMapCapped(sampler2D tex,
                                 vec2 minUV,
                                 vec2 maxUV,
                                 vec2 uv,
                                 float depth,
                                 vec3 worldDir,
                                 vec3 dPdu,
                                 vec3 dPdv,
                                 vec3 baseNormal,
                                 float maxDepth,
                                 int maxTraceSteps,
                                 out HeightMapHit hit) {
    initVanillaParallaxMiss(uv, depth, baseNormal, hit);
    if (maxTraceSteps <= 0 || maxDepth <= heightMapMinWorldDepth) { return false; }

    ivec2 size = textureSize(tex, 0);
    if (size.x <= 0 || size.y <= 0) { return false; }

    vec2 boundsMin = heightMapMinUV(minUV, maxUV, size);
    vec2 boundsMax = heightMapMaxUV(minUV, maxUV, size);
    uv = wrapUvInRect(uv, boundsMin, boundsMax);

    vec2 rateUV = directionToRateUv(worldDir, dPdu, dPdv);
    float depthRate = dot(worldDir, -baseNormal);

    ivec2 atlasTexelMin = clampTexelCoord(ivec2(floor(min(minUV, maxUV) * vec2(size))), size);
    ivec2 atlasTexelMax = clampTexelCoord(ivec2(ceil(max(minUV, maxUV) * vec2(size)) - vec2(1.0)), size);
    ivec2 texel = wrapTexelInRect(ivec2(floor(uv * vec2(size))), atlasTexelMin, atlasTexelMax);

    int maxSteps = min(maxTraceSteps, heightMapNearestMaxSteps);
    if (maxSteps <= 0) { return false; }

    float tCurrent = 0.0;
    vec2 uvCurrent = uv;
    float depthCurrent = depth;

    int step = 0;
    while (step < maxSteps) {
        if (depthCurrent < -heightMapTraceBias) { break; }

        float surfaceDepth = sampleHeightDepthNearest(tex, texel, atlasTexelMin, atlasTexelMax, uvCurrent, maxDepth);
        if (depthCurrent >= surfaceDepth - heightMapTraceBias) {
            hit.hit = true;
            hit.t = tCurrent;
            hit.uv = uvCurrent;
            hit.depth = max(depthCurrent, surfaceDepth);
            return true;
        }

        float tTop = INF_DISTANCE;
        if (depthRate > 1e-6 && surfaceDepth > depthCurrent) { tTop = (surfaceDepth - depthCurrent) / depthRate; }

        float tU = INF_DISTANCE;
        float tV = INF_DISTANCE;
        bool stepU = false;
        bool stepV = false;

        if (rateUV.x > 1e-9) {
            float boundary = (float(texel.x + 1)) / float(size.x);
            tU = (boundary - uvCurrent.x) / rateUV.x;
            stepU = true;
        } else if (rateUV.x < -1e-9) {
            float boundary = float(texel.x) / float(size.x);
            tU = (boundary - uvCurrent.x) / rateUV.x;
            stepU = true;
        }

        if (rateUV.y > 1e-9) {
            float boundary = (float(texel.y + 1)) / float(size.y);
            tV = (boundary - uvCurrent.y) / rateUV.y;
            stepV = true;
        } else if (rateUV.y < -1e-9) {
            float boundary = float(texel.y) / float(size.y);
            tV = (boundary - uvCurrent.y) / rateUV.y;
            stepV = true;
        }

        float tAbove = INF_DISTANCE;
        if (depthRate < -1e-6) { tAbove = -depthCurrent / depthRate; }

        float tBoundary = min(tU, tV);
        if (tTop <= min(tBoundary, tAbove) + heightMapTraceBias && tTop < INF_DISTANCE) {
            tCurrent += max(tTop, 0.0);
            hit.hit = true;
            hit.t = tCurrent;
            hit.uv = uvCurrent + rateUV * max(tTop, 0.0);
            hit.depth = surfaceDepth;
            return true;
        }

        if (tAbove <= tBoundary + heightMapTraceBias) { break; }
        if (tBoundary == INF_DISTANCE) { break; }

        float dt = max(tBoundary, 0.0);
        tCurrent += dt;

        vec2 boundaryUv = uvCurrent + rateUV * dt;
        float boundaryDepth = depthCurrent + depthRate * dt;
        bool cornerStep = stepU && stepV && abs(tU - tV) <= heightMapTraceBias;
        if (cornerStep) {
            ivec2 texelD = texel + ivec2(rateUV.x > 0.0 ? 1 : -1, rateUV.y > 0.0 ? 1 : -1);
            texel = wrapTexelInRect(texelD, atlasTexelMin, atlasTexelMax);
            uvCurrent = wrapBoundaryUvInRect(boundaryUv, texelD, atlasTexelMin, atlasTexelMax, boundsMin, boundsMax);
            depthCurrent = boundaryDepth;
            ++step;
            continue;
        }

        bool processU = false;
        if (tU <= tV + heightMapTraceBias) {
            processU = tU < tV - heightMapTraceBias ? true : abs(rateUV.x) >= abs(rateUV.y);
        }

        ivec2 nextTexel = texel;
        bool enteredTop = false;

        if (processU) {
            nextTexel += ivec2(rateUV.x > 0.0 ? 1 : -1, 0);
            boundaryUv =
                wrapBoundaryUvInRect(boundaryUv, nextTexel, atlasTexelMin, atlasTexelMax, boundsMin, boundsMax);
            nextTexel = wrapTexelInRect(nextTexel, atlasTexelMin, atlasTexelMax);

            float nextDepth =
                sampleHeightDepthNearest(tex, nextTexel, atlasTexelMin, atlasTexelMax, boundaryUv, maxDepth);
            if (shouldContinueTexelTop(surfaceDepth, nextDepth, boundaryDepth)) { enteredTop = true; }
        }

        if (!processU) {
            nextTexel = texel + ivec2(0, rateUV.y > 0.0 ? 1 : -1);
            boundaryUv =
                wrapBoundaryUvInRect(boundaryUv, nextTexel, atlasTexelMin, atlasTexelMax, boundsMin, boundsMax);
            nextTexel = wrapTexelInRect(nextTexel, atlasTexelMin, atlasTexelMax);

            float nextDepth =
                sampleHeightDepthNearest(tex, nextTexel, atlasTexelMin, atlasTexelMax, boundaryUv, maxDepth);
            if (shouldContinueTexelTop(surfaceDepth, nextDepth, boundaryDepth)) { enteredTop = true; }
        }

        if (enteredTop) {
            hit.hit = true;
            hit.t = tCurrent;
            hit.uv = boundaryUv;
            hit.depth = surfaceDepth;
            return true;
        }

        texel = nextTexel;
        uvCurrent = boundaryUv;
        depthCurrent = boundaryDepth;
        ++step;
    }

    return false;
}

bool traceBilinearHeightMapCapped(sampler2D tex,
                                  vec2 minUV,
                                  vec2 maxUV,
                                  vec2 uv,
                                  float depth,
                                  vec3 worldDir,
                                  vec3 dPdu,
                                  vec3 dPdv,
                                  vec3 baseNormal,
                                  float maxDepth,
                                  int maxTraceSteps,
                                  out HeightMapHit hit) {
    initVanillaParallaxMiss(uv, depth, baseNormal, hit);
    if (maxTraceSteps <= 0 || maxDepth <= heightMapMinWorldDepth) { return false; }

    ivec2 size = textureSize(tex, 0);
    if (size.x <= 0 || size.y <= 0) { return false; }

    vec2 boundsMin = heightMapMinUV(minUV, maxUV, size);
    vec2 boundsMax = heightMapMaxUV(minUV, maxUV, size);
    uv = wrapUvInRect(uv, boundsMin, boundsMax);

    vec2 rateUV = directionToRateUv(worldDir, dPdu, dPdv);
    float depthRate = dot(worldDir, -baseNormal);
    float texelTravel = max(abs(rateUV.x) * float(size.x), abs(rateUV.y) * float(size.y));
    float depthTravel = abs(depthRate) / max(maxDepth, heightMapMinWorldDepth) * float(max(size.x, size.y));
    float stepWorld = 0.5 / max(max(texelTravel, depthTravel), 1e-4);

    float tPrev = 0.0;
    vec2 uvPrev = uv;
    float depthPrev = depth;
    float surfacePrev = sampleHeightDepth(tex, uvPrev, boundsMin, boundsMax, 0, 1u, maxDepth);
    float fPrev = depthPrev - surfacePrev;
    if (fPrev >= -heightMapTraceBias) {
        hit.hit = true;
        hit.t = 0.0;
        hit.uv = uvPrev;
        hit.depth = surfacePrev;
        return true;
    }

    int maxSteps = min(maxTraceSteps, heightMapBilinearMaxSteps);
    for (int step = 0; step < maxSteps; ++step) {
        float tCurr = tPrev + stepWorld;
        vec2 uvCurr = uv + rateUV * tCurr;
        float depthCurr = depth + depthRate * tCurr;
        if (depthCurr < -heightMapTraceBias) { break; }

        float surfaceCurr = sampleHeightDepth(tex, uvCurr, boundsMin, boundsMax, 0, 1u, maxDepth);
        float fCurr = depthCurr - surfaceCurr;
        if (fCurr >= -heightMapTraceBias) {
            float lo = tPrev;
            float hi = tCurr;
            for (int refine = 0; refine < heightMapBilinearBinarySteps; ++refine) {
                float mid = 0.5 * (lo + hi);
                vec2 uvMid = uv + rateUV * mid;
                float depthMid = depth + depthRate * mid;
                float surfaceMid = sampleHeightDepth(tex, uvMid, boundsMin, boundsMax, 0, 1u, maxDepth);
                if (depthMid >= surfaceMid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            float tHit = hi;
            vec2 uvHit = wrapUvInRect(uv + rateUV * tHit, boundsMin, boundsMax);
            float depthHit = sampleHeightDepth(tex, uvHit, boundsMin, boundsMax, 0, 1u, maxDepth);
            hit.hit = true;
            hit.t = tHit;
            hit.uv = uvHit;
            hit.depth = depthHit;
            return true;
        }

        tPrev = tCurr;
        uvPrev = uvCurr;
        depthPrev = depthCurr;
        fPrev = fCurr;
    }

    return false;
}

bool traceHeightMapCapped(sampler2D tex,
                          vec2 minUV,
                          vec2 maxUV,
                          vec2 uv,
                          float depth,
                          vec3 worldDir,
                          vec3 dPdu,
                          vec3 dPdv,
                          vec3 baseNormal,
                          float maxDepth,
                          uint samplingMode,
                          int maxTraceSteps,
                          out HeightMapHit hit) {
    maxTraceSteps = heightMapTraceStepBudget(tex, minUV, maxUV, maxTraceSteps);
    if (samplingMode == 0u) {
        return traceNearestHeightMapCapped(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal, maxDepth,
                                           maxTraceSteps, hit);
    }

    return traceBilinearHeightMapCapped(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal, maxDepth,
                                        maxTraceSteps, hit);
}

bool traceLocalHeightIntersectionAndExit(int normalTextureID,
                                         vec2 atlasUvMin,
                                         vec2 atlasUvMax,
                                         vec3 dPduWorld,
                                         vec3 dPdvWorld,
                                         vec3 baseGeoNormal,
                                         float maxDepthWorld,
                                         SampledSurface surface,
                                         vec3 worldDir,
                                         int maxTraceSteps,
                                         out HeightMapHit hit,
                                         out vec2 exitUv,
                                         out float exitDepth,
                                         out float exitDistance) {
    initVanillaParallaxMiss(surface.uv, surface.depth, baseGeoNormal, hit);
    exitUv = surface.uv;
    exitDepth = surface.depth;
    exitDistance = 0.0;
    if (normalTextureID < 0 || maxDepthWorld <= heightMapMinWorldDepth) { return false; }
    ivec2 heightMapSize = textureSize(textures[nonuniformEXT(normalTextureID)], 0);
    if (heightMapSize.x <= 0 || heightMapSize.y <= 0) { return false; }
    vec2 traceAtlasUvMin = heightMapMinUV(atlasUvMin, atlasUvMax, heightMapSize);
    vec2 traceAtlasUvMax = heightMapMaxUV(atlasUvMin, atlasUvMax, heightMapSize);
    int traceStepBudget = heightMapTraceStepBudget(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin,
                                                   traceAtlasUvMax, maxTraceSteps);
    vec2 traceSurfaceUv = wrapUvInRect(surface.uv, traceAtlasUvMin, traceAtlasUvMax);

    initVanillaParallaxMiss(traceSurfaceUv, surface.depth, baseGeoNormal, hit);
    exitUv = traceSurfaceUv;

    float startBias = 2.0 * heightMapTraceBias;
    vec2 rateUV = directionToRateUv(worldDir, dPduWorld, dPdvWorld);
    float depthRate = dot(worldDir, -baseGeoNormal);
    vec2 uv = traceSurfaceUv + rateUV * startBias;
    float depth = surface.depth + depthRate * startBias;
    if (dot(surface.geometricNormal, baseGeoNormal) > 0.5 && depthRate < 0.0) {
        depth = min(depth, surface.depth - startBias);
    }

    bool fallbackToBilinear =
        VPT_PBR_SAMPLING_MODE == 0u && shouldFallbackNearestSecondaryToBilinear(worldDir, baseGeoNormal);
    bool localHit;
    if (depthRate < -1e-6) {
        float maxTraceDistance = max(depth / -depthRate, 0.0);
        if (fallbackToBilinear) {
            localHit = traceBilinearHeightMapLimited(
                textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv, depth, worldDir,
                dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld, maxTraceDistance, traceStepBudget, hit);
        } else {
            localHit = traceHeightMapLimited(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax,
                                             uv, depth, worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                             maxTraceDistance, VPT_PBR_SAMPLING_MODE, traceStepBudget, hit);
        }
    } else if (fallbackToBilinear) {
        localHit = traceBilinearHeightMapCapped(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin,
                                                traceAtlasUvMax, uv, depth, worldDir, dPduWorld, dPdvWorld,
                                                baseGeoNormal, maxDepthWorld, traceStepBudget, hit);
    } else {
        localHit = traceHeightMapCapped(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv,
                                        depth, worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                        VPT_PBR_SAMPLING_MODE, traceStepBudget, hit);
    }

    if (localHit) { return true; }

    if (depthRate >= -1e-6) { return true; }

    float exitT = max(depth / -depthRate, 0.0);
    exitUv = wrapUvInRect(uv + rateUV * exitT, traceAtlasUvMin, traceAtlasUvMax);
    exitDepth = 0.0;
    exitDistance = exitT;
    return false;
}

vec3 sampleSurfaceDirectLight(SampledSurface surface,
                              vec3 viewDir,
                              vec2 referenceUv,
                              vec3 referenceWorldPos,
                              vec2 atlasUvMin,
                              vec2 atlasUvMax,
                              vec3 dPduWorld,
                              vec3 dPdvWorld,
                              vec3 baseGeoNormal,
                              bool traceLocalHeight,
                              int normalTextureID,
                              float maxDepthWorld,
                              bool isFftWaterSurface) {
    if (worldUBO.skyType != 1) { return vec3(0.0); }

    bool isOpaqueSurface = surface.mat.transmission <= EPS;
    vec3 lightDir = celestialSunDirection();
    if (lightDir.y < 0.0) { lightDir = -lightDir; }

    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, 3000.0);
    float sampledLightNoL = dot(sampledLightDir, surface.geometricNormal);
    if (isOpaqueSurface && sampledLightNoL <= 0.0) { return vec3(0.0); }

    float lightPdf;
    vec3 lightBRDF = DisneyEval(surface.mat, viewDir, surface.shadingNormal, sampledLightDir, lightPdf);
    if (lightPdf <= 1e-6) { return vec3(0.0); }

    vec3 shadowOriginPos = surface.worldPos;
    vec3 shadowOriginGeomNormal = surface.geometricNormal;
    if (traceLocalHeight) {
        HeightMapHit shadowSelfHit;
        vec2 exitUv;
        float exitDepth;
        float exitDistance;
        if (traceLocalHeightIntersectionAndExit(
                normalTextureID, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld, surface,
                sampledLightDir, VPT_PARALLAX_SECONDARY_MAX_STEPS, shadowSelfHit, exitUv, exitDepth, exitDistance)) {
            return vec3(0.0);
        }
        shadowOriginPos = basePlaneWorldPosAtUv(exitUv, referenceUv, referenceWorldPos, dPduWorld, dPdvWorld) -
                          baseGeoNormal * exitDepth;
        shadowOriginGeomNormal = baseGeoNormal;
    }

    shadowRay.radiance = vec3(0.0);
    shadowRay.throughput = vec3(1.0);
    shadowRay.insideBoat = rayInsideBoat(mainRay) ? 1u : 0u;
    shadowRay.pad0 = 0u;

    vec3 baseWorldPos = shadowOriginPos;
    vec3 exitGeoNormal = shadowOriginGeomNormal;
    vec3 visibilityNormal = dot(sampledLightDir, exitGeoNormal) >= 0.0 ? exitGeoNormal : -exitGeoNormal;
    vec3 shadowOrigin = baseWorldPos + visibilityNormal * 0.0002;
    uint shadowMask = WORLD_MASK | PLAYER_MASK;
    if (VPT_CLOUD_MODE != 2u) { shadowMask |= CLOUD_MASK; }
    traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, shadowMask, 0, 0, 0,
                shadowOrigin, 0.0001, sampledLightDir, 1000.0, 1);

    float progress = skyUBO.rainGradient;
    vec3 lightRadiance = shadowRay.radiance * mainRay.throughput * lightBRDF;
    if (VPT_CLOUD_MODE == 2u) {
        vec3 absoluteShadowOriginPos = baseWorldPos + vec3(worldUBO.cameraPos.xyz);
        float cloudVisibility = volumetricCloudLightVisibility(absoluteShadowOriginPos, lightDir,
                                                               max(VPT_VOLUMETRIC_CLOUD_LIGHT_STEPS, 1), 0.175);
        lightRadiance *= cloudVisibility;
    }
    return mix(lightRadiance, vec3(0.0), progress);
}

bool loadPreviousScenePos(uint geometryBufferIndex, uint primitiveID, vec3 baryCoords, out vec3 prevScenePos) {
    uint64_t lastPbrAddr = lastPbrBufferAddrs.addrs[geometryAddressIndex(geometryBufferIndex)];
    uint64_t lastIndexAddr = lastIndexBufferAddrs.addrs[geometryAddressIndex(geometryBufferIndex)];
    if (lastPbrAddr == 0 || lastIndexAddr == 0) { return false; }

    IndexBuffer lastIndexBuffer = IndexBuffer(lastIndexAddr);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = lastIndexBuffer.indices[indexBaseID];
    uint i1 = lastIndexBuffer.indices[indexBaseID + 1u];
    uint i2 = lastIndexBuffer.indices[indexBaseID + 2u];

    vec3 p0, p1, p2;
    if (isChunkGeometryBufferIndex(geometryBufferIndex)) {
        ChunkPBRBuffer lastPbrBuffer = ChunkPBRBuffer(lastPbrAddr);
        p0 = vec3(lastPbrBuffer.vertices[i0].posX, lastPbrBuffer.vertices[i0].posY, lastPbrBuffer.vertices[i0].posZ);
        p1 = vec3(lastPbrBuffer.vertices[i1].posX, lastPbrBuffer.vertices[i1].posY, lastPbrBuffer.vertices[i1].posZ);
        p2 = vec3(lastPbrBuffer.vertices[i2].posX, lastPbrBuffer.vertices[i2].posY, lastPbrBuffer.vertices[i2].posZ);
    } else {
        EntityPBRBuffer lastPbrBuffer = EntityPBRBuffer(lastPbrAddr);
        p0 = lastPbrBuffer.vertices[i0].pos;
        p1 = lastPbrBuffer.vertices[i1].pos;
        p2 = lastPbrBuffer.vertices[i2].pos;
    }
    vec3 prevLocalPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;
    mat4 lastModelMat = lastObjToWorldMats.mat[gl_InstanceCustomIndexEXT];
    prevScenePos = mat3(lastModelMat) * prevLocalPos + lastModelMat[3].xyz;
    return true;
}

void main() {
    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;
    uint geometryBufferIndex = getGeometryBufferIndex(instanceID, geometryID);

    uint i0, i1, i2;
    MaterialVertex m0, m1, m2;
    loadTriangleIndices(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2);
    loadTriangleMaterial(geometryBufferIndex, i0, i1, i2, m0, m1, m2);
    PositionVertex p0, p1, p2;
    loadTrianglePositions(geometryBufferIndex, i0, i1, i2, p0, p1, p2);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 planeHitWorldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 baseViewDir = -mainRay.direction;

    uint packedData = m0.packedData;
    bool useColorLayer = hasColorLayer(packedData);
    bool useTexture = hasTexture(packedData);
    bool useGlint = hasGlint(packedData);
    bool useOverlay = hasOverlay(packedData);
    vec4 colorLayerValue =
        useColorLayer ? baryCoords.x * m0.colorLayer + baryCoords.y * m1.colorLayer + baryCoords.z * m2.colorLayer :
                        vec4(1.0);
    vec3 colorLayer = colorLayerValue.rgb;
    if (hasVistaMaterial(m0) && !hasWaterMaterial(m0) && worldUBO.vistaTextureID != 0u) {
        vec2 vistaUv = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        vec3 normal = normalize(cross(p1.pos - p0.pos, p2.pos - p0.pos));
        float footprint = max(mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread, 0.0) /
            max(abs(dot(normal, gl_ObjectRayDirectionEXT)), 0.05);
        colorLayer = vistaAlbedo(textures[nonuniformEXT(worldUBO.vistaTextureID)], m0, vistaUv, colorLayer, footprint);
        colorLayerValue.rgb = colorLayer;
    }
    uint bounce = rayBounce(mainRay);

    float albedoEmission =
        baryCoords.x * m0.albedoEmission + baryCoords.y * m1.albedoEmission + baryCoords.z * m2.albedoEmission;
    uint textureID = m0.textureID;
    uint alphaMode = getAlphaMode(packedData);
    uint coordinate = getCoordinate(packedData);

    vec2 textureUV = vec2(0.0);
    TextureMapEntry textureMap = TextureMapEntry(-1, -1, -1);
    vec2 atlasUvMin = vec2(0.0);
    vec2 atlasUvMax = vec2(0.0);
    float lod = 0.0;
    vec3 dposdu = vec3(1.0, 0.0, 0.0);
    vec3 dposdv = vec3(0.0, 1.0, 0.0);
    bool hasHeightMapSurface = false;
    float maxDepthWorld = 0.0;
    mat3 objectToWorld = mat3(gl_ObjectToWorld3x4EXT);
    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 baseGeoNormal = normalizeF(normalMatrix * cross(p1.pos - p0.pos, p2.pos - p0.pos), vec3(0.0, 1.0, 0.0));
    vec3 planeGeoNormal = baseGeoNormal;
    vec3 dPduWorld = objectToWorld * dposdu;
    vec3 dPdvWorld = objectToWorld * dposdv;
    bool isWaterMaterial = false;
    bool hasFftWaterSurface = false;
    bool parallaxEnabled = false;
    bool useRealisticWaterSurface = VPT_WATER_SURFACE_MODE == 1u;

    if (useTexture) {
        textureMap = mapping.entries[textureID];
        textureUV = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        atlasUvMin = min(m0.textureUV, min(m1.textureUV, m2.textureUV));
        atlasUvMax = max(m0.textureUV, max(m1.textureUV, m2.textureUV));

        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        computedposduDv(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, dposdu, dposdv);
        lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        dPduWorld = objectToWorld * dposdu;
        dPdvWorld = objectToWorld * dposdv;
        planeGeoNormal = normalizeF(cross(dPduWorld, dPdvWorld), baseGeoNormal);
        baseGeoNormal = planeGeoNormal;
        if (dot(baseGeoNormal, baseViewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }

        if (textureMap.flag >= 0) {
            ivec4 flags = ivec4(
                round(sampleTexture(textures[nonuniformEXT(textureMap.flag)], textureUV, ceil(lod), false) * 255.0));
            isWaterMaterial = (flags.r & 0x1) > 0;
        }
        hasFftWaterSurface = useRealisticWaterSurface && isWaterMaterial && abs(planeGeoNormal.y) > 0.75;

        if (!isWaterMaterial && parallaxEnabled && bounce == 0u && textureMap.normal >= 0 && coordinate != 1u) {
            maxDepthWorld = heightMapMaxDepthWorld(atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld);
            hasHeightMapSurface =
                maxDepthWorld > heightMapMinWorldDepth && dot(baseViewDir, baseGeoNormal) > VPT_PARALLAX_MIN_VIEW_DOT;
        }
    } else {
        dPduWorld = objectToWorld * dposdu;
        dPdvWorld = objectToWorld * dposdv;
    }
    if (dot(baseGeoNormal, baseViewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }
    vec3 cameraOrigin = vec3(worldUBO.cameraEffectedViewMatInv * vec4(0.0, 0.0, 0.0, 1.0));
    bool traceLocalHeight = hasHeightMapSurface &&
                            (lod == 0.0 || distance(planeHitWorldPos, cameraOrigin) <= VPT_PARALLAX_CLOSE_DISTANCE) &&
                            !hasFftWaterSurface;

    HeightMapHit initialHit;
    initialHit.hit = false;
    initialHit.t = 0.0;
    initialHit.uv = textureUV;
    initialHit.depth = 0.0;
    if (traceLocalHeight) {
        HeightMapHit tracedInitialHit;
        if (traceHeightMapCapped(textures[nonuniformEXT(textureMap.normal)], atlasUvMin, atlasUvMax, textureUV, 0.0,
                                 gl_WorldRayDirectionEXT, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                 VPT_PBR_SAMPLING_MODE, VPT_PARALLAX_PRIMARY_MAX_STEPS, tracedInitialHit)) {
            initialHit = tracedInitialHit;
        }
    }

    vec3 hitWorldPos = planeHitWorldPos + gl_WorldRayDirectionEXT * initialHit.t;
    float actualHitT = gl_HitTEXT + initialHit.t;

    mainRay.hitT = actualHitT;
    mainRay.coneWidth += actualHitT * mainRay.coneSpread;
    mainRay.directLightRadiance = vec3(0.0);
    mainRay.hasPrevScenePos = 0u;
    if (bounce == 0u) {
        vec3 prevScenePos;
        if (loadPreviousScenePos(geometryBufferIndex, gl_PrimitiveID, baryCoords, prevScenePos)) {
            mainRay.prevScenePos = prevScenePos;
            mainRay.hasPrevScenePos = 1u;
        }
    }

    vec3 glint = vec3(0.0);
    if (useGlint) {
        vec2 glintUV = baryCoords.x * m0.glintUV + baryCoords.y * m1.glintUV + baryCoords.z * m2.glintUV;
        glintUV = (worldUBO.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = sampleTexture(textures[nonuniformEXT(m0.glintTexture)], glintUV, false).rgb;
    }
    glint *= glint;

    vec3 viewDir = -gl_WorldRayDirectionEXT;

    SampledSurface surface;
    sampleSurfaceState(useTexture, textureID, textureMap, atlasUvMin, atlasUvMax, initialHit.uv, lod, alphaMode,
                       colorLayerValue, colorLayer, glint, useOverlay, m0.overlayUV, dPduWorld, dPdvWorld,
                       baseGeoNormal, hasHeightMapSurface, maxDepthWorld, initialHit, hitWorldPos, viewDir,
                       isWaterMaterial, hasFftWaterSurface, surface);

    mainRay.normal = surface.shadingNormal;
    rayStoreMaterial(mainRay, surface.albedoValue, surface.mat.f0, surface.mat.roughness, surface.mat.metallic,
                     surface.mat.transmission, surface.mat.ior, surface.mat.emission);
    rayStoreAux(mainRay, isWaterMaterial ? vec2(VPT_FFT_WATER_ORIGIN_BIAS, 1.0) : vec2(0.0));

    SampledSurface currentSurface = surface;
    vec3 currentViewDir = viewDir;
    bool storedLobeType = false;
    for (int localBounce = 0; localBounce < 1; ++localBounce) {
        float emissionFactor =
            (bounce == 0u && localBounce == 0) ? VPT_DIRECT_LIGHT_STRENGTH : VPT_INDIRECT_LIGHT_STRENGTH;
        vec3 emissionRadiance = emissionFactor * currentSurface.tint * currentSurface.mat.emission * mainRay.throughput;
        emissionRadiance += currentSurface.tint * albedoEmission * mainRay.throughput;
        mainRay.radiance += emissionRadiance;

        vec3 directLight = sampleSurfaceDirectLight(
            currentSurface, currentViewDir, textureUV, planeHitWorldPos, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld,
            baseGeoNormal, traceLocalHeight, textureMap.normal, maxDepthWorld, hasFftWaterSurface);
        if (localBounce == 0) { mainRay.directLightRadiance = directLight; }
        mainRay.radiance += directLight;

        bool isOpaqueSurface = currentSurface.mat.transmission <= EPS;
        vec3 sampleDir;
        float pdf;
        uint lobeType;
        vec3 bsdf;
        if (isWaterMaterial && currentSurface.mat.transmission > EPS) {
            vec3 incident = -currentViewDir;
            vec3 waterNormal = currentSurface.shadingNormal;
            vec3 waterUpNormal = waterNormal.y >= 0.0 ? waterNormal : -waterNormal;
            float eta = dot(incident, waterUpNormal) < 0.0 ? (1.0 / currentSurface.mat.ior) : currentSurface.mat.ior;
            float fresnel = clamp(DielectricFresnel(abs(dot(currentViewDir, waterNormal)), eta), 0.0, 1.0);
            vec3 refractionDir = refract(incident, waterNormal, eta);
            bool hasRefraction = dot(refractionDir, refractionDir) > 1e-6;
            bool chooseReflection =
                !hasRefraction || (VPT_TRANSPARENT_SPLIT_MODE != VPT_TRANSPARENT_SPLIT_MODE_DETERMINISTIC &&
                                   rand(mainRay.seed) < fresnel);
            if (chooseReflection) {
                sampleDir = normalize(reflect(incident, waterNormal));
                pdf = max(fresnel, 1e-4);
                lobeType = 1u;
                bsdf = vec3(fresnel);
            } else {
                sampleDir = normalize(refractionDir);
                pdf = max(1.0 - fresnel, 1e-4);
                lobeType = 2u;
                bsdf = pow(max(currentSurface.albedoValue.rgb, vec3(0.0)), vec3(0.5)) * (1.0 - fresnel);
            }
        } else {
            bsdf = DisneySample(currentSurface.mat, currentViewDir, currentSurface.shadingNormal, sampleDir, pdf,
                                mainRay.seed, lobeType);
        }

        if (!storedLobeType) {
            raySetLobeType(mainRay, lobeType);
            raySetIndirectVolumetricCloud(
                mainRay, vptShouldCaptureIndirectVolumetricCloud(lobeType, currentSurface.mat.roughness));
            storedLobeType = true;
        }
        raySetNoisy(mainRay, true);

        if (pdf <= 1e-6 || max(bsdf.r, max(bsdf.g, bsdf.b)) <= 1e-6) {
            raySetStop(mainRay, true);
            return;
        }

        if (isOpaqueSurface && dot(sampleDir, currentSurface.geometricNormal) <= 0.0) {
            raySetStop(mainRay, true);
            return;
        }

        mainRay.throughput *= bsdf / max(pdf, 1e-4);

        if (!traceLocalHeight) {
            if (hasFftWaterSurface) {
                vec3 exitNormal = dot(sampleDir, currentSurface.geometricNormal) >= 0.0 ?
                                      currentSurface.geometricNormal :
                                      -currentSurface.geometricNormal;
                mainRay.origin = currentSurface.worldPos + exitNormal * VPT_FFT_WATER_ORIGIN_BIAS;
            } else if (hasHeightMapSurface) {
                vec3 exitBasePos =
                    basePlaneWorldPosAtUv(currentSurface.uv, textureUV, planeHitWorldPos, dPduWorld, dPdvWorld);
                vec3 exitNormal = dot(sampleDir, baseGeoNormal) >= 0.0 ? baseGeoNormal : -baseGeoNormal;
                mainRay.origin = exitBasePos + exitNormal * 0.0002;
            } else {
                vec3 exitNormal = dot(sampleDir, currentSurface.geometricNormal) >= 0.0 ?
                                      currentSurface.geometricNormal :
                                      -currentSurface.geometricNormal;
                mainRay.origin = currentSurface.worldPos + exitNormal * 0.0002;
            }
            mainRay.direction = sampleDir;
            raySetStop(mainRay, false);
            return;
        }

        HeightMapHit localBounceHit;
        vec2 exitUv;
        float exitDepth;
        float exitDistance;
        bool localBlocked = traceLocalHeightIntersectionAndExit(textureMap.normal, atlasUvMin, atlasUvMax, dPduWorld,
                                                                dPdvWorld, baseGeoNormal, maxDepthWorld, currentSurface,
                                                                sampleDir, VPT_PARALLAX_SECONDARY_MAX_STEPS,
                                                                localBounceHit, exitUv, exitDepth, exitDistance);
        if (!localBlocked) {
            vec3 exitBasePos = basePlaneWorldPosAtUv(exitUv, textureUV, planeHitWorldPos, dPduWorld, dPdvWorld) -
                               baseGeoNormal * exitDepth;
            vec3 exitNormal = dot(sampleDir, baseGeoNormal) >= 0.0 ? baseGeoNormal : -baseGeoNormal;
            mainRay.origin = exitBasePos + exitNormal * 0.0002;
            mainRay.direction = sampleDir;
            raySetStop(mainRay, false);
            return;
        }

        vec3 nextWorldPos = currentSurface.worldPos + sampleDir * localBounceHit.t;
        sampleSurfaceState(useTexture, textureID, textureMap, atlasUvMin, atlasUvMax, localBounceHit.uv, lod, alphaMode,
                           colorLayerValue, colorLayer, glint, useOverlay, m0.overlayUV, dPduWorld, dPdvWorld,
                           baseGeoNormal, hasHeightMapSurface, maxDepthWorld, localBounceHit, nextWorldPos, -sampleDir,
                           isWaterMaterial, hasFftWaterSurface, currentSurface);
        currentViewDir = -sampleDir;
    }

    raySetStop(mainRay, true);
}
