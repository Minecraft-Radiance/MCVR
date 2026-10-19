#ifndef RAY_PAYLOAD_GLSL
#define RAY_PAYLOAD_GLSL

#include "common/mapping.hpp"

struct MainRay {
    T_VEC3 origin;
    T_FLOAT hitT;

    T_VEC3 direction;
    T_UINT seed;

    T_VEC3 radiance;
    T_FLOAT coneWidth;

    T_VEC3 throughput;
    T_FLOAT coneSpread;

    T_VEC3 prevScenePos;
    T_UINT hasPrevScenePos;

    T_VEC3 normal;
    T_UINT stateBits;

    T_VEC3 directLightRadiance;
    T_UINT pad0;

#if defined(MCVR_RAY_MATERIAL_PAYLOAD) && MCVR_RAY_MATERIAL_PAYLOAD
    // Keep each ray's material with its payload across visible function calls.
    T_VEC4 materialAlbedo;
    T_VEC4 materialF0Roughness;
    T_VEC4 materialProperties;
#endif
};

struct MaterialInfo {
    T_VEC4 albedoValue;
    T_VEC3 f0;
    T_FLOAT roughness;
    T_FLOAT metallic;
    T_FLOAT transmission;
    T_FLOAT ior;
    T_FLOAT emission;
};

struct ShadowRay {
    T_VEC3 radiance;
    T_UINT insideBoat;
    T_VEC3 throughput;
    T_UINT pad0;
};

#endif
