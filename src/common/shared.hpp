#ifndef SHARED_HPP
#define SHARED_HPP

#include "common/mapping.hpp"

#define INF_DISTANCE 65504.0
#define PI 3.14159265358979323
#define INV_PI 0.31830988618379067
#define TWO_PI 6.28318530717958648
#define INV_TWO_PI 0.15915494309189533
#define INV_4_PI 0.07957747154594766

#ifdef __cplusplus
namespace vk {
#endif
#ifdef __cplusplus
namespace VertexFormat {
#endif
    struct Triangle {
        T_VEC3 pos;
        T_VEC3 color;
    };

    struct TexturedTriangle {
        T_VEC3 pos;
        T_VEC2 uv;
    };

    struct ArrayTexturedTriangle {
        T_VEC3 pos;
        T_FLOAT metallic;
        T_VEC3 norm;
        T_FLOAT roughness;
        T_VEC2 uv;
        T_FLOAT textureLayer;
        T_FLOAT pad0;
        T_VEC3 color;
        T_FLOAT intensity;
    };

    struct PositionOnly {
        T_VEC3 position;
    };

    struct PositionTexColor {
        T_VEC3 position;
        T_VEC2 uv;
        T_UINT color;
    };

    struct PositionColor {
        T_VEC3 position;
        T_UINT color;
    };

    struct PositionColorNormal {
        T_VEC3 position;
        T_UINT color;
        T_UINT normal; // first 3 bytes
    };

    struct PositionTex {
        T_VEC3 position;
        T_VEC2 uv;
    };

    struct PositionColorTexLight {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;
        T_UINT uv2;
    };

    struct PositionColorLight {
        T_VEC3 position;
        T_UINT color;
        T_UINT uv2;
    };

    struct PositionTexColorLight {
        T_VEC3 position;
        T_VEC2 uv0;
        T_UINT color;
        T_UINT uv2;
    };

    struct PositionTexColorNormal {
        T_VEC3 position;
        T_VEC2 uv0;
        T_UINT color;
        T_UINT normal; // first 3 bytes
    };

    struct PositionTexLightColor {
        T_VEC3 position;
        T_VEC2 uv0;
        T_UINT uv2;
        T_UINT color;
    };

    struct PositionColorTexLightNormal {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;    // texture
        T_UINT uv2;    // lightmap
        T_UINT normal; // first 3 bytes
    };

    struct PositionColorTexOverlayLightNormal {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;    // texture
        T_UINT uv1;    // overlay
        T_UINT uv2;    // lightmap
        T_UINT normal; // first 3 bytes
    };

    struct PBRTriangle {
        T_VEC3 pos;
        T_UINT useNorm;

        T_VEC3 norm;
        T_UINT useColorLayer;

        T_VEC4 colorLayer;

        T_UINT useTexture;
        T_UINT useOverlay;
        T_VEC2 textureUV;

        T_IVEC2 overlayUV;
        T_UINT useGlint;
        T_UINT textureID;

        T_VEC2 glintUV;
        T_UINT glintTexture;
        T_UINT useLight;

        T_IVEC2 lightUV;
        T_UINT coordinate;
        T_FLOAT albedoEmission;

        T_VEC3 postBase;
        T_UINT pad1;
    };
#ifdef __cplusplus
}; // namespace VertexFormat
#endif

#ifdef __cplusplus
namespace Data {
#endif
    struct Camera {
        T_MAT4 viewMatrix;
        T_MAT4 projMatrix;
        T_MAT4 viewMatrixInv;
        T_MAT4 projMatrixInv;
        T_VEC2 jitter;
        T_VEC2 pad0;
    };

    struct DirectionalLight {
        T_VEC3 direction;  // 12 bytes
        T_FLOAT pad0;      // 4 bytes
        T_VEC3 color;      // 12 bytes
        T_FLOAT intensity; // 4 bytes
    };

    struct World {
        DirectionalLight directionalLight; // 32 bytes
        T_FLOAT time;                      // 4 bytes
        T_UINT seed;                       // 4 bytes
    };

    struct OverlayUBO {
        T_UINT texIndices[12];

        T_MAT4 modelViewMat;

        T_MAT4 projectionMat;

        T_VEC4 colorModulator;

        T_FLOAT glintAlpha;
        T_FLOAT fogStart;
        T_FLOAT fogEnd;
        T_UINT fogShape;

        T_VEC4 fogColor;

        T_MAT4 textureMat;

        T_FLOAT gameTime;
        T_FLOAT lineWidth;
        T_VEC2 screenSize;

        T_VEC3 light0Direction;
        T_FLOAT pad0;

        T_VEC3 light1Direction;
        T_FLOAT pad1;
    };

    struct OverlayPostUBO {
        T_MAT4 projectionMat;
        T_VEC2 inSize;
        T_VEC2 outSize;
        T_VEC2 blurDir;
        T_FLOAT radius;
        T_FLOAT radiusMultiplier;
    };

    struct WorldUBO {
        T_MAT4 cameraViewMat;

        T_MAT4 cameraEffectedViewMat;

        T_MAT4 cameraProjMat;

        T_MAT4 cameraViewMatInv;

        T_MAT4 cameraEffectedViewMatInv;

        T_MAT4 cameraProjMatInv;

        T_VEC2 cameraJitter;
        T_FLOAT gameTime;
        T_UINT seed;

        T_MAT4 textureMat;

        T_UINT overlayTextureID;
        T_UINT isFirstPerson;
        T_FLOAT fogStart;
        T_FLOAT fogEnd;

        T_VEC4 fogColor;

        T_UINT fogType;
        T_UINT skyType;
        T_UINT rayBounces;
        T_FLOAT pad3;

        T_DVEC4 cameraPos; // w for padding

        T_UINT endSkyTextureID;
        T_UINT endPortalTextureID;
        T_UINT pad4;
        T_UINT pad5;

        // VR stereo fields
        T_MAT4 eyeViewOffsets[2];     // per-eye view offset (translate ±ipd/2)
        T_MAT4 eyeProjOffsets[2];     // per-eye projection offset (identity for symmetric)
        T_FLOAT ipd;
        T_UINT  stereoEnabled;        // 0=mono, 1=stereo
        T_FLOAT foveatedInnerRadius;  // normalized radius of full-res inner circle (1.0 = full screen)
        T_UINT  foveatedOuterBlockSize; // block size for outer region (1=off, 2=2x2, 4=4x4)
        T_VEC2  foveatedCenter;        // normalised gaze centre (0.5,0.5 = screen centre)
    };

    struct SkyUBO {
        T_VEC3 baseColor;
        T_UINT skyType;

        T_VEC4 horizonColor;

        T_VEC3 sunDirection;
        T_UINT isSunRisingOrSetting;

        T_UINT isSkyDark;
        T_UINT hasBlindnessOrDarkness;
        T_UINT cameraSubmersionType;
        T_UINT moonPhase;

        T_FLOAT rainGradient;
        T_FLOAT pad0;
        T_FLOAT pad1;
        T_FLOAT pad2;

        // AtmosphereParams

        T_FLOAT Rg;
        T_FLOAT Rt;
        T_FLOAT Hr;
        T_FLOAT Hm;

        T_VEC3 betaR;
        T_FLOAT mieG;

        T_VEC3 betaM;
        T_FLOAT minViewCos;

        T_VEC3 sunRadiance;
        T_UINT sunTextureID;

        T_VEC3 moonRadiance;
        T_UINT moonTextureID;
    };

    struct TextureMapEntry {
        T_INT specular;
        T_INT normal;
        T_INT flag;
    };

    struct TextureMapping {
        TextureMapEntry entries[4096];
    };

    struct ExposureData {
        T_INT width;
        T_INT height;
        T_INT stride;
        T_FLOAT exposure;

        T_FLOAT minL;
        T_FLOAT maxL;
        T_UINT total;
        T_UINT pad0;

        T_UINT bins[256];
    };

    struct LightMapUBO {
        T_FLOAT ambientLightFactor;
        T_FLOAT skyFactor;
        T_FLOAT blockFactor;
        T_INT useBrightLightmap;

        T_VEC3 skyLightColor;
        T_FLOAT nightVisionFactor;

        T_FLOAT darknessScale;
        T_FLOAT darkenWorldFactor;
        T_FLOAT brightnessFactor;
        T_FLOAT pad0;
    };
#ifdef __cplusplus
}; // namespace Data
#endif
#ifdef __cplusplus
}; // namespace vk
#endif

#endif