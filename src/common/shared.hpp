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
        T_UINT normal;
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
        T_UINT normal;
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
        T_VEC2 uv0;
        T_UINT uv2;
        T_UINT normal;
    };

    struct PositionColorTexOverlayLightNormal {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;
        T_UINT uv1;
        T_UINT uv2;
        T_UINT normal;
    };

    struct EntityPBRVertex {
        T_VEC3 pos;
        T_UINT normalOct;

        T_VEC2 textureUV;
        T_UINT colorRGBA8;
        T_UINT lightUV;

        T_UINT overlayUV;
        T_UINT glintUVHalf;
        T_UINT textureID;
        T_UINT glintTexture;

        T_VEC3 postBase;
        T_UINT packedData;
    };

    struct EntityInstanceMaterial {
        T_VEC4 color;
        T_VEC4 uvTransform;
        T_UINT light;
        T_UINT overlay;
        T_UINT flags;
        T_UINT pad;
    };

    struct EntityGeometryHeader {
        T_UINT addressLo;
        T_UINT addressHi;
        T_UINT addressIndex;
        T_UINT pad;
        EntityInstanceMaterial material;
    };

    struct EntityGeometryInstance {
        T_UINT addressLo;
        T_UINT addressHi;
        T_UINT addressIndex;
        T_UINT pad;
        EntityInstanceMaterial material;
        T_UINT rawAddressLo;
        T_UINT rawAddressHi;
        T_UINT rawPad0;
        T_UINT rawPad1;
        T_VEC4 normalToWorld[3];
        T_VEC4 normalToObject[3];
        T_VEC4 attributes[65];
    };

    struct ChunkPBRVertex {
        T_FLOAT posX;
        T_FLOAT posY;
        T_FLOAT posZ;
        T_UINT normalOct;

        T_VEC2 textureUV;
        T_UINT colorRGBA8;
        T_UINT lightUV;

        T_UINT textureID;
        T_UINT packedData;
    };

    struct PositionVertex {
        T_VEC3 pos;
        T_UINT pad0;
    };

    struct MaterialVertex {
        T_VEC3 norm;
        T_UINT textureID;

        T_VEC4 colorLayer;

        T_VEC2 textureUV;
        T_IVEC2 overlayUV;

        T_VEC2 glintUV;
        T_UINT glintTexture;
        T_FLOAT albedoEmission;

        T_IVEC2 lightUV;
        T_UINT packedData;
        T_UINT pad0;
    };
#ifdef __cplusplus
};

static_assert(sizeof(VertexFormat::EntityInstanceMaterial) == 48);
static_assert(sizeof(VertexFormat::EntityGeometryHeader) == 64);
static_assert(sizeof(VertexFormat::EntityGeometryHeader) == offsetof(VertexFormat::EntityGeometryInstance, rawAddressLo));
static_assert(sizeof(VertexFormat::EntityGeometryInstance) == 1216);
static_assert(offsetof(VertexFormat::EntityGeometryInstance, material) == 16);
static_assert(sizeof(VertexFormat::EntityPBRVertex) == 64);
static_assert(offsetof(VertexFormat::EntityPBRVertex, normalOct) == 12);
static_assert(offsetof(VertexFormat::EntityPBRVertex, postBase) == 48);
static_assert(offsetof(VertexFormat::EntityPBRVertex, packedData) == 60);
static_assert(sizeof(VertexFormat::ChunkPBRVertex) == 40);
static_assert(offsetof(VertexFormat::ChunkPBRVertex, normalOct) == 12);
static_assert(offsetof(VertexFormat::ChunkPBRVertex, textureUV) == 16);
static_assert(offsetof(VertexFormat::ChunkPBRVertex, colorRGBA8) == 24);
static_assert(offsetof(VertexFormat::ChunkPBRVertex, lightUV) == 28);
static_assert(offsetof(VertexFormat::ChunkPBRVertex, textureID) == 32);
static_assert(offsetof(VertexFormat::ChunkPBRVertex, packedData) == 36);
static_assert(sizeof(VertexFormat::MaterialVertex) == 80);
static_assert(offsetof(VertexFormat::MaterialVertex, lightUV) == 64);
static_assert(offsetof(VertexFormat::MaterialVertex, packedData) == 72);
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
        T_VEC3 direction;
        T_FLOAT pad0;
        T_VEC3 color;
        T_FLOAT intensity;
    };

    struct World {
        DirectionalLight directionalLight;
        T_FLOAT time;
        T_UINT seed;
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
        T_INT worldBottomY;
        T_INT worldTopY;

        T_VEC4 cameraPos;
        T_VEC4 cameraPosLow;
        T_IVEC4 chunkGridInfo;
        T_IVEC4 chunkStorageSectionPos;

        T_UINT endSkyTextureID;
        T_UINT endPortalTextureID;
        T_UINT lightMapTextureID;
        T_UINT renderDistanceBlocks;
        T_UINT vistaDistanceBlocks;
        T_UINT vistaInstanceOffset;
        T_UINT vistaInstanceCount;
        T_UINT vistaTextureID;
        T_IVEC4 vistaCoverage;
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

        T_UINT sunTextureID;
        T_UINT moonTextureID;
        T_UINT rainData;
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
};
#endif
#ifdef __cplusplus
};
#endif

#endif
