#ifndef VISTA_GLSL
#define VISTA_GLSL

#include "util/vista_coverage.glsl"

bool isVistaBoundaryWater(uint geometryBufferIndex, uint i0, uint i1, uint i2) {
    MaterialVertex material = loadMaterialVertex(geometryBufferIndex, i0);
    if (hasWaterMaterial(material)) { return true; }
    if (!hasTexture(material.packedData)) { return false; }
    int flagTexture = mapping.entries[material.textureID].flag;
    if (flagTexture < 0) { return false; }
    vec2 uv = (material.textureUV + loadMaterialVertex(geometryBufferIndex, i1).textureUV +
               loadMaterialVertex(geometryBufferIndex, i2).textureUV) / 3.0;
    uint flags = uint(round(textureLod(textures[nonuniformEXT(flagTexture)], uv, 0.0).r * 255.0));
    return (flags & 1u) != 0u;
}

bool isVistaCovered(WorldUBO world, uint instanceIndex, vec2 attributes) {
    uint64_t address = uint64_t(uint(world.vistaCoverage.x)) | (uint64_t(uint(world.vistaCoverage.y)) << 32);
    if (address == uint64_t(0)) { return false; }
    if (instanceIndex >= uint(world.vistaCoverage.z) &&
        instanceIndex - uint(world.vistaCoverage.z) < uint(world.vistaCoverage.w)) { return false; }
    bool isVista = instanceIndex >= world.vistaInstanceOffset &&
                   instanceIndex - world.vistaInstanceOffset < world.vistaInstanceCount;
    VistaCoverage coverage = VistaCoverage(address);
    uint geometryBufferIndex = getGeometryBufferIndex(instanceIndex, gl_GeometryIndexEXT);
    if (!isChunkGeometryBufferIndex(geometryBufferIndex)) { return false; }
    vec3 origin = round(vec3(gl_ObjectToWorldEXT[3]) + world.cameraPos.xyz);
    if (!isVista && vistaColumnBorders(world, coverage, ivec2(floor(origin.xz / 16.0))) == 0u) { return false; }
    uint i0, i1, i2;
    loadTriangleIndices(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2);
    vec3 p0 = loadPositionVertex(geometryBufferIndex, i0).pos;
    vec3 p1 = loadPositionVertex(geometryBufferIndex, i1).pos;
    vec3 p2 = loadPositionVertex(geometryBufferIndex, i2).pos;
    vec3 localPosition = p0 + attributes.x * (p1 - p0) + attributes.y * (p2 - p0);
    vec2 position = origin.xz + vec2(localPosition.xz);
    ivec2 column = ivec2(floor(position / 16.0));
    vec3 normal = cross(p1 - p0, p2 - p0);
    bool xFace = p0.x == p1.x && p0.x == p2.x && normal.x != 0.0;
    bool zFace = p0.z == p1.z && p0.z == p2.z && normal.z != 0.0;
    vec2 planes = round(position / 16.0) * 16.0;
    vec2 planeDistance = abs(position - planes);
    bool offsetWater = !isVista &&
        ((xFace && planeDistance.x > 0.0 && planeDistance.x <= 0.0011) ||
         (zFace && planeDistance.y > 0.0 && planeDistance.y <= 0.0011)) &&
        isVistaBoundaryWater(geometryBufferIndex, i0, i1, i2);
    if (xFace && (planeDistance.x == 0.0 || (offsetWater && planeDistance.x <= 0.0011))) {
        ivec2 faceColumn = ivec2(int(planes.x / 16.0), column.y);
        if ((vistaColumnBorders(world, coverage, faceColumn) & 1u) != 0u ||
            (vistaColumnBorders(world, coverage, faceColumn - ivec2(1, 0)) & 2u) != 0u) { return true; }
    }
    if (zFace && (planeDistance.y == 0.0 || (offsetWater && planeDistance.y <= 0.0011))) {
        ivec2 faceColumn = ivec2(column.x, int(planes.y / 16.0));
        if ((vistaColumnBorders(world, coverage, faceColumn) & 4u) != 0u ||
            (vistaColumnBorders(world, coverage, faceColumn - ivec2(0, 1)) & 8u) != 0u) { return true; }
    }
    if (!isVista) { return false; }
    if (mod(position.x, 16.0) == 0.0 && normal.x > 0.0) { --column.x; }
    if (mod(position.y, 16.0) == 0.0 && normal.z > 0.0) { --column.y; }
    return vistaNearColumnReady(world, coverage, column);
}

#endif
