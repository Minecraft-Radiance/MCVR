#ifndef ADV_SCENE_MATERIALS_WETNESS_GLSL
#define ADV_SCENE_MATERIALS_WETNESS_GLSL

#ifndef ADV_RAIN_WETNESS_ENABLED
#define ADV_RAIN_WETNESS_ENABLED 1
#endif

ivec4 rainColumn(ivec2 column, uint textureId) {
    return ivec4(round(texelFetch(textures[nonuniformEXT(textureId)], column & ivec2(511), 0) * 255.0));
}

bool rainColumnBlocks(ivec4 data, int surfaceBlockY) {
    return data.a == 0 || (data.r + 256 * data.g - 32768) > surfaceBlockY;
}

float groundWetness(vec3 position, vec3 outwardNormal) {
#if ADV_RAIN_WETNESS_ENABLED == 0
    return 0.0;
#else
    uint textureId = skyUBO.rainData & 65535u;
    float rain = float(skyUBO.rainData >> 16u) / 65535.0;
    if (textureId == 0u || rain <= 0.0001 || outwardNormal.y < 0.5) return 0.0;
    if (max(abs(position.x), abs(position.z)) > 238.0) return 0.0;
    ivec3 cameraCell = ivec3(worldUBO.cameraPos.xyz);
    vec3 relativePosition = position + vec3(worldUBO.cameraPos.xyz - vec3(cameraCell));
    ivec2 column = cameraCell.xz + ivec2(floor(relativePosition.xz));
    int surfaceBlockY = cameraCell.y + int(floor(relativePosition.y - 0.001));
    ivec4 center = rainColumn(column, textureId);
    if (center.b == 0 || rainColumnBlocks(center, surfaceBlockY)) return 0.0;

    vec2 withinColumn = fract(relativePosition.xz);
    ivec2 nearestSide = ivec2(step(vec2(0.5), withinColumn)) * 2 - 1;
    vec2 edge = min(withinColumn, vec2(1.0) - withinColumn);
    float edgeDistanceSquared = 0.25;
    if (rainColumnBlocks(rainColumn(column + ivec2(nearestSide.x, 0), textureId), surfaceBlockY))
        edgeDistanceSquared = min(edgeDistanceSquared, edge.x * edge.x);
    if (rainColumnBlocks(rainColumn(column + ivec2(0, nearestSide.y), textureId), surfaceBlockY))
        edgeDistanceSquared = min(edgeDistanceSquared, edge.y * edge.y);
    if (rainColumnBlocks(rainColumn(column + nearestSide, textureId), surfaceBlockY))
        edgeDistanceSquared = min(edgeDistanceSquared, dot(edge, edge));
    rain *= 1.0 - smoothstep(220.0, 238.0, max(abs(position.x), abs(position.z)));
    return rain * smoothstep(0.0, 0.5, sqrt(edgeDistanceSquared)) * smoothstep(0.5, 0.95, outwardNormal.y);
#endif
}

void applyGroundWetness(inout LabPBRMat material, float wetness) {
    if (wetness <= 0.0 || material.transmission > 0.0) return;
    float absorbency = (1.0 - material.metallic) * clamp(material.roughness * 2.0, 0.0, 1.0);
    material.albedo *= 1.0 - 0.35 * wetness * absorbency;
    material.roughness = mix(material.roughness, min(material.roughness, 0.08), wetness);
}

#endif
