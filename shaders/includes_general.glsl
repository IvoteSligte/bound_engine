#define EPSILON 1e-4

struct GridCell {
    // combination of position and strength
    // position = vector / (float(counter) * length(vector))
    // energy = length(vector) / float(counter)
    vec3 vector;
    uint counter;
};

struct DynamicParticle {
    // uvec3 position as 3 * 16 bits
    // uvec3 direction as 3 * 16 bits
    uvec3 data;
    float energy;
};

struct StaticParticle {
    // uvec3 position as 3 x 16 bits xyz
    // uint reflectance in range (0, 1) as 16 bits
    uvec2 data;
    // emittance
    float emittance;
    // absorbed energy (only non-constant)
    float energy;
};

uvec3 unpackShorts3x16(uvec2 data) {
    return uvec3(
        data.x & 65535,
        data.x >> 16,
        data.y & 65535
    );
}

uvec4 unpackShorts4x16(uvec2 data) {
    return uvec4(
        data.x & 65535,
        data.x >> 16,
        data.y & 65535,
        data.y >> 16
    );
}

uvec2 packShorts4x16(uvec4 data) {
    return uvec2(
        data.x | (data.y << 16),
        data.z | (data.w << 16)
    );
}

void unpackDynamicParticle(
    DynamicParticle particle,
    out ivec3 position,
    out vec3 direction,
    out float energy
) {
    uvec3 shorts = unpackShorts3x16(particle.data.xy);
    position = ivec3(shorts.xyz);
    direction = vec3(
        unpackHalf2x16(particle.data.y).y,
        unpackHalf2x16(particle.data.z)
    );
    energy = particle.energy;
}

DynamicParticle packDynamicParticle(ivec3 position, vec3 direction, float energy) {
    uint directionX = packHalf2x16(vec2(direction.x, 0.0));
    uint directionYZ = packHalf2x16(direction.yz);
    uvec3 data = uvec3(
        packShorts4x16(uvec4(position, directionX)),
        directionYZ
    );
    return DynamicParticle(data, energy);
}

void unpackStaticParticle(
    StaticParticle particle,
    out uvec3 position,
    out float reflectance,
    out float energy,
    out float emittance
) {
    uvec4 shorts = unpackShorts4x16(particle.data.xy);
    position = shorts.xyz;
    reflectance = float(shorts.w) * (1.0 / 65536.0);
    energy = particle.energy;
    emittance = particle.emittance;
}
