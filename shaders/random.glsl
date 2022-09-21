// https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f * 2.0 - 3.0;                  // Range [-1:1]
}

float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
vec2  random( vec2  v ) { return vec2(
    floatConstruct(hash(floatBitsToUint(v.x))),
    floatConstruct(hash(floatBitsToUint(v.y)))
); }
vec3  random( vec3  v ) { return vec3(
    floatConstruct(hash(floatBitsToUint(v.x))),
    floatConstruct(hash(floatBitsToUint(v.y))),
    floatConstruct(hash(floatBitsToUint(v.z)))
); }
vec4  random( vec4  v ) { return vec4(
    floatConstruct(hash(floatBitsToUint(v.x))),
    floatConstruct(hash(floatBitsToUint(v.y))),
    floatConstruct(hash(floatBitsToUint(v.z))), 
    floatConstruct(hash(floatBitsToUint(v.w)))
); }
