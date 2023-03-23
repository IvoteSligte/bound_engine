// adapted from: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
uint TausStep(inout uint z, int S1, int S2, int S3, uint M) {
    uint b = (((z << S1) ^ z) >> S2);
    return z = (((z & M) << S3) ^ b);
} 

// A and C are constants
uint LCGStep(inout uint z, uint A, uint C) {
    return z = A * z + C;
}

const float HYBRID_TAUS_NORMALIZER = 2.3283064365387e-10;

float HybridTausUnnormalized(inout uvec4 z) {
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121
    return (
        // Periods
        TausStep(z.x, 13, 19, 12, 4294967294) ^
        // p1=2^31-1
        TausStep(z.y, 2, 25, 4, 4294967288) ^
        // p2=2^30-1
        TausStep(z.z, 3, 11, 17, 4294967280) ^
        // p3=2^28-1
        LCGStep(z.w, 1664525, 1013904223)
        // p4=2^32
    );
}

float HybridTaus(inout uvec4 z) {
    return HYBRID_TAUS_NORMALIZER * HybridTausUnnormalized(z);
}