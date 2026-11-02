// include/strata/kernels/cpu/iq_avx1_kern.inc - expert rows for CPUs with AVX but no AVX2 (Sandy / Ivy Bridge, Xeon E5
// v1 / v2), included by ds4_iq_cpu.cpp (DeepSeek's CPU lane) and iq_avx1.cpp (GLM's native rows); the includer brings
// <immintrin.h>, <cstring> and ggml-common.h (GGML_COMMON_IMPL_CPP).  Without them such a CPU runs ggml-cpu's one-token
// AVX dots for every expert.
// Each 32-value chunk is decoded once a pass and applied to up to 4 tokens in 128-bit integer lanes (pshufb, sign_epi8,
// maddubs, madd: SSSE3 / SSE2), the two 16-value halves being the lanes of ggml's AVX dot; the float steps are ggml's
// AVX order (256-bit mul then add, no FMA, then K * hsum_float_8), so a token's dot is ggml-cpu's bit for bit at any
// token count: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ1_S (19), Q2_K (10), and (GLM's
// down rows) Q5_K (13) and IQ4_XS (23).  The includers check them against ggml's dot before use.
#define IQ_AVX1 __attribute__((target("avx,sse4.2,ssse3")))

namespace avx1 {

inline uint16_t u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint64_t u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
inline float h2f(uint16_t h) {   // F16 -> F32, exactly (no F16C needed)
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t b;
    if (e == 0) {
        const float f = (float) m * 5.9604644775390625e-8f;
        std::memcpy(&b, &f, 4);
        b |= s;
    } else if (e == 31) {
        b = s | 0x7f800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// 16 sign bytes (-1 / +1) from two sign bytes of w (byte_pair 0: bytes 0-1, 1: bytes 2-3)
IQ_AVX1 inline __m128i sgn16(uint32_t w, int byte_pair) {
    const __m128i bits = _mm_shuffle_epi8(_mm_set1_epi32((int) w),
                                          byte_pair ? _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)
                                                    : _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i sel = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    return _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(bits, sel), sel), _mm_set1_epi8(1));
}

// keven_signs_q2xs: byte k 0xFF where bit k of ksigns_iq2xs[i] is set, else 0x01
struct EvenSigns {
    uint64_t v[128];
    constexpr EvenSigns() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k) par ^= (i >> k) & 1;
            const int s = i | (par << 7);
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
constexpr EvenSigns even_signs{};

// a half's high index bits one group per byte (IQ3_S: bit k, IQ2_S: bits 2k, 2k+1), next to the low index bytes
struct HiSpread {
    uint64_t iq3s[256];
    uint64_t iq2s[256];
    constexpr HiSpread() : iq3s{}, iq2s{} {
        for (int h = 0; h < 256; ++h) {
            uint64_t r3 = 0, r2 = 0;
            for (int k = 0; k < 8; ++k) r3 |= (uint64_t) ((h >> k) & 1) << (8 * k);
            for (int k = 0; k < 4; ++k) r2 |= (uint64_t) ((h >> (2 * k)) & 3) << (8 * k);
            iq3s[h] = r3;
            iq2s[h] = r2;
        }
    }
};
constexpr HiSpread hi_spread{};

IQ_AVX1 inline __m128i grid_idx(const uint8_t* q, uint64_t hi) {   // eight 16-bit grid indices
    return _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) q), _mm_cvtsi64_si128((long long) hi));
}

// one 32-value half (values 64 j + 32 half ..): grid magnitudes, signs and scales as two 16-value halves
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    IQ_AVX1 static inline void decode(const uint8_t* b, int j, int half, __m128i& g0, __m128i& g1, __m128i& s0,
                                      __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint8_t* q = b + 2 + 8 * (2 * j + half);
        g0 = _mm_set_epi64x((long long) iq2xxs_grid[q[1]], (long long) iq2xxs_grid[q[0]]);
        g1 = _mm_set_epi64x((long long) iq2xxs_grid[q[3]], (long long) iq2xxs_grid[q[2]]);
        const uint32_t w = u32(q + 4);
        s0 = _mm_set_epi64x((long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        s1 = _mm_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127]);
        c0 = c1 = _mm_set1_epi16((short) (2 * (int) (w >> 28) + 1));
    }
};

template <> struct Fmt<17> {   // IQ2_XS
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    IQ_AVX1 static inline void decode(const uint8_t* b, int j, int half, __m128i& g0, __m128i& g1, __m128i& s0,
                                      __m128i& s1, __m128i& c0, __m128i& c1) {
        const int c = 2 * j + half;
        const uint64_t q = u64(b + 2 + 8 * c);
        const uint32_t q0 = (uint32_t) q & 0xffff, q1 = (uint32_t) (q >> 16) & 0xffff;
        const uint32_t q2 = (uint32_t) (q >> 32) & 0xffff, q3 = (uint32_t) (q >> 48);
        g0 = _mm_set_epi64x((long long) iq2xs_grid[q1 & 511], (long long) iq2xs_grid[q0 & 511]);
        g1 = _mm_set_epi64x((long long) iq2xs_grid[q3 & 511], (long long) iq2xs_grid[q2 & 511]);
        s0 = _mm_set_epi64x((long long) even_signs.v[q1 >> 9], (long long) even_signs.v[q0 >> 9]);
        s1 = _mm_set_epi64x((long long) even_signs.v[q3 >> 9], (long long) even_signs.v[q2 >> 9]);
        const uint8_t sb = b[66 + c];
        c0 = _mm_set1_epi16((short) (2 * (sb & 15) + 1));
        c1 = _mm_set1_epi16((short) (2 * (sb >> 4) + 1));
    }
};

template <> struct Fmt<18> {   // IQ3_XXS
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    IQ_AVX1 static inline void decode(const uint8_t* b, int j, int half, __m128i& g0, __m128i& g1, __m128i& s0,
                                      __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g0 = _mm_set_epi32((int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]],
                           (int) iq3xxs_grid[q[0]]);
        g1 = _mm_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]],
                           (int) iq3xxs_grid[q[4]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        s0 = _mm_set_epi64x((long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        s1 = _mm_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127]);
        c0 = c1 = _mm_set1_epi16((short) (2 * (int) (w >> 28) + 1));
    }
};

template <> struct Fmt<21> {   // IQ3_S
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    IQ_AVX1 static inline void decode(const uint8_t* b, int j, int half, __m128i& g0, __m128i& g1, __m128i& s0,
                                      __m128i& s1, __m128i& c0, __m128i& c1) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const __m128i idx = grid_idx(q, hi_spread.iq3s[b[66 + 2 * j + half]]);
#define G3(k) (int) iq3s_grid[_mm_extract_epi16(idx, k)]
        g0 = _mm_set_epi32(G3(3), G3(2), G3(1), G3(0));
        g1 = _mm_set_epi32(G3(7), G3(6), G3(5), G3(4));
#undef G3
        const uint32_t sg = u32(b + 74 + 8 * j + 4 * half);
        s0 = sgn16(sg, 0);
        s1 = sgn16(sg, 1);
        const uint8_t s = b[106 + j];
        c0 = c1 = _mm_set1_epi16((short) (2 * (int) (half ? s >> 4 : s & 15) + 1));
    }
};

template <> struct Fmt<22> {   // IQ2_S
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    IQ_AVX1 static inline void decode(const uint8_t* b, int j, int half, __m128i& g0, __m128i& g1, __m128i& s0,
                                      __m128i& s1, __m128i& c0, __m128i& c1) {
        const __m128i idx = grid_idx(b + 2 + 8 * j + 4 * half, hi_spread.iq2s[b[66 + 2 * j + half]]);
        g0 = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(idx, 1)], (long long) iq2s_grid[_mm_extract_epi16(idx, 0)]);
        g1 = _mm_set_epi64x((long long) iq2s_grid[_mm_extract_epi16(idx, 3)], (long long) iq2s_grid[_mm_extract_epi16(idx, 2)]);
        const uint32_t sg = u32(b + 2 + 32 + 8 * j + 4 * half);
        s0 = sgn16(sg, 0);
        s1 = sgn16(sg, 1);
        const uint8_t sb = b[74 + 2 * j + half];
        c0 = _mm_set1_epi16((short) (2 * (sb & 15) + 1));
        c1 = _mm_set1_epi16((short) (2 * (sb >> 4) + 1));
    }
};

IQ_AVX1 inline __m128i madd_add(__m128i acc, __m128i g, __m128i sgn, __m128i yv, __m128i sc) {
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(g, _mm_sign_epi8(yv, sgn)), sc));
}

template <int TY, int NT>
IQ_AVX1 inline void row_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        __m128i acci[NT][2];
        for (int t = 0; t < NT; ++t) acci[t][0] = acci[t][1] = _mm_setzero_si128();
        for (int j = 0; j < 4; ++j)
            for (int half = 0; half < 2; ++half) {
                __m128i g0, g1, s0, s1, c0, c1;
                Fmt<TY>::decode(blk, j, half, g0, g1, s0, s1, c0, c1);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const uint8_t* q8 = (const uint8_t*) (y[t][i].qs + off);
                    acci[t][0] = madd_add(acci[t][0], g0, s0, _mm_loadu_si128((const __m128i*) q8), c0);
                    acci[t][1] = madd_add(acci[t][1], g1, s1, _mm_loadu_si128((const __m128i*) (q8 + 16)), c1);
                }
            }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const __m256 f = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(acci[t][0]), acci[t][1], 1));
            accf[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dx * y[t][i].d), f), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {   // ggml's hsum_float_8
        __m128 s = _mm_add_ps(_mm256_extractf128_ps(accf[t], 1), _mm256_castps256_ps128(accf[t]));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        res[t] = Fmt<TY>::K * _mm_cvtss_f32(s);
    }
}

IQ_AVX1 inline float hsum8(__m256 a) {   // ggml's hsum_float_8
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(a, 1), _mm256_castps256_ps128(a));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// Q5_K (176-byte blocks: d, dmin, 12 scale bytes, 32 high-bit bytes, 128 nibble bytes): ggml's AVX dot, the 32-value
// sub-blocks decoded once for all tokens; the min term per token as ggml's scalar sum
template <int NT>
IQ_AVX1 inline void q5k_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 accf[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
        summs[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * 176;
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 128, _MM_HINT_T0);
        uint32_t utmp[4];
        std::memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
        const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
        utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f3f3f;
        const uint8_t* sc = (const uint8_t*) utmp;   // bytes 0-7 the scales, 8-15 the mins
        const __m128i mins = _mm_cvtepu8_epi16(_mm_set_epi32(0, 0, (int) utmp[3], (int) utmp[2]));
        const float xd = h2f(u16(blk)), xm = h2f(u16(blk + 2));
        for (int t = 0; t < NT; ++t) {
            const __m128i q8s = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]),
                                               _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hs = _mm_hadd_epi32(_mm_hadd_epi32(prod, _mm_setzero_si128()), _mm_setzero_si128());
            summs[t] += (-y[t][i].d * xm) * _mm_extract_epi32(hs, 0);
        }
        const __m128i hb0 = _mm_loadu_si128((const __m128i*) (blk + 16));
        const __m128i hb1 = _mm_loadu_si128((const __m128i*) (blk + 32));
        __m128i acci[NT][2];
        for (int t = 0; t < NT; ++t) acci[t][0] = acci[t][1] = _mm_setzero_si128();
        for (int s = 0; s < 8; ++s) {
            const __m128i b0 = _mm_loadu_si128((const __m128i*) (blk + 48 + 32 * (s >> 1)));
            const __m128i b1 = _mm_loadu_si128((const __m128i*) (blk + 48 + 32 * (s >> 1) + 16));
            const __m128i hm = _mm_set1_epi8((char) (1 << s));
            const __m128i l0 = (s & 1) ? _mm_and_si128(_mm_srli_epi16(b0, 4), m4) : _mm_and_si128(b0, m4);
            const __m128i l1 = (s & 1) ? _mm_and_si128(_mm_srli_epi16(b1, 4), m4) : _mm_and_si128(b1, m4);
            const __m128i q0 = _mm_add_epi8(l0, _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hb0, hm), s), 4));
            const __m128i q1 = _mm_add_epi8(l1, _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hb1, hm), s), 4));
            const __m128i scv = _mm_set1_epi16((short) sc[s]);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 32 * s;
                acci[t][0] = _mm_add_epi32(acci[t][0], _mm_madd_epi16(scv, _mm_maddubs_epi16(q0, _mm_loadu_si128((const __m128i*) q8))));
                acci[t][1] = _mm_add_epi32(acci[t][1], _mm_madd_epi16(scv, _mm_maddubs_epi16(q1, _mm_loadu_si128((const __m128i*) (q8 + 16)))));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 f = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(acci[t][0]), acci[t][1], 1));
            accf[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(y[t][i].d * xd), f), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]) + summs[t];
}

// IQ4_XS (136-byte blocks: d, scales_h, 4 scales_l, 128 nibble bytes): ggml's AVX dot, |w| and the table lookup once
template <int NT>
IQ_AVX1 inline void iq4xs_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * 136;
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        const uint16_t sh = u16(blk + 2);
        __m128i acci[NT][2];
        for (int t = 0; t < NT; ++t) acci[t][0] = acci[t][1] = _mm_setzero_si128();
        for (int ib = 0; ib < 8; ++ib) {
            const __m128i qb = _mm_loadu_si128((const __m128i*) (blk + 8 + 16 * ib));
            const __m128i w0 = _mm_shuffle_epi8(values, _mm_and_si128(qb, m4));
            const __m128i w1 = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(qb, 4), m4));
            const __m128i a0 = _mm_sign_epi8(w0, w0), a1 = _mm_sign_epi8(w1, w1);
            const int ls = (((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4)) - 32;
            const __m128i lsv = _mm_set1_epi16((short) ls);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 32 * ib;
                const __m128i p0 = _mm_maddubs_epi16(a0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), w0));
                const __m128i p1 = _mm_maddubs_epi16(a1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), w1));
                acci[t][0] = _mm_add_epi32(acci[t][0], _mm_madd_epi16(p0, lsv));
                acci[t][1] = _mm_add_epi32(acci[t][1], _mm_madd_epi16(p1, lsv));
            }
        }
        const float xd = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const __m256 f = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(acci[t][0]), acci[t][1], 1));
            accf[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(xd * y[t][i].d), f), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// IQ1_S (50-byte blocks: d, 32 index bytes, 8 high words): ggml's AVX dot, the grid looked up once for all tokens.
// The scale rides in the magnitudes (|g| ls <= 15), so a block's 8 chunks sum in int16 (<= 8 x 3840) before ggml's
// int32 lanes; the delta term per token as ggml's scalar int sum and float accum1
template <int NT>
IQ_AVX1 inline void iq1s_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    const __m128i ones = _mm_set1_epi16(1);
    __m256 accf[NT];
    float acc1[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
        acc1[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * 50;
        _mm_prefetch((const char*) blk + 1024, _MM_HINT_T0);
        __m128i acc16[NT][2];
        int sumi1[NT];
        for (int t = 0; t < NT; ++t) {
            acc16[t][0] = acc16[t][1] = _mm_setzero_si128();
            sumi1[t] = 0;
        }
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* q = blk + 2 + 4 * ib;
            const unsigned h = u16(blk + 34 + 2 * ib);
            const __m128i g0 = _mm_set_epi64x((long long) iq1s_grid[q[1] | ((h << 5) & 0x700)],
                                              (long long) iq1s_grid[q[0] | ((h << 8) & 0x700)]);
            const __m128i g1 = _mm_set_epi64x((long long) iq1s_grid[q[3] | ((h >> 1) & 0x700)],
                                              (long long) iq1s_grid[q[2] | ((h << 2) & 0x700)]);
            const int ls = 2 * (int) ((h >> 12) & 7) + 1, sl = (h & 0x8000) ? -ls : ls;
            const __m128i lsv = _mm_set1_epi8((char) ls);
            const __m128i a0 = _mm_sign_epi8(lsv, _mm_sign_epi8(g0, g0)), a1 = _mm_sign_epi8(lsv, _mm_sign_epi8(g1, g1));
            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = y[t][i].qs + 32 * ib;
                acc16[t][0] = _mm_add_epi16(acc16[t][0], _mm_maddubs_epi16(a0, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) q8), g0)));
                acc16[t][1] = _mm_add_epi16(acc16[t][1], _mm_maddubs_epi16(a1, _mm_sign_epi8(_mm_loadu_si128((const __m128i*) (q8 + 16)), g1)));
                sumi1[t] += (y[t][i].bsums[2 * ib] + y[t][i].bsums[2 * ib + 1]) * sl;
            }
        }
        const float dx = h2f(u16(blk));
        for (int t = 0; t < NT; ++t) {
            const float d = y[t][i].d * dx;
            const __m128i s0 = _mm_madd_epi16(acc16[t][0], ones), s1 = _mm_madd_epi16(acc16[t][1], ones);
            const __m256 f = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(s0), s1, 1));
            accf[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), f), accf[t]);
            acc1[t] += d * (float) sumi1[t];
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]) + IQ1S_DELTA * acc1[t];
}

// Q2_K (84-byte blocks: 16 scale / min bytes, 64 bytes of 2-bit codes, d, dmin): ggml's AVX dot, the codes decoded
// once for all tokens with their scale in them (q sc <= 45, a pshufb), two groups summed in int16 before ggml's int32
// lanes; per block the min term is added before the scaled sums, as ggml does
template <int NT>
IQ_AVX1 inline void q2k_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    const __m128i m3 = _mm_set1_epi8(3), m4 = _mm_set1_epi8(0xF), ones = _mm_set1_epi16(1);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* blk = row + (size_t) i * 84;
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        const __m128i ms = _mm_loadu_si128((const __m128i*) blk);
        const __m128i mn = _mm_and_si128(_mm_srli_epi16(ms, 4), m4);
        const __m128i mins0 = _mm_cvtepi8_epi16(mn), mins1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mn, mn));
        const float xd = h2f(u16(blk + 80)), xm = h2f(u16(blk + 82));
        for (int t = 0; t < NT; ++t) {
            const __m128i s0 = _mm_madd_epi16(mins0, _mm_loadu_si128((const __m128i*) &y[t][i].bsums[0]));
            const __m128i s1 = _mm_madd_epi16(mins1, _mm_loadu_si128((const __m128i*) &y[t][i].bsums[8]));
            const float dmin = -y[t][i].d * xm;
            const __m256 f = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(s0), s1, 1));
            accf[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(dmin), f), accf[t]);
        }
        __m128i acci[NT][2];
        for (int t = 0; t < NT; ++t) acci[t][0] = acci[t][1] = _mm_setzero_si128();
        for (int j = 0; j < 2; ++j) {
            const __m128i qa = _mm_loadu_si128((const __m128i*) (blk + 16 + 32 * j));
            const __m128i qb = _mm_loadu_si128((const __m128i*) (blk + 32 + 32 * j));
            for (int k = 0; k < 8; k += 2) {   // (groups k, k + 1: bytes 0-15 and 16-31 of the 32, bits 2 (k / 2))
                const __m128i t0 = _mm_set1_epi32((int) ((unsigned) (blk[8 * j + k] & 0xF) * 0x03020100u));
                const __m128i t1 = _mm_set1_epi32((int) ((unsigned) (blk[8 * j + k + 1] & 0xF) * 0x03020100u));
                const __m128i w0 = _mm_shuffle_epi8(t0, _mm_and_si128(_mm_srli_epi16(qa, k), m3));
                const __m128i w1 = _mm_shuffle_epi8(t1, _mm_and_si128(_mm_srli_epi16(qb, k), m3));
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t][i].qs + 128 * j + 16 * k;
                    const __m128i p = _mm_add_epi16(_mm_maddubs_epi16(w0, _mm_loadu_si128((const __m128i*) q8)),
                                                    _mm_maddubs_epi16(w1, _mm_loadu_si128((const __m128i*) (q8 + 16))));
                    acci[t][k >> 2] = _mm_add_epi32(acci[t][k >> 2], _mm_madd_epi16(p, ones));
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 f = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(acci[t][0]), acci[t][1], 1));
            accf[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(y[t][i].d * xd), f), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int TY, int NT>
IQ_AVX1 inline void any_dot(const uint8_t* row, int nb, const block_q8_K* const* y, float* res) {
    if constexpr (TY == 13) q5k_dot<NT>(row, nb, y, res);
    else if constexpr (TY == 23) iq4xs_dot<NT>(row, nb, y, res);
    else if constexpr (TY == 19) iq1s_dot<NT>(row, nb, y, res);
    else if constexpr (TY == 10) q2k_dot<NT>(row, nb, y, res);
    else row_dot<TY, NT>(row, nb, y, res);
}

template <int TY, int NT>
IQ_AVX1 void rows_nt(const uint8_t* w, size_t rb, int nb, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        any_dot<TY, NT>(w + (size_t) r * rb, nb, y, res);
        for (int t = 0; t < NT; ++t) out[t][r - r0] = res[t];
    }
}

template <int TY>
void rows_t(const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {   // (4 tokens a pass: more spill the 16 xmm registers)
        const int c = nt - t0 < 4 ? nt - t0 : 4;
        switch (c) {
            case 1: rows_nt<TY, 1>(w, rb, n / QK_K, act + t0, out + t0, r0, r1); break;
            case 2: rows_nt<TY, 2>(w, rb, n / QK_K, act + t0, out + t0, r0, r1); break;
            case 3: rows_nt<TY, 3>(w, rb, n / QK_K, act + t0, out + t0, r0, r1); break;
            default: rows_nt<TY, 4>(w, rb, n / QK_K, act + t0, out + t0, r0, r1); break;
        }
    }
}

inline bool supported(int type) {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 19 || type == 10;
}
inline bool supported_glm(int type) { return supported(type) || type == 13 || type == 23; }   // (+ GLM's down types)

// out[t][r - r0], as own_rows
inline void rows(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    switch (type) {
        case 16: rows_t<16>(w, rb, n, act, nt, out, r0, r1); break;
        case 17: rows_t<17>(w, rb, n, act, nt, out, r0, r1); break;
        case 18: rows_t<18>(w, rb, n, act, nt, out, r0, r1); break;
        case 21: rows_t<21>(w, rb, n, act, nt, out, r0, r1); break;
        case 22: rows_t<22>(w, rb, n, act, nt, out, r0, r1); break;
        case 13: rows_t<13>(w, rb, n, act, nt, out, r0, r1); break;
        case 23: rows_t<23>(w, rb, n, act, nt, out, r0, r1); break;
        case 19: rows_t<19>(w, rb, n, act, nt, out, r0, r1); break;
        case 10: rows_t<10>(w, rb, n, act, nt, out, r0, r1); break;
        default: break;
    }
}

}  // namespace avx1
#undef IQ_AVX1
