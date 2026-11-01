// src/ds4/ds4_iq128_kern.inc - the CPU lane's i-quant rows for CPUs with AVX but no AVX2 (Sandy / Ivy Bridge, Xeon E5
// v1 / v2), included by ds4_iq_cpu.cpp.  Without them such a CPU runs ggml-cpu's one-token AVX dots for every expert.
// Each 32-value chunk is decoded once a pass and applied to up to 4 tokens in 128-bit integer lanes (pshufb, sign_epi8,
// maddubs, madd: SSSE3 / SSE2), the two 16-value halves being the lanes of ggml's AVX dot; the float steps are ggml's
// AVX order (256-bit mul then add, no FMA, then K * hsum_float_8), so a token's dot is ggml-cpu's bit for bit at any
// token count: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  Checked against ggml's dot (own()).
#define IQ_AVX1 __attribute__((target("avx,sse4.2,ssse3")))

namespace avx1 {

inline uint16_t u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint64_t u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
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

template <int TY, int NT>
IQ_AVX1 void rows_nt(const uint8_t* w, size_t rb, int nb, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * rb, nb, y, res);
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

bool supported(int type) { return type == 16 || type == 17 || type == 18 || type == 21 || type == 22; }

// out[t][r - r0], as own_rows
void rows(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    switch (type) {
        case 16: rows_t<16>(w, rb, n, act, nt, out, r0, r1); break;
        case 17: rows_t<17>(w, rb, n, act, nt, out, r0, r1); break;
        case 18: rows_t<18>(w, rb, n, act, nt, out, r0, r1); break;
        case 21: rows_t<21>(w, rb, n, act, nt, out, r0, r1); break;
        case 22: rows_t<22>(w, rb, n, act, nt, out, r0, r1); break;
        default: break;
    }
}

}  // namespace avx1
#undef IQ_AVX1
