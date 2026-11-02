// src/kernels/cpu/iq_avx1.cpp - GLM's native expert rows on AVX-only CPUs: the shared AVX1 kernels (iq_avx1_kern.inc).
#include "strata/kernels/cpu/iq_avx1.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cstring>

namespace strata::kernels::cpu {

#if defined(_MSC_VER) && !defined(__clang__)   // (the kernels set their ISA with GCC / Clang target attributes)
bool cpu_avx1_only() noexcept { return false; }
bool iq128_supported(int) noexcept { return false; }
void iq128_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
#else
namespace {
#include "strata/kernels/cpu/iq_avx1_kern.inc"
}  // namespace

bool cpu_avx1_only() noexcept {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)   // (clang-cl)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        const unsigned max_leaf = r[0];
        cpuid(1, 0);
        const unsigned ecx = r[2];
        // SSSE3, SSE4.1, SSE4.2, OSXSAVE, AVX
        if (!((ecx >> 9) & 1u) || !((ecx >> 19) & 1u) || !((ecx >> 20) & 1u) || !((ecx >> 27) & 1u) || !((ecx >> 28) & 1u))
            return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;
        const bool fma = (ecx >> 12) & 1u;
        bool avx2 = false;
        if (max_leaf >= 7) {
            cpuid(7, 0);
            avx2 = (r[1] >> 5) & 1u;
        }
        return !(fma && avx2);   // (cpu_avx2_ok's CPUs take the AVX-2 rows)
    }();
    return ok;
}

bool iq128_supported(int ggml_type) noexcept { return avx1::supported_glm(ggml_type); }

void iq128_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1) {
    float* o[64];
    for (int t0 = 0; t0 < nt; t0 += 64) {   // (out[t][r], not avx1::rows' out[t][r - r0])
        const int c = nt - t0 < 64 ? nt - t0 : 64;
        for (int t = 0; t < c; ++t) o[t] = out[t0 + t] + r0;
        avx1::rows(ggml_type, w, row_bytes, n, act + t0, c, o, r0, r1);
    }
}
#endif

}  // namespace strata::kernels::cpu
