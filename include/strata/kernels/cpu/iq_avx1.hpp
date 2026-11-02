// include/strata/kernels/cpu/iq_avx1.hpp - multi-token expert rows for CPUs with AVX but no AVX2 (Sandy / Ivy Bridge):
// IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, Q5_K against Q8_K, each token bit-identical to ggml-cpu's AVX dot.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// AVX (the OS saving the YMM state) and SSE4.1 / SSSE3 without AVX2 + FMA: the CPUs these rows are for.
bool cpu_avx1_only() noexcept;
bool iq128_supported(int ggml_type) noexcept;
/// out[t][r] = w_r . a[t], rows [r0, r1) (iq256_rows' arguments).
void iq128_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
