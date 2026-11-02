// tests/ds4_gguf_expert_bench.cpp - how fast does this CPU run one routed expert of a DeepSeek GGUF's layer, through
// the CPU lane's calls (iq_cpu_gate_up / iq_cpu_down, whichever rows they take here)?  glm_cpu_bench for ds4.
//
//   ds4_gguf_expert_bench <shard.gguf> <layer> [threads...]      DS4_BENCH_TOKENS=1,2,4,8 (default)
//
// Reads the layer's first 64 experts from the shard (gate / up / down _exps, any types the lane takes) - far past the
// L3, so each comes from DRAM like a RAM-tier pick - and times gate/up then down in 64-row jobs on a pool (the caller
// helping), as the lane runs them.  DS4_IQ_OWN=0 STRATA_NO_IQ128=1: ggml-cpu's dots, to compare.
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <shard.gguf> <layer> [threads...]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int layer = std::atoi(argv[2]);
    ggml_context* meta = nullptr;
    gguf_init_params ip{true, &meta};
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (g == nullptr) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        return 1;
    }
    const char* parts[3] = {"gate", "up", "down"};
    int ty[3];
    uint64_t off[3];
    int64_t ne[3][3];
    for (int p = 0; p < 3; ++p) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + parts[p] + "_exps.weight";
        const int64_t id = gguf_find_tensor(g, name.c_str());
        const ggml_tensor* t = id < 0 ? nullptr : ggml_get_tensor(meta, name.c_str());
        if (t == nullptr) {
            std::fprintf(stderr, "%s not in %s\n", name.c_str(), path.c_str());
            return 1;
        }
        ty[p] = (int) t->type;
        off[p] = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
        for (int k = 0; k < 3; ++k) ne[p][k] = t->ne[k];
    }
    if (ty[0] != ty[1]) {
        std::fprintf(stderr, "gate %s and up %s differ\n", ggml_type_name((ggml_type) ty[0]), ggml_type_name((ggml_type) ty[1]));
        return 1;
    }
    const int E = (int) ne[0][0], FF = (int) ne[0][1], n_exp = (int) std::min<int64_t>(64, ne[0][2]);
    gg::IqCpu f;
    std::string err;
    if (!gg::iq_cpu_fmt(ty[0], ty[2], E, FF, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const size_t gub = f.gu_row * (size_t) FF, dnb = f.d_row * (size_t) E, eb = 2 * gub + dnb;
    std::vector<uint8_t> blobs(eb * (size_t) n_exp);
    const int fd = open(path.c_str(), O_RDONLY);
    for (int e = 0; e < n_exp; ++e) {
        uint8_t* b = blobs.data() + eb * (size_t) e;
        if (fd < 0 || pread(fd, b, gub, (off_t) (off[0] + e * gub)) != (ssize_t) gub ||
            pread(fd, b + gub, gub, (off_t) (off[1] + e * gub)) != (ssize_t) gub ||
            pread(fd, b + 2 * gub, dnb, (off_t) (off[2] + e * dnb)) != (ssize_t) dnb) {
            std::fprintf(stderr, "short read\n");
            return 1;
        }
    }
    std::vector<int> toks = {1, 2, 4, 8};
    if (const char* v = std::getenv("DS4_BENCH_TOKENS")) {
        toks.clear();
        std::stringstream ss(v);
        for (std::string s; std::getline(ss, s, ',');) toks.push_back(std::max(1, std::min(16, std::atoi(s.c_str()))));
    }
    const int K = *std::max_element(toks.begin(), toks.end());
    std::vector<uint8_t> act((size_t) K * f.act_bytes), hq((size_t) K * f.h_bytes);
    std::vector<float> x((size_t) E), ff((size_t) K * FF), dn((size_t) K * E);
    std::vector<const void*> ap(K), hp(K);
    std::vector<float*> fp(K), dp(K);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < E; ++i) x[(size_t) i] = std::sin(0.37f * (float) i + 0.11f * (float) k) * 0.5f;
        gg::iq_cpu_quant(f.gu_act, x.data(), E, act.data() + (size_t) k * f.act_bytes);
        ap[k] = act.data() + (size_t) k * f.act_bytes;
        hp[k] = hq.data() + (size_t) k * f.h_bytes;
        fp[k] = ff.data() + (size_t) k * FF;
        dp[k] = dn.data() + (size_t) k * E;
    }
    std::vector<int> counts;
    for (int i = 3; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
    if (counts.empty()) counts = {1, 5};
    std::printf("layer %d: gate/up %s, down %s, %d x %d, expert %.2f MB, %d experts cycled from DRAM; own rows gate/up %s "
                "down %s, native multi-token rows gate/up %s down %s\n", layer, ggml_type_name((ggml_type) ty[0]),
                ggml_type_name((ggml_type) ty[2]), E, FF, (double) eb / 1e6, n_exp, gg::iq_cpu_own_rows(ty[0]) ? "on" : "off",
                gg::iq_cpu_own_rows(ty[2]) ? "on" : "off",
                strata::kernels::cpu::native_multi_rows_exact(ty[0]) ? "on" : "off",
                strata::kernels::cpu::native_multi_rows_exact(ty[2]) ? "on" : "off");
    constexpr int RB = 64;
    for (int nth : counts) {
        strata::ds4::Pool pool(std::max(0, nth - 1));
        for (int nt : toks) {
            const int reps = 4 * n_exp;
            double t_gu = 0, t_dn = 0;
            for (int rep = 0; rep < reps; ++rep) {
                const uint8_t* b = blobs.data() + eb * (size_t) (rep % n_exp);
                const auto t0 = std::chrono::steady_clock::now();
                pool.run((FF + RB - 1) / RB, [&](int j) {
                    gg::iq_cpu_gate_up(f, b, b + gub, ap.data(), nt, fp.data(), j * RB, std::min(FF, (j + 1) * RB), 10.0f);
                });
                const auto t1 = std::chrono::steady_clock::now();
                for (int k = 0; k < nt; ++k) gg::iq_cpu_quant(f.d_act, fp[k], FF, (void*) hp[k]);
                const auto t2 = std::chrono::steady_clock::now();
                pool.run((E + RB - 1) / RB, [&](int j) {
                    gg::iq_cpu_down(f, b + 2 * gub, hp.data(), nt, dp.data(), j * RB, std::min(E, (j + 1) * RB));
                });
                const auto t3 = std::chrono::steady_clock::now();
                if (rep < n_exp) continue;   // (the first lap warms the pool and the clocks)
                t_gu += std::chrono::duration<double, std::micro>(t1 - t0).count();
                t_dn += std::chrono::duration<double, std::micro>(t3 - t2).count();
            }
            const double n = reps - n_exp;
            std::printf("threads %2d tokens %2d: gate/up %8.1f us  down %8.1f us  expert %8.1f us (%.1f us a token)\n", nth,
                        nt, t_gu / n, t_dn / n, (t_gu + t_dn) / n, (t_gu + t_dn) / n / nt);
        }
    }
    gguf_free(g);
    ggml_free(meta);
    return 0;
}
