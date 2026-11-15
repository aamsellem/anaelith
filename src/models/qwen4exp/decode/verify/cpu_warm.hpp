#pragma once

#include "common.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include <atomic>
#include <vector>

namespace basalt::core {

// The CPU tier's cache warm-up.  After layer l's doorbell the parked pool workers score layer l + 1's router on layer
// l's FFN input (the ring's q8_1 rows; the router in 4-bit with one scale per 32 values, a prediction only: half the
// bytes of an 8-bit copy, so the scoring leaves more of the gap to the read), take every token's top k, rank the
// predicted experts only the CPU holds by their summed weight, and read the likeliest ones into their caches, best
// first (`n_read` of them; one unless --exp warm_n says more).  The next batch stops the work; the experts are computed
// as before.  A pack's layers may differ in their experts' formats, so a blob's size and its down rows' offset are the
// layer's.
struct CpuWarm {
    static constexpr int kMaxRead = 4;                ///< the experts one warm-up reads at most
    std::vector<uint8_t> rq;  ///< [n_layers][n_expert][n_embd / 2]: per 32 values, value i in the low nibble of byte i
                              ///< and value i + 16 in its high nibble, each the code plus 8
    std::vector<float> rs;    ///< [n_layers][n_expert][n_embd / 32]
    int n_expert = 0, n_embd = 0, k = 0;
    int n_read = 1;                                   ///< experts read per layer, likeliest first (1 .. kMaxRead)
    std::vector<size_t> bytes, down_off;              ///< per layer: a blob's bytes, where its down rows start
    ExpertSource* src = nullptr;
    // the job in flight
    int layer = 0, n_tok = 0;
    const uint8_t* xq = nullptr;                      ///< the ring's q8_1 rows, tokens back to back
    const int32_t* res0 = nullptr, *res1 = nullptr;   ///< the layer's CUDA0 / peer residency (res1 may be null)
    std::vector<float> logits;                        ///< [kVerifyMaxT][n_expert]
    alignas(64) std::atomic<int> arrive{0};
    alignas(64) std::atomic<int> ready{0};
    const uint8_t* blob[kMaxRead] = {};               ///< the experts to read, likeliest first (null: none)
    int pick[kMaxRead] = {-1, -1, -1, -1};            ///< their ids
    int n_pick = 0;
    std::atomic<uint64_t> sink{0};
    // How the predictions turned out, for --decode-timing: layers a warm-up was started for, the ones whose selection
    // was done when the layer rang, the experts those picked, the picks the layer then routed to the CPU, and the
    // distinct CPU experts of all those layers.
    uint64_t st_layers = 0, st_ready = 0, st_picks = 0, st_right = 0, st_cpu = 0;
};

namespace verify_detail {
void warm_job(void* ctx, int w, int W, const std::atomic<bool>& stop);
/// Counts how the warm-up of ring `R`'s layer turned out. The pool's idle work must be stopped.
void warm_note(CpuWarm& c, const basalt::kernels::DmRing& R, int k);
bool router_codes(const WeightTable& wt, const ModelGeometry& g, std::vector<int8_t>* q8, std::vector<float>* s8,
                  std::vector<uint8_t>* q4, std::vector<float>* s4);
bool warm_router(const std::vector<int8_t>& q, const std::vector<float>& sc, int8_t** rq, float** rs);
}

}  // namespace basalt::core
