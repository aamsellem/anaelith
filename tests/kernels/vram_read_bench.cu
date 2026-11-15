// tests/kernels/vram_read_bench.cu - what a kernel that only reads VRAM gets on a card, for a few access shapes and
// block counts: the rate the expert kernels' own can be read against.  Not built by CMake:
//
//     nvcc -O3 -std=c++17 -o vram_read_bench tests/kernels/vram_read_bench.cu && ./vram_read_bench [MiB] [iters]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// every thread reads W consecutive 32-bit words of its own; the 32 lanes of a warp read 32 * 4 * W contiguous bytes
template<int W>
__global__ void read_words(const uint32_t* __restrict__ p, size_t n_words, uint32_t* out) {
    const size_t per = (size_t) blockDim.x * W, stride = (size_t) gridDim.x * per;
    uint32_t s = 0;
    for (size_t i = (size_t) blockIdx.x * per + (size_t) threadIdx.x * W; i + W <= n_words; i += stride)
#pragma unroll
        for (int k = 0; k < W; ++k) s += __ldg(p + i + k);
    if (s == 0xdeadbeefu) out[0] = s;
}
// one 16-byte load a thread: a warp reads 512 contiguous bytes
__global__ void read_u4(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint32_t s = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 v = __ldg(p + i);
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 0xdeadbeefu) out[0] = s;
}
// the expert rows' shape: each lane reads one 4-byte word 10 bytes from the next lane's (two aligned loads and a shift)
__global__ void read_unaligned(const uint8_t* __restrict__ p, size_t n_bytes, uint32_t* out) {
    const size_t per = (size_t) blockDim.x * 10, stride = (size_t) gridDim.x * per;
    uint32_t s = 0;
    for (size_t i = (size_t) blockIdx.x * per + (size_t) threadIdx.x * 10; i + 16 <= n_bytes; i += stride) {
        const uint8_t* q = p + i + 2;
        const uint32_t* a = (const uint32_t*) ((uintptr_t) q & ~(uintptr_t) 3);
        s += __funnelshift_r(__ldg(a), __ldg(a + 1), ((uint32_t) (uintptr_t) q & 2u) << 3);
        s += __ldg(a + 1) ^ __ldg(a + 2);
    }
    if (s == 0xdeadbeefu) out[0] = s;
}

template<typename F> static double time_us(F&& launch, int iters) {
    launch(); cudaDeviceSynchronize();
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) launch();
        cudaDeviceSynchronize();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        if (us < best) best = us;
    }
    return best;
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::atoi(argv[1]) : 71;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const size_t pool = (size_t) 4096 << 20, bytes = mib << 20;       // read a different 71 MiB of a 4 GiB pool each time
    uint8_t* buf; uint32_t* out;
    if (cudaMalloc(&buf, pool) != cudaSuccess || cudaMalloc(&out, 64) != cudaSuccess) { std::puts("alloc failed"); return 1; }
    cudaMemset(buf, 1, pool);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    size_t off = 0;
    auto next = [&] { off = (off + bytes + (64 << 20)) % (pool - bytes); off &= ~(size_t) 4095; return buf + off; };
    std::printf("%d SMs, %zu MiB per launch\n", sms, mib);
    for (int bps : {1, 2, 4, 8}) {
        const int blocks = sms * bps;
        const double a = time_us([&] { read_words<1><<<blocks, 256>>>((const uint32_t*) next(), bytes / 4, out); }, iters);
        const double b = time_us([&] { read_words<4><<<blocks, 256>>>((const uint32_t*) next(), bytes / 4, out); }, iters);
        const double c = time_us([&] { read_u4<<<blocks, 256>>>((const uint4*) next(), bytes / 16, out); }, iters);
        const double d = time_us([&] { read_unaligned<<<blocks, 256>>>(next(), bytes, out); }, iters);
        std::printf("  %d blocks an SM: 4-byte words %6.1f us %4.0f GB/s | 4 words a thread %6.1f us %4.0f GB/s | 16-byte loads %6.1f us %4.0f GB/s"
                    " | expert-like unaligned %6.1f us %4.0f GB/s\n", bps, a, bytes / a / 1e3, b, bytes / b / 1e3, c,
                    bytes / c / 1e3, d, bytes / d / 1e3);
    }
    return 0;
}
