// r4d_ar_add_rmsnorm.hip -- ar_add_rmsnorm: the exact two-rank all-reduce of a block's partial
// output with the residual add and the RMS norm that follow it, in one kernel.
//
// Two forms, selected by `mode`, each byte-identical to the chain it replaces:
//
//   mode 0  `all_reduce (exact) -> rmsnorm -> add`:   y = rmsnorm(a);  x += y
//   mode 1  `all_reduce (exact) -> add -> rmsnorm`:   x += a;          y = rmsnorm(x)
//
// Both are the drafter's two reductions a layer (hyv4flash_dfly.h): the output projection's partial
// sum, post-normed and folded into the residual stream, and the MLP's partial sum folded into the
// stream before the next layer's input norm. Each is three launches unfused; this is one.
//
// Bit-identity, piece by piece:
//   * the fold is r4d_ar_oneshot_2rank_exact's arithmetic: the f32 sum of the two bf16 values,
//     rounded once through (__hip_bfloat16) (bf16_hi_rne_nan, r4d_common.h), stored back into `a`
//     as the standalone all-reduce stores it -- so a later reader of `a` sees the same bytes;
//   * `x += a` (or `x += y`) is add_bf16's word: bf16(f32(x) + f32(a)), f32_to_bf16;
//   * the norm is rmsnorm_bf16's: the sum of squares in kR4dSumLanes' partition (r4d_common.h),
//     block_sum's wave butterfly then the waves from the first, rsqrtf(ss/n + eps), and
//     bf16(f32(v) * s * (g + wadd) * mult) with the same expression order (libr4d builds with
//     -ffp-contract=off, so no product is contracted into an FMA).
// The width is r4d_act_threads(M, n), rmsnorm_bf16's; the sums fold in the same partition at every
// width, so the width and the block count change no bytes.
//
// A block owns whole rows (a norm closes over a row), so the block count is clamped to M and both
// ranks derive the partition from (M, gridDim.x) with integer arithmetic. Exact wire only: the
// plugin's all_reduce declares `exact` 1, and a norm reads the sum.
//
// Returns a negative code rather than throwing, as the other fused all-reduces do.
#include "r4d.h"
#include "r4d_ar_seq.h"   // the handshake, the push check and the device-resident sequence
/* The coded push's slot layout is device code; the host pass (tests/r4d_emu) has no
 * shuffles and no make_int4, and the coded path is guarded out of it below. */
#if defined(__HIP_DEVICE_COMPILE__)
#include "r4d_ar_arc2.h"
#endif

namespace {
/* Whether a message of n16 chunks fits the coded form in a slot of slot_stride16 words: whole
 * 32-chunk waves, a P2P instance (`stage` 0), and the coded image (16.125 bytes a chunk) inside the
 * slot. The same rule r4d_ar_coded2_ok applies (r4d_ar_oneshot_2rank_exact.hip); repeated here
 * because that symbol is not in the host build of this file, and the launcher has to refuse on the
 * host pass too. */
bool arar_coded_ok(long n16, long slot_stride16, long stage) {
    if (stage != 0 || n16 <= 0 || (n16 & 31L)) return false;
    const long spill = (12L * n16 + n16 / 8L + 15L) & ~15L;
    return spill + 4L * n16 <= slot_stride16 * 16L;
}
}  // namespace
#include "r4d_chain.h"    // R4dUnchained / the chained form
#include "r4d_common.h"   // r4d_act_threads, kR4dSumLanes, r4d_lanes_sumsq, f32_to_bf16
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kSumLanes = kR4dSumLanes;

#define R4D_ARAR_SPIN_MAX 4000000000ULL

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// Round-to-nearest-even, as f32_to_bf16 and every bf16 store in this library round.
__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// The all-reduce's rounding: what (__hip_bfloat16) does, which is what vadd in
// r4d_ar_oneshot_2rank_exact.hip stores and the chain's norm then reads.
__device__ __forceinline__ uint16_t ar_bf16(float f) {
    return (uint16_t)(bf16_hi_rne_nan(f) >> 16);
}

__device__ __forceinline__ void arap_store_rel(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arap_load_acq(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// rmsnorm_bf16's block_sum (r4d_model_bf16.hip): the wave butterfly, then the wave sums from 0.0f
// in order. The waves past the eighth hold 0.0f at any width (kR4dSumLanes' partition), so the
// width changes no bits.
template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// MODE 0: y = rmsnorm(a), then x += y.  MODE 1: x += a, then y = rmsnorm(x).
// `a` is the message and is reduced in place; `x` is the residual stream; `w` the gain.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <bool WF32, int MODE, uint32_t THR, class CH>
__global__ __launch_bounds__(THR) void ar_add_rmsnorm_kernel(
    uint16_t* __restrict__ a, uint16_t* __restrict__ x, const void* __restrict__ w,
    uint16_t* __restrict__ y, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, int M, uint32_t n, float eps, float wadd,
    float mult, uint32_t x_ld, uint32_t y_ld, int slot_stride16, int drain, int acq,
    unsigned int nbw, long stage, int coded) {
    if (CH::on) r4d_chain_wait();
    __shared__ float lds[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
#if defined(__HIP_DEVICE_COMPILE__)
    extern __shared__ uint8_t arar_stage[];   // coded: one decoded row (n bf16)
#endif
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    /* Integer arithmetic only, so both ranks land on the same split. */
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);

    r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int sq = s_seq;
    int4* const ps = peer_base + (size_t)s_slot * slot_stride16;
    const int4* ms = my_base + (size_t)s_slot * slot_stride16;
    const long md = r4d_archk_md(stage, my_base);    // the host mailbox's VRAM mirror

    /* The message: my rows of `a`, pushed. `a` is a tight [M][n] of bf16 (the launcher refuses a
     * pitched one), so a row range is a 16B-word range. */
    const int wpr = (int)(n >> 3);                   // 16B words a row
    const int4* const a4 = (const int4*)a;
    /* The coded form (wire 6): whole 32-chunk groups, a wave a group, the slot the launcher checked
     * (r4d_ar_coded2_ok). Each row is whole groups (the launcher refuses n % 256), so a block's row
     * range is a group range and the reader sums and decodes exactly what was pushed. */
    const int gpr = (int)(wpr >> 5);                 // 32-chunk groups a row
    const uint32_t lane = tid & (kWave - 1u), wv = tid >> 5;
    constexpr uint32_t NWV = THR / kWave;
    unsigned int hck = 0u;
#if defined(__HIP_DEVICE_COMPILE__)
    if (coded) {
        const R4dArc2 psc = r4d_arc2_at(ps, (int)((long)M * wpr));
        for (int row = r0; row < r1; ++row)
            for (int g = (int)wv; g < gpr; g += (int)NWV)
                hck += r4d_arc2_put(psc, row * wpr + g * (int)kWave, a4[row * wpr + g * (int)kWave + (int)lane]);
    } else
#endif
    for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR) {
        const int4 v = a4[k];
        ps[k] = v;
        hck += r4d_archk_u4(v, 4u * (unsigned)k);
    }
    r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)b, sq);
    /* The drain modes are r4d_ar_oneshot_2rank_exact's, selected by the same variant table. The
     * s_wait_storecnt is an instruction, so the host build of this file (tests/r4d_emu) skips it:
     * there is no posted write to drain on one thread. */
    if (drain == 1) __threadfence_system();
#if defined(__HIP_DEVICE_COMPILE__)
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
    __syncthreads();
    if (tid == 0) {
        if (drain == 2) __threadfence_system();
        arap_store_rel(&peer_flags[b], sq);
        unsigned long long z = 0;
        /* A handshake that cannot complete is a lockstep violation; trap rather than norm a slot
         * the peer never filled. */
        while (r4d_ar_seq_behind(arap_load_acq(&my_flags[b]), sq)) {
            if (++z > R4D_ARAR_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    if (acq) __threadfence_system();
    r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)b, sq, [&](unsigned int& h) {
#if defined(__HIP_DEVICE_COMPILE__)
        if (coded) {
            const R4dArc2 msc = r4d_arc2_at(ms, (int)((long)M * wpr));
            for (int row = r0; row < r1; ++row)
                for (int g = (int)wv; g < gpr; g += (int)NWV)
                    h += r4d_arc2_sum(msc, row * wpr + g * (int)kWave);
            return;
        }
#endif
        const int4* const m4 = (const int4*)ms;
        for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR)
            h += r4d_archk_u4(r4d_archk_keep(&m4[k], md), 4u * (unsigned)k);
    });
    ms = r4d_archk_moved(ms, md);
    const uint16_t* __restrict__ P = (const uint16_t*)ms;   // the peer's rows, in a's layout

    for (int row = r0; row < r1; ++row) {
        uint16_t* __restrict__ ar = a + (size_t)row * n;
        /* Coded: decode this row's groups into shared first, so pass one reads the same bf16 words
         * in the same thread order and the sum still folds in kR4dSumLanes' partition. */
        const uint16_t* __restrict__ pr;
#if defined(__HIP_DEVICE_COMPILE__)
        if (coded) {
            const R4dArc2 msc = r4d_arc2_at(ms, (int)((long)M * wpr));
            for (int g = (int)wv; g < gpr; g += (int)NWV)
                *reinterpret_cast<int4*>(arar_stage + (size_t)(g * (int)kWave + (int)lane) * 16u) =
                    r4d_arc2_get(msc, row * wpr + g * (int)kWave);
            __syncthreads();
            pr = reinterpret_cast<const uint16_t*>(arar_stage);
        } else
#endif
        pr = P + (size_t)row * n;
        uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
        uint16_t* __restrict__ yr = y + (size_t)row * y_ld;
        /* MODE 0 norms the reduced message; MODE 1 norms the stream after the add. */
        const uint16_t* __restrict__ nr = MODE == 0 ? ar : xr;

        /* Pass one: the fold (and, mode 1, the residual add), then the sum of squares. */
        float ss = 0.0f;
        for (uint32_t i = tid; i < n; i += THR) {
            const uint16_t av = ar_bf16(bf2f(ar[i]) + bf2f(pr[i]));
            ar[i] = av;
            if (MODE == 1) {
                const uint16_t h = f2bf(bf2f(xr[i]) + bf2f(av));
                xr[i] = h;
                if constexpr (THR == kSumLanes) { const float v = bf2f(h); ss += v * v; }
            } else {
                if constexpr (THR == kSumLanes) { const float v = bf2f(av); ss += v * v; }
            }
        }
        if constexpr (THR != kSumLanes) {
            __syncthreads();   /* the row, stored across the block's threads */
            ss = r4d_lanes_sumsq(nr, n, [](uint16_t h) { return bf2f(h); });
        }
        /* block_sum's barrier is also what publishes pass one's stores to pass two's reads. */
        const float rs = rsqrtf(block_sum<THR / kWave>(ss, lds) / (float)n + eps);

        /* Pass two: the norm, and (mode 0) the add of its output into the stream. */
        for (uint32_t i = tid; i < n; i += THR) {
            const float g = !w ? 1.0f : WF32 ? ((const float*)w)[i]
                                             : bf2f(((const uint16_t*)w)[i]);
            const uint16_t h = f2bf(bf2f(nr[i]) * rs * (g + wadd) * mult);
            yr[i] = h;
            if (MODE == 0) xr[i] = f2bf(bf2f(xr[i]) + bf2f(h));
        }
        /* lds is reused by this block's next row; without the barrier a thread still reading it
         * would race the next row's write. */
        __syncthreads();
    }
    if (CH::on) r4d_chain_done();
}

}  // namespace

/* `w` is the norm gain, f32 when `w_f32` and bf16 otherwise (as rmsnorm_bf16 reads it). `a` is the
 * message and takes no pitch, since it is also the pushed buffer; `x` and `y` may be pitched.
 *
 *   a  [M][n]   bf16, inout: this rank's partial sum, reduced in place
 *   x  [M][n]   bf16, inout: the residual stream, `x_ld` elements a row (0 = n)
 *   w  [n]      f32 or bf16
 *   y  [M][n]   bf16, `y_ld` elements a row (0 = n)
 *
 * `mode` 0: y = rmsnorm(a), then x += y.  `mode` 1: x += a, then y = rmsnorm(x). */
extern "C" int r4d_ar_add_rmsnorm(long peer_scratch, long my_scratch, long peer_flags,
                                  long my_flags, long seq_ctrs, long slot_stride16,
                                  long a, long x, long w, long y,
                                  int M, int n, float eps, float wadd, float mult, int w_f32,
                                  int x_ld, int y_ld, long stream, int nblocks, int nthreads,
                                  int drain, int acq, int mode, long stage, long wire) {
    if (!a || !x || !w || !y) return -1;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (mode != 0 && mode != 1) return -3;
    /* n a multiple of 8: n bf16 is whole 16B words, which the push and the message need. */
    if (n % 8) return -5;
    /* Wire 6 is the coded push: whole 256-value waves (every row whole 32-chunk groups), a P2P
     * instance, and the coded image inside the slot. A caller that asks for it and does not fit gets
     * the refusal, not a silent fall back to exact. */
    const int coded = wire == 6;
    if (wire != 0 && !coded) return -7;
    if (coded && (n % 256 || !arar_coded_ok((long)M * (n >> 3), slot_stride16, stage)))
        return -7;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;

    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;                       /* a block owns whole rows */
    const int nbw = r4d_ar_max_blocks();      /* the counter capacity; see r4d_ar_seq.h */
    if (nb > nbw) nb = nbw;
    /* rmsnorm_bf16's width; the sums fold in kR4dSumLanes' partition at any of the three. */
    const int thr = nthreads ? nthreads : r4d_act_threads(M, n);
    if (thr != 256 && thr != 512 && thr != 1024) return -6;

#define R4D_ARAR_K(wf, md, T)                                                                    \
    hipLaunchKernelGGL((ar_add_rmsnorm_kernel<wf, md, T, R4dUnchained>), dim3((uint32_t)nb),      \
                       dim3(T), coded ? (size_t)n * 2u : 0u, (hipStream_t)stream, (uint16_t*)a, (uint16_t*)x,              \
                       (const void*)w, (uint16_t*)y, (int4*)peer_scratch, (const int4*)my_scratch, \
                       (unsigned int*)peer_flags, (unsigned int*)my_flags,                       \
                       (unsigned int*)seq_ctrs, M, (uint32_t)n, eps, wadd, mult,                 \
                       (uint32_t)x_ld, (uint32_t)y_ld, (int)slot_stride16, drain, acq,           \
                       (unsigned)nbw, stage, coded)
#define R4D_ARAR_M(wf, md)                                                                       \
    do { if (thr == 1024)      R4D_ARAR_K(wf, md, 1024);                                         \
         else if (thr == 512)  R4D_ARAR_K(wf, md, 512);                                          \
         else                  R4D_ARAR_K(wf, md, 256); } while (0)
    if (mode == 0) { if (w_f32) R4D_ARAR_M(true, 0);  else R4D_ARAR_M(false, 0); }
    else           { if (w_f32) R4D_ARAR_M(true, 1);  else R4D_ARAR_M(false, 1); }
#undef R4D_ARAR_M
#undef R4D_ARAR_K
    return hipGetLastError() == hipSuccess ? 0 : -4;
}