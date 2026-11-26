/* tests/r4d_emu: ar_add_rmsnorm (libr4d/r4d_ar_add_rmsnorm.hip) on the host SIMT emulator.
 *
 * The exact two-rank all-reduce with the residual add and the RMS norm beside it, in one kernel, in
 * both orders the drafter's layers have them:
 *
 *   mode 0   all_reduce (exact) -> rmsnorm -> add:   y = rmsnorm(a);  x += y
 *   mode 1   all_reduce (exact) -> add -> rmsnorm:   x += a;          y = rmsnorm(x)
 *
 * Reference: the unfused chain, in libr4d's per-element arithmetic -- the standalone exact
 * all-reduce's fold (an f32 sum of the two bf16 values rounded once), add_bf16's word
 * (bf16(f32(x) + f32(a))), and rmsnorm_bf16's norm (the sum of squares in kR4dSumLanes' partition,
 * a butterfly a wave then the waves from zero, rsqrtf(ss/n + eps), bf16(f32(v) * s * g) -- the
 * same helpers tests/r4d_emu/test_norm.cpp uses). The fused kernel must match it bit for bit:
 * fusing changes only the launch count.
 *
 * What the cases cover: both modes; every width the row may be tuned to (256, 512, 1024), since
 * the sums fold in the same partition at each and no width may change a byte; a block owning
 * several rows as well as one; an f32 and a bf16 gain; rows and a width past the 1024-thread
 * kernel's register-bound shape; and the peer's message differing from this rank's in every element,
 * so a kernel that read its own slot instead of the peer's would fail. The push is checked too: the
 * block must have sent exactly this rank's bytes, before the fold wrote over them.
 *
 * The emulator runs one block after another on one thread, so the two ranks of an instance are one
 * process: the peer's half is written into the slot the kernel reads before the launch, and the
 * handshake's flag and push-check word are prefilled with what the peer's own launch would have
 * left there (r4d_ar_seq.h: seq 1 and checked sum + seq * R4D_ARCHK_SALT, since the counters start
 * at zero, and the parity word every block takes on its first launch is slot 1).
 */
#include "rad_abi.h"
#include "r4d.h"
#include "r4d_ar_seq.h"   /* the push check's word and mix, to prefill what the peer's launch leaves */
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(97);

/* rmsnorm_bf16's sum of squares: 256 lanes strided over the row, a butterfly a wave, the waves in
 * order (tests/r4d_emu/test_norm.cpp's block_factor, the same partition). */
static float block_factor(const std::vector<float>& v, int n, float eps) {
    float part[256];
    for (int t = 0; t < 256; ++t) {
        float s = 0.0f;
        for (int k = 0; k < 32; ++k) { const int i = t + 256 * k; const float x = i < n ? v[(size_t)i] : 0.0f; s += x * x; }
        part[t] = s;
    }
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) {
        float l[32];
        for (int i = 0; i < 32; ++i) l[i] = part[w * 32 + i];
        for (int d = 16; d >= 1; d >>= 1) {
            float nl[32];
            for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
            std::memcpy(l, nl, sizeof l);
        }
        total += l[0];
    }
    return 1.0f / std::sqrt(total / (float)n + eps);
}

static std::vector<uint16_t> randb(size_t k, float sd) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<uint16_t> v(k);
    for (auto& x : v) x = f2b(d(rng));
    return v;
}

struct Case {
    int mode = 0, M = 0, n = 0, thr = 0, nb = 1;
    bool w_bf16 = false;
};

static int run_case(const Case& c) {
    const float eps = 1e-6f;
    const int n = c.n, M = c.M;
    const size_t els = (size_t)M * n;

    std::vector<uint16_t> a0 = randb(els, 3.f);        /* this rank's partial sum */
    std::vector<uint16_t> peer = randb(els, 3.f);      /* the peer's, different in every element */
    std::vector<uint16_t> x0 = randb(els, 1.f);        /* the residual stream */
    std::vector<float> gf((size_t)n);
    { std::normal_distribution<float> d(1.f, 0.4f); for (auto& g : gf) g = d(rng); }
    std::vector<uint16_t> gw((size_t)n);
    for (int i = 0; i < n; ++i) gw[(size_t)i] = f2b(gf[(size_t)i]);
    /* the gain as the kernel reads it: the bf16 weight widened back, or the f32 one as it is */
    for (int i = 0; i < n; ++i)
        if (c.w_bf16) gf[(size_t)i] = b2f(gw[(size_t)i]);

    /* The chain, before the launch: the fold, then the add and the norm in the mode's order. */
    std::vector<uint16_t> ca(els), cx = x0, cy(els);
    for (int m = 0; m < M; ++m) {
        const size_t o = (size_t)m * n;
        std::vector<float> v((size_t)n);
        for (int i = 0; i < n; ++i) {
            ca[o + (size_t)i] = f2b(b2f(a0[o + (size_t)i]) + b2f(peer[o + (size_t)i]));
            v[(size_t)i] = b2f(ca[o + (size_t)i]);
        }
        if (c.mode == 0) {
            const float f = block_factor(v, n, eps);
            for (int i = 0; i < n; ++i) {
                const uint16_t y = f2b(rb((v[(size_t)i] * f) * gf[(size_t)i]));      /* rmsnorm */
                cy[o + (size_t)i] = y;
                cx[o + (size_t)i] = f2b(b2f(cx[o + (size_t)i]) + b2f(y));            /* add */
            }
        } else {
            std::vector<float> xs((size_t)n);
            for (int i = 0; i < n; ++i) {
                cx[o + (size_t)i] = f2b(b2f(cx[o + (size_t)i]) + b2f(ca[o + (size_t)i]));   /* add */
                xs[(size_t)i] = b2f(cx[o + (size_t)i]);
            }
            const float f = block_factor(xs, n, eps);
            for (int i = 0; i < n; ++i)
                cy[o + (size_t)i] = f2b(rb((xs[(size_t)i] * f) * gf[(size_t)i]));       /* rmsnorm */
        }
    }

    /* The instance's buffers: what the peer's launch would have left in this rank's half. The
     * emulator is one process, so the peer's message is written into the slot the kernel reads and
     * the handshake is prefilled. */
    const int nbw = r4d_ar_max_blocks();
    const int nb = c.nb < 1 ? 1 : (c.nb > M ? M : c.nb);
    const int wpr = n / 8;                            /* 16B words a row */
    const size_t slot16 = ((els * 2 + 15) / 16) + 8;  /* the message, a slot, and slack */
    std::vector<int4> my_scratch(slot16 * 2), peer_scratch(slot16 * 2);
    std::vector<unsigned int> my_flags((size_t)3 * nbw + 16, 0u), peer_flags((size_t)3 * nbw + 16, 0u);
    std::vector<unsigned int> ctrs((size_t)2 * nbw, 0u);
    /* slot 1 and seq 1: par[b] starts at 0, so (0 + 1) & 1 = 1, and ctrs[b] + 1 = 1. */
    const unsigned int seq = 1u, slot = 1u;
    std::memcpy((uint16_t*)my_scratch.data() + (size_t)slot * slot16 * 8, peer.data(), els * 2);
    for (int b = 0; b < nb; ++b) {
        my_flags[(size_t)b] = seq;                    /* the peer's flag */
        const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
        unsigned int t = 0u;
        for (int k = r0 * wpr; k < r1 * wpr; ++k)
            t += r4d_archk_u4(((const int4*)peer.data())[k], 4u * (unsigned)k);
        *r4d_archk_word(my_flags.data(), (unsigned)nbw, slot, (unsigned)b) = t + seq * R4D_ARCHK_SALT;
    }

    std::vector<uint16_t> x = x0, y(els, 0u);
    const std::vector<uint16_t> a_orig = a0;   /* what the block must push, before the fold */
    const int rc = r4d_ar_add_rmsnorm(
        (long)(uintptr_t)peer_scratch.data(), (long)(uintptr_t)my_scratch.data(),
        (long)(uintptr_t)peer_flags.data(), (long)(uintptr_t)my_flags.data(),
        (long)(uintptr_t)ctrs.data(), (long)slot16, (long)(uintptr_t)a0.data(),
        (long)(uintptr_t)x.data(), (long)(uintptr_t)(c.w_bf16 ? (const void*)gw.data() : (const void*)gf.data()),
        (long)(uintptr_t)y.data(), M, n, eps, 0.0f, 1.0f, c.w_bf16 ? 0 : 1, 0, 0,
        (long)0, nb, c.thr, 3, 0, c.mode, 0);
    if (rc != 0) {
        std::printf("ar_add_rmsnorm mode %d M %3d n %4d thr %4d nb %2d w %s: launch refused (%d)\n",
                    c.mode, M, n, c.thr, nb, c.w_bf16 ? "bf16" : "f32", rc);
        return 1;
    }

    long bad_a = 0, bad_x = 0, bad_y = 0, bad_push = 0;
    for (size_t i = 0; i < els; ++i) {
        bad_a += a0[i] != ca[i];
        bad_x += x[i] != cx[i];
        bad_y += y[i] != cy[i];
    }
    /* the push: the block sent this rank's bytes, before the fold wrote over them */
    const uint16_t* ps = (const uint16_t*)peer_scratch.data() + (size_t)slot * slot16 * 8;
    for (size_t i = 0; i < els; ++i) bad_push += ps[i] != a_orig[i];

    const bool ok = bad_a == 0 && bad_x == 0 && bad_y == 0 && bad_push == 0;
    std::printf("ar_add_rmsnorm mode %d M %3d n %4d thr %4d nb %2d w %-4s: a %ld x %ld y %ld push %ld "
                "differ -> %s\n", c.mode, M, n, c.thr, nb, c.w_bf16 ? "bf16" : "f32", bad_a, bad_x,
                bad_y, bad_push, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    /* both modes at the drafter's width and heights: one row, a draft block of six, a full 48-row
     * block; one block a row, and two blocks sharing them */
    for (int mode = 0; mode < 2; ++mode) {
        for (int M : { 1, 6, 48 }) {
            for (int nb : { 1, 2 }) {
                if (nb > M) continue;
                for (int thr : { 256, 512, 1024 })
                    fails += run_case({ mode, M, 5120, thr, nb, false });
            }
        }
    }
    /* a width past the 1024-thread kernel's shape, the 1024-thread floor, and a bf16 gain */
    fails += run_case({ 0, 6, 1000, 1024, 6, false });
    fails += run_case({ 1, 6, 1000, 512, 3, true });
    fails += run_case({ 0, 3, 5128, 512, 3, true });
    fails += run_case({ 1, 3, 5128, 1024, 1, false });

    /* the refusals: n not whole 16B words, a mode it does not serve, no rows, no gain */
    {
        std::vector<uint16_t> a(64), x(64), y(64);
        std::vector<int4> s(64);
        std::vector<unsigned int> f(64), c(64);
        float g[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        struct Bad { const char* what; int mode, M, n; int want; } bad[] = {
            { "n not whole 16B words", 0, 1, 12, -5 },
            { "mode 2", 2, 1, 8, -3 },
            { "no rows", 0, 0, 8, -2 },
        };
        for (const Bad& b : bad) {
            const int rc = r4d_ar_add_rmsnorm((long)(uintptr_t)s.data(), (long)(uintptr_t)s.data(),
                                              (long)(uintptr_t)f.data(), (long)(uintptr_t)f.data(),
                                              (long)(uintptr_t)c.data(), 8, (long)(uintptr_t)a.data(),
                                              (long)(uintptr_t)x.data(), (long)(uintptr_t)g,
                                              (long)(uintptr_t)y.data(), b.M, b.n, 1e-6f, 0.f, 1.f,
                                              1, 0, 0, 0, 1, 256, 3, 0, b.mode, 0);
            if (rc != b.want) { std::printf("ar_add_rmsnorm %s: refused %d, wanted %d\n", b.what, rc, b.want); ++fails; }
        }
        const int rc = r4d_ar_add_rmsnorm((long)(uintptr_t)s.data(), (long)(uintptr_t)s.data(),
                                          (long)(uintptr_t)f.data(), (long)(uintptr_t)f.data(),
                                          (long)(uintptr_t)c.data(), 8, (long)(uintptr_t)a.data(),
                                          (long)(uintptr_t)x.data(), 0, (long)(uintptr_t)y.data(),
                                          1, 8, 1e-6f, 0.f, 1.f, 1, 0, 0, 0, 1, 256, 3, 0, 0, 0);
        if (rc != -1) { std::printf("ar_add_rmsnorm no gain: refused %d, wanted -1\n", rc); ++fails; }
    }
    std::printf("ar_add_rmsnorm: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}