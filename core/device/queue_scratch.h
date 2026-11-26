/* queue_scratch.h -- the VRAM a card keeps as queue scratch, from the kernels' own metadata.
 *
 * The runtime hands a queue one scratch allocation at the first dispatch of a kernel that has a
 * private segment: the static scratch a code object declares per work-item
 * (`private_segment_fixed_size`, a spill slot or a local array). It is per queue and it is
 * grow-only: a later dispatch that needs more re-takes it, a smaller one keeps what is there
 * (ROCr aql_queue.cpp, HandleInsufficientScratch -- `device_size` is the queue's figure, and the
 * allocation is only released when a larger one replaces it). So the resident figure is a
 * MAXIMUM over the kernels the plan launches, not a sum over them: summing would charge the same
 * VRAM once per kernel.
 *
 * A kernel's segment is per work-item, so one wave of it is `private * wave` bytes, and the
 * smallest dispatch that covers the card needs one wave on every CU. That is the charge:
 *
 *     align_up(largest_private, 1024 / wave) * wave * n_cu
 *
 * which is 0 when nothing the plan launches declares a private segment -- the answer the flat
 * `n_cu * 5 MiB` never asked the kernels for. It is a floor, not the runtime's worst case: ROCr's
 * own device_size multiplies by the card's whole wave-slot count (32 slots a CU on an R9700),
 * which for the 18 KiB segment DFly2's coupled verify declares would reserve 1.1 GiB -- more than
 * the card has to spare, and not what the lane measured it keeping. What a queue actually holds
 * depends on the dispatch's grid, which no metadata carries; the measurement is what says the
 * floor is enough (notes/backlog/measurements/M-09.md).
 *
 * RADIANCE_QUEUE_SCRATCH_MIB is a deployment's floor on top of that, MiB a CU, default 0. It can
 * only raise the charge: the default behaviour is the metadata's, and a typo cannot lower it.
 * A value that is not a whole number in [0, 64] is refused by name instead of clamped -- a silent
 * clamp to 1 MiB a CU is a budget nobody chose.
 */
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace rad {

/* The most a deployment may add through RADIANCE_QUEUE_SCRATCH_MIB, MiB a CU. Above the largest
 * segment any library here declares (18,256 B, less than a MiB a CU at one wave a CU), so it is
 * room for a kernel a deployment builds itself, not a second copy of the old flat bound. */
static const int64_t kQueueScratchMaxMibPerCu = 64;

/* ROCr's alignment for a private segment: `AlignUp(private_segment_size, 1024 / lanes_per_wave)`
 * (aql_queue.cpp). 1 when `wave` is not a sane lane count, so the arithmetic cannot divide by 0. */
inline uint32_t queue_scratch_align(int wave) {
    if (wave <= 0 || wave > 1024) return 1;
    return (uint32_t)(1024 / wave) ? (uint32_t)(1024 / wave) : 1u;
}

/* The charge in bytes for one card. `largest_private` is the largest private_segment_fixed_size
 * among the kernels the plan launches, `n_cu` the card's compute units, `wave` its lanes a wave.
 * `extra_mib_per_cu` is RADIANCE_QUEUE_SCRATCH_MIB, 0 by default. */
inline int64_t queue_scratch_bytes(uint32_t largest_private, int n_cu, int wave,
                                   int64_t extra_mib_per_cu) {
    if (n_cu <= 0) return 0;
    const uint32_t a = queue_scratch_align(wave);
    int64_t per_thread = (int64_t)largest_private;
    if (per_thread > 0) per_thread = (per_thread + a - 1) / a * a;
    if (extra_mib_per_cu < 0) extra_mib_per_cu = 0;
    if (extra_mib_per_cu > kQueueScratchMaxMibPerCu) extra_mib_per_cu = kQueueScratchMaxMibPerCu;
    const int64_t lanes = wave > 0 && wave <= 1024 ? wave : 1;
    /* The knob is MiB a CU, as the bound it succeeds was (5 MiB a CU): a deployment's own floor
     * on the card, not another per-work-item segment. */
    return per_thread * lanes * (int64_t)n_cu + extra_mib_per_cu * (1LL << 20) * (int64_t)n_cu;
}

/* Parses one RADIANCE_QUEUE_SCRATCH_MIB value. `value` null or empty means unset: 0, no error.
 * Anything else must be a whole decimal number in [0, kQueueScratchMaxMibPerCu]; outside it, or
 * with a trailing character, `*why` names the variable and the value and the answer is false. */
inline bool queue_scratch_knob(const char* value, int64_t* mib_per_cu, std::string* why) {
    if (mib_per_cu) *mib_per_cu = 0;
    if (!value || !*value) return true;
    const char* p = value;
    if (*p == '+' || *p == '-') ++p;              /* a sign is read, so "-1" is refused by value */
    if (!*p) { if (why) *why = std::string("RADIANCE_QUEUE_SCRATCH_MIB=\"") + value +
                               "\" is not a number of MiB a CU (0.." +
                               std::to_string((long long)kQueueScratchMaxMibPerCu) + ")"; return false; }
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(value, &end, 10);
    if (errno != 0 || end == value || (end && *end) || v < 0 || v > kQueueScratchMaxMibPerCu) {
        if (why) *why = std::string("RADIANCE_QUEUE_SCRATCH_MIB=\"") + value +
                        "\" is not a number of MiB a CU (0.." +
                        std::to_string((long long)kQueueScratchMaxMibPerCu) + ")";
        return false;
    }
    if (mib_per_cu) *mib_per_cu = (int64_t)v;
    return true;
}

}  /* namespace rad */