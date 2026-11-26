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
 * No deployment knob: the item asks for the metadata charge, so the metadata is the charge. A box
 * whose own kernels spill more than their metadata declares is out of scope here.
 */
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace rad {

/* ROCr's alignment for a private segment: `AlignUp(private_segment_size, 1024 / lanes_per_wave)`
 * (aql_queue.cpp). 1 when `wave` is not a sane lane count, so the arithmetic cannot divide by 0. */
inline uint32_t queue_scratch_align(int wave) {
    if (wave <= 0 || wave > 1024) return 1;
    return (uint32_t)(1024 / wave) ? (uint32_t)(1024 / wave) : 1u;
}

/* The charge in bytes for one card. `largest_private` is the largest private_segment_fixed_size
 * among the kernels the plan launches, `n_cu` the card's compute units, `wave` its lanes a wave. */
inline int64_t queue_scratch_bytes(uint32_t largest_private, int n_cu, int wave) {
    if (n_cu <= 0) return 0;
    const uint32_t a = queue_scratch_align(wave);
    int64_t per_thread = (int64_t)largest_private;
    if (per_thread > 0) per_thread = (per_thread + a - 1) / a * a;
    const int64_t lanes = wave > 0 && wave <= 1024 ? wave : 1;
    return per_thread * lanes * (int64_t)n_cu;
}

/* The bound for a backend that cannot read code objects: what the term always was, 5 MiB a CU.
 * Split out so the fallback branch of rad_dev_resident_scratch_bytes has an answer with a test. */
inline int64_t queue_scratch_flat_bytes(int n_cu) {
    if (n_cu <= 0) return 0;
    return (int64_t)n_cu * (5LL * 1024 * 1024);
}

/* One kernel's private segment, folded into the scan's running answer. `private_size` 0 means the
 * kernel declares no segment and only counts towards the total; the largest wins, and a name the
 * runtime keeps as "<kernel>.kd" is reported without the suffix. Split out of the AQL backend so
 * the rule -- largest, not sum -- is testable without a card (tests/mem_test.cpp). */
struct ScratchScan {
    uint32_t    largest = 0;
    int         declared = 0;      /* kernels that declare a segment */
    int         total = 0;         /* kernels scanned */
    std::string kernel;            /* the heaviest one, suffix stripped */
};

inline void queue_scratch_take(ScratchScan* s, const std::string& name, uint32_t private_size) {
    if (!s) return;
    ++s->total;
    if (!private_size) return;
    ++s->declared;
    if (private_size > s->largest) {
        s->largest = private_size;
        s->kernel = name;
        if (s->kernel.size() > 3 && s->kernel.compare(s->kernel.size() - 3, 3, ".kd") == 0)
            s->kernel.resize(s->kernel.size() - 3);
    }
}

}  /* namespace rad */