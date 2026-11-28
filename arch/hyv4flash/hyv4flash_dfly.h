/* hyv4flash_dfly.h -- DFly2, Hy4-flash's block drafter (kernels/dfly2.md; vLLM
 * models/hy_v4/nvidia/dfly*.py is the reference, ref/dfly_ref.py its port).
 *
 * A block drafter with no cache of its own (RAD_DRAFT_BLOCK, block = depth + 1, the anchor row
 * predicting nothing). Behind every trunk step the core runs a context pass, which issues nothing
 * here, and a query pass of `block` rows a drafting sequence at positions a .. a+depth past its
 * committed end a, tokens [bonus, mask, ..]:
 *
 *   x[j]  = embed(tok[j]);  x[0] += gate * eh_proj([enorm(x[0]) | hnorm(t)])     t = final_hidden at
 *                                                                               the anchor row
 *   draft layer i: the sparse layers' gated MLA with the draft's weights over the latent cache of
 *     trunk layer Lkv[i] (its 6 rows stored there first, past the committed end), keys = the
 *     selection trunk layer Lidx[i] made at the anchor row plus the block's own rows (dfly_sel,
 *     mla_attn_sel_sink); a dense SwiGLU MLP; pre-norm residual
 *   z     = final_norm(x)
 *   chain k = 0..depth-1:  zc = z[k+1] + down(silu(gate(c)) * up(c)),  c = [hnorm(z[k+1]) |
 *     enorm(embed(prev))], prev the bonus then each proposal; proposal k = argmax(lm_head(zc))
 *     (the trunk's head, row_topk over the bf16-rounded logits, ties to the lower id; one
 *     cast_row_topk where a library serves it, else cast then row_topk)
 *
 * Under load the core may stage the short block instead (D-06: kShortBlock = 4 rows, so 3
 * proposals and 3 chain rounds; --draft-short-from). The draft is causal, so that pass is the
 * full pass's first rows.
 *
 * What the trunk step leaves for it (only while the drafter is served): final_hidden (PERSIST,
 * every possible anchor row), and for each distinct Lidx a snapshot of the indexer's selection by trunk row
 * (`dfly.sel_snap.<l>`, `dfly.nsel_snap.<l>`, nsel 0 for a row that attended densely), taken right
 * after qsa_select and before a streamed group's kv_resolve rewrites it into ring slots.
 *
 * Rollback (kernels/dfly2.md §6): the query pass writes latent rows a..a+depth of layers Lkv only,
 * every one before any row reads it, and nothing else of the trunk's state (no indexer key, tail,
 * Engram history); the next trunk step rewrites whatever positions it carries before reading them.
 *
 * Tensor parallelism (kernels/dfly2.md §3.6): as the trunk's sparse layers, whole heads a rank
 * (q_b, kv_b, the gate and the sinks by rows, the output projection by columns) and the MLP's width
 * evenly (gate_up's rows of each half, down's columns), each followed by an exact all_reduce before
 * the norm or residual that reads it; q_a, kv_a, the norms, fusion, correction, the embedding and
 * dfly_sel replicated, as is the latent cache (every rank stores the same rows). The chain's logits
 * are the trunk head's vocabulary shard: row_topk (global ids, the pair plane), all_gather,
 * row_topk_merge, so every rank proposes the same tokens (ties to the lower id in both stages).
 *
 * Served: 1, 2 or 4 ranks, draft linears bf16 or quantised as the trunk's Lin reads them (int8,
 * IQ8R, rotated forms; WP13.9, sliced at TP like the bf16 ones; kv_b and the embedding stay bf16),
 * depth 1..5, greedy
 * proposals (exact at every temperature under the engine's verify; vLLM's `probabilistic` draft
 * sampling is not modelled) or, under --draft-coupling on (WP13.8), coupled ones: each chain step
 * samples x_k ~ Q_k = softmax of its merged top-8 logits with mtp_couple_propose (round k + 1) and
 * stores Q_k in `mtp_couple_state`, which the coupled / traversal verify and the adaptive prefix
 * read (a greedy request keeps the argmax); no local window (dfly_context_local_window > 0 is
 * refused), the confidence head declared and not read. Anything else declares the manifest only,
 * says why, and leaves speculation to prompt lookup. RADIANCE_HY4_DFLY=0 does the same on purpose
 * (prompt lookup alone, as --spec-ngram needs); RADIANCE_HY4_DFLY_SABOTAGE=1 (tests only) replaces
 * every proposal by the mask id, so that nearly every draft is rejected. */
#ifndef HYV4FLASH_DFLY_H
#define HYV4FLASH_DFLY_H

#include "hyv4flash_indexer.h"
#include "rad_spechub.h"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace hyv4flash {

struct Dfly {
    /* ---------------------------------------------------------------- configuration */
    int64_t layers = 0, depth = 0, block = 0, ff = 0, window = 0, topk = 0;
    /* D-06: the short block the core stages under load (rad_drafter_short_block, --draft-short-from):
     * kShortBlock rows, the anchor and kShortBlock - 1 masks, declared when the drafter's own block
     * is longer (depth 4 and 5). The draft is causal (dfly_sel gives a row the keys up to its own
     * position), so the short pass is the full pass's first rows; 0 when there is none. */
    static constexpr int64_t kShortBlock = 4;
    int64_t block_lo = 0;
    int32_t mask = -1;
    std::vector<int> kv, idx;       /* per draft layer: Lkv[i], Lidx[i] */
    std::vector<int> idx_set;       /* the distinct Lidx */
    std::vector<int> post;          /* per draft layer: post_proj_norm present */
    bool own_embed = false;         /* model.dfly_layers.embed_tokens.weight */
    bool want = false;              /* the drafter's weights are declared (manifest) */
    bool served = false;            /* ... and its passes are built */
    bool sabotage = false;
    /* RADIANCE_HY4_DFLY_QTEMP: the proposal law's temperature, Q = softmax(top-8 / q_temp). 1 (the
     * default) is the shortlist's own law. Exact for any value: the verify couples against the Q the
     * state plane records; only the acceptance moves (WP13.8 open item, G-5a). */
    double q_temp = 1.0;
    /* --draft-coupling on (kernels/dfly2.md §8.2): each chain step samples its proposal from the
     * law of its top-8 shortlist (mtp_couple_propose, round k + 1) and stores that law for the
     * coupled verify. */
    bool coupled = false;
    /* RADIANCE_HY4_DFLY_TEST_BREAK (tests only, kernels/dfly2.md §6.4): `store_late` stores the
     * draft rows after the attention, `nsel_keep` leaves the selection snapshot uncleared on a
     * step that does not select. Each must be caught by the parity check; neither changes text.
     * `chain_rows` (D-02) makes every chain round read round 0's rows of the hoisted gather and
     * norm, which changes the proposals from the second on; the arch test's replay catches it. */
    bool break_store_late = false, break_nsel_keep = false, break_chain_rows = false;
    std::string why;                /* why a declared drafter is not served */
    std::string why_buf;            /* storage for a reason built from an encoding (the manifest) */
    std::vector<std::function<int()>> defer;   /* quantised linears' manifest entries, kept until served is known */
    int n_quant = 0;                /* linears stored quantised (read by Lin) */
    /* Declines the drafter (the first reason stands). After configure, from the manifest. */
    void refuse(const char* r) {
        if (why.empty()) why = r;
        served = false;
    }
    std::map<std::string, rad_weight> w;   /* the manifest's single-plane handles, by name */

    /* Reads the drafter's config (top level, kernels/dfly2.md §1.1) and decides whether to serve.
     * Never fails: a drafter that cannot be served is declared as the manifest and says why. */
    void configure(RadBuilder* b, const RadModelMeta* m, const Shape& s, const RadBuildCtx* ctx) {
        layers = s.dfly_layers;
        want = layers > 0 && ctx->max_spec > 0;
        served = false;
        block_lo = 0;
        if (!want) return;
        depth = ctx->max_spec;
        block = depth + 1;
        block_lo = block > kShortBlock ? kShortBlock : 0;
        mask = (int32_t)rad_meta_geti(m, "dfly_mask_token_id", -1);
        window = rad_meta_geti(m, "dfly_context_local_window", 0);
        topk = rad_meta_geti(m, "dfly_target_index_topk", -1);
        auto ids = [&](const char* key, std::vector<int>& out) {
            out.clear();
            for_each_token(rad_meta_gets(m, key, ""), [&](const char* t, size_t) {
                char* e = nullptr;
                const long v = std::strtol(t, &e, 10);
                if (e == t) return false;
                out.push_back((int)v);
                return true;
            });
            if (out.size() == 1 && layers > 1) out.assign((size_t)layers, out[0]);
        };
        ids("dfly_target_kv_layers", kv);
        ids("dfly_target_index_layer", idx);
        const char* e = std::getenv("RADIANCE_HY4_DFLY");
        const char* sb = std::getenv("RADIANCE_HY4_DFLY_SABOTAGE");
        sabotage = sb && *sb && std::strcmp(sb, "0") != 0;
        const char* br = std::getenv("RADIANCE_HY4_DFLY_TEST_BREAK");
        break_store_late = br && std::strstr(br, "store_late");
        break_nsel_keep = br && std::strstr(br, "nsel_keep");
        break_chain_rows = br && std::strstr(br, "chain_rows");
        char buf[256];
        why.clear();
        auto no = [&](const char* fmt, auto... a) {
            if (!why.empty()) return;
            /* A message with no arguments is taken as written: giving it to std::snprintf would
             * read any % in it as a conversion with nothing behind it. */
            if constexpr (sizeof...(a) == 0) {
                why = fmt;
            } else {
                std::snprintf(buf, sizeof buf, fmt, a...);
                why = buf;
            }
        };
        if (const char* qt = std::getenv("RADIANCE_HY4_DFLY_QTEMP"); qt && *qt) {
            char* end = nullptr;
            const double v = std::strtod(qt, &end);
            if (end != qt && *end == 0 && v >= 0.05 && v <= 20.0) q_temp = v;
            else no("RADIANCE_HY4_DFLY_QTEMP='%s' is no number in [0.05, 20]", qt);
        }
        if (e && !std::strcmp(e, "0")) no("RADIANCE_HY4_DFLY=0");
        coupled = ctx->draft_coupling != 0;
        if (coupled && b && !rad_op_declared(b, "mtp_couple_propose"))
            no("--draft-coupling on: no loaded kernel library serves mtp_couple_propose (the "
               "spechub kernels)");
        if (b && !rad_op_declared(b, "dfly_sel"))
            no("no loaded kernel library serves dfly_sel (libref, libavx and libr4d do)");
        if (depth > 5)
            no("--num-speculative-tokens %d: DFly2 was trained at 5 proposals and serves 1 to 5",
               (int)depth);
        /* The draft's heads are the sparse layers' (split_ranks splits those too); its MLP width
         * is the checkpoint's and is checked in the manifest (hyv4flash.cpp drafter()). */
        if (s.world > 1 && s.sparse.heads % s.world)
            no("the draft's %lld attention heads do not split evenly over %d ranks",
               (long long)s.sparse.heads, s.world);
        /* The correction's gate and up hold this rank's rows and its down this rank's columns, so
         * the chain's width splits over the same ranks. */
        if (s.world > 1 && s.n_embd % s.world)
            no("the draft's %lld-wide correction does not split evenly over %d ranks",
               (long long)s.n_embd, s.world);
        if (s.world > 1 && b && (!rad_op_declared(b, "all_gather") ||
                                 !rad_op_declared(b, "row_topk_merge")))
            no("tensor parallelism (%d ranks) needs all_gather and row_topk_merge for the draft "
               "head's vocabulary shards", s.world);
        if (mask < 0 || mask >= s.n_vocab)
            no("dfly_mask_token_id %d is outside the vocabulary (%lld)", (int)mask,
               (long long)s.n_vocab);
        if (window != 0)
            no("dfly_context_local_window %lld: a local window over the history is not built",
               (long long)window);
        if ((int64_t)kv.size() != layers || (int64_t)idx.size() != layers)
            no("dfly_target_kv_layers / dfly_target_index_layer name %zu / %zu layers for %lld "
               "draft layers", kv.size(), idx.size(), (long long)layers);
        for (size_t i = 0; i < kv.size() && i < idx.size(); ++i)
            for (int l : { kv[i], idx[i] })
                if (l < 1 || l >= s.n_layer || s.is_sliding[(size_t)l])
                    no("dfly target layer %d is not a sparse layer above 0", l);
        for (size_t i = 0; i < kv.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                if (kv[i] == kv[j]) no("dfly_target_kv_layers names layer %d twice", kv[i]);
        if (topk != s.ix_group_topk * s.ix_pool)
            no("dfly_target_index_topk %lld is not group_topk * ratio (%lld)", (long long)topk,
               (long long)(s.ix_group_topk * s.ix_pool));
        idx_set.clear();
        for (int l : idx)
            if (std::find(idx_set.begin(), idx_set.end(), l) == idx_set.end()) idx_set.push_back(l);
        served = why.empty();
    }
    bool snaps(int l) const {
        return served && std::find(idx_set.begin(), idx_set.end(), l) != idx_set.end();
    }

    /* ---------------------------------------------------------------- the trunk's snapshots */
    /* P-09: the snapshot layers' selections live in two layer-major planes, so a single-span
     * step of a group that does not stream lets `qsa_select` write its layer's window directly
     * (no post-copy), and a non-selecting step clears every layer's counts with one fill.
     * Layer li's window starts at row `Snap::base` and is `g.max_tok` rows tall. */
    struct Snap {
        int layer = -1;
        int li = 0;             /* the plane's layer slot (declare order) */
        int64_t base = 0;       /* the plane row this layer's window starts at */
        int64_t width = 0;
    };
    std::vector<Snap> snap;
    rad_buf snap_sel = 0, snap_nsel = 0;      /* [L*T, G] i32 + [L*T] i32, PERSIST */
    rad_buf snap_ids = 0;                     /* taps only, the selection as group ids */
    rad_op op_sel = 0, op_nsel = 0, op_zero = 0, op_ids = 0;   /* shared by the layers */
    int64_t snap_t = 0;                       /* the plane's rows a layer (g.max_tok) */
    const Snap* snap_of(int l) const {
        for (const Snap& sn : snap)
            if (sn.layer == l) return &sn;
        return nullptr;
    }
    /* P-09: the window a snapshot layer's `qsa_select` writes when the step is single-span and
     * the group does not stream (the caller holds both step facts). */
    SelOut snap_out(int l) const {
        const Snap* sn = snap_of(l);
        return SelOut{ snap_sel, snap_nsel, sn->base };
    }
    /* Declared right after layer l's indexer (its `qsa_select` may write the plane's window). */
    int declare_snap(RadBuilder* b, Names& nm, const Shape& s, const Geom& g, int l,
                     const IndexWire& iw) {
        if (!snaps(l)) return RAD_OK;
        const int64_t T = g.max_tok, G = s.ix_group_topk;
        if (snap.empty()) {
            snap_t = T;
            const int64_t L = (int64_t)idx_set.size(), P = L * T;
            snap_sel = decl_b(b, nm.f("dfly.sel_snap"), RAD_I32, { P, G }, RAD_BUF_PERSIST);
            snap_nsel = decl_b(b, nm.f("dfly.nsel_snap"), RAD_I32, { P }, RAD_BUF_PERSIST);
            if (!snap_sel || !snap_nsel) return RAD_E_INVAL;
            op_sel = decl_copy(b, T, G, RAD_I32, iw.sel, snap_sel);
            op_nsel = decl_copy(b, T, 1, RAD_I32, iw.nsel, snap_nsel);
            op_zero = mk(b, "fill", { RAD_RANGE("M", 1, P), RAD_INT("n", 1), RAD_STR("dtype", "i32") },
                         {}, {}, { snap_nsel });
            HY4_NEED(op_sel && op_nsel, "cast (i32)");
            HY4_NEED(op_zero, "fill (i32)");
            if (taps_on() && iw.sel_ids) {
                snap_ids = decl_b(b, nm.f("dfly.sel_snap_ids"), RAD_I32, { P, G }, RAD_BUF_PERSIST);
                if (!snap_ids) return RAD_E_INVAL;
                op_ids = decl_copy(b, T, G, RAD_I32, iw.sel_ids, snap_ids);
                HY4_NEED(op_ids, "cast (i32)");
            }
        }
        Snap sn;
        sn.layer = l;
        sn.li = (int)snap.size();
        sn.base = sn.li * T;
        sn.width = G;
        snap.push_back(sn);
        return RAD_OK;
    }
    /* Issued after layer l's Indexer::step, before its attention (whose kv_resolve may rewrite
     * `ix.selected` in place). `direct`: the step's `qsa_select` wrote the plane's window
     * itself, so a selecting layer copies only the tap's group ids (P-09). */
    void snap_step(RadCtx* c, int l, const IndexWire& iw, bool select, QueryRows rows,
                   bool direct) const {
        const Snap* sn = snap_of(l);
        if (!sn || !rows.n) return;
        if (select) {
            if (direct && !op_ids) return;
            rows.spans([&](int64_t orig, int64_t compact, int64_t n) {
                if (!direct) {
                    RAD_ISSUE_N(c, op_sel, n, brow_slice(iw.sel, compact, n, sn->width),
                                brow_slice(snap_sel, sn->base + orig, n, sn->width));
                    RAD_ISSUE_N(c, op_nsel, n, brow_slice(iw.nsel, compact, n, 1),
                                brow_slice(snap_nsel, sn->base + orig, n, 1));
                }
                /* the tap's group ids (`op_tap_select`) always land on the wire */
                if (op_ids)
                    RAD_ISSUE_N(c, op_ids, n, brow_slice(iw.sel_ids, compact, n, sn->width),
                                brow_slice(snap_ids, sn->base + orig, n, sn->width));
            });
        } else if (!break_nsel_keep && sn == &snap.front()) {
            /* One fill clears every layer's counts (P-09). The rows outside this step's spans
             * ride along: a row the step did not query cannot be an anchor of the draft pass
             * that follows it, so no reader sees a difference from the per-span clear. */
            RAD_ISSUE_N(c, op_zero, (int64_t)snap.size() * snap_t,
                        brows(snap_nsel, (int64_t)snap.size() * snap_t));
        }
    }

    /* ---------------------------------------------------------------- the query pass */
    /* A linear of the draft: a bf16 weight is one `gemm_nt` over the manifest's handle (as ever);
     * a quantised one (IQ8R, int8, the rotated forms the trunk's `Lin` reads, hyv4flash_common.h)
     * is a Lin that quantises its own input into the rank's code pair and runs the int8 GEMM. The
     * encoding is the container's, nothing here chooses it. */
    struct Proj {
        Lin lin{};
        bool q = false;
        rad_op op = 0;
        rad_weight w = 0;
        /* A private code pair (the chain's: its ops are issued once a step, so the planner, which
         * places a buffer by where its ops are declared, must not share the trunk's pair with
         * whatever is declared after them); empty = the rank's shared pair. */
        rad_buf q8 = 0, s8 = 0;
        int64_t g8 = 0;
        bool ok() const { return q || op; }
    };
    struct Layer {
        int kvl = 0, ixl = 0;
        bool post = true;
        rad_weight attn_norm = 0, ffn_norm = 0, q_a = 0, qa_norm = 0, q_b = 0, kv_a = 0,
                   kv_norm = 0, kv_b = 0, gate = 0, out = 0, post_norm = 0, sinks = 0, gu = 0,
                   down = 0;
        rad_buf tab = 0, len = 0, cnt = 0;
        Proj p_qa, p_qb, p_kva, p_gate, p_out, p_gu, p_down;
        rad_op op_tab = 0, op_attn_norm = 0, op_qa_norm = 0, op_qn = 0,
               op_qr = 0, op_q_rope = 0, op_kv_norm = 0, op_k_rope = 0, op_store = 0,
               op_store_stream = 0, op_resolve = 0, op_absorb = 0, op_attn = 0,
               op_expand = 0, op_gmul = 0, op_post = 0, op_add1 = 0, op_ffn_norm = 0,
               op_silu = 0, op_add2 = 0;
        /* D-01: the trunk's fused producers, each byte-identical to the chain of separate
         * launches it replaces (docs/OPS.md), hence 0 where the pair is kept: the norm that
         * writes its readers' codes (rmsnorm_had_quant_i8g), the expansion with the gate and the
         * o_proj quantiser (mla_expand_gate_had_quant_i8g), the MLP's activation with the down
         * quantiser (gated_had_quant_i8g), the cache row's norm, key rotary and store
         * (mla_row_norm_rope_store), and the residual add with the norm after it (add_rmsnorm). */
        rad_op op_attn_norm_q = 0, op_qa_norm_q = 0, op_ffn_norm_q = 0, op_row_q = 0,
               op_expand_q = 0, op_gated = 0, op_final_add = 0;
        /* D-03: the fused reduction twins (ar_add_rmsnorm), each byte-identical to the launches it
         * replaces: `op_ar_add` is the output projection's exact all-reduce with the post-norm and
         * the residual add after it (mode 0), `op_ar_next` the MLP's exact all-reduce with its add
         * and the norm that follows (mode 1), which is the next layer's input norm or, on the last
         * layer, the final one (`ar_next_w` / `ar_next_out`). 0 where the chain is kept. */
        rad_op op_ar_add = 0, op_ar_next = 0;
        rad_weight ar_next_w = 0;
        rad_buf ar_next_out = 0;
        bool ra = false;   /* whether q_a writes `qa_raw` (the fused q residual norm reads it) */
        rad_op op_tap_tab = 0;
        Tap t_in{}, t_qres{}, t_row{}, t_gate{}, t_oproj{}, t_out{}, t_mid{}, t_ffn_in{}, t_mlp{},
            t_resid{}, t_len{};
        rad_buf tap_tab = 0;
    };
    struct Step {
        rad_buf rowk = 0;   /* the chain's one row-index plane (Dfly::rowk); round k fills rows k */
        rad_op op_rowk = 0, op_pe = 0, op_en = 0,
               op_silu = 0, op_add = 0, op_head = 0, op_lcast = 0, op_topk = 0, op_ctopk = 0,
               op_tok = 0, op_prev = 0, op_gcp = 0, op_ucp = 0;
        rad_op op_rowk_lo = 0;   /* rowk at the short block's stride (rounds < block_lo - 1) */
        /* --draft-coupling: the sampled proposal (and, sabotaged, the shortlist's ids replaced
         * first) */
        rad_op op_couple = 0, op_csab = 0;
        rad_op op_gather = 0, op_merge = 0;   /* TP: the vocabulary shards' merge; 0 at one rank */
        Tap t_corr{}, t_logits{}, t_idx{}, t_val{};
    };
    Shape s{};
    Geom g{};
    int64_t Md = 0, W = 0, R = 8, E = 0, Ew = 0, H = 0, L = 0, RO = 0, N = 0, VD = 0, Q = 0, F = 0;
    rad_kvgroup kvg = 0;
    rad_weight token = 0, head = 0, fusion = 0, f_enorm = 0, f_hnorm = 0, f_gate = 0, c_hnorm = 0,
               c_enorm = 0, c_gate = 0, c_up = 0, c_down = 0, out_norm = 0;
    rad_buf final_hidden = 0;
    rad_buf x = 0, xn = 0, y = 0, ffgu = 0, ffo = 0, qa = 0, q = 0, q_nope = 0, q_rope = 0,
            row = 0, q_lat = 0, lat = 0, ao = 0, gt = 0, ag = 0, o = 0, z = 0;
    rad_buf qa_raw = 0;   /* D-01: q_a's output, which a fused q residual norm reads */
    rad_buf row0 = 0, ea = 0, th = 0, cat = 0, fu = 0;
    rad_buf rowk = 0;   /* [depth, seqs] k-major: every round's rows of z, for the one gather */
    rad_buf zk = 0, pe = 0, ccat = 0, cg = 0, cup = 0, cgu = 0, cff = 0, cd = 0, zc = 0, logits = 0, lbf = 0,
            cidx = 0, cval = 0, prev = 0, tokens = 0;
    rad_buf pairs = 0, gath = 0;   /* TP: this rank's top-R as (id, value) pairs, every rank's */
    /* --draft-coupling: the planes the engine binds by name (one draw a drafting row, one state a
     * (chain step, row)) */
    rad_buf couple_draws = 0, couple_state = 0;
    rad_op op_ar_o = 0, op_ar_y = 0, op_ar_c = 0;   /* TP: the layers' two reductions and the chain's; 0 at one rank */
    rad_buf ident = 0;   /* taps: [max_seqs, P] the identity table */
    int64_t ident_w = 0;
    struct ChainCodes { int64_t k = 0, g8 = 0; rad_buf q8 = 0, s8 = 0; };
    std::vector<ChainCodes> chain_codes;
    Proj p_fuse, c_g, c_u, c_d;   /* the fusion and the correction's three linears */
    rad_op op_zk = 0, op_hn = 0;   /* D-02: every round's z rows and their norm, one launch each */
    rad_op op_embed = 0, op_row0 = 0, op_ea = 0, op_th = 0, op_fen = 0, op_fhn = 0,
           op_fgate = 0, op_fadd = 0, op_scatter = 0, op_final = 0, op_sabotage = 0;
    rad_op op_final_add = 0;   /* D-01: the last add with the final norm in one launch */
    rad_op op_row0_lo = 0;   /* row0 at the short block's stride */
    rad_op op_ident = 0;
    std::vector<Layer> lay;
    std::vector<Step> steps;
    Tap t_embed{}, t_fused{}, t_final{}, t_tokens{}, t_slot{}, t_pos{}, t_anchor{};

    /* An i32 copy from an operand the core hands the pass (no buffer to read). */
    static rad_op raw_copy(RadBuilder* b, int64_t rows, rad_buf out) {
        return mk(b, "cast",
                  { RAD_RANGE("M", 1, rows), RAD_INT("n", 1), RAD_STR("from", "i32"),
                    RAD_STR("to", "i32") },
                  {}, {}, { out });
    }
    /* A weight stored as several planes (codes and a scale, ...): a quantised one. */
    static bool quantised(RadBuilder* b, const char* name) {
        RadEncoding e{};
        return b && weight_enc(b, name, &e) && e.n_planes > 1;
    }
    /* The encodings a draft linear is read in: the trunk's (Lin). */
    static bool lin_reads(const RadEncoding& e) {
        return linear_is_w8(e) || (linear_is_w8h(e) && linear_w8h_form(e) == W8hForm::I8);
    }
    rad_weight wt(const char* name) const {
        auto it = w.find(name);
        return it == w.end() ? 0 : it->second;
    }

    /* After the manifest (hyv4flash.cpp drafter(), which recorded the handles in `w`). Checks
     * every handle first and declares nothing when one is missing (a quantised linear), so a
     * declined drafter leaves no op behind. */
    int declare(RadBuilder* b, Names& nm, const Shape& shape, const Geom& geom, rad_weight tok_w,
                rad_weight head_w, rad_buf fh, rad_kvgroup sparse_group, const IndexWire& iw) {
        if (!served) return RAD_OK;
        s = shape;
        g = geom;
        E = s.n_embd;
        Ew = E / s.world;
        const AttnGeom& a = s.sparse_r;   /* this rank's heads */
        H = a.heads;
        F = ff / s.world;                 /* this rank's MLP width (drafter() checked it divides) */
        L = s.kv_lora;
        RO = a.qk_rope;
        N = a.qk_nope;
        VD = a.v;
        Q = a.qk();
        kvg = sparse_group;
        final_hidden = fh;
        head = head_w;
        W = s.ix_group_topk + 2;
        const int64_t S = g.max_seqs;
        Md = std::min<int64_t>(g.max_tok, S * block);
        /* The handles, all or nothing. */
        /* A linear stored quantised has no manifest handle: its Lin declares it (proj_declare);
         * drafter() already declined the encodings none can read. */
        auto need_full = [&](const char* full, rad_weight& h, bool linear = false) {
            h = wt(full);
            if (!h && !(linear && quantised(b, full)) && why.empty())
                why = std::string("'") + full + "' is stored " + (linear ? "in a form no draft "
                      "linear reads" : "quantised, and only a draft linear may be") +
                      " (bf16 or f32 here)";
        };
        auto need = [&](const char* name, rad_weight& h, bool linear = false) {
            need_full(nm.f("%s", name), h, linear);
        };
        token = tok_w;
        if (own_embed) need("dfly.token_embd.weight", token);
        need("dfly.fusion.weight", fusion, true);
        need("dfly.fusion_embed_norm.weight", f_enorm);
        need("dfly.fusion_hidden_norm.weight", f_hnorm);
        need("dfly.fusion_gate", f_gate);
        need("dfly.correct_hidden_norm.weight", c_hnorm);
        need("dfly.correct_embed_norm.weight", c_enorm);
        need("dfly.correct_gate.weight", c_gate, true);
        need("dfly.correct_up.weight", c_up, true);
        need("dfly.correct_down.weight", c_down, true);
        need("dfly.output_norm.weight", out_norm);
        lay.assign((size_t)layers, Layer{});
        for (int64_t i = 0; i < layers; ++i) {
            Layer& d = lay[(size_t)i];
            d.kvl = kv[(size_t)i];
            d.ixl = idx[(size_t)i];
            d.post = post.size() > (size_t)i ? post[(size_t)i] != 0 : true;
            auto n = [&](const char* leaf) { return nm.f("dfly.blk.%lld.%s", (long long)i, leaf); };
            auto get = [&](const char* leaf, rad_weight& h, bool linear = false) {
                need_full(n(leaf), h, linear);
            };
            get("attn_norm.weight", d.attn_norm);
            get("ffn_norm.weight", d.ffn_norm);
            get("attn_q_a.weight", d.q_a, true);
            get("attn_q_a_norm.weight", d.qa_norm);
            get("attn_q_b.weight", d.q_b, true);
            get("attn_kv_a.weight", d.kv_a, true);
            get("attn_kv_a_norm.weight", d.kv_norm);
            get("attn_kv_b.weight", d.kv_b);
            get("attn_gate.weight", d.gate, true);
            get("attn_output.weight", d.out, true);
            if (d.post) get("attn_post_norm.weight", d.post_norm);
            get("attn_sinks.weight", d.sinks);
            get("ffn_gate_up.weight", d.gu, true);
            get("ffn_down.weight", d.down, true);
        }
        if (!why.empty()) {
            served = false;
            return RAD_OK;
        }

        auto bf = [&](const char* n, std::initializer_list<int64_t> sh) {
            return decl_b(b, nm.f("dfly.%s", n), RAD_BF16, sh);
        };
        auto i32 = [&](const char* n, std::initializer_list<int64_t> sh, int flags = 0) {
            return decl_b(b, nm.f("dfly.%s", n), RAD_I32, sh, flags);
        };
        const int64_t Vr = g.n_vocab;
        tokens = i32("tokens", { S, depth }, RAD_BUF_PERSIST);
        x = bf("x", { Md, E });
        xn = bf("xn", { Md, E });
        y = bf("y", { Md, E });
        ffgu = bf("ffn_gate_up", { Md, 2 * F });
        ffo = bf("ffn_act", { Md, F });
        qa = bf("q_a", { Md, s.q_lora });
        q = bf("q", { Md, H, Q });
        q_nope = bf("q_nope", { Md, H * N });
        q_rope = bf("q_rope", { Md, H * RO });
        row = bf("kv_row", { Md, L + RO });
        q_lat = bf("q_latent", { Md, H * L });
        lat = bf("attn_latent", { Md, H * L });
        ao = bf("attn_value", { Md, H * VD });
        gt = bf("attn_gate", { Md, H * VD });
        ag = bf("attn_gated", { Md, H * VD });
        o = bf("o_proj", { Md, E });
        z = bf("final", { Md, E });
        row0 = i32("row0", { S });
        ea = bf("anchor_embed", { S, E });
        th = bf("target_hidden", { S, E });
        cat = bf("fusion_in", { S, 2 * E });
        fu = bf("fusion_out", { S, E });
        /* D-02 (#88): the chain's rows of z do not change once the final norm has run, so every
         * round's are gathered and normalised in one launch each, k-major (round k at rows
         * [k * n, (k + 1) * n) of a pass of n sequences), and each round reads its own rows. A row's
         * norm does not depend on how many rows its launch carries (libr4d's kR4dSumLanes), so the
         * bytes are the per-round launches'. */
        rowk = i32("rowk", { depth * S });
        zk = bf("chain_hidden", { depth * S, E });
        /* The anchors' raw embeddings from the fusion (B) through round 0, then each round's
         * previous proposal's. */
        pe = bf("chain_prev_embed", { S, E });
        ccat = bf("chain_in", { depth * S, 2 * E });
        /* gate and up each into a buffer of its own, then copied into the halves silu_mul reads: a GEMM
         * output as a column half of one [S, 2E] buffer is strided once two sequences draft, and
         * libr4d's GEMMs refuse a strided output (the copies are exact, so the bytes are unchanged).
         * At TP each holds this rank's width: the gate's and the up's rows and the down's input. */
        cg = bf("chain_gate", { S, Ew });
        cup = bf("chain_up", { S, Ew });
        cgu = bf("chain_gate_up", { S, 2 * Ew });
        cff = bf("chain_act", { S, Ew });
        cd = bf("chain_delta", { S, E });
        zc = bf("corrected", { S, E });
        logits = decl_b(b, nm.f("dfly.logits"), RAD_F32, { S, Vr });
        lbf = bf("logits_bf16", { S, Vr });
        cidx = i32("cand_idx", { S, R });
        /* f32: row_topk's and row_topk_merge's `val` is f32 by the op's contract (docs/OPS.md,
         * libref's shape hook), whatever the logits' dtype; libr4d's kernels write f32 there. A bf16
         * plane here took the card's 4-byte values in 2-byte slots (rows overlapping, the last ones
         * past the buffer): harmless to greedy proposals (ids only), fatal to a coupled law. */
        cval = decl_b(b, nm.f("dfly.cand_val"), RAD_F32, { S, R });
        prev = i32("prev", { S });
        if (coupled) {
            /* The engine finds the two planes by these names (core/engine_draft.cpp) and stages
             * the draws in the query pass's sequence order, which is the chain's row order. */
            couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                  { S, (int64_t)RAD_MTP_COUPLE_DRAW_WORDS }, RAD_BUF_PERSIST);
            couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                  { depth, S, (int64_t)RAD_MTP_COUPLE_STATE_WORDS }, RAD_BUF_PERSIST);
            if (!couple_draws || !couple_state) return RAD_E_INVAL;
        }
        if (s.world > 1) {
            pairs = decl_b(b, nm.f("dfly.cand_pairs"), RAD_F32, { S, R, 2 });
            gath = decl_b(b, nm.f("dfly.cand_gathered"), RAD_F32, { S, s.world * R, 2 });
            if (!pairs || !gath) return RAD_E_INVAL;
        }
        if (!tokens || !x || !xn || !y || !ffgu || !ffo || !qa || !q || !q_nope ||
            !q_rope || !row || !q_lat || !lat || !ao || !gt || !ag || !o || !z || !row0 || !ea ||
            !th || !cat || !fu || !rowk || !zk || !pe || !ccat || !cg || !cup || !cgu || !cff || !cd || !zc || !logits ||
            !lbf || !cidx || !cval || !prev)
            return RAD_E_INVAL;
        for (int64_t i = 0; i < layers; ++i) {
            Layer& d = lay[(size_t)i];
            d.tab = decl_b(b, nm.f("dfly.sel.%lld", (long long)i), RAD_I32, { Md, W });
            d.len = decl_b(b, nm.f("dfly.len.%lld", (long long)i), RAD_I32, { Md });
            d.cnt = decl_b(b, nm.f("dfly.count.%lld", (long long)i), RAD_I32, { Md });
            if (!d.tab || !d.len || !d.cnt) return RAD_E_INVAL;
        }

        auto gemm = [&](int64_t rows, int64_t n_out, int64_t k, rad_weight wgt, rad_buf in,
                        rad_buf out) {
            return mk(b, "gemm_nt",
                      { RAD_RANGE("M", 1, rows), RAD_INT("N", n_out), RAD_INT("K", k),
                        RAD_STR("dtype", "bf16") },
                      { wgt }, { in }, { out });
        };
        /* One draft linear: a bf16 weight is the manifest's handle under gemm_nt; a quantised one a
         * Lin over the container's own encoding (this rank's slice at TP, as the trunk's). */
        auto linear = [&](Proj& p, const std::string& base, int grp, int64_t rows, int64_t n_out,
                        int64_t n_in, rad_weight plain_w, rad_buf in, rad_buf out, int shard,
                        std::initializer_list<int64_t> parts, bool persist = false,
                        int64_t code_rows = 0) -> int {
            const std::string wn = base + ".weight";
            const char* wnc = nm.f("%s", wn.c_str());
            if (!quantised(b, wnc)) {
                p.w = plain_w;
                p.op = gemm(rows, n_out, n_in, plain_w, in, out);
                return p.op ? RAD_OK : RAD_E_INVAL;
            }
            p.q = true;
            p.lin.mapped = true;   /* the manifest made the name map */
            if (s.world <= 1) { shard = RAD_SHARD_NONE; parts = {}; }
            ActFP8 a{};
            a.x = in;
            if (persist) {
                RadEncoding e{};
                weight_enc(b, wnc, &e);
                const int64_t g8 = linear_w8h_group(e);
                bool have = false;
                for (const auto& c : chain_codes)
                    if (c.k == n_in && c.g8 == g8) {
                        p.q8 = c.q8;
                        p.s8 = c.s8;
                        have = true;
                    }
                if (!have) {
                    ChainCodes c;
                    c.k = n_in;
                    c.g8 = g8;
                    c.q8 = decl_b(b, nm.f("dfly.chain_codes.%lld.%lld.q8", (long long)n_in, (long long)g8),
                                  RAD_I8, { code_rows ? code_rows : rows, n_in }, RAD_BUF_PERSIST);
                    c.s8 = decl_b(b, nm.f("dfly.chain_codes.%lld.%lld.s8", (long long)n_in, (long long)g8),
                                  RAD_F32, { code_rows ? code_rows : rows, (n_in + g8 - 1) / g8 },
                                  RAD_BUF_PERSIST);
                    if (!c.q8 || !c.s8) return RAD_E_INVAL;
                    chain_codes.push_back(c);
                    p.q8 = c.q8;
                    p.s8 = c.s8;
                }
                p.g8 = g8;
                a.q8 = p.q8;
                a.s8 = p.s8;
                a.g8 = g8;
            }
            return p.lin.declare(b, nm, g, nm.f("%s", base.c_str()), n_out, n_in, shard,
                                 grp_layer(grp), { wnc }, a, out, rows, parts);
        };
        auto add = [&](int64_t rows, rad_buf a1, rad_buf a2, rad_buf out) {
            return mk(b, "add", { RAD_RANGE("M", 1, rows), RAD_INT("n", E), RAD_STR("dtype", "bf16") },
                      {}, { a1, a2 }, { out });
        };
        auto gather = [&](int64_t rows, rad_buf src, rad_buf ix, rad_buf out) {
            const RadParam ps[] = { RAD_RANGE("M", 1, rows), RAD_INT("n", E),
                                    RAD_STR("dtype", "bf16") };
            const rad_op h = rad_decl_op(b, "gather_rows", ps, 3, nullptr, 0);
            return ix ? rw(b, h, { src, ix }, { out }) : rw(b, h, { src }, { out });
        };
        auto iota = [&](int64_t start, rad_buf out, int64_t stride) {
            return mk(b, "iota",
                      { RAD_RANGE("M", 1, S), RAD_INT("start", start), RAD_INT("step", stride) },
                      {}, {}, { out });
        };

        /* A: the block's embeddings, straight into the residual (D-02: no copy of the block). */
        op_embed = mk(b, "embed_lookup",
                      { RAD_RANGE("M", 1, Md), RAD_INT("n_embd", E), RAD_INT("n_vocab", s.n_vocab),
                        RAD_STR("dtype", "bf16") },
                      { token }, {}, { x });
        HY4_NEED(op_embed, "embed_lookup");
        RAD_ARCH_TRY(t_embed.declare(b, nm, Md, "dfly.embed", -1, RAD_BF16, E, x));
        /* B: the fusion into each anchor row. The anchors' raw embeddings are gathered once, before
         * the fusion overwrites those rows of x, and kept in `pe` for the chain's first round. */
        op_row0 = iota(0, row0, block);
        if (block_lo) op_row0_lo = iota(0, row0, block_lo);
        op_ea = gather(S, x, row0, pe);
        op_th = gather(S, final_hidden, 0, th);
        op_fen = decl_rmsnorm(b, S, E, s.eps, f_enorm, pe, cat);
        op_fhn = decl_rmsnorm(b, S, E, s.eps, f_hnorm, th, cat);
        const int gtop = (int)(s.n_layer + layers);   /* the drafter's model-level weights' group */
        RAD_ARCH_TRY(linear(p_fuse, "dfly.fusion", gtop, S, E, 2 * E, fusion, cat, fu, RAD_SHARD_NONE, {}));
        op_fgate = mk(b, "mul_weight", { RAD_RANGE("M", 1, S), RAD_INT("n", E), RAD_STR("dtype", "bf16") },
                      { f_gate }, { fu }, { fu });
        op_fadd = add(S, pe, fu, ea);
        op_scatter = mk(b, "scatter_rows",
                        { RAD_RANGE("M", 1, S), RAD_INT("n", E), RAD_STR("dtype", "bf16") }, {},
                        { ea, row0, x }, { x });
        HY4_NEED(op_row0, "iota");
        HY4_NEED(!block_lo || op_row0_lo, "iota");
        HY4_NEED(op_ea && op_th, "gather_rows");
        HY4_NEED(op_fen && op_fhn, "rmsnorm");
        HY4_NEED(p_fuse.ok() && op_fgate && op_fadd, "gemm_nt / mul_weight / add");
        HY4_NEED(op_scatter, "scatter_rows");
        RAD_ARCH_TRY(t_fused.declare(b, nm, S, "dfly.fused", -1, RAD_BF16, E, ea));

        /* TP: the two exact reductions a layer, the output projection's and the MLP's partial sums
         * (one op each, shared by the layers), before the post norm and the residual add. */
        {
            int rc = RAD_OK;
            op_ar_o = decl_all_reduce(b, g, E, o, &rc);
            RAD_ARCH_TRY(rc);
            op_ar_y = decl_all_reduce(b, g, E, y, &rc);
            RAD_ARCH_TRY(rc);
            op_ar_c = decl_all_reduce(b, g, E, cd, &rc);
            RAD_ARCH_TRY(rc);
        }
        /* C, D: the layers, each with its table. */
        const double scale = 1.0 / std::sqrt((double)Q);
        const int64_t half = (L + RO) / 2;
        for (int64_t i = 0; i < layers; ++i) {
            Layer& d = lay[(size_t)i];
            if (!snap_of(d.ixl)) return RAD_E_INVAL;   /* the draft layer's trunk selection */
            const int li = (int)i;
            d.op_tab = mk(b, "dfly_sel",
                          { RAD_RANGE("M", 1, Md), RAD_INT("blocks_in", s.ix_group_topk),
                            RAD_INT("blocks", W), RAD_INT("ratio", s.ix_pool),
                            RAD_INT("seqs", S) },
                          {}, { snap_sel, snap_nsel }, { d.tab, d.len, d.cnt });
            HY4_NEED(d.op_tab, "dfly_sel");
            d.op_attn_norm = decl_rmsnorm(b, Md, E, s.eps, d.attn_norm, x, xn);
            RAD_ARCH_TRY(d.t_in.declare(b, nm, Md, "dfly.attn_in_norm", li, RAD_BF16, E, xn));
            const std::string ln = nm.f("dfly.blk.%lld", (long long)i);
            const int gi = (int)(s.n_layer + i);
            /* D-01: the trunk's fused producers (docs/OPS.md), each byte-identical to the chain of
             * separate launches it replaces, so a draft layer issues fewer launches for the same
             * bytes. A producer writes the codes its reader's own quantiser would have written and
             * the reader then issues none: the norm of the attention's input (read by q_a, kv_a
             * and gate), the norm of the q residual (q_b), the feed-forward's norm (gate_up), the
             * MLP's activation (down) and the expansion with the gate (o_proj). Each needs the
             * rotated int8 codes at its reader's group; where it does not run the chain stays, and
             * q_a, kv_a and gate still share the one quantiser of xn that q_a issues. */
            const CodeForm f_qa = code_form(b, (ln + ".attn_q_a.weight").c_str());
            const CodeForm f_kva = code_form(b, (ln + ".attn_kv_a.weight").c_str());
            const CodeForm f_gate = code_form(b, (ln + ".attn_gate.weight").c_str());
            const CodeForm f_qb = code_form(b, (ln + ".attn_q_b.weight").c_str());
            const bool share_in = f_qa.i8 && f_qa.rot && f_qa.same(f_kva) && f_qa.same(f_gate) &&
                                  E % f_qa.g8 == 0;
            if (share_in) {
                d.p_qa.lin.lead = "dfly_attn_in";
                d.p_kva.lin.feed = d.p_gate.lin.feed = &d.p_qa.lin;
            }
            /* the q residual's norm writes q_b's codes and reads q_a's output, so q_a writes the
             * norm's input and the norm writes `qa` */
            if (fused_quant_on() && f_qb.i8 && f_qb.rot && f_qb.g8 == RAD_FP8_BLOCK &&
                s.q_lora % f_qb.g8 == 0) {
                if (!qa_raw) {
                    qa_raw = bf("attn_q_a_raw", { Md, s.q_lora });
                    if (!qa_raw) return RAD_E_INVAL;
                }
                d.ra = true;
            }
            RAD_ARCH_TRY(linear(d.p_qa, ln + ".attn_q_a", gi, Md, s.q_lora, E, d.q_a, xn,
                                d.ra ? qa_raw : qa, RAD_SHARD_NONE, {}));
            /* the norm of the attention's input writes q_a's codes -- and the fed ones' of kv_a and
             * gate, which take q_a's pair -- so the three issue no quantiser */
            if (fused_quant_on() && qplans(d.p_qa) && d.p_qa.lin.form.rot &&
                d.p_qa.lin.g8 == RAD_FP8_BLOCK && E % d.p_qa.lin.g8 == 0)
                d.op_attn_norm_q = decl_norm_codes(b, g, Md, E, s.eps, d.attn_norm, x, xn,
                                                   d.p_qa.lin.q8, d.p_qa.lin.s8, d.p_qa.lin.g8);
            d.op_qa_norm =
                decl_rmsnorm(b, Md, s.q_lora, s.eps, d.qa_norm, d.ra ? qa_raw : qa, qa);
            RAD_ARCH_TRY(d.t_qres.declare(b, nm, Md, "dfly.attn_q_resid", li, RAD_BF16, s.q_lora, qa));
            RAD_ARCH_TRY(linear(d.p_qb, ln + ".attn_q_b", gi, Md, H * Q, s.q_lora, d.q_b, qa, q, RAD_SHARD_ROW, {}));
            /* the q residual's norm writes q_b's codes: its quantiser then issues none */
            if (d.ra && fused_quant_on() && qplans(d.p_qb) && d.p_qb.lin.form.rot &&
                s.q_lora % d.p_qb.lin.g8 == 0)
                d.op_qa_norm_q = decl_norm_codes(b, g, Md, s.q_lora, s.eps, d.qa_norm, qa_raw, qa,
                                                 d.p_qb.lin.q8, d.p_qb.lin.s8, d.p_qb.lin.g8);
            /* q_b stays one tensor (the manifest is the checkpoint's): its output is split a head
             * at a time into the nope and rope parts. */
            d.op_qn = decl_copy(b, Md, H * N, RAD_BF16, q, q_nope);
            d.op_qr = decl_copy(b, Md, H * RO, RAD_BF16, q, q_rope);
            /* The sparse layers' rotary (vLLM's `target_rope_parameters`): M-RoPE where the
             * program serves pictures, at the pass's rotary positions (the card keeps each row's
             * offset to its index, as it does for decode rows). */
            const char* sec = s.mrope ? s.mrope_sparse.c_str() : nullptr;
            d.op_q_rope = decl_rope_gptj(b, Md, H, RO, s.theta_sparse, q_rope, sec);
            RAD_ARCH_TRY(linear(d.p_kva, ln + ".attn_kv_a", gi, Md, L + RO, E, d.kv_a, xn, row, RAD_SHARD_NONE, {}));
            d.op_kv_norm = decl_rmsnorm(b, Md, L, s.eps, d.kv_norm, row, row);
            d.op_k_rope = decl_rope_gptj(b, Md, 1, RO, s.theta_sparse, row, sec);
            RAD_ARCH_TRY(d.t_row.declare(b, nm, Md, "dfly.attn_kv_row", li, RAD_BF16, L + RO, row));
            d.op_store = mk(b, "kv_store",
                            { RAD_RANGE("M", 1, Md), RAD_INT("head_dim", half),
                              RAD_INT("n_head_kv", 1), RAD_INT("block_size", s.ix_pool),
                              RAD_STR("kv_dtype", "bf16") },
                            {}, { row }, {});
            /* The row's front in one launch (mla_row_norm_rope_store): the latent norm, the key's
             * rotary and the store, byte-identical to the three. It is not the streamed store's,
             * nor the debug taps' (t_row reads the row between the norm and the store), nor the
             * late store's. */
            if (cache_row_on() && !taps_on() && !break_store_late &&
                rad_op_declared(b, "mla_row_norm_rope_store")) {
                rad_op h = (sec && *sec)
                    ? mk(b, "mla_row_norm_rope_store",
                         { RAD_RANGE("M", 1, Md), RAD_INT("latent", L), RAD_INT("rope_dim", RO),
                           RAD_F64("eps", s.eps), RAD_F64("theta", s.theta_sparse),
                           RAD_F64("scale", 1.0), RAD_STR("mode", "imrope_gptj"),
                           RAD_STR("sections", sec), RAD_INT("block_size", s.ix_pool),
                           RAD_STR("dtype", "bf16") },
                         { d.kv_norm }, { row }, { row })
                    : mk(b, "mla_row_norm_rope_store",
                         { RAD_RANGE("M", 1, Md), RAD_INT("latent", L), RAD_INT("rope_dim", RO),
                           RAD_F64("eps", s.eps), RAD_F64("theta", s.theta_sparse),
                           RAD_F64("scale", 1.0), RAD_STR("mode", "gptj"),
                           RAD_INT("block_size", s.ix_pool), RAD_STR("dtype", "bf16") },
                         { d.kv_norm }, { row }, { row });
                if (const rad_op hr = optional_op(b, h)) d.op_row_q = hr;
            }
            if (rad_op_declared(b, "kv_store_streamed"))
                d.op_store_stream = mk(b, "kv_store_streamed",
                                       { RAD_RANGE("M", 1, Md), RAD_INT("head_dim", half),
                                         RAD_INT("n_head_kv", 1), RAD_INT("block_size", s.ix_pool),
                                         RAD_STR("kv_dtype", "bf16") },
                                       {}, { row }, {});
            RAD_ARCH_TRY(linear(d.p_gate, ln + ".attn_gate", gi, Md, H * VD, E, d.gate, xn, gt, RAD_SHARD_ROW, {}));
            RAD_ARCH_TRY(d.t_gate.declare(b, nm, Md, "dfly.attn_gate_logits", li, RAD_BF16, H * VD, gt));
            auto proj = [&](const char* mode, rad_buf in_, rad_buf out_) {
                return mk(b, "mla_project_kv",
                          { RAD_RANGE("M", 1, Md), RAD_INT("heads", H), RAD_INT("k_dim", N),
                            RAD_INT("v_dim", VD), RAD_INT("latent_dim", L), RAD_STR("mode", mode),
                            RAD_STR("dtype", "bf16") },
                          { d.kv_b }, { in_ }, { out_ });
            };
            d.op_absorb = proj("absorb", q_nope, q_lat);
            d.op_resolve = kv_resolve_declare(b, g, kvg, d.tab, d.cnt, W);
            d.op_attn = mk(b, "mla_attn_sel_sink",
                           { RAD_RANGE("M", 1, Md), RAD_INT("heads", H), RAD_INT("latent_dim", L),
                             RAD_INT("rope_dim", RO), RAD_INT("blocks", W),
                             RAD_INT("block_size", s.ix_pool), RAD_F64("scale", scale),
                             RAD_STR("dtype", "bf16") },
                           { d.sinks }, { q_lat, q_rope, d.tab, d.len }, { lat });
            d.op_expand = proj("expand", lat, ao);
            d.op_gmul = mk(b, "gate_mul",
                           { RAD_RANGE("M", 1, Md), RAD_INT("n", H * VD), RAD_STR("dtype", "bf16") },
                           {}, { ao, gt }, { ag });
            RAD_ARCH_TRY(linear(d.p_out, ln + ".attn_output", gi, Md, E, H * VD, d.out, ag, o,
                                RAD_SHARD_COL, {}));
            /* The expansion with its gate and o_proj's quantiser in one launch
             * (mla_expand_gate_had_quant_i8g, byte-identical to the three): o_proj then reads the
             * codes and issues none. It takes at most kFuseRows rows (its range), so a taller block
             * of drafts keeps o_proj's quantiser at issue; a calibrated run keeps the expansion's
             * own plane. */
            if (fused_quant_on() && calib_dir() == nullptr && qplans(d.p_out) &&
                d.p_out.lin.form.rot && d.p_out.lin.g8 == RAD_FP8_BLOCK &&
                (H * VD) % d.p_out.lin.g8 == 0 &&
                fuse_on("RADIANCE_HY4_FUSE_EXPAND_QUANT") &&
                rad_op_declared(b, "mla_expand_gate_had_quant_i8g")) {
                d.op_expand_q = mk(b, "mla_expand_gate_had_quant_i8g",
                                   { RAD_RANGE("M", 1, std::min(Md, kFuseRows)),
                                     RAD_INT("heads", H), RAD_INT("k_dim", N),
                                     RAD_INT("v_dim", VD), RAD_INT("latent_dim", L),
                                     RAD_INT("group", d.p_out.lin.g8), RAD_STR("dtype", "bf16") },
                                   { d.kv_b }, { lat, gt }, { d.p_out.lin.q8, d.p_out.lin.s8 });
                d.op_expand_q = optional_op(b, d.op_expand_q);
            }
            RAD_ARCH_TRY(d.t_oproj.declare(b, nm, Md, "dfly.attn_o_proj", li, RAD_BF16, E, o));
            if (d.post) d.op_post = decl_rmsnorm(b, Md, E, s.eps, d.post_norm, o, y);
            RAD_ARCH_TRY(d.t_out.declare(b, nm, Md, "dfly.attn_out", li, RAD_BF16, E, d.post ? y : o));
            d.op_add1 = add(Md, x, d.post ? y : o, x);
            /* D-03: the exact reduction of `o` with the post-norm and the residual add in one
             * launch (ar_add_rmsnorm mode 0), byte-identical to the three it replaces
             * (notes/backlog/audit/C-plugin-graph.md F23). The taps read `o`, `y` and `x` between
             * them, so a tapped run keeps the chain, and a layer without a post-norm has no norm
             * to fuse (its pair stays). */
            if (d.post && g.world == 2 && !taps_on() && rad_op_declared(b, "ar_add_rmsnorm")) {
                d.op_ar_add = mk(b, "ar_add_rmsnorm",
                                 { RAD_RANGE("M", 1, Md), RAD_INT("n", E),
                                   RAD_INT("world_size", g.world), RAD_INT("mode", 0),
                                   RAD_F64("eps", s.eps), RAD_STR("dtype", "bf16"),
                                   RAD_INT("wire", E % 256 ? 0 : 6) },
                                 { d.post_norm }, { o, x }, { o, x, y });
                d.op_ar_add = optional_op(b, d.op_ar_add);
            }
            RAD_ARCH_TRY(d.t_mid.declare(b, nm, Md, "dfly.resid_mid", li, RAD_BF16, E, x));
            d.op_ffn_norm = decl_rmsnorm(b, Md, E, s.eps, d.ffn_norm, x, xn);
            RAD_ARCH_TRY(d.t_ffn_in.declare(b, nm, Md, "dfly.ffn_in_norm", li, RAD_BF16, E, xn));
            RAD_ARCH_TRY(linear(d.p_gu, ln + ".ffn_gate_up", gi, Md, 2 * F, E, d.gu, xn, ffgu, RAD_SHARD_ROW, { F, F }));
            /* the feed-forward's norm writes gate_up's codes: its quantiser then issues none */
            if (fused_quant_on() && qplans(d.p_gu) && d.p_gu.lin.form.rot &&
                d.p_gu.lin.g8 == RAD_FP8_BLOCK && E % d.p_gu.lin.g8 == 0)
                d.op_ffn_norm_q = decl_norm_codes(b, g, Md, E, s.eps, d.ffn_norm, x, xn,
                                                  d.p_gu.lin.q8, d.p_gu.lin.s8, d.p_gu.lin.g8);
            d.op_silu = mk(b, "silu_mul", { RAD_RANGE("M", 1, Md), RAD_INT("n", F), RAD_STR("dtype", "bf16") },
                           {}, { ffgu }, { ffo });
            RAD_ARCH_TRY(linear(d.p_down, ln + ".ffn_down", gi, Md, E, F, d.down, ffo, y, RAD_SHARD_COL, {}));
            /* the activation with down's quantiser in one launch (gated_had_quant_i8g, byte-identical
             * to the two): down then reads the codes and issues none */
            if (fused_quant_on() && calib_dir() == nullptr && qplans(d.p_down) &&
                d.p_down.lin.form.rot && d.p_down.lin.g8 == RAD_FP8_BLOCK &&
                F % d.p_down.lin.g8 == 0 && rad_op_declared(b, "gated_had_quant_i8g")) {
                d.op_gated = mk(b, "gated_had_quant_i8g",
                                { RAD_RANGE("M", 1, Md), RAD_INT("n", F),
                                  RAD_INT("group", d.p_down.lin.g8), RAD_STR("dtype", "bf16"),
                                  RAD_STR("act", "silu") },
                                {}, { ffgu }, { d.p_down.lin.q8, d.p_down.lin.s8 });
                d.op_gated = optional_op(b, d.op_gated);
            }
            RAD_ARCH_TRY(d.t_mlp.declare(b, nm, Md, "dfly.mlp_out", li, RAD_BF16, E, y));
            d.op_add2 = add(Md, x, y, x);
            RAD_ARCH_TRY(d.t_resid.declare(b, nm, Md, "dfly.resid", li, RAD_BF16, E, x));
            HY4_NEED(d.op_attn_norm && d.op_qa_norm && d.op_kv_norm && d.op_ffn_norm &&
                         (!d.post || d.op_post),
                     "rmsnorm");
            HY4_NEED(d.p_qa.ok() && d.p_qb.ok() && d.p_kva.ok() && d.p_gate.ok() && d.p_out.ok() &&
                         d.p_gu.ok() && d.p_down.ok(),
                     "gemm_nt (bf16)");
            HY4_NEED(d.op_qn && d.op_qr, "cast");
            HY4_NEED(d.op_q_rope && d.op_k_rope, "rope (gptj)");
            HY4_NEED(d.op_store, "kv_store");
            HY4_NEED(d.op_absorb && d.op_expand, "mla_project_kv");
            HY4_NEED(d.op_attn, "mla_attn_sel_sink");
            HY4_NEED(d.op_gmul && d.op_add1 && d.op_add2 && d.op_silu, "gate_mul / add / silu_mul");
            /* Taps: the table as group indices (dfly_sel again over the group-id snapshot and an
             * identity table, the indexer tap's trick) and its lengths. */
            if (taps_on() && snap_ids) {
                if (!ident) {
                    ident_w = iw.blocks;
                    ident = decl_b(b, nm.f("dfly.identity"), RAD_I32, { S, ident_w }, RAD_BUF_PERSIST);
                    op_ident = mk(b, "iota",
                                  { RAD_RANGE("M", 1, S * ident_w), RAD_INT("start", 0),
                                    RAD_INT("step", 1), RAD_INT("period", ident_w) },
                                  {}, {}, { ident });
                    HY4_NEED(ident && op_ident, "iota");
                }
                d.tap_tab = decl_b(b, nm.f("dbg.dfly.sel_ids.%d", li), RAD_I32, { Md, W }, RAD_BUF_PERSIST);
                rad_buf tl = decl_b(b, nm.f("dbg.dfly.len.%d", li), RAD_I32, { Md, 1 }, RAD_BUF_PERSIST);
                if (!d.tap_tab || !tl) return RAD_E_INVAL;
                d.t_len.buf = tl;
                d.t_len.width = 1;
                d.op_tap_tab = mk(b, "dfly_sel",
                                  { RAD_RANGE("M", 1, Md), RAD_INT("blocks_in", s.ix_group_topk),
                                    RAD_INT("blocks", W), RAD_INT("ratio", s.ix_pool),
                                    RAD_INT("seqs", S) },
                                  {}, { snap_ids, snap_nsel, ident }, { d.tap_tab, tl });
                HY4_NEED(d.op_tap_tab, "dfly_sel");
            }
        }
        /* E: the final norm. The last layer's residual add is folded into it where a library
         * serves add_rmsnorm (byte-identical to the add and the norm on both outputs, docs/OPS.md):
         * this norm feeds no quantiser, so nothing else can take it in. */
        if (!taps_on() && rad_op_declared(b, "add_rmsnorm")) {
            op_final_add = mk(b, "add_rmsnorm",
                              { RAD_RANGE("M", 1, Md), RAD_INT("n", E), RAD_F64("eps", s.eps),
                                RAD_STR("dtype", "bf16") },
                              { out_norm }, { x, y }, { x, z });
            op_final_add = optional_op(b, op_final_add);
            if (op_final_add) {
                op_final = 0;
                if (!lay.empty()) lay.back().op_add2 = 0;   /* the add moved into the norm */
            }
        }
        if (!op_final_add) op_final = decl_rmsnorm(b, Md, E, s.eps, out_norm, x, z);
        /* a layer whose mode-0 twin resolved issues neither the post-norm nor the pair's add (the
         * needs above are already checked, as D-01 leaves them for the last layer's add) */
        for (Layer& d : lay)
            if (d.op_ar_add) { d.op_post = 0; d.op_add1 = 0; }
        /* D-03: the MLP's exact reduction with its residual add and the norm after it in one launch
         * (ar_add_rmsnorm mode 1: all_reduce -> add -> rmsnorm), byte-identical to the three. The
         * norm after a layer's MLP is the next layer's input norm, so the fused op writes `xn` and
         * that layer issues none; after the last layer it is the final norm, which then writes `z`.
         * A codes norm (op_attn_norm_q) writes its readers' codes and has no fused twin yet, a tap
         * reads `x` between the add and the norm, and the op is declared only at two ranks: any of
         * those keeps the chain. The last layer's add lives in op_final_add (D-01) or in op_add2;
         * mode 1 is that add and the norm, so either folds in and neither is issued again. */
        for (int64_t i = 0; i < layers; ++i) {
            Layer& d = lay[(size_t)i];
            const bool last = (i + 1 == layers);
            const bool plain = last || (lay[(size_t)i + 1].op_attn_norm != 0 &&
                                        lay[(size_t)i + 1].op_attn_norm_q == 0);
            if (g.world == 2 && !taps_on() && plain && rad_op_declared(b, "ar_add_rmsnorm")) {
                d.ar_next_w = last ? out_norm : lay[(size_t)i + 1].attn_norm;
                d.ar_next_out = last ? z : xn;
                d.op_ar_next = mk(b, "ar_add_rmsnorm",
                                  { RAD_RANGE("M", 1, Md), RAD_INT("n", E),
                                    RAD_INT("world_size", g.world), RAD_INT("mode", 1),
                                    RAD_F64("eps", s.eps), RAD_STR("dtype", "bf16"),
                                    RAD_INT("wire", E % 256 ? 0 : 6) },
                                  { d.ar_next_w }, { y, x }, { y, x, d.ar_next_out });
                d.op_ar_next = optional_op(b, d.op_ar_next);
            }
            if (d.op_ar_next && last) {
                d.op_add2 = 0;
                op_final_add = 0;
                op_final = 0;
            }
        }
        HY4_NEED(op_final || op_final_add || (!lay.empty() && lay.back().op_ar_next), "rmsnorm");
        RAD_ARCH_TRY(t_final.declare(b, nm, Md, "dfly.final", -1, RAD_BF16, E, z));

        /* F: the chain, one step a proposal. */
        steps.assign((size_t)depth, Step{});
        /* The correction's linears once, their ops issued by every step: gate and up by rows, down
         * by columns, so each rank reads 1/world of each and the down's partial sum is reduced
         * exactly once a step before the residual. */
        /* The codes of gate's and up's input hold a round a row block, as `ccat` does. */
        RAD_ARCH_TRY(linear(c_g, "dfly.correct_gate", gtop, S, Ew, 2 * E, c_gate, ccat, cg, RAD_SHARD_ROW, {}, true, depth * S));
        RAD_ARCH_TRY(linear(c_u, "dfly.correct_up", gtop, S, Ew, 2 * E, c_up, ccat, cup, RAD_SHARD_ROW, {}, true, depth * S));
        RAD_ARCH_TRY(linear(c_d, "dfly.correct_down", gtop, S, E, Ew, c_down, cff, cd, RAD_SHARD_COL, {}, true));
        /* Every round's rows of z (declared in the order they are issued: the rows, the gather,
         * the norm, then the rounds). */
        for (int64_t k = 0; k < depth; ++k) {
            Step& st = steps[(size_t)k];
            st.rowk = rowk;
            st.op_rowk = iota(k + 1, st.rowk, block);
            if (block_lo && k < block_lo - 1) st.op_rowk_lo = iota(k + 1, st.rowk, block_lo);
            HY4_NEED(st.op_rowk, "iota");
            HY4_NEED(!(block_lo && k < block_lo - 1) || st.op_rowk_lo, "iota");
        }
        op_zk = gather(depth * S, z, rowk, zk);
        op_hn = decl_rmsnorm(b, depth * S, E, s.eps, c_hnorm, zk, ccat);
        HY4_NEED(op_zk, "gather_rows");
        HY4_NEED(op_hn, "rmsnorm");
        for (int64_t k = 0; k < depth; ++k) {
            Step& st = steps[(size_t)k];
            const int ki = (int)k;
            /* Round 0 reads the anchors' embeddings the fusion kept in `pe`. */
            if (k > 0)
                st.op_pe = mk(b, "embed_lookup",
                              { RAD_RANGE("M", 1, S), RAD_INT("n_embd", E),
                                RAD_INT("n_vocab", s.n_vocab), RAD_STR("dtype", "bf16") },
                              { token }, { prev }, { pe });
            st.op_en = decl_rmsnorm(b, S, E, s.eps, c_enorm, pe, ccat);
            st.op_gcp = decl_copy(b, S, Ew, RAD_BF16, cg, cgu);
            st.op_ucp = decl_copy(b, S, Ew, RAD_BF16, cup, cgu);
            st.op_silu = mk(b, "silu_mul", { RAD_RANGE("M", 1, S), RAD_INT("n", Ew), RAD_STR("dtype", "bf16") },
                            {}, { cgu }, { cff });
            st.op_add = add(S, zk, cd, zc);
            RAD_ARCH_TRY(st.t_corr.declare(b, nm, S, "dfly.corrected", ki, RAD_BF16, E, zc));
            st.op_head = mk(b, "logits_gemm",
                            { RAD_RANGE("M", 1, S), RAD_INT("n_vocab", Vr), RAD_INT("n_embd", E),
                              RAD_STR("dtype", "bf16") },
                            { head }, { zc }, { logits });
            RAD_ARCH_TRY(st.t_logits.declare(b, nm, S, "dfly.logits", ki, RAD_F32, Vr, logits));
            /* Sharded, row_topk ranks this rank's columns, reports global ids and writes the pair
             * plane; the merge below overwrites cidx / cval with the global top-R. Whole, no
             * `vocab_off` (an optional parameter present at 0 is another declaration).
             *
             * cast_row_topk is that cast and row_topk in one op: it reads the f32 logits and
             * rounds each as the cast does, so its outputs are the pair's bytes without the
             * cast's launch or the bf16 plane. Taken where a library serves it; the pair
             * otherwise. */
            if (rad_op_declared(b, "cast_row_topk")) {
                if (s.world > 1)
                    st.op_ctopk = mk(b, "cast_row_topk",
                                     { RAD_RANGE("M", 1, S), RAD_INT("N", Vr), RAD_INT("R", R),
                                       RAD_INT("vocab_off", g.vocab_off), RAD_STR("dtype", "f32") },
                                     {}, { logits }, { cidx, cval, pairs });
                else
                    st.op_ctopk = mk(b, "cast_row_topk",
                                     { RAD_RANGE("M", 1, S), RAD_INT("N", Vr), RAD_INT("R", R),
                                       RAD_STR("dtype", "f32") },
                                     {}, { logits }, { cidx, cval });
                st.op_ctopk = optional_op(b, st.op_ctopk);
            }
            if (!st.op_ctopk) {
                st.op_lcast = mk(b, "cast",
                                 { RAD_RANGE("M", 1, S), RAD_INT("n", Vr), RAD_STR("from", "f32"),
                                   RAD_STR("to", "bf16") },
                                 {}, { logits }, { lbf });
                if (s.world > 1)
                    st.op_topk = mk(b, "row_topk",
                                    { RAD_RANGE("M", 1, S), RAD_INT("N", Vr), RAD_INT("R", R),
                                      RAD_INT("vocab_off", g.vocab_off), RAD_STR("dtype", "bf16") },
                                    {}, { lbf }, { cidx, cval, pairs });
                else
                    st.op_topk = mk(b, "row_topk",
                                    { RAD_RANGE("M", 1, S), RAD_INT("N", Vr), RAD_INT("R", R),
                                      RAD_STR("dtype", "bf16") },
                                    {}, { lbf }, { cidx, cval });
            }
            if (s.world > 1) {
                /* Row-interleaved, so a row's pairs from every rank are adjacent, as
                 * row_topk_merge reads them. A pair of ops a step (each step's buffers live
                 * between its own ops). */
                st.op_gather = mk(b, "all_gather",
                                  { RAD_INT("world_size", s.world), RAD_INT("numel", S * R * 2),
                                    RAD_STR("dtype", "f32"), RAD_INT("row", R * 2) },
                                  {}, { pairs }, { gath });
                st.op_merge = mk(b, "row_topk_merge",
                                 { RAD_RANGE("M", 1, S), RAD_INT("R", R),
                                   RAD_INT("world_size", s.world), RAD_STR("dtype", "bf16") },
                                 {}, { gath }, { cidx, cval });
                HY4_NEED(st.op_gather, "all_gather (f32 pairs)");
                HY4_NEED(st.op_merge, "row_topk_merge");
            }
            RAD_ARCH_TRY(st.t_idx.declare(b, nm, S, "dfly.cand_idx", ki, RAD_I32, R, cidx));
            RAD_ARCH_TRY(st.t_val.declare(b, nm, S, "dfly.cand_val", ki, RAD_F32, R, cval));
            if (coupled) {
                /* Q_k = the FP64 softmax (temperature 1) of the eight best logits (bf16-rounded as
                 * the greedy chain ranks them, f32 in `cval`); x_k ~ Q_k into column 0 and Q_k
                 * into state row (k, sequence). The shortlist is the merged global one at TP, the
                 * draw the one the engine staged on every rank, so x_k is the same on every rank.
                 * A greedy request (draw mode 0) keeps the argmax: greedy text is unchanged. */
                if (sabotage) {
                    /* Tests: eight distinct ids next to the mask id (never the model's choice in
                     * general), so the coupled states are valid laws over tokens the target
                     * rejects. */
                    const int64_t base = std::min<int64_t>((int64_t)mask, s.n_vocab - R);
                    st.op_csab = mk(b, "iota",
                                    { RAD_RANGE("M", 1, S * R), RAD_INT("start", base),
                                      RAD_INT("step", 1), RAD_INT("period", R) },
                                    {}, {}, { cidx });
                    HY4_NEED(st.op_csab, "iota (period)");
                }
                st.op_couple = mk(b, "mtp_couple_propose",
                                  { RAD_RANGE("M", 1, S), RAD_INT("n_vocab", s.n_vocab),
                                    RAD_INT("round", k + 1), RAD_INT("max_rounds", depth),
                                    RAD_F64("q_temp", q_temp) },
                                  {}, { cidx, cval, couple_draws }, { cidx, couple_state });
                HY4_NEED(st.op_couple, "mtp_couple_propose");
            }
            st.op_tok = decl_copy(b, S, 1, RAD_I32, cidx, tokens);
            st.op_prev = decl_copy(b, S, 1, RAD_I32, cidx, prev);
            HY4_NEED(k == 0 || st.op_pe, "embed_lookup");
            HY4_NEED(st.op_en, "rmsnorm");
            HY4_NEED(c_g.ok() && c_u.ok() && c_d.ok() && st.op_add && st.op_silu,
                     "gemm_nt / silu_mul / add");
            HY4_NEED(st.op_head, "logits_gemm");
            HY4_NEED(st.op_tok && st.op_prev && st.op_gcp && st.op_ucp, "cast");
            HY4_NEED(st.op_ctopk || (st.op_lcast && st.op_topk), "cast and row_topk (bf16)");
        }
        if (sabotage && !coupled) {
            op_sabotage = mk(b, "fill",
                             { RAD_RANGE("M", 1, S), RAD_INT("n", depth), RAD_STR("dtype", "i32"),
                               RAD_F64("value", (double)mask) },
                             {}, {}, { tokens });
            HY4_NEED(op_sabotage, "fill (i32)");
        }
        RAD_ARCH_TRY(t_tokens.declare(b, nm, S, "dfly.tokens", -1, RAD_I32, depth, tokens));
        if (taps_on()) {
            /* What the core handed the pass: positions, the sparse group's slots, the anchors. */
            for (auto* t : { &t_slot, &t_pos }) {
                const char* key = t == &t_slot ? "dbg.dfly.slot" : "dbg.dfly.pos";
                t->buf = decl_b(b, nm.f("%s", key), RAD_I32, { Md, 1 }, RAD_BUF_PERSIST);
                t->width = 1;
                t->op = raw_copy(b, Md, t->buf);
                HY4_NEED(t->buf && t->op, "cast (i32)");
            }
            t_anchor.buf = decl_b(b, nm.f("dbg.dfly.anchor"), RAD_I32, { S, 1 }, RAD_BUF_PERSIST);
            t_anchor.width = 1;
            t_anchor.op = raw_copy(b, S, t_anchor.buf);
            HY4_NEED(t_anchor.buf && t_anchor.op, "cast (i32)");
        }

        RadDrafterDecl dd{};
        dd.name = "dfly2";
        dd.kind = RAD_DRAFT_BLOCK;
        dd.depth = (int)depth;
        dd.proposal = tokens;
        dd.proposal_pitch = depth;
        dd.mask_token = mask;
        /* No cache of its own, so no floor: 1 rather than 0, because the core reads a window of 0
         * after a prefix hit as "the hole never closes" and would never draft (kernels/dfly2.md
         * §3.5). The refill this asks for ends a hit one token early, which costs one row. */
        dd.window = 1;
        dd.block = block;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        /* D-06: the short block, which the core stages per step under load (--draft-short-from). */
        if (block_lo) RAD_ARCH_TRY(rad_drafter_short_block(b, block_lo));
        return RAD_OK;
    }

    /* D-01: the rows mla_expand_gate_had_quant_i8g takes at most (the trunk's kFuseRows). */
    static constexpr int64_t kFuseRows = 64;

    /* rmsnorm_had_quant_i8g: the norm of `in` into `out` that writes `q8`/`s8` -- the pair of the
     * linear that reads them -- as well, byte-identical to the norm and the quantiser of that
     * linear's input it replaces. With the rank's chained quantisers (Geom::chain_q8) it carries the
     * chain, as the trunk's does. Only libr4d declares it: elsewhere (libref, libavx) the plain
     * norm and the linear's own quantiser run, as where it does not resolve. */
    static rad_op decl_norm_codes(RadBuilder* b, const Geom& g, int64_t rows, int64_t n, double eps,
                                  rad_weight gain, rad_buf in, rad_buf out, rad_buf q8,
                                  rad_buf s8, int64_t grp) {
        if (!rad_op_declared(b, "rmsnorm_had_quant_i8g")) return 0;
        rad_op h = g.chain_q8
            ? mk(b, "rmsnorm_had_quant_i8g",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                   RAD_INT("group", grp), RAD_STR("dtype", "bf16"), RAD_INT("chained", 1) },
                 { gain }, { in }, { q8, s8, out })
            : mk(b, "rmsnorm_had_quant_i8g",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                   RAD_INT("group", grp), RAD_STR("dtype", "bf16") },
                 { gain }, { in }, { q8, s8, out });
        return optional_op(b, h);
    }
    /* Whether `p` quantises its input itself (an int8 plan of its own, not a fed one), into the
     * pair a fused producer can write instead. */
    static bool qplans(const Proj& p) { return p.q && p.lin.q8 && p.lin.s8 && !p.lin.fed; }

    /* A draft linear's step over `n` rows of `in` into `out`; `codes_ready` says a fused producer
     * wrote this step's codes, so the step issues its own quantiser not. */
    static void run(RadCtx* c, const Proj& p, rad_buf in, rad_buf out, int64_t n,
                    bool codes_ready = false) {
        if (p.q) {
            ActFP8 a{};
            a.x = in;
            a.q8 = p.q8;   /* the persist chain's pair; a layer linear takes its own */
            a.s8 = p.s8;
            if (p.q8) a.g8 = p.g8;
            p.lin.step(c, a, out, n, 0, 0, -1, codes_ready);
        } else {
            RAD_ISSUE_N(c, p.op, n, brows(in, n), RAD_W(p.w), brows(out, n));
        }
    }

    /* The query pass (draft_pass 1). */
    void query(RadCtx* c, const RadBatch* batch) const {
        const int64_t n = batch->n_seq, M = batch->n_tok;
        /* The drafter's block or, under load, its short one (D-06): as many rounds as proposals. */
        const int64_t blk = block_lo > 0 && n > 0 && M == n * block_lo ? block_lo : block;
        const int64_t rounds = depth - (block - blk);
        if (!served || n <= 0 || M != n * blk || M > Md || !batch->anchor_rows) {
            rad_step_fail(c, "hyv4flash: the DFly2 query pass needs the drafter's block (or its "
                             "short block) of rows a sequence and the anchor rows "
                             "(RadBatch::anchor_rows)");
            return;
        }
        const auto* kb = kv_batch(batch, kvg);
        if (!kb || !kb->slot_mapping || !kb->block_table) {
            rad_step_fail(c, "hyv4flash: the DFly2 query pass carries no sparse latent group");
            return;
        }
        const RadOperand pos = praw(batch->positions, RAD_I32, M);
        const RadOperand rpos = rope_pos_of(s, batch, M);
        const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, n + 1);
        const RadOperand src = praw(batch->anchor_rows, RAD_I32, n);
        const RadOperand bt = praw2(kb->block_table, RAD_I32, n, kb->block_table_pitch);
        if (t_pos.op) {
            RAD_ISSUE_N(c, t_pos.op, M, pos, brows(t_pos.buf, M));
            RAD_ISSUE_N(c, t_slot.op, M, praw(kb->slot_mapping, RAD_I32, M), brows(t_slot.buf, M));
            RAD_ISSUE_N(c, t_anchor.op, n, src, brows(t_anchor.buf, n));
        }
        /* A */
        RAD_ISSUE_N(c, op_embed, M, praw(batch->token_ids, RAD_I32, M), RAD_W(token), brows(x, M));
        t_embed.step(c, brows(x, M), M);
        /* B */
        RAD_ISSUE_N(c, blk == block ? op_row0 : op_row0_lo, n, brows(row0, n));
        RAD_ISSUE_N(c, op_ea, n, brows(x, M), brows(row0, n), brows(pe, n));
        RAD_ISSUE_N(c, op_th, n, RAD_B(final_hidden), src, brows(th, n));
        RAD_ISSUE_N(c, op_fen, n, brows(pe, n), RAD_W(f_enorm), bcol(cat, 0, E, n));
        RAD_ISSUE_N(c, op_fhn, n, brows(th, n), RAD_W(f_hnorm), bcol(cat, E, E, n));
        run(c, p_fuse, cat, fu, n);
        RAD_ISSUE_N(c, op_fgate, n, brows(fu, n), RAD_W(f_gate), brows(fu, n));
        RAD_ISSUE_N(c, op_fadd, n, brows(pe, n), brows(fu, n), brows(ea, n));
        t_fused.step(c, brows(ea, n), n);
        RAD_ISSUE_N(c, op_scatter, n, brows(ea, n), brows(row0, n), brows(x, M));
        if (op_ident) RAD_ISSUE_N(c, op_ident, n * ident_w, bcol_at(ident, 0, ident_w, 0, ident_w, n));
        /* C, D */
        const int64_t half = (L + RO) / 2;
        /* D-03: set once the previous layer's fused reduction has written `xn` (its MLP sum, the
         * residual add and this layer's input norm in one launch), so this layer issues no norm. */
        bool in_norm_done = false;
        for (const Layer& d : lay) {
            const Snap* sn = snap_of(d.ixl);
            RAD_ISSUE_N(c, d.op_tab, M, brow_slice(snap_sel, sn->base, snap_t, sn->width),
                        brow_slice(snap_nsel, sn->base, snap_t, 1), src, bt, pos, cu,
                        brows(d.tab, M), brows(d.len, M), brows(d.cnt, M));
            if (d.op_tap_tab) {
                RAD_ISSUE_N(c, d.op_tap_tab, M, brow_slice(snap_ids, sn->base, snap_t, sn->width),
                            brow_slice(snap_nsel, sn->base, snap_t, 1), src,
                            bcol_at(ident, 0, ident_w, 0, ident_w, n), pos, cu,
                            brows(d.tap_tab, M), brows(d.t_len.buf, M), RAD_NONE);
            }
            if (in_norm_done) {
                /* the previous layer's ar_add_rmsnorm wrote xn: no norm here */
            } else if (d.op_attn_norm_q)
                RAD_ISSUE_N(c, d.op_attn_norm_q, M, brows(x, M), RAD_NONE, RAD_W(d.attn_norm),
                            brows(d.p_qa.lin.q8, M), brows(d.p_qa.lin.s8, M), brows(xn, M));
            else
                RAD_ISSUE_N(c, d.op_attn_norm, M, brows(x, M), RAD_W(d.attn_norm), brows(xn, M));
            d.t_in.step(c, brows(xn, M), M);
            const rad_buf qa_in = d.ra ? qa_raw : qa;
            const bool ac = d.op_attn_norm_q != 0;
            run(c, d.p_qa, xn, qa_in, M, ac);
            if (d.op_qa_norm_q)
                RAD_ISSUE_N(c, d.op_qa_norm_q, M, brows(qa_in, M), RAD_NONE, RAD_W(d.qa_norm),
                            brows(d.p_qb.lin.q8, M), brows(d.p_qb.lin.s8, M), brows(qa, M));
            else
                RAD_ISSUE_N(c, d.op_qa_norm, M, brows(qa_in, M), RAD_W(d.qa_norm), brows(qa, M));
            d.t_qres.step(c, brows(qa, M), M);
            run(c, d.p_qb, qa, q, M, d.op_qa_norm_q != 0);
            RadOperand qn = brows(q, M), qr = qn;
            qn.cols = N;
            qr.offset = N;
            qr.cols = RO;
            RAD_ISSUE_N(c, d.op_qn, M, qn, brows(q_nope, M));
            RAD_ISSUE_N(c, d.op_qr, M, qr, brows(q_rope, M));
            RAD_ISSUE_N(c, d.op_q_rope, M, brows(q_rope, M), rpos);
            run(c, d.p_kva, xn, row, M, ac && d.p_kva.lin.q8 == d.p_qa.lin.q8);
            /* Every row of the block into trunk layer Lkv's cache before any row attends. The
             * row's norm, rotary and store are one launch where it is declared (and the cache
             * streams nothing: that store is kv_store_streamed). */
            const RadOperand slots = praw(kb->slot_mapping, RAD_I32, M);
            const bool row_q = d.op_row_q && !kb->stream;
            if (row_q) {
                RAD_ISSUE_N(c, d.op_row_q, M, brows(row, M), RAD_W(d.kv_norm), rpos, slots,
                            kv_cache(kvg, d.kvl));
            } else {
                RAD_ISSUE_N(c, d.op_kv_norm, M, bcol(row, 0, L, M), RAD_W(d.kv_norm), bcol(row, 0, L, M));
                RAD_ISSUE_N(c, d.op_k_rope, M, bcol(row, L, RO, M), rpos);
            }
            d.t_row.step(c, brows(row, M), M);
            if (!row_q) {
                if (break_store_late && !kb->stream) {
                    /* (test only) the attention first, over the previous pass's rows */
                } else if (!kb->stream) {
                    RAD_ISSUE_N(c, d.op_store, M, bcol(row, 0, half, M), bcol(row, half, half, M),
                                slots, kv_cache(kvg, d.kvl));
                } else {
                    const bool staged = kv_stage_wait(c, kvg, kb, d.kvl);
                    const KVStreamLayer S = kv_stream_layer(c, kvg, d.kvl, kb);
                    if (!d.op_store_stream || !S.host || S.blocks <= 0) {
                        rad_step_fail(c, "hyv4flash: the sparse latent cache streams but the build has "
                                         "no kv_store_streamed row for the DFly2 draft rows");
                        return;
                    }
                    RAD_ISSUE_N(c, d.op_store_stream, M, bcol(row, 0, half, M),
                                bcol(row, half, half, M), slots, kv_cache(kvg, d.kvl),
                                praw(S.res, RAD_I32, S.blocks),
                                praw2(S.host, RAD_U8, S.blocks, S.bbytes));
                    if (staged)
                        RAD_ISSUE_N(c, d.op_store, M, bcol(row, 0, half, M), bcol(row, half, half, M),
                                    slots, kv_cache(kvg, d.kvl));
                }
            }
            run(c, d.p_gate, xn, gt, M, ac && d.p_gate.lin.q8 == d.p_qa.lin.q8);
            d.t_gate.step(c, brows(gt, M), M);
            RAD_ISSUE_N(c, d.op_absorb, M, brows(q_nope, M), RAD_W(d.kv_b), brows(q_lat, M));
            if (kb->stream && !kv_stage_layer(c, kvg, kb, d.kvl)) {
                /* The table names block ids; the ring holds slots (as the trunk's sel). */
                const KVStreamLayer S = kv_stream_layer(c, kvg, d.kvl, kb);
                if (!d.op_resolve || !S.host || S.blocks <= 0 || S.slots <= 0) {
                    rad_step_fail(c, "hyv4flash: the sparse latent cache streams but the build has "
                                     "no kv_resolve for the DFly2 table");
                    return;
                }
                kv_resolve_issue(c, d.op_resolve, S, brows(d.tab, M), brows(d.cnt, M), kvg, d.kvl, M);
            }
            RAD_ISSUE_N(c, d.op_attn, M, brows(q_lat, M), brows(q_rope, M), kv_cache(kvg, d.kvl),
                        brows(d.tab, M), brows(d.len, M), RAD_W(d.sinks), brows(lat, M));
            if (break_store_late && !kb->stream)
                RAD_ISSUE_N(c, d.op_store, M, bcol(row, 0, half, M), bcol(row, half, half, M), slots,
                            kv_cache(kvg, d.kvl));
            /* the fused expansion takes at most kFuseRows rows: a taller block of drafts keeps
             * the plain expansion and o_proj's own quantiser */
            const bool xq = d.op_expand_q && M <= kFuseRows;
            if (xq) {
                RAD_ISSUE_N(c, d.op_expand_q, M, brows(lat, M), RAD_W(d.kv_b), brows(gt, M),
                            brows(d.p_out.lin.q8, M), brows(d.p_out.lin.s8, M));
            } else {
                RAD_ISSUE_N(c, d.op_expand, M, brows(lat, M), RAD_W(d.kv_b), brows(ao, M));
                RAD_ISSUE_N(c, d.op_gmul, M, brows(ao, M), brows(gt, M), brows(ag, M));
            }
            run(c, d.p_out, ag, o, M, xq);
            if (d.op_ar_add) {
                /* D-03 mode 0: the exact reduction of o, the post-norm and the add in one launch */
                RAD_ISSUE_N(c, d.op_ar_add, M, brows(o, M), brows(x, M), RAD_W(d.post_norm),
                            brows(y, M));
            } else {
                issue_all_reduce(c, op_ar_o, o, M, E);
                d.t_oproj.step(c, brows(o, M), M);
                const rad_buf ab = d.post ? y : o;
                if (d.post) RAD_ISSUE_N(c, d.op_post, M, brows(o, M), RAD_W(d.post_norm), brows(y, M));
                d.t_out.step(c, brows(ab, M), M);
                RAD_ISSUE_N(c, d.op_add1, M, brows(x, M), brows(ab, M), brows(x, M));
            }
            d.t_mid.step(c, brows(x, M), M);
            if (kb->stream) kv_stage_next(c, kvg, d.kvl);
            if (d.op_ffn_norm_q)
                RAD_ISSUE_N(c, d.op_ffn_norm_q, M, brows(x, M), RAD_NONE, RAD_W(d.ffn_norm),
                            brows(d.p_gu.lin.q8, M), brows(d.p_gu.lin.s8, M), brows(xn, M));
            else
                RAD_ISSUE_N(c, d.op_ffn_norm, M, brows(x, M), RAD_W(d.ffn_norm), brows(xn, M));
            d.t_ffn_in.step(c, brows(xn, M), M);
            run(c, d.p_gu, xn, ffgu, M, d.op_ffn_norm_q != 0);
            if (d.op_gated)
                RAD_ISSUE_N(c, d.op_gated, M, brows(ffgu, M), brows(d.p_down.lin.q8, M),
                            brows(d.p_down.lin.s8, M));
            else
                RAD_ISSUE_N(c, d.op_silu, M, brows(ffgu, M), brows(ffo, M));
            run(c, d.p_down, ffo, y, M, d.op_gated != 0);
            if (d.op_ar_next) {
                /* D-03 mode 1: the MLP's exact reduction, its add and the norm after it (the next
                 * layer's input norm, or the final one) in one launch */
                RAD_ISSUE_N(c, d.op_ar_next, M, brows(y, M), brows(x, M), RAD_W(d.ar_next_w),
                            brows(d.ar_next_out, M));
                in_norm_done = true;
            } else {
                issue_all_reduce(c, op_ar_y, y, M, E);
                d.t_mlp.step(c, brows(y, M), M);
                if (d.op_add2)   /* 0 where the last layer's add moved into the final norm (D-01) */
                    RAD_ISSUE_N(c, d.op_add2, M, brows(x, M), brows(y, M), brows(x, M));
                d.t_resid.step(c, brows(x, M), M);
                in_norm_done = false;
            }
        }
        /* E */
        if (!lay.empty() && lay.back().op_ar_next) {
            /* the last layer's add and the final norm ran in its fused reduction (D-03) */
        } else if (op_final_add)
            RAD_ISSUE_N(c, op_final_add, M, brows(x, M), brows(y, M), RAD_W(out_norm), brows(z, M));
        else
            RAD_ISSUE_N(c, op_final, M, brows(x, M), RAD_W(out_norm), brows(z, M));
        t_final.step(c, brows(z, M), M);
        /* F. Every round's rows of z and their norm first, one launch each over rounds * n rows
         * (D-02), round k at rows [k * n, (k + 1) * n). */
        const int64_t RN = rounds * n;
        for (int64_t k = 0; k < rounds; ++k) {
            const Step& st = steps[(size_t)k];
            RAD_ISSUE_N(c, blk == block ? st.op_rowk : st.op_rowk_lo, n, brow_slice(rowk, k * n, n, 1));
        }
        RAD_ISSUE_N(c, op_zk, RN, brows(z, M), brows(rowk, RN), brows(zk, RN));
        RAD_ISSUE_N(c, op_hn, RN, brows(zk, RN), RAD_W(c_hnorm), bcol(ccat, 0, E, RN));
        /* gate and up read one input: where both are int8 linears of one code form they share the
         * chain's code pair, so up reads the codes gate's quantiser has just written (the same op
         * over the same rows) instead of writing them again. */
        const bool up_fed = c_g.q && c_u.q && c_g.q8 && c_g.q8 == c_u.q8 && c_g.s8 == c_u.s8 &&
                            c_g.lin.op_q && c_g.lin.form.same(c_u.lin.form);
        /* A correction linear over round rows [r0, r0 + n) of `ccat`, into rows [0, n) of `out`. */
        auto corr = [&](const Proj& p, int64_t r0, rad_buf out, bool codes_ready) {
            if (!p.q) {
                RAD_ISSUE_N(c, p.op, n, brow_slice(ccat, r0, n, 2 * E), RAD_W(p.w), brows(out, n));
                return;
            }
            ActFP8 a{};
            a.x = ccat;
            a.q8 = p.q8;
            a.s8 = p.s8;
            if (p.q8) a.g8 = p.g8;
            p.lin.step(c, a, out, 0, r0, n, 0, codes_ready);
        };
        for (int64_t k = 0; k < rounds; ++k) {
            const Step& st = steps[(size_t)k];
            const int64_t r0 = break_chain_rows ? 0 : k * n;
            if (k > 0) RAD_ISSUE_N(c, st.op_pe, n, brows(prev, n), RAD_W(token), brows(pe, n));
            RAD_ISSUE_N(c, st.op_en, n, brows(pe, n), RAD_W(c_enorm), bcol_at(ccat, r0, 2 * E, E, E, n));
            corr(c_g, r0, cg, false);
            corr(c_u, r0, cup, up_fed);
            RAD_ISSUE_N(c, st.op_gcp, n, brows(cg, n), bcol(cgu, 0, Ew, n));
            RAD_ISSUE_N(c, st.op_ucp, n, brows(cup, n), bcol(cgu, Ew, Ew, n));
            RAD_ISSUE_N(c, st.op_silu, n, brows(cgu, n), brows(cff, n));
            run(c, c_d, cff, cd, n);
            issue_all_reduce(c, op_ar_c, cd, n, E);
            RAD_ISSUE_N(c, st.op_add, n, brow_slice(zk, r0, n, E), brows(cd, n), brows(zc, n));
            st.t_corr.step(c, brows(zc, n), n);
            RAD_ISSUE_N(c, st.op_head, n, brows(zc, n), RAD_W(head), brows(logits, n));
            st.t_logits.step(c, brows(logits, n), n);
            if (!st.op_ctopk) RAD_ISSUE_N(c, st.op_lcast, n, brows(logits, n), brows(lbf, n));
            /* cast_row_topk reads the f32 logits where row_topk reads their bf16 copy */
            const rad_op topk = st.op_ctopk ? st.op_ctopk : st.op_topk;
            const rad_buf tin = st.op_ctopk ? logits : lbf;
            if (st.op_merge) {
                RAD_ISSUE_N(c, topk, n, brows(tin, n), brows(cidx, n), brows(cval, n),
                            brows(pairs, n));
                RAD_ISSUE(c, st.op_gather, RAD_B(pairs), RAD_B(gath));
                RAD_ISSUE_N(c, st.op_merge, n, brows(gath, n), brows(cidx, n), brows(cval, n));
            } else {
                RAD_ISSUE_N(c, topk, n, brows(tin, n), brows(cidx, n), brows(cval, n), RAD_NONE);
            }
            st.t_idx.step(c, brows(cidx, n), n);
            st.t_val.step(c, brows(cval, n), n);
            if (st.op_couple) {
                if (st.op_csab) RAD_ISSUE_N(c, st.op_csab, n * R, bcol_at(cidx, 0, R, 0, R, n));
                RAD_ISSUE_N(c, st.op_couple, n, brows(cidx, n), brows(cval, n),
                            praw(batch->seq_ids, RAD_I32, batch->n_seq),
                            brows(couple_draws, n), RAD_B(couple_state));
            }
            RAD_ISSUE_N(c, st.op_tok, n, bcol(cidx, 0, 1, n), bcol(tokens, k, 1, n));
            /* The next round's embedding reads `prev`; the last round has no reader. */
            if (k + 1 < rounds) RAD_ISSUE_N(c, st.op_prev, n, bcol(cidx, 0, 1, n), brows(prev, n));
        }
        if (op_sabotage) RAD_ISSUE_N(c, op_sabotage, n, brows(tokens, n));
        t_tokens.step(c, brows(tokens, n), n);
    }
};

}  // namespace hyv4flash
#endif
