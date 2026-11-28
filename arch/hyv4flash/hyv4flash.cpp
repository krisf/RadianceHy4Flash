/* hyv4flash -- Tencent Hy4-flash (`model_type: hy_v4_flash`, HYV4FlashForConditionalGeneration)
 * from its bf16 checkpoint: the text trunk's serving graph at 1, 2 or 4 ranks, and the weight
 * manifest and checkpoint name map rad-convert needs. docs/ARCHITECTURES.md describes the model and its traps.
 *
 *   42 layers over four hyper-connection streams (iHC: a read gate and a write gate a block, no
 *     mixing matrix), hidden 3072, vocab 120832, embedding and lm_head separate
 *   attention: gated MLA (q_lora 1024, kv_lora 512, a shared 64-wide rotary key, a sigmoid gate
 *     of the value width on every head, a learned sink a head, a sandwich norm behind o_proj);
 *     12 sparse layers of 32 heads behind a compressed indexer (16 heads of 128 over keys pooled
 *     four tokens a group) and 30 sliding-window layers of 64 heads (window 512)
 *   feed-forward: layer 0 dense (13824), 41 latent MoE layers: the router and the shared expert
 *     read the hidden state, 512 routed experts at top-16 run in a 1536-wide latent
 *     (routed_expert_down_proj in front, routed_expert_up_proj behind), a sandwich norm closes
 *   Engram: before layers 1, 11, 21 and 31 a hashed bigram/trigram lookup (16 heads of 96 over a
 *     165M-row table a layer, 63.4B parameters in all) gated per stream
 *   DFly2: a 5-layer drafter over the sparse layers' cache and selection
 *   vision: a 27-layer ViT (1152 wide, patch 16) and a two-convolution projector with a newline
 *     row closing each merged row (hyv4flash_vision.h); the trunk then rotates at M-RoPE positions
 *
 * ============================== THE GRAPH (PLAN §1.2) ==============================
 *
 * A step: the token embedding copied into every stream; per layer the Engram add in front of
 * layers 1, 11, 21 and 31 (hyv4flash_engram.h), the attention connection (ihc_pre, attn_norm),
 * the attention (hyv4flash_attn.h, the sparse layers' indexer hyv4flash_indexer.h), ihc_post, the
 * feed-forward connection, the feed-forward (hyv4flash_moe.h), ihc_post; at the end the head's
 * collapse (ihc_pre over hc_head), the final norm and the lm_head into the declared logits.
 *
 * Served: one, two or four ranks (PLAN §1.5: whole heads a rank, the dense and shared widths
 * split, whole experts a rank, the lm_head by vocabulary, two exact all-reduces a layer, the rest
 * replicated; split_ranks, Ffn::place), the text trunk, bf16 or quantised (PLAN §4.3): the routed
 * experts as MoeFP8 reads them (the w4nl64 / w5nl64 / w6nl64 codebooks on the rotated E4M3 arm,
 * rotated int8 and the IQR trellis on the rotated int8 arm, clamped in every layer; a slice of every
 * expert a rank at two and four ranks), the trunk linears bf16 or int8 (Lin: rotated or not, IQ8R,
 * the IQR trellis such as u5), the lm_head bf16 or int8, the Engram table bf16 or E4M3 rows. Refused
 * by name: other worlds, a trunk linear in another form, a quantised embedding, kv_b (DECISIONS U2),
 * Engram projection or f32 parameter. Pictures and video through the tower
 * (hyv4flash_vision.h; the trunk then rotates at M-RoPE positions, `imrope_gptj`). DFly2 is
 * served at one, two or four ranks (split as the sparse layers) as a block drafter (hyv4flash_dfly.h)
 * when the deployment speculates
 * (--num-speculative-tokens 1..5; the probe states no depth, so speculation stays off unless
 * asked for); elsewhere it is declared as the checkpoint holds it, and says why it is not
 * served.
 *
 * The caches (PLAN §1.3): the sliding layers' latent rows in a window group (block 16), the sparse
 * layers' in a full group of block 4 (a page is one indexer group, so the indexer's table is the
 * attention's), the indexer's pooled keys (block 4, one rotated key a page) and its raw-key tail
 * (a linear state of ratio + speculation rows), and each Engram layer's token history (a CONV
 * state of ngram - 1 ids).
 *
 * The geometry is read from the checkpoint's config and nothing is assumed about its size, so
 * the convert test and the parity harness run on a model of a few kilobytes. */
#include "hyv4flash_common.h"
#include "hyv4flash_attn.h"
#include "hyv4flash_indexer.h"
#include "hyv4flash_moe.h"
#include "hyv4flash_engram.h"
#include "hyv4flash_calib.h"
#include "hyv4flash_vision.h"
#include "hyv4flash_dfly.h"
#include <arch/rad_block_hc.h>   // HcProject; hyv4flash_common.h imports rad::arch

#include "chat.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hyv4flash {

/* ------------------------------------------------------------------ the pipelined prefill
 * A prefill chunk wider than two shape bands runs as two row pieces: one piece's junction (join,
 * exact all-reduce, sandwich norm) is issued on the second lane and the other piece's arms run
 * beside it, so half of the two reductions a layer pays rides under GEMMs (docs/TP4.md). It is the
 * same trade qwen35_fp8's step_pipelined makes at two ranks; here the layer's whole collective is
 * already one Junction, so nothing but that junction and Ffn::body has to know about pieces.
 *
 * Off unless RADIANCE_HY4_PIPE=1 says otherwise, off with taps (they are declared at the step's
 * height, not per piece), off for a calibration run (its hooks read between the ops) and off for a
 * media step (the encoder's own reduction would have to ride the other lane). */
/* A piece must stay inside the shape band the whole step's kernels were picked from, so the floor
 * is the band's lower edge: a 4096-row chunk halves into two 2048-row pieces, a 2048-row chunk does
 * not (two 1024-row pieces would pick other kernels and other numbers). */
constexpr int64_t kPipeMinRows = 1024;

inline bool pipe_env_on() {
    const char* e = std::getenv("RADIANCE_HY4_PIPE");
    return e && *e && std::strcmp(e, "0") != 0;
}
/* The floor, overridable to sweep a smaller piece than the band allows (RADIANCE_HY4_PIPE_MIN). */
inline int64_t pipe_min_rows() {
    const char* e = std::getenv("RADIANCE_HY4_PIPE_MIN");
    const int64_t v = (e && *e) ? std::atoll(e) : 0;
    return v > 0 ? v : kPipeMinRows;
}
/* Half the chunk, a multiple of 64 (the GEMMs' row grain); RADIANCE_HY4_PIPE_ROWS cuts elsewhere. */
inline int64_t pipe_rows(int64_t T) {
    const char* e = std::getenv("RADIANCE_HY4_PIPE_ROWS");
    if (e && *e) {
        const int64_t v = std::atoll(e);
        if (v > 0 && v < T) return ((v + 63) / 64) * 64;
    }
    return ((T + 1) / 2 + 63) / 64 * 64;
}

/* Both pieces at or above the floor (#282: the gate checked n0 alone, so 1984 rows cut 1024 + 960). */
inline bool pipe_cut_ok(int64_t n0, int64_t n1) {
    return n0 >= pipe_min_rows() && n1 >= pipe_min_rows();
}

/* The second lane (abi/rad_runtime.h): issue on lane 1 while lane 0 runs on, then join. */
inline void pipe_on1(RadCtx* c, const std::function<void()>& fn) {
    rad_lane(c, 1);
    rad_lane_join(c, 0, 1);
    fn();
    rad_lane(c, 0);
}
inline void pipe_wait1(RadCtx* c) { rad_lane_join(c, 1, 0); }
/* RADIANCE_HY4_PIPE_LANE=0 issues the same pieces on lane 0 only: it separates what the pieces cost
 * from what riding the second lane gains. */
inline bool pipe_lane() {
    static const bool on = [] {
        const char* e = std::getenv("RADIANCE_HY4_PIPE_LANE");
        return !(e && *e && e[0] == '0');
    }();
    return on;
}
inline void pipe_issue(RadCtx* c, const std::function<void()>& fn) {
    if (pipe_lane())
        pipe_on1(c, fn);
    else
        fn();
}
inline void pipe_join(RadCtx* c) {
    if (pipe_lane()) pipe_wait1(c);
}
/* P-17: RADIANCE_HY4_PIPE_ATTN=1 runs the attention core in the pipeline's pieces as well
 * (Attn::core_piece), so piece 0's attention junction rides lane 1 beside piece 1's attention instead
 * of running exposed. Off unless set; it needs the pipeline (RADIANCE_HY4_PIPE=1). */
inline bool pipe_attn_on() {
    const char* e = std::getenv("RADIANCE_HY4_PIPE_ATTN");   /* read once a step */
    return e && *e && e[0] == '1';
}
/* The cut's rows are one sequence's consecutive positions (the bounds below take n0 from them). */
inline bool pipe_positions_run(const RadBatch* batch, int64_t T) {
    const int32_t* ph = batch->positions_h;
    if (!ph || !batch->positions || T < 2) return false;
    for (int64_t i = 1; i < T; ++i)
        if (ph[i] != ph[0] + i) return false;
    return true;
}
/* One line per process when the pipeline first runs, so a log says which prefill shape took it. */
inline bool& pipe_said() {
    static bool said = false;
    return said;
}

/* Rows the control-vector derivation tap holds. Every collection prompt is under this many
 * tokens (the chat template included), and the probe reads the same count back. */
constexpr int64_t kCvecTapRows = 192;
/* Most directions per layer the plugin will declare. The package's rank is bounded here so the
 * chained projection count is a compile-time array; a larger file steers its first kMaxCvecRank
 * directions and says so. */
constexpr int64_t kMaxCvecRank = RAD_CONTROL_VECTOR_MAX_RANK;

struct Layer {
    bool sliding = false;
    int engram = -1;      /* index into Model::engrams, -1 for none */
    Hyper hc_attn{}, hc_ffn{};
    rad_weight attn_gain = 0, ffn_gain = 0;
    rad_op op_attn_norm = 0, op_ffn_norm = 0;
    Attn attn{};
    Indexer ix{};
    Ffn ffn{};
    Tap t_stream_in{}, t_attn_x{}, t_attn_post{}, t_mid{}, t_ffn_x{}, t_ffn_post{}, t_ffn_in{},
        t_resid{};
    /* The iHC junctions (WP16.2): attention -> feed-forward, and the feed-forward -> the next
     * layer's attention (or the head; tail 0 when an Engram opens the next layer). */
    Junction j_mid{}, j_end{};
    /* Control-vector projections of the wide stream, after this layer's final write. Up to
     * kMaxCvecRank chained directions; inert when no vector is loaded, this layer is outside its
     * range, or no row asked for it. */
    HcProject proj[kMaxCvecRank]{};
    /* The same stream a projection would read, for deriving a vector offline
     * (RADIANCE_HY4_TAPS=1 + RADIANCE_DUMP_BUF=dbg.cvec.<l>:<rows>). Off, not declared. */
    Tap t_cvec{};
};

struct Model {
    Names nm{};
    Shape s{};
    Geom g{};
    int64_t n_weights = 0;
    Trunk tr{};
    AttnWire aw_sparse{}, aw_sliding{};
    /* P-04: the prefill sequences' bounds rebased to 0, once a step (op_rebase, issued before the
     * layers and declared there too: declaration order is liveness), read by both wires. */
    rad_buf cu_prefill = 0;
    rad_op op_rebase = 0;
    IndexWire iw{};
    MoeWire mw{};
    EngramWire ew{};
    rad_buf x0 = 0, dense_gu = 0, dense_ff = 0, dense_out = 0, final_hidden = 0, logit_input = 0,
            logits = 0;
    rad_kvgroup kv_sparse = 0, kv_sliding = 0, kv_index = 0, kv_tail = 0, kv_engram = 0;
    rad_weight token = 0, head = 0, final_gain = 0;
    rad_op embed = 0, op_stream = 0, op_norm = 0, op_gather = 0, op_head = 0;
    Hyper hc_head{};
    std::vector<Layer> layers;
    std::vector<Engram> engrams;
    /* M-18b: the layers behind the first register here (EngramWire::pf), so the first Engram
     * layer's step can issue their step-start prefetch. Holds pointers into `engrams`, which is
     * sized before any declare and never resized again. */
    std::vector<Engram*> eg_pf;
    Tap t_embed{}, t_hc_head{}, t_final{};
    bool force_absorbed = false;
    Calib calib{};   /* --calib-dir: the trunk's Hessians (hyv4flash_calib.h) */
    Codes codes{};   /* the int8 trunk linears' code pair (hyv4flash_common.h, Lin) */
    Vision vision{}; /* the tower, served on encoder passes (hyv4flash_vision.h) */
    /* A media step's encoder rows, every rank's: the core hands rank 0 the rows and the others
     * zeros (RadBatch::mm_rows), and the embedding here is replicated, so the rows are summed
     * over the ranks (exactly: one non-zero term) before they replace the lookup's. */
    rad_buf mm_rows = 0;
    rad_buf proj_on = 0;   /* [T] int32: 1 = project this row this step */
    int64_t proj_rank = 1; /* direction GROUPS per layer declared from ctx->cvec (clamped) */
    rad_op op_mm_scatter = 0, op_mm_copy = 0, op_mm_reduce = 0;
    rad_buf tail_meta = 0;
    rad_op pack_h = 0, pack_post = 0, unpack_hidden = 0;
    Dfly df{};       /* DFly2, the block drafter (hyv4flash_dfly.h) */
    int arm = kArmBf16;   /* the routed arms' activation codes (Ffn::arm) */
    /* The pipelined prefill (docs/TP4.md, "the pipelined prefill"): a chunk wider than two shape
     * bands runs as two row pieces, so the layer's junction (join, reduction, sandwich norm) of one
     * piece rides the second lane beside the other piece's GEMMs. `pipe_on` is the declare-time
     * half (the junction's narrow reduction exists, the buffers are concurrent); the step adds the
     * per-batch half. */
    bool pipe_on = false;
    /* P-17 (RADIANCE_HY4_PIPE_ATTN): the attention core in the pipeline's two pieces too, so piece
     * 0's junction rides lane 1 beside piece 1's attention. The bounds plane and its two launches,
     * issued once a step (pipe_cut). */
    PieceCut cut{};
    rad_op op_cut_fill = 0, op_cut_sub = 0;
};
static std::unique_ptr<Model> models[MAX_SLOTS];
/* Where a sizing declare goes (RadBuildCtx::shape_probe): the engine declares again at smaller step
 * sizes to size the activation arena's levels, lends the memory above a level to the expert slab
 * while steps stay that small, and arms the prefill stager above the largest level. Such a declare
 * must leave the state step() reads alone, so it fills this model instead (nemotronh's and
 * glm5next's probe_model); the engine runs one per (level, rank) in parallel threads. The friend's
 * plugin measured the loan at 10.6 -> 24 tok/s decode on the real model at TP2 (tiered experts). */
static thread_local std::unique_ptr<Model> probe_model;

/* ------------------------------------------------------------------ the config */
static int64_t top(const RadModelMeta* m, const char* key, int64_t dflt) {
    return rad_meta_geti(m, key, dflt);
}
static int64_t txt(const RadModelMeta* m, const char* key, int64_t dflt) {
    return meta_text_geti(m, key, dflt);
}
/* The sliding layers' value where the config states one (`swa_<key>`), else the sparse layers'. */
static int64_t swa(const RadModelMeta* m, const char* key, int64_t sparse) {
    char k[96];
    std::snprintf(k, sizeof k, "swa_%s", key);
    return txt(m, k, sparse);
}

static int read_shape(Shape& s, const RadModelMeta* m) {
    s.n_layer = m->n_layers;
    s.n_embd = m->n_embd;
    s.n_vocab = m->n_vocab;
    s.hc = txt(m, "hc_mult", 0);
    s.q_lora = txt(m, "q_lora_rank", 0);
    s.kv_lora = txt(m, "kv_lora_rank", 0);
    s.sparse.heads = txt(m, "num_attention_heads", 0);
    s.sparse.qk_nope = txt(m, "qk_nope_head_dim", 0);
    s.sparse.qk_rope = txt(m, "qk_rope_head_dim", 0);
    s.sparse.v = txt(m, "v_head_dim", 0);
    s.sliding.heads = swa(m, "num_attention_heads", s.sparse.heads);
    s.sliding.qk_nope = swa(m, "qk_nope_head_dim", s.sparse.qk_nope);
    s.sliding.qk_rope = swa(m, "qk_rope_head_dim", s.sparse.qk_rope);
    s.sliding.v = swa(m, "v_head_dim", s.sparse.v);
    s.n_ff = txt(m, "intermediate_size", 0);
    s.n_expert = m->n_expert;
    s.n_ff_exp = txt(m, "moe_intermediate_size", 0);
    s.latent = txt(m, "routed_expert_hidden_size", 0);
    s.n_ff_shared = s.n_ff_exp * txt(m, "n_shared_experts", 0);
    s.ix_heads = txt(m, "index_n_heads", 0);
    s.ix_dim = txt(m, "index_head_dim", 0);
    s.ix_pool = txt(m, "compressed_indexer_compress_ratio", 0);
    s.ix_group_topk = txt(m, "compressed_indexer_group_topk", 0);
    s.dense_limit = s.ix_group_topk * s.ix_pool;
    s.top_k = txt(m, "num_experts_per_tok", m->n_expert_used);
    s.route_scale = meta_text_getf(m, "routed_scaling_factor", 1.0);
    s.swiglu_limit = meta_text_getf(m, "swiglu_limit", 0.0);
    s.norm_topk = txt(m, "norm_topk_prob", 1) != 0;
    s.eps = m->rms_eps;
    s.hc_eps = meta_text_getf(m, "hc_eps", 1e-6);
    s.hc_mag = meta_text_getf(m, "hc_magnitude", 2.0);
    /* The two layer kinds' rotary bases (HF keys them by layer type; the release config spells
     * the sparse kind `deepseek_sparse_attention`, the port `indexed_attention`). */
    s.theta_sliding = meta_text_getf(m, "rope_parameters.sliding_attention.rope_theta", 1e4);
    s.theta_sparse = meta_text_getf(
        m, "rope_parameters.deepseek_sparse_attention.rope_theta",
        meta_text_getf(m, "rope_parameters.indexed_attention.rope_theta", 1e7));
    /* A sliding row sees `sliding_window` keys, its own included (HF's sliding mask;
     * `window_size` [511, 0] says the same as 511 before it). */
    s.window = txt(m, "sliding_window", 0);
    s.engram_pad = txt(m, "engram_pad_id", -1);
    s.engram_fp8 = txt(m, "engram_fp8_embedding", 0) != 0;
    s.engram_map = txt(m, "engram_lookup_table_size", s.n_vocab);
    s.image_token = top(m, "image_token_id", -1);
    s.video_token = top(m, "video_token_id", -1);

    if (s.n_layer < 1 || s.n_embd < 1 || s.n_vocab < 1 || s.hc < 1 || s.q_lora < 1 ||
        s.kv_lora < 1 || s.sparse.heads < 1 || s.sparse.qk_nope < 1 || s.sparse.qk_rope < 1 ||
        s.sparse.v < 1 || s.sliding.heads < 1 || s.n_ff < 1)
        return refuse("the config lacks one of hidden_size, num_hidden_layers, vocab_size, "
                      "hc_mult, q_lora_rank, kv_lora_rank, num_attention_heads, "
                      "qk_nope_head_dim, qk_rope_head_dim, v_head_dim or intermediate_size");
    /* The graph this plugin maps: MLA with the head-wise gate, sinks and sandwich norms, iHC,
     * and the compressed indexer on the sparse layers. A checkpoint without one of them has
     * other tensors, and mapping it by these names would leave them out without an error. */
    const struct { const char* key; int64_t want; } flags[] = {
        { "use_mla", 1 }, { "gated_mla", 1 }, { "learnable_sink", 1 }, { "sandwich_norm", 1 },
        { "enable_ihc", 1 }, { "use_dsa", 1 }, { "use_compressed_indexer", 1 },
        { "attention_bias", 0 }, { "moe_router_use_sigmoid", 1 },
        { "moe_router_enable_expert_bias", 1 },
    };
    for (const auto& f : flags)
        if (txt(m, f.key, -1) != f.want) {
            fprintf(stderr, "radiance: hyv4flash: the config's %s is %lld; this plugin maps the "
                            "Hy4-flash graph, where it is %lld\n",
                    f.key, (long long)txt(m, f.key, -1), (long long)f.want);
            return RAD_E_UNSUPPORTED;
        }

    RAD_ARCH_TRY(parse_layer_kinds("hyv4flash", meta_text_gets(m, "layer_types", ""),
                                   { "sliding_attention" }, "deepseek_sparse_attention",
                                   &s.is_sliding));
    RAD_ARCH_TRY(parse_layer_kinds("hyv4flash", meta_text_gets(m, "mlp_layer_types", ""),
                                   { "sparse" }, "dense", &s.is_moe));
    if ((int64_t)s.is_sliding.size() != s.n_layer || (int64_t)s.is_moe.size() != s.n_layer)
        return refuse("layer_types and mlp_layer_types must name every layer");
    bool any_sparse = false, any_moe = false;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        any_sparse = any_sparse || !s.is_sliding[(size_t)l];
        any_moe = any_moe || s.is_moe[(size_t)l];
    }
    if (any_sparse && (s.ix_heads < 1 || s.ix_dim < 1 || s.ix_pool < 1 || s.ix_group_topk < 2))
        return refuse("the config has sparse layers and lacks index_n_heads, index_head_dim, "
                      "compressed_indexer_compress_ratio or compressed_indexer_group_topk");
    bool any_sliding = false;
    for (int64_t l = 0; l < s.n_layer; ++l) any_sliding = any_sliding || s.is_sliding[(size_t)l];
    if (any_sliding && s.window < 1) return refuse("the config has sliding layers and no sliding_window");
    if (any_moe && (s.n_expert < 1 || s.n_ff_exp < 1 || s.latent < 1 || s.n_ff_shared < 1))
        return refuse("the config has MoE layers and lacks n_routed_experts, "
                      "moe_intermediate_size, routed_expert_hidden_size or n_shared_experts");

    /* Engram. `engram_layer_ids` are one above the layer the memory sits in front of (the
     * reference builds layer l's from id l + 1), and each id's head moduli are a config row:
     * (max_ngram - 1) orders of n_head_per_ngram primes, whose sum is that layer's table. */
    s.engram_ngram = txt(m, "engram_max_ngram_size", 0);
    const int64_t per = txt(m, "engram_n_head_per_ngram", 0);
    const int64_t embed = txt(m, "engram_n_embed_per_ngram", 0);
    int ids[64];
    const int n_ids = parse_layer_ids(meta_text_gets(m, "engram_layer_ids", ""), s.n_layer + 1,
                                      ids, 64);
    if (n_ids > 0) {
        if (s.engram_ngram < 2 || per < 1 || embed < per || embed % per != 0)
            return refuse("the config has engram_layer_ids and lacks engram_max_ngram_size, "
                          "engram_n_head_per_ngram or an engram_n_embed_per_ngram they divide");
        s.engram_heads = (s.engram_ngram - 1) * per;
        s.engram_head_dim = embed / per;
    }
    for (int i = 0; i < n_ids; ++i) {
        if (ids[i] < 1) return refuse("engram_layer_ids names layer id 0, which has no layer");
        char key[96];
        std::snprintf(key, sizeof key, "engram_layer_head_vocab_sizes.%d", ids[i]);
        int64_t rows = 0, heads = 0;
        for_each_token(meta_text_gets(m, key, ""), [&](const char* t, size_t) {
            rows += std::strtoll(t, nullptr, 10);
            ++heads;
            return true;
        });
        if (heads != s.engram_heads || rows < 1) {
            fprintf(stderr, "radiance: hyv4flash: %s names %lld head moduli; an Engram layer has "
                            "%lld\n", key, (long long)heads, (long long)s.engram_heads);
            return RAD_E_UNSUPPORTED;
        }
        s.engram_layers.push_back(ids[i] - 1);
        s.engram_rows.push_back(rows);
    }

    s.dfly_layers = top(m, "dfly_num_hidden_layers", 0);
    s.v_layers = top(m, "vision_config.num_hidden_layers", 0);
    s.v_embd = top(m, "vision_config.hidden_size", 0);
    s.v_ff = top(m, "vision_config.intermediate_size", 0);
    s.v_patch = top(m, "vision_config.patch_size", 0);
    s.v_chan = top(m, "vision_config.num_channels", 3);
    s.v_merge = top(m, "vision_config.spatial_merge_size", 0);
    s.v_pos = top(m, "vision_config.max_vit_seq_len", 0) + 1;
    s.v_heads = top(m, "vision_config.num_attention_heads", 0);
    /* HF reads `layer_norm_eps` through attribute_map as `rms_norm_eps`: one eps for the blocks'
     * LayerNorms and the merger's RMSNorm. */
    s.v_eps = rad_meta_getf(m, "vision_config.rms_norm_eps",
                            rad_meta_getf(m, "vision_config.layer_norm_eps", 1e-5));
    /* The position grid's side: max_image_size / patch (HF's position_edge), else the table's own
     * (max_vit_seq_len = side^2, a class row on top). */
    s.v_side = s.v_patch > 0 ? top(m, "vision_config.max_image_size", 0) / s.v_patch : 0;
    if (s.v_side <= 0 && s.v_pos > 1) {
        const int64_t r = (int64_t)std::llround(std::sqrt((double)(s.v_pos - 1)));
        if (r * r == s.v_pos - 1) s.v_side = r;
    }
    s.mrope_sliding = meta_text_gets(m, "rope_parameters.sliding_attention.mrope_section", "");
    s.mrope_sparse = meta_text_gets(
        m, "rope_parameters.deepseek_sparse_attention.mrope_section",
        meta_text_gets(m, "rope_parameters.indexed_attention.mrope_section", ""));
    return RAD_OK;
}

/* ------------------------------------------------------------------ declaring one weight */
/* The manifest's declarations, for what this build declares as the checkpoint holds it (DFly2,
 * whose ops then read these handles, and the tower):
 * every weight at the encoding it is stored in, so rad-convert writes the
 * whole release. The served graph's weights are declared by the blocks that read them. */
struct Decl {
    RadBuilder* b;
    Model&      m;
    int         layer;   /* the movement group; -1 for a model-level weight */
    int         access;
    /* Where a single-plane weight's handle is recorded, by declared name (the drafter's ops read
     * them); null records nothing. */
    std::map<std::string, rad_weight>* rec = nullptr;
    /* The ranks a RAD_SHARD_ROW / RAD_SHARD_COL weight splits over (the drafter at TP,
     * kernels/dfly2.md §3.6). At one rank every weight is declared whole and unsharded, as the
     * container holds it. */
    int world = 1;
    /* A drafter linear stored quantised in a form the trunk's `Lin` reads is not declared here: the
     * drafter's own declare takes it over (hyv4flash_dfly.h proj), sliced at TP. Until the drafter
     * is known to be served, the planes are kept to declare as views after all (`defer`). */
    std::vector<std::function<int()>>* defer = nullptr;

    /* The checkpoint's extents of `name` where the model behind the builder has it, else the
     * config's (`dflt`): a builder with no model (a test's) answers nothing. */
    int64_t extent(const char* name, int dim, int64_t dflt) const {
        RadEncoding e{};
        int64_t shape[RAD_MAX_RANK] = {};
        uint32_t rank = 0;
        if (rad_weight_encoding(b, name, &e, shape, &rank) != RAD_OK || (uint32_t)dim >= rank)
            return dflt;
        return shape[dim];
    }
    /* A matrix: bf16 as the checkpoint ships it, or whatever a recipe made of it. */
    int mat(const char* name, const std::vector<int64_t>& shape, int shard = RAD_SHARD_NONE,
            const std::vector<int64_t>& parts = {}) {
        return put(name, RAD_BF16, shape, grp_layer(layer), 0, shard, parts);
    }
    /* A drafter linear: bf16 as `mat`, or (quantised and read by Lin) left to the drafter. */
    int linear(const char* name, const std::vector<int64_t>& shape, int shard = RAD_SHARD_NONE,
               const std::vector<int64_t>& parts = {}) {
        RadEncoding e{};
        if (defer && m.df.served && weight_enc(b, name, &e) && e.n_planes > 1) {
            if (Dfly::lin_reads(e)) {
                ++m.df.n_quant;
                Decl self = *this;
                defer->push_back([self, name, shape, shard, parts]() mutable {
                    return self.mat(name, shape, shard, parts);
                });
                return RAD_OK;
            }
            char en[256];
            rad_enc_format(&e, en, sizeof en);
            m.df.why_buf = std::string("'") + name + "' is " + en +
                           ", which a draft linear does not read (bf16, int8 i8*bf16[1x128] "
                           "optionally /fwht128, IQ8R, or a form the rotated int8 GEMM decodes)";
            m.df.refuse(m.df.why_buf.c_str());
        }
        return mat(name, shape, shard, parts);
    }
    /* A gain, a bias a norm adds, a hyper-connection table: f32, widened from bf16 at load where
     * the checkpoint has it so (the norms and the HC arithmetic run in f32). */
    int f32(const char* name, const std::vector<int64_t>& shape, int shard = RAD_SHARD_NONE) {
        return put(name, RAD_F32, shape, grp_layer(layer), 0, shard);
    }
    /* One stored weight, at the encoding the model behind the builder holds it in.
     *
     * A plain weight is one declaration, at `want` where that is the stored float or an exact
     * widening of it, else at the stored dtype. A quantised weight is several planes (codes, a
     * scale or two, a codebook, a zero point), and only a kernel's layout can join planes into one
     * operand; with no op to name that kernel, each plane is declared as its own view, at its own
     * dtype and extents, the codes under the weight's name and the others as `<stem>.<role>`
     * (`blk.3.ffn_down_exps.7.scale` beside `.weight`, the spelling rad_block_moe_fp8.h uses).
     * Every view takes the weight's movement group. The block that will serve the weight declares
     * the views its ops take; these only keep the manifest whole.
     *
     * `shape` is the stored (whole) extent. A plain weight declared RAD_SHARD_ROW or _COL at more
     * than one rank takes this rank's share of dim 0 or dim 1 (and of each stacked row part);
     * a quantised one stays whole and unsharded: only the views of a declined drafter's weights come
     * through here; a served drafter's quantised linears are declared by its Lin, sliced (`linear`). */
    int put(const char* name, uint32_t want, const std::vector<int64_t>& shape,
            RadWeightGroup g, int optional = 0, int shard = RAD_SHARD_NONE,
            const std::vector<int64_t>& parts = {}) {
        if (world <= 1) shard = RAD_SHARD_NONE;
        ++m.n_weights;
        RadEncoding e{};
        const bool known = weight_enc(b, name, &e) && e.n_planes > 0;
        if (!known || e.n_planes == 1) {
            uint32_t dt = want;
            if (known) {
                const uint32_t have = e.plane[0].dtype;
                const bool widens = want == RAD_F32 && (have == RAD_BF16 || have == RAD_F16);
                if (!widens) dt = have;
            }
            std::vector<int64_t> dims(shape), rp(parts);
            if (shard == RAD_SHARD_ROW) {
                dims[0] /= world;
                for (int64_t& r : rp) r /= world;
            } else if (shard == RAD_SHARD_COL && dims.size() > 1) {
                dims[1] /= world;
            }
            if (shard != RAD_SHARD_ROW) rp.clear();   /* parts only say how rows split */
            RadWeightDecl wd = weight_decl(dt, {}, access, shard, g, rp.data(), (int)rp.size());
            wd.rank = (uint32_t)dims.size();
            for (size_t i = 0; i < dims.size(); ++i) wd.shape[i] = dims[i];
            wd.optional = optional;
            const rad_weight h = rad_decl_weight(b, name, &wd);
            if (h && rec) (*rec)[name] = h;
            return h ? RAD_OK : RAD_E_INVAL;
        }
        int64_t rows = 1, cols = 1;
        std::vector<int64_t> dims(shape);
        rad_enc_view((uint32_t)dims.size(), dims.data(), &rows, &cols);
        std::string stem(name);
        if (stem.size() > 7 && stem.compare(stem.size() - 7, 7, ".weight") == 0)
            stem.resize(stem.size() - 7);
        for (int k = 0; k < e.n_planes; ++k) {
            const RadEncPlane& pl = e.plane[k];
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&pl, rows, cols, &pr, &pc);
            /* Built from the declared name, so a view carries the scope its weight does. */
            const char* view = k == 0 ? name : m.nm.ckpt("%s.%s", stem.c_str(), pl.role);
            RadWeightDecl d = weight_decl(pl.dtype, { pr, pc }, access, RAD_SHARD_NONE, g, nullptr, 0);
            d.optional = optional;
            d.source = name;
            d.planes = pl.role;
            if (!rad_decl_weight(b, view, &d)) return RAD_E_INVAL;
        }
        return RAD_OK;
    }
};

/* ------------------------------------------------------------------ the drafter's manifest */
/* The drafter's gated MLA, declared as the checkpoint holds it. `scope` is
 * "dfly.blk.<l>", `ap` the checkpoint's `...self_attn`. */
static int attention(Decl& d, Names& nm, const Shape& s, const AttnGeom& a, const char* scope,
                     const char* ap, int* has_post = nullptr) {
    RadBuilder* b = d.b;
    auto n = [&](const char* leaf) { return nm.f("%s.%s", scope, leaf); };
    const char* q_a = n("attn_q_a.weight");
    const char* q_an = n("attn_q_a_norm.weight");
    const char* q_b = n("attn_q_b.weight");
    const char* kv_a = n("attn_kv_a.weight");
    const char* kv_an = n("attn_kv_a_norm.weight");
    const char* kv_b = n("attn_kv_b.weight");
    const char* gate = n("attn_gate.weight");
    const char* out = n("attn_output.weight");
    const char* post = n("attn_post_norm.weight");
    const char* sinks = n("attn_sinks.weight");
    RAD_ARCH_TRY(map_copy(b, q_a, nm.ckpt("%s.q_a_proj.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, q_an, nm.ckpt("%s.q_a_layernorm.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, q_b, nm.ckpt("%s.q_b_proj.weight", ap)));
    /* kv_a's rows are the latent (kv_lora) then the shared rotary key (qk_rope): one projection,
     * split by the reader, as the checkpoint keeps it (`kv_a_proj_with_mqa`). */
    RAD_ARCH_TRY(map_copy(b, kv_a, nm.ckpt("%s.kv_a_proj_with_mqa.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, kv_an, nm.ckpt("%s.kv_a_layernorm.weight", ap)));
    /* kv_b's rows are a head's key (qk_nope) then its value (v), head after head. */
    RAD_ARCH_TRY(map_copy(b, kv_b, nm.ckpt("%s.kv_b_proj.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, gate, nm.ckpt("%s.linear_gate.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, out, nm.ckpt("%s.o_proj.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, post, nm.ckpt("%s.post_proj_norm.weight", ap)));
    RAD_ARCH_TRY(map_copy(b, sinks, nm.ckpt("%s.learnable_sink_param", ap)));
    RAD_ARCH_TRY(d.linear(q_a, { s.q_lora, s.n_embd }));
    RAD_ARCH_TRY(d.f32(q_an, { s.q_lora }));
    /* Whole heads a rank at TP (as the trunk's sparse layers): q_b, kv_b and the gate by rows, the
     * sinks by entry, the output projection by columns; the rest replicated. */
    RAD_ARCH_TRY(d.linear(q_b, { a.q_width(), s.q_lora }, RAD_SHARD_ROW));
    RAD_ARCH_TRY(d.linear(kv_a, { s.kv_lora + a.qk_rope, s.n_embd }));
    RAD_ARCH_TRY(d.f32(kv_an, { s.kv_lora }));
    RAD_ARCH_TRY(d.mat(kv_b, { a.kv_width(), s.kv_lora }, RAD_SHARD_ROW));
    RAD_ARCH_TRY(d.linear(gate, { a.v_width(), s.n_embd }, RAD_SHARD_ROW));
    RAD_ARCH_TRY(d.linear(out, { s.n_embd, a.v_width() }, RAD_SHARD_COL));
    /* post_proj_norm is optional in vLLM's draft (present iff the checkpoint has it): a model
     * that has q_a and lacks it is declared without; a builder with no model declares it. */
    {
        RadEncoding e{};
        const bool model = weight_enc(b, q_a, &e);
        const bool post_on = !model || weight_enc(b, post, &e);
        if (has_post) *has_post = post_on;
        if (post_on) RAD_ARCH_TRY(d.f32(post, { s.n_embd }));
    }
    RAD_ARCH_TRY(d.f32(sinks, { a.heads }, RAD_SHARD_ROW));
    return RAD_OK;
}

/* A gated MLP: gate and up fused, gate rows first, as every block of the tree reads them.
 * `scope` is the declared stem, `mp` the checkpoint's prefix; `shexp` names the shared expert.
 * A width of 0 is read from the checkpoint's down projection, `ff_dflt` where no model answers. */
static int mlp(Decl& d, Names& nm, const Shape& s, const char* scope, const char* mp, int64_t ff,
               bool shexp, int64_t ff_dflt = 0) {
    RadBuilder* b = d.b;
    const char* gu = nm.f(shexp ? "%s.ffn_gate_up_shexp.weight" : "%s.ffn_gate_up.weight", scope);
    const char* dn = nm.f(shexp ? "%s.ffn_down_shexp.weight" : "%s.ffn_down.weight", scope);
    RAD_ARCH_TRY(map_concat(b, gu, { nm.ckpt("%s.gate_proj.weight", mp),
                                     nm.ckpt("%s.up_proj.weight", mp) }));
    RAD_ARCH_TRY(map_copy(b, dn, nm.ckpt("%s.down_proj.weight", mp)));
    if (ff <= 0) ff = d.extent(dn, 1, ff_dflt);
    /* At TP each rank a share of the width: gate_up's rows of each half, down's columns. A width
     * that does not divide is declared whole (drafter() then declines the drafter). */
    const int shard = d.world > 1 && ff % d.world == 0 ? 1 : 0;
    RAD_ARCH_TRY(d.linear(gu, { 2 * ff, s.n_embd }, shard ? RAD_SHARD_ROW : RAD_SHARD_NONE, { ff, ff }));
    RAD_ARCH_TRY(d.linear(dn, { s.n_embd, ff }, shard ? RAD_SHARD_COL : RAD_SHARD_NONE));
    return RAD_OK;
}

/* DFly2, the release's block drafter (`model.dfly_layers`): `dfly_num_hidden_layers` layers of
 * the sparse layers' attention geometry over the target's cache and selection, each with a dense
 * MLP and no hyper-connection, between a fusion of the target's hidden state with the token
 * embedding (`mtp_fusion`), a hidden-state correction and a confidence head. The transformers
 * port ignores these tensors and their forward is in the serving engines, so this declares them
 * as the checkpoint holds them, fused nowhere but an MLP's gate and up. Its groups follow the
 * trunk's layers. */
static int drafter(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    const char* dp = "model.dfly_layers";
    /* Served, the drafter's weights are read every step; declared only, maybe never. The access
     * class is a placement hint and not part of the container (hyv4flash_convert_test). */
    const int access = m.df.served ? RAD_ACCESS_PER_TOKEN : RAD_ACCESS_RARE;
    m.df.post.clear();
    m.df.w.clear();
    m.df.n_quant = 0;
    auto& defer = m.df.defer;
    defer.clear();
    for (int64_t i = 0; i < s.dfly_layers; ++i) {
        const int grp = (int)(s.n_layer + i);
        Decl d{ b, m, grp, access, &m.df.w, s.world, &defer };
        const char* pre = nm.ckpt("%s.%lld", dp, (long long)i);
        const char* scope = nm.f("dfly.blk.%lld", (long long)i);
        const char* an = nm.f("%s.attn_norm.weight", scope);
        const char* fnorm = nm.f("%s.ffn_norm.weight", scope);
        RAD_ARCH_TRY(map_copy(b, an, nm.ckpt("%s.input_layernorm.weight", pre)));
        RAD_ARCH_TRY(map_copy(b, fnorm, nm.ckpt("%s.post_attention_layernorm.weight", pre)));
        RAD_ARCH_TRY(d.f32(an, { s.n_embd }));
        int post_on = 1;
        RAD_ARCH_TRY(attention(d, nm, s, s.sparse, scope, nm.ckpt("%s.self_attn", pre), &post_on));
        m.df.post.push_back(post_on);
        RAD_ARCH_TRY(d.f32(fnorm, { s.n_embd }));
        /* The config does not state the drafter's MLP width; the checkpoint's down projection
         * does (9216, three times the hidden size, in the release). */
        RAD_ARCH_TRY(mlp(d, nm, s, scope, nm.ckpt("%s.mlp", pre), 0, false, 3 * s.n_embd));
        const int64_t ff = d.extent(nm.f("%s.ffn_down.weight", scope), 1, 3 * s.n_embd);
        if (i == 0) m.df.ff = ff;
        if (ff != m.df.ff || ff % s.world)
            m.df.refuse(ff != m.df.ff ? "the draft layers' MLP widths differ"
                                      : "the draft MLP's width does not split evenly over the ranks");
    }
    Decl d{ b, m, (int)(s.n_layer + s.dfly_layers), access, &m.df.w, 1, &defer };
    const int64_t E = s.n_embd;
    /* The correction's three linears split over the ranks as the drafter's own declare reads them
     * (hyv4flash_dfly.h): gate and up by rows, down by columns. A plain weight takes this rank's
     * share here, which the drafter's bf16 gemm_nt reads directly; a quantised one stays whole and
     * the drafter's Lin slices its views. */
    Decl ds{ b, m, (int)(s.n_layer + s.dfly_layers), access, &m.df.w, s.world, &defer };
    const bool chain_split = s.world > 1 && E % s.world == 0;
    const int chain_row = chain_split ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int chain_col = chain_split ? RAD_SHARD_COL : RAD_SHARD_NONE;
    struct { const char* name; const char* src; bool gain; int64_t rows, cols; bool linear; int shard; } one[] = {
        { "dfly.fusion.weight", "mtp_fusion.eh_proj.weight", false, E, 2 * E, true, RAD_SHARD_NONE },
        { "dfly.fusion_embed_norm.weight", "mtp_fusion.enorm.weight", true, E, 0, false, RAD_SHARD_NONE },
        { "dfly.fusion_hidden_norm.weight", "mtp_fusion.hnorm.weight", true, E, 0, false, RAD_SHARD_NONE },
        { "dfly.fusion_gate", "mtp_fusion.gate", true, E, 0, false, RAD_SHARD_NONE },
        { "dfly.correct_embed_norm.weight", "hidden_correction.embed_norm.weight", true, E, 0, false, RAD_SHARD_NONE },
        { "dfly.correct_hidden_norm.weight", "hidden_correction.hidden_norm.weight", true, E, 0, false, RAD_SHARD_NONE },
        { "dfly.correct_gate.weight", "hidden_correction.gate_proj.weight", false, E, 2 * E, true, chain_row },
        { "dfly.correct_up.weight", "hidden_correction.up_proj.weight", false, E, 2 * E, true, chain_row },
        { "dfly.correct_down.weight", "hidden_correction.down_proj.weight", false, E, E, true, chain_col },
        { "dfly.output_norm.weight", "final_norm.weight", true, E, 0, false, RAD_SHARD_NONE },
        { "dfly.confidence.weight", "confidence_head.proj.weight", false, 1, E, false, RAD_SHARD_NONE },
        { "dfly.confidence.bias", "confidence_head.proj.bias", true, 1, 0, false, RAD_SHARD_NONE },
    };
    for (const auto& w : one) {
        const char* name = nm.f("%s", w.name);
        RAD_ARCH_TRY(map_copy(b, name, nm.ckpt("%s.%s", dp, w.src)));
        Decl& dd = w.shard == RAD_SHARD_NONE ? d : ds;
        if (w.gain) RAD_ARCH_TRY(dd.f32(name, { w.rows }));
        else RAD_ARCH_TRY(w.linear ? dd.linear(name, { w.rows, w.cols }, w.shard)
                                   : dd.mat(name, { w.rows, w.cols }, w.shard));
    }
    /* A draft-owned embedding (vLLM loads `dfly_layers.embed_tokens` where the checkpoint has
     * one, else shares the target's). Declared only where the model has it. */
    {
        const char* name = nm.f("dfly.token_embd.weight");
        RAD_ARCH_TRY(map_copy(b, name, nm.ckpt("%s.embed_tokens.weight", dp)));
        RadEncoding e{};
        m.df.own_embed = weight_enc(b, name, &e);
        if (m.df.own_embed) RAD_ARCH_TRY(d.mat(name, { s.n_vocab, s.n_embd }));
    }
    /* A linear left to the drafter's own declare while it looked served: if it is not after all,
     * keep the planes as views, so the manifest stays whole. */
    if (!m.df.served) {
        for (auto& f : defer) RAD_ARCH_TRY(f());
        defer.clear();
    }
    return RAD_OK;
}

/* "25 29 .." -> "25/25, 29/29": each draft layer's cache layer / selection layer. */
static std::string dfly_layers_str(const Dfly& d) {
    std::string r;
    for (size_t i = 0; i < d.kv.size() && i < d.idx.size(); ++i)
        r += (i ? ", " : "") + std::to_string(d.kv[i]) + "/" + std::to_string(d.idx[i]);
    return r;
}

/* ------------------------------------------------------------------ declare */
/* The KV groups, each declared right before the first op that claims its block size (the first
 * op after a group that carries `block_size` binds it, rad_builder.cpp find_kv_consumer). */
static rad_kvgroup latent_group(RadBuilder* b, Names& nm, bool sliding, int64_t window,
                                int64_t row) {
    RadKVGroupDecl d{};
    d.kind = sliding ? RAD_KV_WINDOW : RAD_KV_FULL;
    d.dtype = RAD_BF16;
    d.window = sliding ? window : 0;
    d.n_head_kv = 1;
    d.head_dim = row / 2;
    return rad_decl_kv_group(b, nm.f(sliding ? "kv_latent_sliding" : "kv_latent_sparse"), &d);
}

static int buffers(RadBuilder* b, Model& m) {
    auto& s = m.s;
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, E = s.n_embd;
    auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, shape);
    };
    m.tr.h = bf("streams", { T, s.hc * E });
    m.tr.x = bf("hc_x", { T, E });
    m.tr.post = decl_b(b, nm.f("hc_post"), RAD_F32, { T, s.hc });
    m.tr.xn = bf("block_in", { T, E });
    m.tr.y = bf("block_out", { T, E });
    m.x0 = bf("embed", { T, E });
    /* P-09: the drafter's query pass reads this plane's anchor rows written by an earlier trunk
     * step, so it persists only when the drafter is served; without a drafter every reader
     * (head, taps, unpack, calibration) reads it inside its own step. */
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, E },
                            m.df.served ? RAD_BUF_PERSIST : RAD_BUF_TRANSIENT);
    m.logit_input = bf("logit_input", { g.max_logit_rows, E });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    if (!m.tr.h || !m.tr.x || !m.tr.post || !m.tr.xn || !m.tr.y || !m.x0 || !m.final_hidden ||
        !m.logit_input || !m.logits)
        return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));
    bool any_sparse = false, any_sliding = false, any_moe = false, any_dense = false;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        (s.is_sliding[(size_t)l] ? any_sliding : any_sparse) = true;
        (s.is_moe[(size_t)l] ? any_moe : any_dense) = true;
    }
    /* P-17: with RADIANCE_HY4_PIPE_ATTN at more than one rank, the attention pieces' bounds
     * (PieceCut) are eight more ints behind the rebased bounds: every attention launch that reads a
     * piece's bounds reads w.cu_p already, so the plane's planned life covers them, and no other
     * buffer is added (a PERSIST one moved every activation of the arena by its alignment and cost
     * the switch-off path its speed). */
    const bool cut_on = g.world > 1 && pipe_attn_on();
    m.cu_prefill = decl_b(b, nm.f("cu_prefill"), RAD_I32,
                          { g.max_seqs + 1 + (cut_on ? PieceCut::kSlots : 0) });
    if (cut_on) {
        m.cut.buf = m.cu_prefill;
        m.cut.base = g.max_seqs + 1;
    }
    if (!m.cu_prefill) return RAD_E_INVAL;
    if (any_sparse)
        RAD_ARCH_TRY(m.aw_sparse.declare(b, nm, s, s.sparse_r, T, m.cu_prefill, "sparse"));
    if (any_sliding)
        RAD_ARCH_TRY(m.aw_sliding.declare(b, nm, s, s.sliding_r, T, m.cu_prefill, "sliding"));
    if (any_sparse) RAD_ARCH_TRY(m.iw.declare(b, nm, s, g));
    if (any_moe) RAD_ARCH_TRY(m.mw.declare(b, nm, s, g, m.arm));
    if (any_dense) {
        m.dense_gu = bf("dense_gate_up", { T, 2 * s.n_ff_r });
        /* P-11b: the product plane is declared nowhere here. Where the gate writes the codes
         * itself (GatedMlp::gfuse) no op writes or reads a bf16 product, and a plane in the model's
         * wire is arena held for the whole program; GatedMlp::declare declares it into `dense_ff`
         * once, for every layer, the first time a layer takes the unfused pair. The shared expert's
         * plane of a MoE layer (`m.mw.sff`) works the same way and gets its slot passed the same
         * way -- both are the model's, because a plane in a wire is charged for the whole program. */
        m.dense_out = bf("dense_out", { T, E });
        if (!m.dense_gu || !m.dense_out) return RAD_E_INVAL;
    }
    m.ew.pf = &m.eg_pf;   /* M-18b: the step-start prefetch registry (Engram::declare) */
    if (!s.engram_layers.empty()) RAD_ARCH_TRY(m.ew.declare(b, nm, s, g));
    return RAD_OK;
}

static int trunk_layer(RadBuilder* b, Model& m, int l, MoeFP8::Src* msrc,
                       const RadControlVector* cvec) {
    Names& nm = m.nm;
    const Shape& s = m.s;
    const Geom& g = m.g;
    const int64_t T = g.max_tok, E = s.n_embd;
    Layer& lay = m.layers[(size_t)l];
    lay.sliding = s.is_sliding[(size_t)l] != 0;
    const auto grp = grp_layer(l);
    const char* pre = nm.ckpt("model.layers.%d", l);
    const char* ap = nm.ckpt("%s.self_attn", pre);

    for (size_t i = 0; i < s.engram_layers.size(); ++i)
        if (s.engram_layers[i] == l) {
            lay.engram = (int)i;
            Engram& eg = m.engrams[i];
            RAD_ARCH_TRY(eg.declare(b, nm, s, g, l, s.engram_rows[i], m.ew, m.tr.h, m.kv_engram));
        }
    RAD_ARCH_TRY(lay.t_stream_in.declare(b, nm, T, "stream_in", l, RAD_BF16, s.hc * E, m.tr.h));

    RAD_ARCH_TRY(lay.hc_attn.declare(b, nm, s, T, nm.f("blk.%d.hc_attn", l),
                                     nm.ckpt("%s.hc_attn_layer.hc_pre.hc_fn", pre),
                                     nm.ckpt("%s.hc_attn_layer.hc_pre.hc_base", pre),
                                     nm.ckpt("%s.hc_attn_layer.hc_pre.hc_scale", pre), grp, false,
                                     m.tr.h, m.tr.x, m.tr.post));
    RAD_ARCH_TRY(lay.t_attn_x.declare(b, nm, T, "attn_hc_x", l, RAD_BF16, E, m.tr.x));
    RAD_ARCH_TRY(lay.t_attn_post.declare(b, nm, T, "attn_hc_post", l, RAD_F32, s.hc, m.tr.post));
    lay.attn_gain = decl_f32(b, nm.f("blk.%d.attn_norm.weight", l),
                             nm.ckpt("%s.input_layernorm.weight", pre), { E }, grp);
    if (!lay.attn_gain) return RAD_E_INVAL;
    lay.op_attn_norm = decl_rmsnorm(b, T, E, s.eps, lay.attn_gain, m.tr.x, m.tr.xn);
    HY4_NEED(lay.op_attn_norm, "rmsnorm");
    /* The previous layer's junction into this connection (one with an Engram in front of it
     * declared its tail 0 form behind the previous layer). */
    if (l > 0 && lay.engram < 0) {
        Layer& pv = m.layers[(size_t)l - 1];
        /* G-3b: the junction writes the rotated int8 codes the attention's three input linears read,
         * where their weights take them (q_a's form; each linear checks its own). */
        const Codes::Pair* jp = nullptr;
        if (fuse_on("RADIANCE_HY4_FUSE_JUNCTION_Q") && E % 128 == 0) {
            const CodeForm f = lay.attn.premap_in(b, nm, s, l, ap);
            Codes* cs = codes_of(b);
            if (cs && f.i8 && f.rot && f.g8 == RAD_FP8_BLOCK) jp = cs->get(E, f.g8, "junc_attn_in");
        }
        RAD_ARCH_TRY(pv.j_end.declare(b, s, g, T, m.tr, pv.ffn.junction_a(), pv.ffn.junction_b(),
                                      pv.ffn.post_gain, &lay.hc_attn, lay.attn_gain, m.tr.xn,
                                      pv.ffn.op_join, pv.ffn.op_ar, jp));
    }

    /* The layer's latent group, declared before its first store; a sparse layer's indexer
     * groups before the first pool. */
    rad_kvgroup& kv = lay.sliding ? m.kv_sliding : m.kv_sparse;
    const bool first_kv = !kv;
    const AttnGeom& a = s.local(lay.sliding);
    if (!kv) kv = latent_group(b, nm, lay.sliding, s.window, s.kv_lora + a.qk_rope);
    if (!kv) return RAD_E_INVAL;
    const int64_t block = lay.sliding ? (int64_t)RAD_KV_BLOCK : s.ix_pool;
    RAD_ARCH_TRY(lay.attn.declare(b, nm, s, g, l, lay.sliding,
                                  lay.sliding ? m.aw_sliding : m.aw_sparse, m.tr.xn, m.tr.y, kv,
                                  block, ap, first_kv,
                                  l > 0 && lay.engram < 0 ? m.layers[(size_t)l - 1].j_end.codes_if()
                                                          : nullptr));
    if (!lay.sliding) {
        if (!m.kv_tail) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_LINEAR;
            d.dtype = RAD_BF16;
            d.n_head_kv = 1;
            d.state_dim[0] = m.iw.ring;
            d.state_dim[1] = s.ix_dim;
            m.kv_tail = rad_decl_kv_group(b, nm.f("kv_index_tail"), &d);
            if (!m.kv_tail) return RAD_E_INVAL;
        }
        const bool first_ix = !m.kv_index;
        if (!m.kv_index) {
            /* One pooled key of ix_dim a page of four token slots: four rows of ix_dim / 4,
             * K and V halves of ix_dim / 8 each. */
            RadKVGroupDecl d{};
            d.kind = RAD_KV_FULL;
            d.dtype = RAD_BF16;
            d.n_head_kv = 1;
            d.head_dim = s.ix_dim / 8;
            if (s.ix_fp8) {
                /* dsa_pool's packed page: 128 E4M3 (or int8) codes and the f32 scale, four rows of
                 * 34 bytes (GLM's kv_dsa_index). */
                d.dtype = RAD_U8;
                d.head_dim = 17;
            }
            m.kv_index = rad_decl_kv_group(b, nm.f("kv_index_keys"), &d);
            if (!m.kv_index) return RAD_E_INVAL;
            /* The scorer reads every group of the history every step: resident. */
            RAD_ARCH_TRY(rad_kv_group_dense_read(b, m.kv_index));
        }
        RAD_ARCH_TRY(lay.ix.declare(b, nm, s, g, l, m.iw, m.tr.xn, lay.attn.w.qa, m.kv_index,
                                    m.kv_tail, kv, ap, first_ix));
        RAD_ARCH_TRY(m.df.declare_snap(b, nm, s, g, l, m.iw));
    }
    RAD_ARCH_TRY(lay.attn.declare_core(b, nm, ap, lay.sliding ? 0 : m.iw.sel,
                                       lay.sliding ? 0 : m.iw.len, lay.sliding ? 0 : m.iw.nsel));
    if (l + 1 == s.n_layer) {
        m.pack_h = decl_copy(b, T, s.hc * E, RAD_BF16, m.tr.h, m.tr.h);
        m.pack_post = decl_copy(b, T, s.hc, RAD_F32, m.tr.post, m.tr.post);
        HY4_NEED(m.pack_h && m.pack_post, "cast (last layer streams and post)");
    }
    RAD_ARCH_TRY(lay.hc_attn.declare_post(b, s, T, m.tr.h, m.tr.post, m.tr.y));
    RAD_ARCH_TRY(lay.t_mid.declare(b, nm, T, "resid_mid", l, RAD_BF16, s.hc * E, m.tr.h));

    RAD_ARCH_TRY(lay.hc_ffn.declare(b, nm, s, T, nm.f("blk.%d.hc_ffn", l),
                                    nm.ckpt("%s.hc_mlp_layer.hc_pre.hc_fn", pre),
                                    nm.ckpt("%s.hc_mlp_layer.hc_pre.hc_base", pre),
                                    nm.ckpt("%s.hc_mlp_layer.hc_pre.hc_scale", pre), grp, false,
                                    m.tr.h, m.tr.x, m.tr.post));
    RAD_ARCH_TRY(lay.t_ffn_x.declare(b, nm, T, "ffn_hc_x", l, RAD_BF16, E, m.tr.x));
    RAD_ARCH_TRY(lay.t_ffn_post.declare(b, nm, T, "ffn_hc_post", l, RAD_F32, s.hc, m.tr.post));
    lay.ffn_gain = decl_f32(b, nm.f("blk.%d.ffn_norm.weight", l),
                            nm.ckpt("%s.post_attention_layernorm.weight", pre), { E }, grp);
    if (!lay.ffn_gain) return RAD_E_INVAL;
    lay.op_ffn_norm = decl_rmsnorm(b, T, E, s.eps, lay.ffn_gain, m.tr.x, m.tr.xn);
    HY4_NEED(lay.op_ffn_norm, "rmsnorm");
    /* G-3b: the codes of the feed-forward's input a MoE layer's latent_in (and the shared expert's
     * gate_up) read. */
    const Codes::Pair* jq = nullptr;
    if (fuse_on("RADIANCE_HY4_FUSE_JUNCTION_Q") && E % 128 == 0) {
        const CodeForm f = lay.ffn.premap_in(b, nm, s, l);
        Codes* cs = codes_of(b);
        if (cs && f.i8 && f.rot && f.g8 == RAD_FP8_BLOCK) jq = cs->get(E, f.g8, "junc_ffn_in");
    }
    RAD_ARCH_TRY(lay.j_mid.declare(b, s, g, T, m.tr, lay.attn.w.o, 0, lay.attn.post_gain,
                                   &lay.hc_ffn, lay.ffn_gain, m.tr.xn, 0, lay.attn.op_ar, jq));
    RAD_ARCH_TRY(lay.t_ffn_in.declare(b, nm, T, "ffn_in_norm", l, RAD_BF16, E, m.tr.xn));
    RAD_ARCH_TRY(lay.ffn.declare(b, nm, s, g, l, m.tr.xn, m.tr.y, m.mw, m.dense_gu, &m.dense_ff,
                                 m.dense_out, msrc, lay.j_mid.codes_if(), &m.mw.sff));
    RAD_ARCH_TRY(lay.hc_ffn.declare_post(b, s, T, m.tr.h, m.tr.post, m.tr.y));
    /* An Engram opens the next layer: the junction stops behind the write (tail 0). */
    for (int e : s.engram_layers)
        if (e == l + 1)
            RAD_ARCH_TRY(lay.j_end.declare(b, s, g, T, m.tr, lay.ffn.junction_a(),
                                           lay.ffn.junction_b(), lay.ffn.post_gain, nullptr, 0, 0,
                                           lay.ffn.op_join, lay.ffn.op_ar, nullptr));
    /* The projection reads the streams after this layer's final write, so it is declared after
     * j_end / hc_ffn.declare_post and before the next reader, t_resid. */
    for (int64_t k = 0; k < m.proj_rank; ++k)
        RAD_ARCH_TRY(lay.proj[k].declare(b, nm.f("blk.%lld.proj.%lld", (long long) l, (long long) k),
                                         g, l, cvec, m.proj_on, m.tr.h, s.hc, k));
    /* The stream the projection reads, for deriving a vector from real prompts. Gated by
     * RADIANCE_HY4_CVEC_TAPS, not RADIANCE_HY4_TAPS: the full tap set costs a ~12.4 GiB arena. */
    /* Gate the declare itself, not Tap::declare's `force`: `force` is checked only when taps are
     * off, so passing cvec_taps_on() there declared the cvec buffer under plain
     * RADIANCE_HY4_TAPS=1 as well, adding ~225 MB to an arena that already does not fit. */
    if (cvec_taps_on())
        RAD_ARCH_TRY(lay.t_cvec.declare(b, nm, kCvecTapRows, "cvec", l, RAD_BF16, s.hc * E,
                                        m.tr.h, true));
    RAD_ARCH_TRY(lay.t_resid.declare(b, nm, T, "resid", l, RAD_BF16, s.hc * E, m.tr.h));
    RAD_ARCH_TRY(m.calib.layer(b, g, s, l, m.tr.xn, lay.attn, !lay.sliding, lay.ffn,
                               lay.engram >= 0 ? &m.engrams[(size_t)lay.engram] : nullptr));
    return RAD_OK;
}

/* This rank's share of the model (PLAN §1.5, Shape's comment): whole heads a rank on both
 * attention kinds, the dense MLP's and the shared expert's widths evenly. 1, 2 or 4 ranks. The
 * routed experts' placement follows their format (Ffn::place). */
static int split_ranks(Shape& s, int world, int rank) {
    s.world = world;
    s.rank = rank;
    if (world != 1 && world != 2 && world != 4) {
        fprintf(stderr, "radiance: hyv4flash: --tp %d; the head and width splits are built for 1, "
                        "2 or 4 ranks\n", world);
        return RAD_E_UNSUPPORTED;
    }
    s.sparse_r = s.sparse;
    s.sliding_r = s.sliding;
    s.n_ff_r = s.n_ff;
    s.n_ff_shared_r = s.n_ff_shared;
    s.n_expert_r = s.n_expert;
    s.n_ff_exp_r = s.n_ff_exp;
    bool any_sparse = false, any_sliding = false, any_moe = false, any_dense = false;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        (s.is_sliding[(size_t)l] ? any_sliding : any_sparse) = true;
        (s.is_moe[(size_t)l] ? any_moe : any_dense) = true;
    }
    const struct { bool on; const char* what; int64_t n; } parts[] = {
        { any_sparse, "the sparse layers' attention heads", s.sparse.heads },
        { any_sliding, "the sliding layers' attention heads", s.sliding.heads },
        { any_dense, "the dense MLP's intermediate_size", s.n_ff },
        { any_moe, "the shared expert's width", s.n_ff_shared },
        { true, "the vocabulary (lm_head)", s.n_vocab },
    };
    for (const auto& p : parts)
        if (p.on && p.n % world) {
            fprintf(stderr, "radiance: hyv4flash: %s (%lld) do not split evenly over %d ranks\n",
                    p.what, (long long)p.n, world);
            return RAD_E_UNSUPPORTED;
        }
    s.sparse_r.heads = s.sparse.heads / world;
    s.sliding_r.heads = s.sliding.heads / world;
    s.n_ff_r = s.n_ff / world;
    s.n_ff_shared_r = s.n_ff_shared / world;
    return RAD_OK;
}

/* The prefill_tail derivation's layout (abi/rad_runtime.h, RadBatch::prefill_tail_h; written by
 * core/sched/derive.cpp, d_prefill_tail): [rows, decode_rows, cu[S+1], prefill_cu[P+1],
 * original_row[m], position[m], three rotary planes of m when the step carries rope_pos (an
 * encoder is declared), ckpt_cu[3] when it carries ckpt_cut]. The reader in step() and the
 * capacity declare() gives the buffer both take their offsets from here. */
struct TailLayout {
    int64_t cu, pf_cu, ids, pos, rope, ckpt, end;
    TailLayout(int64_t S, int64_t P, int64_t m, bool rope_planes)
        : cu(2), pf_cu(cu + S + 1), ids(pf_cu + P + 1), pos(ids + m), rope(pos + m),
          ckpt(rope + (rope_planes ? 3 * m : 0)), end(ckpt + 3) {}
};

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS ||
        rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->kv_dtype != RAD_DT_INVALID && ctx->kv_dtype != RAD_BF16)
        return refuse("the latent and indexer caches are bf16; run with --kv-cache-dtype bf16");
    const bool sizing = ctx->shape_probe != 0;
    /* A calibration run records into host squares and files from its declare's sites; a sizing
     * declare of it would pin those again. Its levels are dropped (the engine notes it and lends
     * nothing): a calibration run is not served for speed. */
    if (sizing && ctx->calib_dir && *ctx->calib_dir) return RAD_E_UNSUPPORTED;
    auto& slot = sizing ? probe_model : models[rad_build_slot(ctx)];
    slot = std::make_unique<Model>();
    Model& m = *slot;
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    Shape& s = m.s;
    Geom& g = m.g;
    if (!sizing) RAD_ARCH_TRY(map_cli_to_env(ctx));   /* before the first block reads RADIANCE_CALIB_* */
    RAD_ARCH_TRY(read_shape(s, meta));
    RAD_ARCH_TRY(split_ranks(s, ctx->world_size ? ctx->world_size : 1, ctx->rank));
    s.calib_lo = ctx->calib_layer_lo;
    s.calib_hi = ctx->calib_layer_hi;
    RAD_ARCH_TRY(geom_from(g, meta, ctx, "bf16", RAD_BF16));
    m.df.configure(b, meta, s, ctx);   /* before the trunk: the snapshots follow its indexers */
    /* The producers write bf16 planes only: a bf16 trunk linear reads them, an int8 one quantises
     * its input itself into the rank's code pair (Lin), declared at the first one. */
    g.w_bf16 = true;
    /* Every GEMM of the int8 decode pass chains, so the 128-group rotated quantisers between them
     * run chained too (Geom::chain_q8; gpu-mini int8 decode 0.86 -> 0.75 ms a token at TP1).
     * RADIANCE_HY4_CHAIN_Q8=0 keeps them plain. */
    g.chain_q8 = chain_q8_on();
    m.codes.b = b;
    m.codes.nm = &nm;
    m.codes.rows = g.max_tok;
    const CodesScope codes_scope(&m.codes);
    {
        const char* e = std::getenv("RADIANCE_HY4_PREFILL");
        m.force_absorbed = e && !std::strcmp(e, "absorbed");
        /* The indexer's numerics: HF's (the default) or vLLM's FWHT + FP8 (hyv4flash_indexer.h). */
        const char* ix = std::getenv("RADIANCE_HY4_INDEXER");
        if (ix && *ix && std::strcmp(ix, "exact") && std::strcmp(ix, "fp8") &&
            std::strcmp(ix, "int8"))
            return refuse("RADIANCE_HY4_INDEXER is 'exact' (the default), 'fp8' or 'int8'");
        s.ix_i8 = ix && !std::strcmp(ix, "int8");
        s.ix_fp8 = s.ix_i8 || (ix && !std::strcmp(ix, "fp8"));   /* the packed-page wiring */
        if (s.ix_fp8 && s.ix_dim != 128)
            return refuse("RADIANCE_HY4_INDEXER=fp8/int8 needs an indexer head of 128 (the packed page)");
    }

    /* The token embedding and the head. The release ships its own lm_head (`tie_word_embeddings`
     * false); a tied fine-tune reads the embedding. The reference runs the head in f32: the
     * logits are f32 accumulations over the bf16 rows. */
    const char* token = nm.f("token_embd.weight");
    const char* headn = nm.f("output.weight");
    RAD_ARCH_TRY(map_copy(b, token, "model.embed_tokens.weight"));
    RAD_ARCH_TRY(map_copy_alt(b, headn, "lm_head.weight", "model.embed_tokens.weight"));
    if (!is_plain_bf16(b, token))
        return refuse("the token embedding is read by embed_lookup as plain bf16; convert it "
                      "without a recipe rule");
    /* The head is bf16 or what a recipe made of it: declared over all its planes at the codes'
     * dtype, read by the `logits_gemm` whose layout takes that encoding (int8 codes with a bf16
     * scale per [1 x 128], qwen4exp_fp8's head). */
    {
        RadEncoding e{};
        if (weight_enc(b, headn, &e) && !is_plain_bf16(b, headn) && !linear_is_w8(e)) {
            char en[256];
            rad_enc_format(&e, en, sizeof en);
            fprintf(stderr, "radiance: hyv4flash: the lm_head is %s; it is served bf16 or int8 "
                            "(i8*bf16[1x128], unrotated)\n", en);
            return RAD_E_DTYPE;
        }
    }
    /* The embedding is replicated (no reduction behind the lookup); the head is split by
     * vocabulary, each rank's logits [rows, n_vocab / world] (Geom::n_vocab, vocab_off). */
    m.token = decl_w(b, token, RAD_BF16, { s.n_vocab, s.n_embd }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                     grp_model());
    m.head = decl_w(b, headn, weight_codes_dtype(b, headn, RAD_BF16), { g.n_vocab, s.n_embd },
                    RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.final_gain = decl_f32(b, nm.f("output_norm.weight"), "model.norm.weight", { s.n_embd },
                            grp_model());
    if (!m.token || !m.head || !m.final_gain) return RAD_E_INVAL;
    m.n_weights += 3;

    /* The routed arms' format, before any buffer (MoeFP8::probe maps each probed expert). */
    std::vector<MoeFP8::Src> msrc((size_t)s.n_layer);
    bool any_quant = false, calib = false;
    for (int l = 0; l < (int)s.n_layer; ++l)
        if (s.is_moe[(size_t)l]) {
            RAD_ARCH_TRY(Ffn::probe(b, nm, s, l, &msrc[(size_t)l]));
            const int arm = Ffn::arm(msrc[(size_t)l]);
            any_quant = any_quant || arm != kArmBf16;
            m.arm = std::max(m.arm, arm);   /* one int8 layer makes the arm int8 */
            calib = calib || msrc[(size_t)l].cfg.calib;
        }
    /* bf16 experts and a calibration run whole a rank, quantised ones a slice of every expert's
     * width (Ffn::place); the probe read layer l's first expert this rank maps either way. */
    RAD_ARCH_TRY(Ffn::place(s, calib || !any_quant));
    for (int l = 0; l < (int)s.n_layer; ++l)
        if (s.is_moe[(size_t)l]) Ffn::apply(s, msrc[(size_t)l].cfg);

    /* Pictures: the tower is served wherever the checkpoint has one and the caller left room for
     * a pass (--mm-max-patches; rad-convert declares it too, so the container carries it). The
     * trunk then rotates at the step's three-component positions (PLAN D6), which only a program
     * declaring an encoder is handed. */
    const bool vit_weights = s.v_layers > 0 && s.v_embd > 0 && s.v_patch > 0 &&
                             s.v_merge > 0 && ctx->max_enc_patches > 0;
    /* Served when the config states what the tower and the 3-D positions need; otherwise the
     * weights alone are declared (a converted container still carries them), and why is noted. */
    const char* why_not = !vit_weights ? nullptr
        : (s.mrope_sliding.empty() || s.mrope_sparse.empty()) ? "no mrope_section for both layer kinds"
        : s.v_heads <= 0 ? "no vision_config.num_attention_heads"
        : (s.image_token < 0 || s.video_token < 0) ? "no image_token_id / video_token_id"
        : nullptr;
    const bool vit = vit_weights && !why_not;
    s.mrope = vit;
    RAD_ARCH_TRY(buffers(b, m));
    m.proj_on = HcProject::declare_mask(b, nm.f("projection_on"), g, ctx->cvec);
    /* Directions per layer, clamped to the array the Layer holds. */
    if (ctx->cvec && ctx->cvec->dir) {
        m.proj_rank = ctx->cvec->rank > 0 ? ctx->cvec->rank : 1;
        /* A per-stream package carries one direction per stream inside each group, so the number
         * of chained projections is the group count, not the row count. */
        if (ctx->cvec->per_stream) {
            if (m.proj_rank % s.hc != 0) {
                fprintf(stderr, "radiance: a per-stream control vector of %lld rows a layer does "
                                "not split into this model's %lld streams\n",
                        (long long) m.proj_rank, (long long) s.hc);
                return RAD_E_INVAL;
            }
            m.proj_rank /= s.hc;
        }
        if (m.proj_rank > kMaxCvecRank) {
            fprintf(stderr, "radiance: control vector declares %lld directions per layer; "
                            "steering the first %lld\n",
                    (long long) m.proj_rank, (long long) kMaxCvecRank);
            m.proj_rank = kMaxCvecRank;
        }
    }
    const int64_t T = g.max_tok, E = s.n_embd;
    m.embed = mk(b, "embed_lookup",
                 { RAD_RANGE("M", 1, T), RAD_INT("n_embd", E), RAD_INT("n_vocab", s.n_vocab),
                   RAD_STR("dtype", "bf16") },
                 { m.token }, {}, { m.x0 });
    if (vit) {
        /* The encoder's rows replace their placeholders' lookups (spec §11). */
        m.mm_rows = decl_b(b, nm.f("mm_rows"), RAD_BF16, { T, E });
        if (!m.mm_rows) return RAD_E_INVAL;
        /* The core's rows (a raw operand) into a buffer the reduction may write. */
        m.op_mm_copy = mk(b, "cast",
                          { RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("from", "bf16"),
                            RAD_STR("to", "bf16") },
                          {}, {}, { m.mm_rows });
        int arc = RAD_OK;
        m.op_mm_reduce = decl_all_reduce(b, g, E, m.mm_rows, &arc);
        RAD_ARCH_TRY(arc);
        m.op_mm_scatter = mk(b, "scatter_rows",
                             { RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("dtype", "bf16") },
                             {}, { m.mm_rows, m.x0 }, { m.x0 });
        HY4_NEED(m.op_mm_copy && m.op_mm_scatter, "the media rows' cast and scatter_rows");
    }
    m.op_stream = decl_copy(b, T, E, RAD_BF16, m.x0, m.tr.h);
    HY4_NEED(m.embed, "embed_lookup");
    HY4_NEED(m.op_stream, "cast");
    /* P-04: one rebase a step for every layer's prefill attention, declared where step() issues it. */
    m.op_rebase = mk(b, "sub_i32", { RAD_RANGE("M", 1, g.max_seqs + 1) }, {}, {}, { m.cu_prefill });
    HY4_NEED(m.op_rebase, "sub_i32");
    /* P-17: the two launches that write the pieces' bounds (buffers() put them behind the rebased
     * bounds), declared where step() issues them, with the switch only, so the sizing declare is the
     * same with the pipeline on and off. A library without the rows leaves the attention whole. */
    if (m.cut.buf) {
        m.op_cut_fill = optional_op(b, mk(b, "fill", { RAD_RANGE("M", 1, 1), RAD_INT("n", PieceCut::kSlots),
                                                      RAD_STR("dtype", "i32") },
                                          {}, {}, { m.cu_prefill }));
        m.op_cut_sub = optional_op(b, mk(b, "sub_i32", { RAD_RANGE("M", 1, 1) }, {}, { m.cu_prefill },
                                         { m.cu_prefill }));
    }
    RAD_ARCH_TRY(m.t_embed.declare(b, nm, T, "embed", -1, RAD_BF16, E, m.x0));

    if (!s.engram_layers.empty()) {
        /* P-06: one CONV state for all four Engram layers -- the last ngram - 1 token ids, i32, the
         * same for each of them, since they hash the same tokens. The first Engram layer binds it
         * and the last one's engram_ids advances it once a step; the group has one bound layer, so
         * a checkpoint or a rollback copies one window per sequence, not four. */
        RadKVGroupDecl d{};
        d.kind = RAD_KV_CONV;
        d.dtype = RAD_I32;
        d.conv_width = s.engram_ngram;
        d.n_head_kv = 1;
        d.head_dim = 1;
        m.kv_engram = rad_decl_kv_group(b, nm.f("kv_engram_tokens"), &d);
        if (!m.kv_engram) return RAD_E_INVAL;
        if (s.engram_pad < 0) return refuse("the config has Engram layers and no engram_pad_id");
    }
    RAD_ARCH_TRY(m.calib.begin(b, nm, g, ctx, s, m.mw.lat.x));
    m.engrams.resize(s.engram_layers.size());
    m.layers.resize((size_t)s.n_layer);
    /* The declare-time half of the pipeline: no taps, no calibration, more than one rank (a
     * reduction to hide). No `T` here, and the row floor is the step's alone (kPipeMinRows): the
     * graph a plugin declares must be the same at every `max_tok`, because the arena sizes its
     * levels by declaring it again at smaller steps -- a floor in the declare dropped them. The
     * step adds its own half, and a library that serves no narrow reduction clears this below. */
    m.pipe_on = pipe_env_on() && g.world > 1 && !taps_on() && calib_dir() == nullptr;
    for (int l = 0; l < (int)s.n_layer; ++l)
        RAD_ARCH_TRY(trunk_layer(b, m, l, &msrc[(size_t)l], ctx->cvec));

    /* The head: the streams' collapse, the final norm, the lm_head over the sampled rows. */
    RAD_ARCH_TRY(m.hc_head.declare(b, nm, s, T, nm.f("hc_head"), "model.hc_head.hc_head_fn",
                                   "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale",
                                   grp_model(), true, m.tr.h, m.tr.x, 0));
    RAD_ARCH_TRY(m.t_hc_head.declare(b, nm, T, "hc_head_out", -1, RAD_BF16, E, m.tr.x));
    m.op_norm = decl_rmsnorm(b, T, E, s.eps, m.final_gain, m.tr.x, m.final_hidden);
    if (m.op_norm && !m.layers.empty()) {
        Layer& lt = m.layers.back();
        RAD_ARCH_TRY(lt.j_end.declare(b, s, g, T, m.tr, lt.ffn.junction_a(), lt.ffn.junction_b(),
                                      lt.ffn.post_gain, &m.hc_head, m.final_gain, m.final_hidden,
                                      lt.ffn.op_join, lt.ffn.op_ar, nullptr));
        /* A library without a narrow reduction leaves the junctions whole: no pipeline. */
        if (!lt.j_end.pipe_ok()) m.pipe_on = false;
    }
    /* Every junction needs its narrow reduction for the pipeline to have one form to issue on the
     * second lane; then name what that lane touches to the planner (Ctx::lane1_ rule 3: a transient
     * there could otherwise share arena bytes with a run of lane 0's that overlaps it in time).
     * Every buffer of every junction (Junction::bufs): the connection buffers are PERSIST already,
     * so this costs the transients -- the routed/shared sums, the attention's output, the G-3b
     * codes the junctions write and the last junction's `final_hidden` (#282: those two were left
     * out, and the last layer's piece wrote `final_hidden` on lane 1 into bytes lane 0's arms were
     * still using). */
    for (const Layer& l : m.layers)
        if (!l.j_mid.pipe_ok() || !l.j_end.pipe_ok()) m.pipe_on = false;
    if (m.pipe_on)
        for (const Layer& l : m.layers)
            for (const Junction* j : { &l.j_mid, &l.j_end })
                for (rad_buf v : j->bufs())
                    if (v) rad_buf_concurrent(b, v);
    /* --prefill-span-pass step is the routed pass this architecture already takes: Ffn::step issues
     * moe.step over the whole batch (batch->n_tok), so a spanned step sorts its routes once and
     * reads each expert's weight once for the step's rows. Say so where the flag is set, and say
     * where the pipeline keeps the pass a piece, so an operator is not left looking for a switch
     * that changes nothing here. */
    if (ctx->span_pass_step)
        rad_note(b, "--prefill-span-pass step: %s", m.pipe_on
                        ? "a spanned step is pipelined here, so the routed pass stays one "
                          "checkpoint interval a piece (the junction overlap beside it is worth "
                          "more than the pass)"
                        : "the routed block already takes the whole step in one pass");
    RAD_ARCH_TRY(m.t_final.declare(b, nm, T, "final_hidden", -1, RAD_BF16, E, m.final_hidden));
    m.unpack_hidden = decl_copy(b, T, E, RAD_BF16, m.final_hidden, m.final_hidden);
    HY4_NEED(m.unpack_hidden, "cast (last layer hidden)");
    m.op_gather = mk(b, "gather_rows",
                     { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", E),
                       RAD_STR("dtype", "bf16") },
                     {}, { m.final_hidden }, { m.logit_input });
    /* P-08: the direct head reads `final_hidden` in place of the gather; the declared read keeps
     * it live at this op now that a step-lifetime plane may be transient (P-09). */
    m.op_head = mk(b, "logits_gemm",
                   { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                     RAD_INT("n_embd", E), RAD_STR("dtype", "bf16") },
                   { m.head }, { m.logit_input, m.final_hidden }, { m.logits });
    HY4_NEED(m.op_norm && m.op_gather && m.op_head, "rmsnorm / gather_rows / logits_gemm");
    RAD_ARCH_TRY(m.calib.output(b, g, s, m.final_hidden));

    /* Every rank computes the whole latent rows (kv_a, its norm and rope are replicated; only the
     * heads split) and pools the whole indexer keys, so the host tier keeps each paged group once
     * a slot (WP16.0c). The sliding window's latent is the same rows. The indexer's raw-key tail
     * (LINEAR) and the Engram history (CONV) travel in the checkpoint store and stay per rank. */
    m.cut.groups[0] = m.kv_sparse;
    m.cut.groups[1] = m.kv_sliding;
    if (m.kv_sparse) RAD_ARCH_TRY(rad_kv_group_replicated(b, m.kv_sparse));
    if (m.kv_sliding) RAD_ARCH_TRY(rad_kv_group_replicated(b, m.kv_sliding));
    if (m.kv_index) RAD_ARCH_TRY(rad_kv_group_replicated(b, m.kv_index));

    /* A container carries what the model can serve, so the drafter and the tower are declared
     * wherever the checkpoint has them and the caller left room (rad-convert's --max-spec 0
     * leaves the drafter out). Neither is served by this build. */
    const bool dfly = s.dfly_layers > 0 && ctx->max_spec > 0;
    if (dfly) RAD_ARCH_TRY(drafter(b, m));
    if (dfly && m.df.served)
        RAD_ARCH_TRY(m.df.declare(b, nm, s, g, m.token, m.head, m.final_hidden, m.kv_sparse, m.iw));
    if (dfly && !m.df.served) {
        /* Declined after the manifest (its declare found a handle it could not read): the linears
         * left to it are declared as their planes' views after all. */
        for (auto& f : m.df.defer) RAD_ARCH_TRY(f());
        m.df.defer.clear();
    }
    if (dfly && !m.df.served && ctx->rank == 0 && !sizing)
        fprintf(stderr, "radiance: hyv4flash: DFly2 is declared and not served: %s; speculation "
                        "falls back to prompt lookup\n", m.df.why.c_str());
    if (vit_weights) {
        RAD_ARCH_TRY(m.vision.declare(b, nm, s, ctx, (int)(s.n_layer + s.dfly_layers + 1), vit));
        if (why_not) rad_note(b, "the vision tower is declared and not served: %s", why_not);
    }

    if (!taps_on() && !m.calib.on()) {
        RadBufDecl d{};
        d.dtype = RAD_I32;
        d.rank = 1;
        /* The largest tail: no decode sequence (P = max_seqs), every logit row kept, a
         * checkpoint cut. The derivation writes rotary planes when the step carries rope_pos,
         * which the core hands only a program that declares an encoder (batch.cpp, mm_on_); the
         * tower declares one only when served (hyv4flash_vision.h: `if (!serve) return`), so
         * s.mrope (served) covers every step whose tail has the planes. */
        d.shape[0] = TailLayout(g.max_seqs, g.max_seqs, g.max_logit_rows, s.mrope).end;
        d.kind = RAD_BUF_DERIVED;
        d.domain = RAD_DOMAIN_DEVICE;
        d.derive = m.df.served ? "prefill_tail_anchors" : "prefill_tail";
        m.tail_meta = rad_decl_buffer(b, nm.f("prefill_tail"), &d);
        if (!m.tail_meta) return RAD_E_INVAL;
    }
    int64_t n_sparse = 0, n_moe = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        n_sparse += !s.is_sliding[(size_t)l];
        n_moe += s.is_moe[(size_t)l] != 0;
    }
    rad_note(b,
             "Hy4-flash (hyv4flash): %lld layers over %lld streams, %lld sparse + %lld sliding "
             "MLA (window %lld, dense to %lld keys), %lld latent MoE of %lld experts top-%lld, "
             "%zu Engram tables (eps %g), DFly2 %s, vision %s; TP%d (rank %d: %lld/%lld sparse "
             "and %lld/%lld sliding heads, experts %s), bf16%s",
             (long long)s.n_layer, (long long)s.hc, (long long)n_sparse,
             (long long)(s.n_layer - n_sparse), (long long)s.window, (long long)s.dense_limit,
             (long long)n_moe, (long long)s.n_expert, (long long)s.top_k, s.engram_layers.size(),
             s.engram_layers.empty() ? 0.0 : engram_eps(),
             !dfly         ? "left out"
             : m.df.served ? nm.f("served (depth %lld, block %lld, %lld layer(s) over trunk layers "
                                  "%s, %s, %s)",
                                  (long long)m.df.depth, (long long)m.df.block,
                                  (long long)m.df.layers, dfly_layers_str(m.df).c_str(),
                                  m.df.coupled
                                      ? "coupled proposals (x~Q, Q the T1 softmax of each chain "
                                        "step's top 8, mtp_couple_propose a step)"
                                      : "greedy proposals",
                                  m.df.n_quant ? nm.f("%d of its linears int8", m.df.n_quant)
                                               : "bf16 linears")
                           : "declared (not served)",
             vit ? "served" : vit_weights ? "declared (not served)" : "left out",
             s.world, s.rank, (long long)s.sparse_r.heads, (long long)s.sparse.heads,
             (long long)s.sliding_r.heads, (long long)s.sliding.heads,
             s.world == 1 ? "all" : s.experts_whole ? "whole per rank" : "sliced per rank",
             taps_on() ? ", debug taps" : "");
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
static StepPlan plan(const Model& m, const RadBatch* batch) {
    StepPlan p;
    p.T = batch->n_tok;
    p.S = batch->n_seq;
    batch_split(batch, &p.D, &p.DT);
    p.P = p.S - p.D;
    /* The longest context after the step, from the host mirrors. */
    int64_t ctx_max = (int64_t)batch->max_ctx_len + batch->max_q_len;
    int64_t zeros = 0;
    if (batch->positions_h) {
        ctx_max = 0;
        for (int64_t t = 0; t < p.T; ++t) {
            ctx_max = std::max<int64_t>(ctx_max, (int64_t)batch->positions_h[t] + 1);
            if (t >= p.DT && batch->positions_h[t] == 0) ++zeros;
        }
    }
    p.select = m.s.dense_limit > 0 && ctx_max > m.s.dense_limit;
    /* Every prefill sequence of the step starts at position 0 (one row at 0 each): its keys are
     * this step's rows, which the non-absorbed path reads. */
    p.nonabs = !m.force_absorbed && batch->positions_h && p.P > 0 && zeros == p.P;
    /* The indexer's live block columns: the reach the batch bounds on the host. */
    const int64_t r = m.s.ix_pool > 0 ? m.s.ix_pool : 4;
    int64_t live = ((int64_t)batch->max_ctx_len + p.T + batch->n_spec + r) / r + 1;
    if (live > m.iw.blocks) live = m.iw.blocks;
    if (m.kv_index)
        if (const auto* ib = kv_batch(batch, m.kv_index))
            if (live > ib->block_table_pitch) live = ib->block_table_pitch;
    p.live = live < 1 ? 1 : live;
    return p;
}

/* P-08: a step whose every row is an output row. The scheduler appends out_ids per entry from that
 * entry's own rows only, so n_out == T forces every entry to contribute all of its rows, in entry
 * order = 0..T-1. With no prefill entries (D == S, DT == T) this is a plain decode/verify step and
 * the gather before the head is an identity copy; the head can read final_hidden directly. */
static bool head_reads_all_rows(const RadBatch* batch, const StepPlan& p) {
    return batch->n_out == p.T && p.D == p.S && p.DT == p.T;
}

/* The compact pass reads its rows [0, DT) as the batch's decode rows: paged() hands them the
 * batch's block table, seqused and decode bounds, and the stable instance. The derivation keeps
 * every decode row (d_prefill_tail); a tail whose kept rows do not start with exactly those runs
 * the full pass instead. The kept rows ascend, so the last decode row in place means all are. */
static bool tail_keeps_decode(const int32_t* meta, const TailLayout& tl, int64_t DT) {
    if (DT == 0) return true;
    return meta[0] >= DT && meta[1] == DT && meta[tl.ids + DT - 1] == DT - 1;
}

/* In-place row moves issue disjoint slices: kernels copy in parallel within a launch.
 * Pack low to high, expand high to low, including when two spans overlap each other. */
static void move_rows(RadCtx* c, rad_op op, rad_buf buf, int64_t width, int64_t src,
                      int64_t dst, int64_t n) {
    if (src == dst) return;
    const int64_t gap = std::abs(dst - src);
    for (int64_t done = 0; done < n;) {
        const int64_t count = std::min(n - done, gap);
        const int64_t at = dst < src ? done : n - done - count;
        RAD_ISSUE_N(c, op, count, brow_slice(buf, src + at, count, width),
                    brow_slice(buf, dst + at, count, width));
        done += count;
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = *models[rad_slot(c)];
    const auto& s = m.s;
    const int64_t T = batch->n_tok, E = s.n_embd;
    /* DFly2's passes: the context pass has nothing to do (the drafter keeps no cache; what it
     * reads the trunk step left in final_hidden and the selection snapshots), the query pass
     * drafts. */
    if (batch->draft_pass) {
        if (!m.df.served) {
            rad_step_fail(c, "hyv4flash: a drafter pass, and DFly2 is not served");
            return;
        }
        if (batch->draft_pass > 0) m.df.query(c, batch);
        return;
    }
    if (batch->enc) {
        m.vision.step(c, batch);
        return;
    }
    if (batch->gen) {
        rad_step_fail(c, "hyv4flash serves the text trunk, its tower and DFly2; generator passes "
                         "are not built");
        return;
    }
    const int64_t nmm = batch->n_mm_rows;
    if (nmm > 0 && (!m.op_mm_scatter || nmm > T)) {
        rad_step_fail(c, "hyv4flash: this step carries media rows and the program declares no "
                         "tower to take them");
        return;
    }
    const StepPlan p = plan(m, batch);
    /* The step half of the pipeline gate: one sequence (the pieces are one chunk's rows), no media
     * rows (their reduction would have to pick a lane too), and both pieces above the band floor so
     * each one's kernels are the ones the whole step's shape would have picked. */
    const int64_t n0 = pipe_rows(T), n1 = T - n0;
    const bool pipe = pipe_cut_ok(n0, n1) && m.pipe_on && nmm == 0 && batch->n_seq == 1;
    RAD_ISSUE_N(c, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.x0, T));
    if (nmm > 0) {
        /* Rank 0 holds the rows, the others zeros: the exact sum gives every rank the rows. */
        RAD_ISSUE_N(c, m.op_mm_copy, nmm, praw2(batch->mm_embd, RAD_BF16, nmm, E),
                    brows(m.mm_rows, nmm));
        issue_all_reduce(c, m.op_mm_reduce, m.mm_rows, nmm, E);
        RAD_ISSUE_N(c, m.op_mm_scatter, nmm, brows(m.mm_rows, nmm),
                    praw(batch->mm_rows, RAD_I32, nmm), brows(m.x0, T));
    }
    m.t_embed.step(c, brows(m.x0, T), T);
    for (int64_t j = 0; j < s.hc; ++j)
        RAD_ISSUE_N(c, m.op_stream, T, brows(m.x0, T), bcol(m.tr.h, j * E, E, T));

    /* P-04: the prefill sequences' bounds rebased to 0, once for every layer (both geometries'
     * absorbed prefill instance and the non-absorbed attention read m.cu_prefill). */
    if (p.P > 0 && p.T > p.DT) {
        const int32_t* cu = batch->cu_seqlens + p.D;
        RAD_ISSUE_N(c, m.op_rebase, p.P + 1, praw(cu, RAD_I32, p.P + 1), praw(cu, RAD_I32, 1),
                    brows(m.cu_prefill, p.P + 1));
    }

    /* P-17: the attention pieces' bounds (PieceCut), once a step on a step the pipeline cuts, all on
     * lane 0 ahead of the layers: zeros, n0 = positions[n0] - positions[0] (the cut lies inside one
     * sequence's consecutive positions), n1 = T - n0 from the rebased bounds, and per cache group its
     * seqused less n1. Device values, because one program serves every step height. */
    PieceCut cut = m.cut;
    const bool attn_pieces = pipe && pipe_attn_on() && m.op_cut_fill && m.op_cut_sub && cut.buf &&
                             p.D == 0 && p.P == 1 && p.T == T && pipe_positions_run(batch, T);
    if (attn_pieces) {
        cut.n0 = n0;
        cut.n1 = n1;
        auto slot = [&](int64_t k) { return brow_slice(cut.buf, cut.base + k, 1, 1); };
        RAD_ISSUE_N(c, m.op_cut_fill, 1, brow_slice(cut.buf, cut.base, PieceCut::kSlots, 1));
        RAD_ISSUE_N(c, m.op_cut_sub, 1, praw(batch->positions + n0, RAD_I32, 1),
                    praw(batch->positions, RAD_I32, 1), slot(1));
        RAD_ISSUE_N(c, m.op_cut_sub, 1, brow_slice(m.cu_prefill, 1, 1, 1), slot(1), slot(3));
        for (int k = 0; k < 2; ++k) {
            const auto* kb = cut.groups[k] ? kv_batch(batch, cut.groups[k]) : nullptr;
            if (kb && kb->seqused)
                RAD_ISSUE_N(c, m.op_cut_sub, 1, praw(kb->seqused, RAD_I32, 1), slot(3), slot(4 + k));
        }
    }

    /* A junction (WP16.2) runs the chain from a block's reduction to the next block's input norm
     * in one launch; the taps behind it read what the chain would have left. */
    RadBatch compact{};
    const RadBatch* queries = batch;
    QueryRows rows{nullptr, T};
    bool entered = false;   /* the previous junction ran this layer's connection and input norm */
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer& l = m.layers[i];
        if (l.engram >= 0) {
            if (!m.engrams[(size_t)l.engram].step(c, batch)) return;
            m.calib.after_engram(c, (int)i, T);
        }
        l.t_stream_in.step(c, brows(m.tr.h, T), T);
        if (!entered) l.hc_attn.pre(c, m.tr.h, m.tr.x, m.tr.post, T);
        l.t_attn_x.step(c, brows(m.tr.x, T), T);
        l.t_attn_post.step(c, brows(m.tr.post, T), T);
        if (!entered)
            RAD_ISSUE_N(c, l.op_attn_norm, T, brows(m.tr.x, T), RAD_W(l.attn_gain), brows(m.tr.xn, T));
        StepPlan lp = p;
        const int32_t* meta = batch->prefill_tail_h;
        const TailLayout tl(p.S, p.P, meta ? meta[0] : 0, batch->rope_pos != nullptr);
        if (i + 1 == m.layers.size() && m.tail_meta && meta && p.P && meta[0] < T &&
            tail_keeps_decode(meta, tl, p.DT)) {
            lp.T = meta[0];
            rows.n = lp.T;
            if (rows.n) {
                lp.DT = meta[1];
                const auto* device = static_cast<const int32_t*>(rad_buf_ptr(c, m.tail_meta));
                rows.ids = meta + tl.ids;
                rows.pf_cu = device + tl.pf_cu;
                compact = *batch;
                compact.n_tok = lp.T;
                compact.n_tok_decode = lp.DT;
                compact.cu_seqlens = device + tl.cu;
                compact.positions = device + tl.pos;
                compact.positions_h = meta + tl.pos;
                /* An interior checkpoint is a single sequence. The retained queries are a
                 * suffix. Only a cut inside that suffix can split its MoE pass. */
                if (batch->ckpt_cut) {
                    const int cut = meta[tl.ckpt + 1];
                    compact.ckpt_cut = cut > 0 && cut < lp.T ? cut : 0;
                    compact.ckpt_cu = compact.ckpt_cut ? device + tl.ckpt : nullptr;
                }
                compact.rope_pos = batch->rope_pos ? device + tl.rope : nullptr;
                queries = &compact;
            }
        }
        if (!l.attn.front(c, batch, queries, rows, lp)) return;
        m.calib.after_front(c, (int)i, l.attn, T);
        /* A pass on a streamed group's ring attends through the selection at every context
         * (hyv4flash_attn.h); every sparse layer of a pass agrees, the view being the pass's. */
        if (!l.sliding && !lp.select && l.attn.ring_bound(c, batch)) lp.select = true;
        /* P-09: on a snapshot layer of a single-span step whose group does not stream (the only
         * other writer of the selection is the streamed group's kv_resolve, rewriting `sel` in
         * place), `qsa_select` writes the selection snapshot's window and the attention reads it
         * from there: the post-copy has nothing left to do. A compact tail pass (multiple spans)
         * and a streamed group keep the copy. */
        const auto* kb0 = l.sliding ? nullptr : kv_batch(batch, l.attn.kv);
        const bool direct = !l.sliding && lp.select && !rows.ids && !(kb0 && kb0->stream) &&
                            m.df.snaps((int)i);
        SelOut so{};
        if (direct) so = m.df.snap_out((int)i);
        if (!l.sliding && !l.ix.step(c, batch, lp, queries, rows, direct ? &so : nullptr)) return;
        if (!rows.n) {
            const auto* kb = kv_batch(batch, l.attn.kv);
            if (!l.j_mid && kb && kb->stream) kv_stage_next(c, l.attn.kv, l.attn.layer);
            return;
        }
        if (!l.sliding) m.df.snap_step(c, (int)i, m.iw, lp.select, rows, direct);
        const bool jm = (bool)l.j_mid;
        const bool je = (bool)l.j_end;
        /* P-17: core(0), then piece 0's junction on lane 1 beside core(1); the pipeline below joins
         * it before piece 1's junction and piece 0's arms (which read what it wrote). */
        const bool split = attn_pieces && pipe && jm && je && !rows.ids &&
                           l.attn.piece_ok(batch, lp, rows, cut);
        if (split) {
            const rad_buf sel = direct ? so.sel : m.iw.sel;
            const int64_t sel0 = direct ? so.first : 0;
            l.attn.core_piece(c, batch, lp, sel, m.iw.len, cut, 0, sel0);
            pipe_issue(c, [&] { l.j_mid.step(c, 0, n0, pipe_lane()); });
            l.attn.core_piece(c, batch, lp, sel, m.iw.len, cut, 1, sel0);
            static bool said = false;
            if (!said) {
                said = true;
                fprintf(stderr, "hyv4flash: pipelined prefill, attention in pieces (%lld + %lld rows)\n",
                        (long long)n0, (long long)n1);
            }
        } else {
            l.attn.core(c, batch, lp, direct ? so.sel : m.iw.sel, m.iw.len,
                        direct ? so.nsel : m.iw.nsel, jm, queries, rows,
                        direct ? so.first : 0, direct ? so.first : 0);
        }
        if (rows.ids)
            rows.spans([&](int64_t src, int64_t dst, int64_t n) {
                move_rows(c, m.pack_h, m.tr.h, s.hc * E, src, dst, n);
                move_rows(c, m.pack_post, m.tr.post, s.hc, src, dst, n);
            });
        const bool pipe_tail = pipe && jm && je;
        auto junction = [&](const Junction& j) {
            if (pipe_tail) {
                pipe_issue(c, [&] { j.step(c, 0, rows.n, pipe_lane()); });
                pipe_join(c);
            } else {
                j.step(c, rows.n);
            }
        };
        if (jm) {
            /* With both junctions split, the pieces below are this junction: running it whole here
             * as well would add the connection twice (and reduce twice). */
            if (!pipe_tail || rows.ids) junction(l.j_mid);
            l.attn.tail_taps(c, rows.n);
        }
        m.calib.after_attn(c, (int)i, rows.n);
        if (!jm) l.hc_attn.post_add(c, m.tr.y, m.tr.post, m.tr.h, rows.n);
        l.t_mid.step(c, brows(m.tr.h, rows.n), rows.n);

        if (!jm) l.hc_ffn.pre(c, m.tr.h, m.tr.x, m.tr.post, rows.n);
        l.t_ffn_x.step(c, brows(m.tr.x, rows.n), rows.n);
        l.t_ffn_post.step(c, brows(m.tr.post, rows.n), rows.n);
        if (!jm)
            RAD_ISSUE_N(c, l.op_ffn_norm, rows.n, brows(m.tr.x, rows.n), RAD_W(l.ffn_gain), brows(m.tr.xn, rows.n));
        l.t_ffn_in.step(c, brows(m.tr.xn, rows.n), rows.n);
        m.calib.after_ffn_norm(c, (int)i, rows.n);
        if (pipe_tail && !rows.ids) {
            /* The pipelined prefill: piece 0's junction exposed, then piece 1's junction beside
             * piece 0's arms, then piece 0's junction beside piece 1's arms. Every piece's chain
             * steps inside its rows (the reduction sums the same rank order element by element),
             * so the result is the serial step's. Every collective rides lane 1, the exposed pieces
             * too: the ranks must agree on one lane (abi/rad_runtime.h), so a piece on lane 0 would
             * end the lane-1 run. */
            if (!pipe_said()) {
                pipe_said() = true;
                fprintf(stderr, "hyv4flash: pipelined prefill, %lld + %lld rows\n", (long long)n0,
                        (long long)n1);
            }
            if (!split) pipe_issue(c, [&] { l.j_mid.step(c, 0, n0, pipe_lane()); });
            pipe_join(c);
            pipe_issue(c, [&] { l.j_mid.step(c, n0, n1, pipe_lane()); });
            l.ffn.body(c, rows.n, 0, n0);
            pipe_join(c);
            pipe_issue(c, [&] { l.j_end.step(c, 0, n0, pipe_lane()); });
            l.ffn.body(c, rows.n, n0, n1);
            l.ffn.route_report(c);
            pipe_join(c);
            pipe_issue(c, [&] { l.j_end.step(c, n0, n1, pipe_lane()); });
            pipe_join(c);
            for (int64_t k = 0; k < m.proj_rank; ++k) l.proj[k].step(c, batch, 0, rows.n);
            l.t_cvec.step(c, brows(m.tr.h, rows.n), rows.n);
            entered = je && l.j_end.tail;
            continue;
        }
        if (!l.ffn.step(c, queries, je)) return;
        if (je) {
            junction(l.j_end);
            l.ffn.tail_taps(c, rows.n);
        }
        m.calib.after_ffn(c, (int)i, rows.n);
        if (!je) l.hc_ffn.post_add(c, m.tr.y, m.tr.post, m.tr.h, rows.n);
        for (int64_t k = 0; k < m.proj_rank; ++k) l.proj[k].step(c, batch, 0, rows.n);
        l.t_cvec.step(c, brows(m.tr.h, rows.n), rows.n);
        l.t_resid.step(c, brows(m.tr.h, rows.n), rows.n);
        entered = je && l.j_end.tail;
    }
    if (!entered) m.hc_head.pre(c, m.tr.h, m.tr.x, 0, rows.n);
    m.t_hc_head.step(c, brows(m.tr.x, rows.n), rows.n);
    if (!entered)
        RAD_ISSUE_N(c, m.op_norm, rows.n, brows(m.tr.x, rows.n), RAD_W(m.final_gain), brows(m.final_hidden, rows.n));
    m.t_final.step(c, brows(m.final_hidden, rows.n), rows.n);
    if (rows.ids) {
        /* The sampler and the next draft pass both address the original trunk rows. */
        for (int64_t end = rows.n; end > 0;) {
            int64_t start = end - 1;
            while (start && rows.ids[start - 1] + 1 == rows.ids[start]) --start;
            move_rows(c, m.unpack_hidden, m.final_hidden, E, start, rows.ids[start], end - start);
            end = start;
        }
    }
    if (batch->n_out > 0) {
        if (head_reads_all_rows(batch, p)) {
            /* P-08: every row outputs, so out_ids is 0..T-1 and the gather would copy
             * final_hidden onto itself; the head reads final_hidden directly. */
            RAD_ISSUE_N(c, m.op_head, batch->n_out, brows(m.final_hidden, T), RAD_W(m.head),
                        brows(m.logits, batch->n_out));
        } else {
            RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.final_hidden, T),
                        praw(batch->out_ids, RAD_I32, batch->n_out),
                        brows(m.logit_input, batch->n_out));
            RAD_ISSUE_N(c, m.op_head, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                        brows(m.logits, batch->n_out));
        }
    }
    m.calib.after_step(c, T);
}

/* Chained launches, with a drafter and without: the decode step is launch-bound and the rows that
 * have a chained form (GEMM, norm, rope, kv_store, add, silu_mul, MoE, gather) chain; the others
 * keep their barrier. Bit-identical: the same kernels launched with the barrier bit (WP16.1a).
 * The sparse latent group can stream (--kv-resident): the attention declares the streamed store
 * and the resolve (hyv4flash_attn.h).
 *
 * No draft depth: DFly2 serves 1 to 5 (hyv4flash_dfly.h), but what a draft pass costs against a
 * verify row is not measured (kernels/dfly2.md §9, R-2), so `--num-speculative-tokens auto` keeps
 * speculation off and a deployment asks for the depth it wants.
 *
 * With DFly2 in the container the depth is fixed in the sense the core reads it for
 * `--draft-suffix`: the block is the verify window (block = max_spec + 1), so a suffix chain is
 * cut to num-speculative-tokens rather than widening the window past the block, which DFly2 would
 * decline (kernels/dfly2.md §8.4). With no depth stated, `auto` and rad-convert are unaffected. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth_fixed = top(meta, "dfly_num_hidden_layers", 0) > 0 ? 1 : 0;
    out->chain_pays = RAD_CHAIN_ALWAYS;
    out->kv_stream_ok = 1;
    /* S-02: whether this step can run the trunk over a layer range and resume the next sub-step
     * from a persistent set. S-02a only wires the report: the step still runs the whole trunk and
     * declares no such set (RadBuildCtx::layer_slices), so it must say 0 and a --chunk-slices
     * request keeps m = 1. S-02b declares the set and reports 1 here when the trunk can stop
     * between two layers (no taps, no calibration window). */
    out->slice_layers_ok = 0;
    /* The gated streams are projected after every block's write (HcProject). */
    out->control_vector_ok = 1;
    /* declare() honours RadBuildCtx::shape_probe (probe_model above): the core sizes the arena's
     * levels, lends their tops to the expert slab and arms the prefill stager. */
    {   /* RADIANCE_HY4_SHAPE_PROBE=0: no sizing declares (an A/B switch, hy4-perf). */
        const char* e = std::getenv("RADIANCE_HY4_SHAPE_PROBE");
        out->shape_probe_ok = (e && *e && std::strcmp(e, "0") == 0) ? 0 : 1;
    }
    return RAD_OK;
}

/* The recording key's own part (WP13.0). The core keys a pass by shapes and the context bound it
 * rounds; what this step issues also depends on exact positions: whether any row is past the dense
 * limit (a selecting step issues the query side and qsa_select on the sparse layers, plan().select)
 * and whether the prefill sequences start at 0 (plan().nonabs). Without this, a verify of 6 rows at
 * context 2042 and one at 2043 share a key, and the second one is refused as issued differently
 * from its recording (or, between audits, replays the dense pass on a selecting step). */
static uint64_t pass_key(RadCtx* c, const RadBatch* batch, const int32_t*) {
    if (!c || !batch) return 0;
    /* Prefill and mixed passes split their KV by the live context, which no key here carries:
     * unrecorded, as they were without this callback. */
    /* A DFly2 pass issues the same ops whatever the context (the table carries the trunk's
     * dense-or-selecting choice on the card); the context pass issues none. */
    if (batch->draft_pass) return batch->draft_pass > 0 ? 9u : 17u;
    if (batch->phase != RAD_PHASE_DECODE) return 0;
    const StepPlan p = plan(*models[rad_slot(c)], batch);
    /* The gather-or-direct head choice (P-08) changes what the step issues; the key separates it. */
    return 1u | (p.select ? 2u : 0u) | (p.nonabs ? 4u : 0u) |
           (head_reads_all_rows(batch, p) ? 8u : 0u);
}
} // namespace hyv4flash

#ifndef RAD_ARCH_NO_EXPORTS
/* The model's state is indexed by slot (rad_build_slot). */
RAD_ARCH_PROBE(hyv4flash)
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(hyv4flash, "hyv4flash", "", "0.2.0",
                "Tencent Hy4-flash: the text trunk at 1, 2 or 4 ranks, pictures and video through its "
                "tower, the weight manifest and checkpoint name map")
RAD_ARCH_CHAT_FORMAT(hyv4flash)
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {
    return hyv4flash::pass_key(c, batch, used_max_h);
}
#endif
