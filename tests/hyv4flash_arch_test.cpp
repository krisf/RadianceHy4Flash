/* Hy4-flash's declared graph and step (arch/hyv4flash), without a GPU or checkpoint: the harness's
 * `mini` geometry (ref/minicfg.py, the convert test's fixture) at one and two ranks, and its
 * `mini4` twin (twice the heads and experts) at one, two and four ranks, bf16. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/hyv4flash/hyv4flash.cpp"

#include <cstdlib>
#include <memory>

using namespace rad_fake;

namespace {

/* The `mini` preset as rad-convert records its config: 3 layers (sparse, sliding, sparse; layer 0
 * dense, 1-2 latent MoE), two streams, an Engram in front of layer 1, dense limit 16. `wide` is
 * `mini4`: 4 sparse and 8 sliding heads and 8 experts, so four ranks hold whole heads. */
MetaKV mini_meta(bool wide = false, int64_t ff_exp = 128, int64_t n_embd = 128,
                 int64_t latent = 0, int64_t top_k = 0, int64_t n_exp = 0) {
    RadModelMeta m{};
    m.arch_id = "hyv4flash";
    m.name = "test-hy4-mini";
    m.quant = "";
    m.n_layers = 3;
    m.n_embd = n_embd;
    m.n_head = 2;
    m.n_head_kv = 2;
    m.head_dim = 8;
    m.n_ff = 256;
    m.n_vocab = 320;
    m.n_ctx_train = 64;
    m.rms_eps = 1e-5f;
    m.rope_theta = 10000.0f;
    m.n_expert = 4;
    m.n_expert_used = 2;
    MetaKV k(m);
    const char* kv[][2] = {
        { "hc_mult", "2" }, { "q_lora_rank", "64" }, { "kv_lora_rank", "32" },
        { "num_attention_heads", "2" }, { "qk_nope_head_dim", "24" }, { "qk_rope_head_dim", "8" },
        { "v_head_dim", "32" }, { "swa_num_attention_heads", "4" },
        { "swa_qk_nope_head_dim", "24" }, { "swa_qk_rope_head_dim", "8" },
        { "swa_v_head_dim", "32" }, { "sliding_window", "8" },
        { "layer_types", "deepseek_sparse_attention sliding_attention deepseek_sparse_attention" },
        { "mlp_layer_types", "dense sparse sparse" }, { "intermediate_size", "256" },
        { "n_routed_experts", "4" }, { "n_shared_experts", "1" }, { "num_experts_per_tok", "2" },
        { "moe_intermediate_size", "128" }, { "routed_expert_hidden_size", "128" },
        { "routed_scaling_factor", "3.998" }, { "swiglu_limit", "10" }, { "norm_topk_prob", "1" },
        { "index_n_heads", "2" }, { "index_head_dim", "16" },
        { "compressed_indexer_compress_ratio", "4" }, { "compressed_indexer_group_topk", "4" },
        { "use_mla", "1" }, { "gated_mla", "1" }, { "learnable_sink", "1" },
        { "sandwich_norm", "1" }, { "enable_ihc", "1" }, { "use_dsa", "1" },
        { "use_compressed_indexer", "1" }, { "attention_bias", "0" },
        { "moe_router_use_sigmoid", "1" }, { "moe_router_enable_expert_bias", "1" },
        { "hc_eps", "1e-06" }, { "hc_magnitude", "2" },
        { "rope_parameters.sliding_attention.rope_theta", "10000" },
        { "rope_parameters.deepseek_sparse_attention.rope_theta", "10000000" },
        { "engram_layer_ids", "2" }, { "engram_max_ngram_size", "3" },
        { "engram_n_head_per_ngram", "5" }, { "engram_n_embed_per_ngram", "40" },
        { "engram_layer_head_vocab_sizes.2", "11 13 17 19 23 29 31 37 41 43" },
        { "engram_pad_id", "2" }, { "engram_lookup_table_size", "300" },
        { "engram_fp8_embedding", "1" }, { "image_token_id", "120120" },
        { "video_token_id", "120124" },
    };
    for (const auto& p : kv) k.set(p[0], p[1]);
    if (wide) {
        k.m.n_head = k.m.n_head_kv = 4;
        k.m.n_expert = 8;
        k.set("num_attention_heads", "4").set("swa_num_attention_heads", "8");
        k.set("n_routed_experts", "8");
    }
    if (ff_exp != 128) k.set("moe_intermediate_size", std::to_string(ff_exp).c_str());
    /* A trunk whose routed MoE is wider than the preset's (K-19's fusion wants a latent a 256-group
     * divides and the sixteen-slot selection), and one with a wider routed arm. */
    if (latent) k.set("routed_expert_hidden_size", std::to_string(latent).c_str());
    if (top_k) k.set("num_experts_per_tok", std::to_string(top_k).c_str());
    if (n_exp) {
        k.m.n_expert = n_exp;
        k.set("n_routed_experts", std::to_string(n_exp).c_str());
    }
    return k;
}

/* A bf16 container: the experts plain (MoeFP8 reads the format off the first expert). */
void bf16_encs(RadBuilder* b) {
    b->encs.clear();
    b->encs.push_back({ "ffn_gate_up_exps", rad_enc_plain(RAD_BF16) });
    b->encs.push_back({ "ffn_down_exps", rad_enc_plain(RAD_BF16) });
    b->encs.push_back({ "blk.1.glm_router.weight", rad_enc_plain(RAD_BF16) });
    b->encs.push_back({ "blk.2.glm_router.weight", rad_enc_plain(RAD_F32) });
    b->kv_block_by_consumer = true;
}

/* The quantised forms of PLAN §4.3 (libquant's encodings, as rad-convert records them). */
RadEncoding enc_i8(bool rot) {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    if (rot) rad_enc_copy_str(e.transform, "fwht128");
    return e;
}
/* The codebook at E4M3 scales a 64 of K under a fixed f32: w4nl64, w5nl64, w6nl64. */
RadEncoding enc_nl64(int bits) {
    RadEncoding e = rad_enc_affine(bits == 4 ? RAD_U4 : bits == 5 ? RAD_U5 : RAD_U6, RAD_F8E4M3,
                                   1, 64);
    rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&e, "table", RAD_F32, 1, 1 << bits);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}
/* The rotated computed trellis under one f32 scale a row (IQR; tcq_code=mad), its run (libquant's
 * tcq_rows: the rows a run of the plane's table) as given. */
RadEncoding enc_trellis(uint32_t codes, bool colf, uint32_t run = 128) {
    RadEncoding e{};
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "trellis");
    rad_enc_add_plane(&e, "codes", codes, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F32, 1, 0);
    const int t = rad_enc_add_table(&e, "trellis", RAD_I8, 1 << 16, 1);
    e.plane[t].block[0] = run;
    rad_enc_add_table(&e, "trellis.mad", RAD_U32, 2, 1);
    if (colf) rad_enc_add_plane(&e, "scale.col", RAD_F32, 0, 128);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* G-3's launch fusions are declared only by the tests that ask (g3_fusions); every other test keeps
 * the unfused chains its assertions count. */
bool g3_fusions = false;
struct G3Env {
    std::vector<const char*> set;
    G3Env() {
        if (g3_fusions) return;
        for (const char* k : { "RADIANCE_HY4_FUSE_SWIGLU", "RADIANCE_HY4_FUSE_EXPAND_QUANT",
                               "RADIANCE_HY4_FUSE_JUNCTION_Q", "RADIANCE_HY4_FUSE_QPROJ",
                               "RADIANCE_HY4_FUSE_QA_GATE" })
            if (!std::getenv(k)) { setenv(k, "0", 1); set.push_back(k); }
    }
    ~G3Env() { for (const char* k : set) unsetenv(k); }
};
struct Declared {
    MetaKV mk;
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    int rc = 0;
    explicit Declared(void (*encs)(RadBuilder*) = bf16_encs, int world = 1, int rank = 0,
                      bool wide = false, int64_t ff_exp = 128, int64_t latent = 0,
                      int64_t top_k = 0, int64_t n_exp = 0)
        : mk(mini_meta(wide, ff_exp, 128, latent, top_k, n_exp)) {
        encs(&b);
        ctx = build_ctx(rank, world);
        G3Env g3env;
        std::string err;
        err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        if (rc != RAD_OK && std::getenv("HY4_TEST_STDERR")) fputs(err.c_str(), stderr);
    }
};

const RecKV* kv_named(const RadBuilder& b, const char* name) {
    for (const auto& k : b.kv_groups)
        if (k.name == name) return &k;
    return nullptr;
}

int64_t block_of(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.kv_groups.size(); ++i)
        if (b.kv_groups[i].name == name) return fake_kv_block(&b, (rad_kvgroup)(i + 1));
    return -1;
}

int count_issued(const RadBuilder& b, const RadCtx& c, const char* op) {
    int n = 0;
    for (const auto& s : issued_ops(b, c)) n += s == op;
    return n;
}

}  // namespace

/* WP16.0c / WP16.1a: the whole-on-every-rank paged groups are declared replicated at every world
 * (the host tier keeps each once a slot), the per-rank state groups are not, and the plugin asks
 * for chained launches. */
TEST(the_replicated_kv_groups_and_the_chain_request) {
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            Declared d(bf16_encs, world, rank);
            REQUIRE_EQ(d.rc, RAD_OK);
            for (const char* n : { "kv_latent_sparse", "kv_latent_sliding", "kv_index_keys" }) {
                const RecKV* k = kv_named(d.b, n);
                REQUIRE(k != nullptr);
                CHECK(k->replicated);
            }
            for (const char* n : { "kv_index_tail", "kv_engram_tokens" }) {
                const RecKV* k = kv_named(d.b, n);
                REQUIRE(k != nullptr);
                CHECK(!k->replicated);
            }
        }
    }
    Declared d;
    RadArchProbe p{};
    CHECK_OK(hyv4flash::probe(d.mk.meta(), &p));
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);
}

/* S-02: the plugin reports whether its step's trunk can be stopped between two layers and resumed
 * from a persistent set. This step still runs the whole trunk and declares no persistent set, so a
 * --chunk-slices request must not slice a chunk on it: the report is 0, and the core keeps m = 1.
 * S-02b declares the set and reports 1 here when the trunk can stop between two layers. */
TEST(the_probe_reports_no_resumable_layer_range_on_the_trunk_step) {
    RadArchProbe p{};
    CHECK_OK(hyv4flash::probe(mini_meta().meta(), &p));
    CHECK_EQ(p.slice_layers_ok, 0);
}

/* Every layer kind's ops, once each where the plan says so: the connections, both MLA forms on
 * both layer kinds (the sparse ones with the selection form), the indexer on the sparse layers, the
 * dense MLP and the latent MoE, the Engram in front of layer 1, the head. */
TEST(the_mini_graph_declares_every_layer_kind) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const RadBuilder& b = d.b;
    CHECK_EQ(count_op(b, "ihc_pre"), 7);                /* 2 a layer and the head's collapse */
    CHECK_EQ(count_op(b, "ihc_post"), 6);
    CHECK_EQ(count_op(b, "mla_project_kv"), 6);         /* absorb and expand a layer */
    CHECK_EQ(count_op(b, "mla_attn_paged_sink"), 6);    /* decode-class (stable) and prefill */
    CHECK_EQ(count_op(b, "mla_attn_sel_sink"), 4);      /* the sparse layers, both instances */
    CHECK_EQ(count_op(b, "attn_kv_sink"), 3);
    CHECK_EQ(count_op(b, "mla_kv_split"), 3);
    CHECK_EQ(count_op(b, "gate_mul"), 3);
    CHECK_EQ(count_op(b, "kv_store"), 3);
    CHECK_EQ(count_op(b, "qsa_work"), 2);
    CHECK_EQ(count_op(b, "idx_pool_gated"), 2);
    CHECK_EQ(count_op(b, "qsa_tail_store"), 2);
    CHECK_EQ(count_op(b, "idx_score"), 2);
    CHECK_EQ(count_op(b, "qsa_select"), 2);
    CHECK_EQ(count_op(b, "engram_ids"), 2);             /* the decode and the prefill layout */
    CHECK_EQ(count_op(b, "embed_lookup_q"), 1);
    CHECK_EQ(count_op(b, "quant_act_fp8"), 1);          /* the bf16 table's E4M3 round trip */
    CHECK_EQ(count_op(b, "dequant"), 1);
    CHECK_EQ(count_op(b, "engram_norm"), 1);
    CHECK_EQ(count_op(b, "engram_gate"), 1);
    CHECK_EQ(count_op(b, "router_sigmoid_topk"), 2);
    CHECK_EQ(count_op(b, "swiglu_clamped"), 2);         /* the routed arms; the shared ones are not */
    CHECK_EQ(count_op(b, "logits_gemm"), 1);
    CHECK_EQ(count_op(b, "cast") > 0, true);
    CHECK(find_b(b, "dbg.resid.0") == nullptr);         /* no taps unless asked */

    /* The sink attention's geometry: a sliding layer's window, the scale 1/sqrt(nope + rope). */
    const RecOp* sw = find_op(b, "mla_attn_paged_sink", 2);
    CHECK_EQ(pint(sw, "heads"), 4);
    CHECK_EQ(pint(sw, "window"), 8);
    CHECK_EQ(pint(sw, "block_size"), 16);
    CHECK(std::fabs(pflt(sw, "scale") - 1.0 / std::sqrt(32.0)) < 1e-12);
    const RecOp* sp = find_op(b, "mla_attn_paged_sink", 0);
    CHECK_EQ(pint(sp, "window"), 0);
    CHECK_EQ(pint(sp, "block_size"), 4);
    /* WP16 3.4.1: the decode-class instance stable, the prefill one fast */
    CHECK_EQ(pint(sp, "stable"), 1);
    CHECK_EQ(pint(find_op(b, "mla_attn_paged_sink", 1), "stable"), 0);
    CHECK_EQ(pint(find_op(b, "mla_attn_sel_sink", 0), "stable"), 1);
    CHECK_EQ(pint(find_op(b, "mla_attn_sel_sink", 1), "stable"), 0);
    const RecOp* sel = find_op(b, "qsa_select");
    CHECK_EQ(pint(sel, "topk"), 3);
    CHECK_EQ(pint(sel, "own"), 1);
    CHECK_EQ(pint(sel, "signed"), 1);
    const RecOp* r = find_op(b, "router_sigmoid_topk");
    CHECK_EQ(pint(r, "top_k"), 2);
    CHECK(std::fabs(pflt(r, "route_scale") - 3.998) < 1e-12);
    CHECK(std::fabs(pflt(find_op(b, "swiglu_clamped"), "limit") - 10.0) < 1e-12);
    const RecOp* ih = find_op(b, "ihc_pre");
    CHECK(std::fabs(pflt(ih, "post_mag") - 2.0) < 1e-12);
    CHECK(std::fabs(pflt(ih, "hc_eps") - 1e-6) < 1e-15);
    /* The Engram eps is the f32 default (DECISIONS U1) unless the switch says bf16. */
    CHECK(std::fabs(pflt(find_op(b, "engram_norm"), "eps") - 1.1920928955078125e-07) < 1e-20);

    /* The caches (PLAN §1.3). */
    const RecKV* swk = kv_named(b, "kv_latent_sliding");
    const RecKV* spk = kv_named(b, "kv_latent_sparse");
    const RecKV* ixk = kv_named(b, "kv_index_keys");
    const RecKV* tl = kv_named(b, "kv_index_tail");
    const RecKV* eg = kv_named(b, "kv_engram_tokens");
    REQUIRE(swk && spk && ixk && tl && eg);
    CHECK_EQ(swk->d.kind, RAD_KV_WINDOW);
    CHECK_EQ(swk->d.window, 8);
    CHECK_EQ(swk->d.head_dim, 20);                      /* (32 + 8) / 2 */
    CHECK_EQ(block_of(b, "kv_latent_sliding"), 16);
    CHECK_EQ(spk->d.kind, RAD_KV_FULL);
    CHECK_EQ(block_of(b, "kv_latent_sparse"), 4);
    CHECK_EQ(block_of(b, "kv_index_keys"), 4);
    CHECK_EQ(ixk->d.head_dim, 2);                       /* a page is one 16-wide key */
    CHECK_EQ(tl->d.kind, RAD_KV_LINEAR);
    CHECK_EQ(tl->d.state_dim[0], 4);
    CHECK_EQ(eg->d.kind, RAD_KV_CONV);
    CHECK_EQ(eg->d.dtype, RAD_I32);

    /* The weights the graph reads under the names it declares them (PLAN D10: the router is
     * GlmRouter's). Both its projection and kv_b are stored bf16. */
    CHECK(find_w(b, "blk.1.glm_router.weight") != nullptr);
    CHECK(find_w(b, "blk.1.glm_router.correction") != nullptr);
    CHECK(find_w(b, "blk.1.ffn_gate_inp.weight") == nullptr);
    CHECK_EQ(find_w(b, "blk.1.glm_router.weight")->d.dtype, RAD_BF16);
    CHECK_EQ(find_w(b, "blk.2.glm_router.weight")->d.dtype, RAD_F32);
    CHECK_EQ(find_w(b, "blk.1.glm_router.correction")->d.dtype, RAD_F32);
    CHECK_EQ(find_w(b, "blk.0.attn_kv_b.weight")->d.dtype, RAD_BF16);
    CHECK(find_w(b, "blk.0.attn_q_nope.weight") != nullptr);
    CHECK(find_w(b, "blk.0.attn_q_rope.weight") != nullptr);
    CHECK(find_w(b, "blk.0.index_query_nope.weight") != nullptr);
    CHECK(find_w(b, "blk.1.index_query_nope.weight") == nullptr);
    CHECK(find_w(b, "blk.1.engram_mult") != nullptr);
    /* The Engram table is model-level, so --ngram-placement keeps it off the cards. */
    CHECK_EQ(find_w(b, "blk.1.engram_table.weight")->d.group.layer, -1);
    /* ...and with it on disk libavx's host gather reads the ids and writes the rows in the host
     * arena (a device gather reads them over the link): host-domain planes, as rad_block_ple.h's.
     * Device buffers there crashed the first prefill. The read-ahead ids are host planes too. */
    for (const char* n : { "eg.ids", "eg.rows", "eg.ids_ahead" }) {
        const RecBuf* eb = find_b(b, n);
        REQUIRE(eb != nullptr);
        CHECK_EQ(eb->d.domain, (int)RAD_DOMAIN_HOST);
    }
    CHECK_EQ(find_b(b, "eg.mask")->d.domain, (int)RAD_DOMAIN_DEVICE);
    /* The prefill hash writes the following prompt's ids, which the gather reads. */
    const RecOp* gat = find_op(b, "embed_lookup_q");
    REQUIRE(gat != nullptr);
    rad_buf ahead_h = 0;
    for (size_t k = 0; k < b.bufs.size(); ++k)
        if (b.bufs[k].name == "eg.ids_ahead") ahead_h = (rad_buf)(k + 1);
    CHECK(in(gat->reads, ahead_h));
    /* q_b's rows per head: nope then rope, a piece a head. */
    const RecMap* qn = find_map(b, "blk.1.attn_q_nope.weight");
    REQUIRE(qn != nullptr);
    CHECK_EQ(qn->src.size(), (size_t)4);
    CHECK_EQ(qn->src_index[1], 32);
}

/* The steps the core builds, each lint clean (liveness by declaration order, spans, KV binds):
 * decode past the dense limit (the selection), a fresh prefill below it (non-absorbed on every
 * layer), a mixed step, a chunk with a cached prefix (absorbed), and the absorbed switch. */
TEST(every_step_kind_is_lint_clean) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    RadBuilder& b = d.b;
    struct Case { const char* what; std::vector<int> dq, pq; int32_t ctx; bool host_pos; };
    const Case cases[] = {
        { "decode", { 1, 1, 1 }, {}, 0, true },
        { "fresh prefill", {}, { 12 }, 0, true },
        { "fresh prefill, long", {}, { 40 }, 0, true },
        { "mixed", { 1 }, { 10 }, 0, true },
        { "mixed, prefix", { 1, 1 }, { 4 }, 8, true },
        { "prefix chunk", {}, { 16 }, 16, true },
        { "no host positions", {}, { 12 }, 0, false },
    };
    for (const Case& cs : cases) {
        auto k = kit(b, cs.dq, cs.pq, cs.ctx);
        if (cs.host_pos) k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(b, &c);
        if (!c.step_fail.empty()) fprintf(stderr, "%s: %s\n", cs.what, c.step_fail.c_str());
        CHECK(c.step_fail.empty());
        const int dense = count_issued(b, c, "attn_kv_sink");
        const int sel = count_issued(b, c, "mla_attn_sel_sink");
        /* P-04: the prefill sequences' bounds are rebased once a step, into the one buffer
         * every layer of both geometries reads; a step with no prefill sequence rebases nothing. */
        CHECK_EQ(count_issued(b, c, "sub_i32"), cs.pq.empty() ? 0 : 1);
        if (!std::strcmp(cs.what, "fresh prefill")) {
            CHECK_EQ(dense, 3);       /* every layer non-absorbed */
            CHECK_EQ(sel, 0);
            CHECK_EQ(count_issued(b, c, "idx_score"), 0);
        } else if (!std::strcmp(cs.what, "fresh prefill, long")) {
            CHECK_EQ(dense, 1);       /* the sliding layer; the sparse ones select */
            CHECK_EQ(sel, 2);
        } else if (!std::strcmp(cs.what, "decode")) {
            CHECK_EQ(dense, 0);
            CHECK_EQ(sel, 2);         /* contexts of 40 and more pass the limit of 16 */
            CHECK_EQ(count_issued(b, c, "mla_attn_paged_sink"), 1);   /* the stable one only */
        } else if (!std::strcmp(cs.what, "mixed, prefix")) {
            /* both instances (WP16 3.4.1): decode rows stable, the absorbed prefill rows fast */
            CHECK_EQ(dense, 0);
            CHECK_EQ(sel, 4);
            CHECK_EQ(count_issued(b, c, "mla_attn_paged_sink"), 2);
        } else if (!std::strcmp(cs.what, "prefix chunk") ||
                   !std::strcmp(cs.what, "no host positions")) {
            CHECK_EQ(dense, 0);       /* a cached prefix (or none known) is absorbed */
        }
    }
    setenv("RADIANCE_HY4_PREFILL", "absorbed", 1);
    Declared a;
    unsetenv("RADIANCE_HY4_PREFILL");
    REQUIRE_EQ(a.rc, RAD_OK);
    auto k = kit(a.b, {}, { 12 }, 0);
    k->b.positions_h = k->pos.data();
    RadCtx c{ .b = &a.b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    CHECK_LINT(a.b, &c);
    CHECK_EQ(count_issued(a.b, c, "attn_kv_sink"), 0);
    CHECK_EQ(count_issued(a.b, c, "sub_i32"), 1);   /* P-04: absorbed on every layer, rebased once */
}

/* P-05 (audit F15): the ahead ids are hashed only where a gather reads them. A prefill step with
 * a following prompt hashes them into eg.ids_ahead and hands them to the gather only when the
 * step's embed_lookup_q dispatches to the host domain (the table on disk, read by libavx's
 * reader thread); a device gather ignores them, so their hashing is skipped. RADIANCE_HY4_ENGRAM_-
 * AHEAD=1 hashes them at every placement (the A/B arm), =0 never. */
TEST(the_ahead_ids_are_hashed_only_for_a_host_gather) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    rad_op gather_h = 0;
    for (size_t i = 0; i < d.b.ops.size() && !gather_h; ++i)
        if (d.b.ops[i].op == "embed_lookup_q") gather_h = (rad_op)(i + 1);
    REQUIRE(gather_h != 0);
    auto k = kit(d.b, {}, { 12 }, 0);
    k->b.positions_h = k->pos.data();
    k->b.n_ahead = 7;
    const int64_t T = k->b.n_tok;
    auto run = [&](bool host, const char* env) {
        if (env) setenv("RADIANCE_HY4_ENGRAM_AHEAD", env, 1);
        else     unsetenv("RADIANCE_HY4_ENGRAM_AHEAD");
        RadCtx c{ .b = &d.b, .batch = &k->b };
        if (host) c.host_ops.push_back(gather_h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(d.b, &c);
        const RecIssue* ids = issue_of(d.b, c, "engram_ids");
        const RecIssue* g = issue_of(d.b, c, "embed_lookup_q");
        REQUIRE(ids != nullptr);
        REQUIRE(g != nullptr);
        const bool forced = env && *env && std::strcmp(env, "0") != 0;
        const bool off = env && std::strcmp(env, "0") == 0;
        const bool on = !off && (forced || host);
        if (on) {
            CHECK_EQ(ids->opd[0].rows, T + 7);          /* the ahead tokens ride the same read */
            CHECK_EQ(g->opd.back().kind, (uint8_t)RAD_OPK_BUF);
            CHECK_EQ(g->opd.back().rows, (int64_t)7);   /* the ahead ids go to the host gather */
        } else {
            CHECK_EQ(ids->opd[0].rows, T);              /* the device gather ignores them */
            CHECK_EQ(g->opd.back().kind, (uint8_t)RAD_OPK_NONE);
        }
    };
    run(false, nullptr);        /* device gather: no ahead hashing */
    run(true, nullptr);         /* host gather (disk-placed table): the read-ahead runs */
    run(false, "1");            /* the A/B arm: hashed for a device gather too */
    run(true, "0");             /* never hashed */
    unsetenv("RADIANCE_HY4_ENGRAM_AHEAD");
}

TEST(last_layer_keeps_cache_writers_at_chunk_height_and_queries_at_output_height) {
    RadBuilder b;
    bf16_encs(&b);
    b.encs.push_back({ "attn_q_a.", enc_i8(true) });
    b.encs.push_back({ "attn_kv_a.", enc_i8(true) });
    auto mk = mini_meta();
    mk.set("kv_lora_rank", "56");  /* kv_a's 64 output channels admit an int8 weight */
    auto ctx = build_ctx();
    G3Env env;
    REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto& last = hyv4flash::models[0]->layers.back();
    REQUIRE(last.attn.kv_a.fed);
    REQUIRE(last.attn.q_a.op_q);
    for (int outputs : {0, 1}) {
        auto k = kit(b, {}, {12}, 10);
        k->b.positions_h = k->pos.data();
        k->b.n_out = outputs;
        int32_t tail[] = {outputs, 0, 0, outputs, 0, outputs, 11, 21};
        k->b.prefill_tail_h = tail;
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        auto height = [&](rad_op op) {
            int64_t total = 0;
            for (const auto& issue : c.issues)
                if (issue.op == op) total += issue.n;
            return total;
        };
        CHECK_EQ(height(last.attn.q_a.op_q), 12);
        CHECK_EQ(height(last.attn.kv_a.op), 12);
        CHECK_EQ(height(last.ix.key.op), 12);
        CHECK_EQ(height(last.ix.op_work), 12);
        CHECK_EQ(height(last.ix.op_tail), 12);
        CHECK_EQ(height(last.attn.q_a.op), outputs);
        CHECK_EQ(height(last.ix.op_select), outputs);
        CHECK_EQ(height(last.attn.op_sel_pf), outputs);
        CHECK_EQ(height(last.attn.out.op), outputs);
        CHECK_EQ(height(hyv4flash::models[0]->op_head), outputs);
    }
}

TEST(compact_dense_prefill_keeps_full_keys_and_restores_gather_indices) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    const auto& last = m.layers.back();
    auto k = kit(d.b, {}, {6, 4});
    k->b.positions_h = k->pos.data();
    int32_t tail[] = {2, 0, 0, 1, 2, 0, 1, 2, 5, 9, 5, 3};
    k->b.prefill_tail_h = tail;
    RadCtx c{ .b = &d.b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    CHECK_LINT(d.b, &c);
    CHECK(c.step_fail.empty());
    int dense = 0;
    std::vector<int64_t> restored;
    for (const auto& issue : c.issues) {
        if (issue.op == last.attn.op_dense) {
            ++dense;
            CHECK_EQ(issue.n, 2);
            CHECK_EQ(issue.opd[0].rows, 2);
            CHECK_EQ(issue.opd[1].rows, 10);
            CHECK_EQ(issue.opd[2].rows, 10);
            CHECK(issue.opd[3].kind != issue.opd[4].kind);
        }
        if (issue.op == m.unpack_hidden)
            restored.push_back(issue.opd[1].offset / m.s.n_embd);
    }
    CHECK_EQ(dense, 1);
    CHECK((restored == std::vector<int64_t>{9, 5}));
}

/* #74 review, point 3 (and the P-04 rebase): the last layer's compact pass reads its query
 * bounds from the tail's prefill_cu, at the offset the derivation writes it (TailLayout), in both
 * the absorbed (paged) and the non-absorbed (dense) prefill. Every other layer, and the keys,
 * read the trunk's once-a-step rebase of the full batch (m.cu_prefill). */
TEST(the_compact_pass_reads_its_query_bounds_from_the_tail) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    const auto& last = m.layers.back();
    auto raw_at = [&](RadCtx& c, int64_t off) {
        return (const void*)((const int32_t*)rad_buf_ptr(&c, m.tail_meta) + off);
    };
    /* Absorbed: one chunk continuing 2 tokens of history, its last 2 rows kept. */
    {
        auto k = kit(d.b, {}, {6}, 2);
        k->b.positions_h = k->pos.data();
        int32_t tail[] = {2, 0, 0, 2, 0, 2, 4, 5, 6, 7};
        k->b.prefill_tail_h = tail;
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(d.b, &c);
        CHECK(c.step_fail.empty());
        const int64_t pf_cu = hyv4flash::TailLayout(1, 1, 2, false).pf_cu;
        CHECK_EQ(pf_cu, 4);
        int last_pf = 0, other_pf = 0;
        for (const auto& issue : c.issues) {
            if (issue.op == last.attn.op_paged_pf) {
                ++last_pf;
                CHECK_EQ(issue.n, 2);
                CHECK(issue.opd[5].raw == raw_at(c, pf_cu));
            } else {
                for (const auto& l : m.layers)
                    if (&l != &last && issue.op == l.attn.op_paged_pf) {
                        ++other_pf;
                        CHECK_EQ((rad_buf)issue.opd[5].handle, m.cu_prefill);
                    }
            }
        }
        CHECK_EQ(last_pf, 1);
        CHECK(other_pf > 0);
    }
    /* Non-absorbed: the dense test's step (two chunks from position 0). */
    {
        auto k = kit(d.b, {}, {6, 4});
        k->b.positions_h = k->pos.data();
        int32_t tail[] = {2, 0, 0, 1, 2, 0, 1, 2, 5, 9, 5, 3};
        k->b.prefill_tail_h = tail;
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        int dense = 0;
        for (const auto& issue : c.issues)
            if (issue.op == last.attn.op_dense) {
                ++dense;
                CHECK(issue.opd[3].raw == raw_at(c, hyv4flash::TailLayout(2, 2, 2, false).pf_cu));
                CHECK_EQ((rad_buf)issue.opd[4].handle, m.cu_prefill);
            }
        CHECK_EQ(dense, 1);
    }
}

/* #74 review, point 2: the compact pass reads its rows [0, DT) as the batch's decode rows, so a
 * tail that does not keep them whole and in place (here decode row 1 dropped) runs the full pass
 * rather than mis-map them; the same step with the decode rows kept compacts. */
TEST(a_tail_without_every_decode_row_in_place_runs_the_full_pass) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    const auto& last = m.layers.back();
    for (bool whole : {true, false}) {
        auto k = kit(d.b, {1, 1}, {6});
        k->b.positions_h = k->pos.data();
        /* [m, DT, cu[4], pf_cu[2], ids[m], pos[m]] */
        std::vector<int32_t> tail = whole
            ? std::vector<int32_t>{3, 2, 0, 1, 2, 3, 0, 1, 0, 1, 7, 40, 47, 5}
            : std::vector<int32_t>{2, 1, 0, 1, 1, 2, 0, 1, 0, 7, 40, 5};
        k->b.prefill_tail_h = tail.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(d.b, &c);
        CHECK(c.step_fail.empty());
        int64_t q_rows = 0, unpacked = 0;
        for (const auto& issue : c.issues) {
            if (issue.op == last.attn.op_q_rope) q_rows += issue.n;
            if (issue.op == m.unpack_hidden) ++unpacked;
        }
        CHECK_EQ(q_rows, whole ? 3 : 8);
        CHECK_EQ(unpacked, whole ? 1 : 0);
    }
}

/* K-26: the decode query's rotary is applied inside the sinks' stable instances (`rope_pos`), so
 * a step with no prefill rows needs no `rope` over its query rows. Where the rotation happens is
 * the whole of it: a decode row rotated in the sink and again by a launch is a different
 * attention, and so is one rotated nowhere, while the rest of the step is identical. So the
 * choice is held here. Decode only: no `rope` launch a layer, and both sinks carry the step's own
 * rotary positions, one a row -- the two sparse layers attend through the selection at this
 * context, the sliding layer through the paged instance. Mixed, or RADIANCE_HY4_FOLD_QROPE=0: the
 * launch runs over every query row the layer keeps (the fast instances and the non-absorbed path
 * read the rotated buffer) and the sinks carry no positions. */
TEST(a_decode_only_step_folds_the_query_rotary_into_the_sinks) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    /* The per-layer ops the choice sits between: the query's rotary launch, and the two stable
     * sinks it can fold into. */
    auto rope_op = [&](rad_op op) {
        for (const auto& l : m.layers)
            if (l.attn.op_q_rope == op) return true;
        return false;
    };
    auto sink_op = [&](rad_op op, bool& through_sel) {
        for (const auto& l : m.layers)
            if (l.attn.op_sel == op || l.attn.op_paged == op) {
                through_sel = l.attn.op_sel == op;
                return true;
            }
        return false;
    };
    struct Got {
        int launches = 0, sinks = 0, positioned = 0, bare = 0, through_sel = 0, through_paged = 0;
        int64_t rows = 0, pos_rows = 0;
    };
    /* fold: "1", "0", or nullptr for the shipped default. */
    auto run = [&](const char* fold, const std::vector<int>& dq, const std::vector<int>& pq) {
        if (fold) setenv("RADIANCE_HY4_FOLD_QROPE", fold, 1);
        auto k = kit(d.b, dq, pq);
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        unsetenv("RADIANCE_HY4_FOLD_QROPE");
        CHECK_LINT(d.b, &c);
        CHECK(c.step_fail.empty());
        CHECK(k->b.phase == (pq.empty() ? RAD_PHASE_DECODE : RAD_PHASE_MIXED));
        Got g;
        for (const auto& issue : c.issues) {
            if (rope_op(issue.op)) {
                ++g.launches;
                g.rows += issue.n;
                continue;
            }
            bool sel_op = false;
            if (!sink_op(issue.op, sel_op)) continue;
            ++g.sinks;
            (sel_op ? g.through_sel : g.through_paged) += 1;
            const RadOperand& rp = issue.opd.back();
            if (rp.kind == RAD_OPK_NONE) {
                ++g.bare;
                CHECK(rp.raw == nullptr);
            } else {
                ++g.positioned;
                g.pos_rows += rp.rows;
                /* the step's own rotary positions, one a row it folds */
                CHECK(rp.raw == (const void*)k->b.positions);
                CHECK_EQ(rp.rows, issue.n);
            }
        }
        return g;
    };

    const Got fold = run(nullptr, { 1, 1 }, {});
    CHECK_EQ(fold.launches, 0);
    CHECK_EQ(fold.sinks, 3);
    CHECK_EQ(fold.through_sel, 2);
    CHECK_EQ(fold.through_paged, 1);
    CHECK_EQ(fold.positioned, 3);
    CHECK_EQ(fold.bare, 0);
    CHECK_EQ(fold.pos_rows, 6);

    const Got off = run("0", { 1, 1 }, {});
    CHECK_EQ(off.launches, 3);
    CHECK_EQ(off.rows, 6);
    CHECK_EQ(off.sinks, 3);
    CHECK_EQ(off.positioned, 0);
    CHECK_EQ(off.bare, 3);

    /* One decode row and four prompt rows: the launch carries all five of them, a layer. */
    const Got mixed = run(nullptr, { 1 }, { 4 });
    CHECK_EQ(mixed.launches, 3);
    CHECK_EQ(mixed.rows, 15);
    CHECK_EQ(mixed.sinks, 3);
    CHECK_EQ(mixed.positioned, 0);
    CHECK_EQ(mixed.bare, 3);
}

/* #74 review, point 1: the tail's capacity counts the three rotary planes exactly when the
 * derivation writes them, which is when the step carries rope_pos: the core hands it to a program
 * that declares an encoder (batch.cpp, mm_on_). Served tower: an encoder, five planes; declared
 * and not served (weights only): no encoder, two; left out: two. */
TEST(the_tail_capacity_counts_the_rotary_planes_the_core_derives) {
    auto check = [](const RadBuilder& b, bool weights) -> int64_t {
        const auto& g = hyv4flash::models[0]->g;
        const RecBuf* tail = find_b(b, "prefill_tail");
        CHECK(tail != nullptr);
        if (!tail) return 0;
        CHECK_EQ(find_w(b, "v.blk.0.attn_qkv.weight") != nullptr, weights);
        const int64_t planes = b.encoder.modalities ? 5 : 2;
        CHECK_EQ(tail->d.shape[0], 7 + 2 * g.max_seqs + planes * g.max_logit_rows);
        return planes;
    };
    for (int state : {0, 1, 2}) {   /* left out, declared and not served, served */
        Declared d;
        if (state) {
            d.mk.set("vision_config.num_hidden_layers", "1").set("vision_config.hidden_size", "32")
                .set("vision_config.intermediate_size", "64").set("vision_config.patch_size", "4")
                .set("vision_config.num_channels", "3").set("vision_config.spatial_merge_size", "2")
                .set("vision_config.num_attention_heads", "2")
                .set("vision_config.max_image_size", "16")
                .set("vision_config.max_vit_seq_len", "16")
                .set("vision_config.rms_norm_eps", "1e-05");
            if (state == 2)
                d.mk.set("rope_parameters.sliding_attention.mrope_section", "2 1 1")
                    .set("rope_parameters.deepseek_sparse_attention.mrope_section", "2 1 1");
            d.b = RadBuilder();
            bf16_encs(&d.b);
            d.ctx = build_ctx();
            d.ctx.max_enc_patches = 64;
            d.rc = hyv4flash::declare(&d.b, d.mk.meta(), &d.ctx);
        }
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK_EQ(check(d.b, state != 0), state == 2 ? 5 : 2);
    }
}

/* WP13.0: a verify step that crosses the dense limit issues other ops (the selection) than one of
 * the same shape just below it, and the core's recording key holds only the rounded context bound.
 * The plugin's pass key must therefore tell the two apart, or the second step of a request that
 * crosses the limit is refused as issued differently from its recording (found on gpu-mini at
 * context 2042, six rows). Prefill and mixed passes stay unkeyed. */
TEST(the_pass_key_tells_a_selecting_step_from_a_dense_one) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    uint64_t keys[3] = {};
    int sel[3] = {};
    const int first[3] = { 8, 11, 12 };           /* rows first..first+5; the dense limit is 16 */
    for (int i = 0; i < 3; ++i) {
        auto k = kit(d.b, { 6 }, {});
        for (int j = 0; j < 6; ++j) k->pos[(size_t)j] = first[i] + j;
        k->b.max_ctx_len = 16;                    /* what the core keys: a rounded bound, equal for all three */
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        keys[i] = hyv4flash::pass_key(&c, &k->b, nullptr);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        sel[i] = count_issued(d.b, c, "mla_attn_sel_sink");
    }
    CHECK_EQ(sel[0], 0);                          /* rows 8..13: dense */
    CHECK_EQ(sel[1], 2);                          /* rows 11..16: the last passes the limit */
    CHECK_EQ(sel[2], 2);
    CHECK(keys[0] != 0);
    CHECK(keys[0] != keys[1]);
    CHECK_EQ(keys[1], keys[2]);
    auto pf = kit(d.b, {}, { 12 });
    pf->b.positions_h = pf->pos.data();
    RadCtx pc{ .b = &d.b, .batch = &pf->b };
    CHECK_EQ(hyv4flash::pass_key(&pc, &pf->b, nullptr), (uint64_t)0);
}

/* P-08: a decode step whose every row is an output row skips the gather and reads the head off
 * final_hidden; the choice changes what the step issues, so the pass key carries it. Two batches
 * with the same choice (one verify row per sequence or per step, whatever the split) share a key;
 * a step with a row that does not output differs from one where every row does. */
TEST(the_pass_key_tells_a_direct_head_step_from_a_gathered_one) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    auto full = kit(d.b, { 6 }, {});
    full->b.positions_h = full->pos.data();
    RadCtx fc{ .b = &d.b, .batch = &full->b };
    const uint64_t k_full = hyv4flash::pass_key(&fc, &full->b, nullptr);
    CHECK(k_full != 0);
    CHECK((k_full & 8u) != 0);
    /* the same choice in another shape (two verifies): the same key */
    auto split = kit(d.b, { 3, 3 }, {});
    split->b.positions_h = split->pos.data();
    RadCtx sc{ .b = &d.b, .batch = &split->b };
    CHECK_EQ(hyv4flash::pass_key(&sc, &split->b, nullptr), k_full);
    /* one row of the step is not an output row: the gather runs, another key */
    auto part = kit(d.b, { 6 }, {});
    part->b.positions_h = part->pos.data();
    part->b.n_out = part->b.n_tok - 1;
    RadCtx pc{ .b = &d.b, .batch = &part->b };
    const uint64_t k_part = hyv4flash::pass_key(&pc, &part->b, nullptr);
    CHECK_EQ(k_full ^ k_part, (uint64_t)8u);      /* the choice bit only */
}

/* P-08: on those steps the head reads final_hidden directly, one launch and the T x E round trip
 * a step gone; a step with a row that does not output (or a prefill entry) keeps the gather. */
TEST(the_head_reads_final_hidden_when_every_row_outputs) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    auto full = kit(d.b, { 6 }, {});
    full->b.positions_h = full->pos.data();
    RadCtx fc{ .b = &d.b, .batch = &full->b };
    hyv4flash::step(&fc, &full->b);
    CHECK(fc.step_fail.empty());
    CHECK_LINT(d.b, &fc);
    CHECK_EQ(count_issued(d.b, fc, "logits_gemm"), 1);
    CHECK_EQ(count_issued(d.b, fc, "gather_rows"), 0);
    /* a decode step with a row that does not output keeps the gather */
    auto part = kit(d.b, { 6 }, {});
    part->b.positions_h = part->pos.data();
    part->b.n_out = part->b.n_tok - 1;
    RadCtx pc{ .b = &d.b, .batch = &part->b };
    hyv4flash::step(&pc, &part->b);
    CHECK(pc.step_fail.empty());
    CHECK_LINT(d.b, &pc);
    CHECK_EQ(count_issued(d.b, pc, "gather_rows"), 1);
    CHECK_EQ(count_issued(d.b, pc, "logits_gemm"), 1);
    /* a mixed step: the prefill rows are not all outputs, the gather stays */
    auto mix = kit(d.b, { 1 }, { 6 });
    mix->b.positions_h = mix->pos.data();
    RadCtx mc{ .b = &d.b, .batch = &mix->b };
    hyv4flash::step(&mc, &mix->b);
    CHECK(mc.step_fail.empty());
    CHECK_LINT(d.b, &mc);
    CHECK_EQ(count_issued(d.b, mc, "gather_rows"), 1);
    CHECK_EQ(count_issued(d.b, mc, "logits_gemm"), 1);
}

/* K-49: the rotaries that turn over a whole prefill chunk are declared under the row-form name.
 * `rope_rows` is the same op with the same parameters -- one thread per (token, pair) looping the
 * heads, so one `sincosf` a (token, pair) instead of one a (token, head, pair) -- byte-identical,
 * and libr4d's launch takes it at 64+ rows and 4+ heads (r4d_model_bf16.hip:1530-1541), so one
 * declaration still runs the per-(head, pair) kernel at decode. What stays on `rope` is the key's
 * rotary: one head a row, where the row form only cuts the thread count. The wide fixture is the
 * one whose head counts reach the launch's bound; the mini's do not, and the declaration is the
 * same either way. */
TEST(the_rotaries_of_a_prefill_chunk_are_declared_under_the_row_form_name) {
    for (const bool wide : { false, true }) {
        Declared d(bf16_encs, 1, 0, wide);
        REQUIRE_EQ(d.rc, RAD_OK);
        const RadBuilder& b = d.b;
        /* The mini program, wide or not: a query rope and a key rope a layer (three layers), and
         * the two sparse layers' indexer query rope. */
        int rows_form = 0, per_pair = 0;
        for (const auto& o : b.ops) {
            if (o.op == "rope_rows") {
                /* The query rotaries: the heads that share one position's angles. */
                CHECK(pint(&o, "n_head") > 1);
                CHECK_EQ(pint(&o, "n_head_kv"), 0);
                CHECK(pflt(&o, "scale") > 0.0);
                ++rows_form;
            } else if (o.op == "rope") {
                CHECK_EQ(pint(&o, "n_head"), 1);       /* the key part's rotary, alone */
                ++per_pair;
            }
        }
        CHECK_EQ(rows_form, 5);
        CHECK_EQ(per_pair, 3);
    }
}

/* The latent cache row's front (WP16.5): where a library declares mla_row_norm_rope_store the step
 * issues it once a layer instead of the row's norm, its rotary and the store; RADIANCE_HY4_CACHE_ROW=0,
 * the debug taps and a library without the op keep the three. Taps need the unfused ops (they
 * read between them). */
TEST(the_cache_row_front_is_one_launch_where_a_library_has_it) {
    auto run = [&](const char* env, bool undeclared, int* fused, int* norm, int* krope, int* store) {
        if (env) setenv("RADIANCE_HY4_CACHE_ROW", env, 1);
        Declared d;
        unsetenv("RADIANCE_HY4_CACHE_ROW");
        REQUIRE_EQ(d.rc, RAD_OK);
        (void)undeclared;
        auto k = kit(d.b, { 1 }, {}, 0);
        k->pos[0] = 5; k->used[0] = 6; k->ctxl[0] = 5;
        k->b.positions_h = k->pos.data();
        k->b.max_ctx_len = 6;
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(d.b, &c);
        CHECK(c.step_fail.empty());
        *fused = count_issued(d.b, c, "mla_row_norm_rope_store");
        *norm = count_issued(d.b, c, "rmsnorm");
        *krope = count_issued(d.b, c, "rope");
        *store = count_issued(d.b, c, "kv_store");
    };
    int f1, n1, r1, s1, f0, n0, r0, s0;
    run(nullptr, false, &f1, &n1, &r1, &s1);
    run("0", false, &f0, &n0, &r0, &s0);
    CHECK_EQ(f0, 0);
    CHECK_EQ(f1, 3);                    /* every layer: the sparse and the sliding */
    CHECK_EQ(s0 - s1, 3);               /* a store a layer folded in */
    CHECK_EQ(n0 - n1, 3);               /* ...and the latent's norm */
    CHECK_EQ(r0 - r1, 3);               /* ...and the key part's rotary (q_rope stays) */
}

/* RADIANCE_HY4_TAPS=1: the harness's buffers (ref/README.md "Tap contract"), PERSIST, and a step
 * that writes them stays lint clean. */
TEST(the_debug_taps_are_persistent_buffers_and_lint_clean) {
    setenv("RADIANCE_HY4_TAPS", "1", 1);
    Declared d;
    unsetenv("RADIANCE_HY4_TAPS");
    REQUIRE_EQ(d.rc, RAD_OK);
    for (const char* n : { "dbg.resid.0", "dbg.resid_mid.2", "dbg.attn_hc_x.1", "dbg.attn_hc_post.0",
                           "dbg.attn_out.1", "dbg.mlp_out.0", "dbg.engram_out.1",
                           "dbg.router_topk_idx.1", "dbg.index_token_keys.0",
                           "dbg.index_group_keys.2", "dbg.index_sel.0", "dbg.index_nsel.2", "dbg.index_score.0",
                           "dbg.embed", "dbg.final_hidden" }) {
        const RecBuf* t = find_b(d.b, n);
        CHECK(t != nullptr);
        if (t) CHECK_EQ(t->d.kind, RAD_BUF_PERSIST);
    }
    CHECK(find_b(d.b, "dbg.engram_out.0") == nullptr);
    CHECK(find_b(d.b, "dbg.index_sel.1") == nullptr);
    CHECK_EQ(find_b(d.b, "dbg.router_topk_idx.1")->d.dtype, RAD_I32);
    for (const auto& pq : { std::vector<int>{ 12 }, std::vector<int>{ 40 } }) {
        auto k = kit(d.b, {}, pq, 0);
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(d.b, &c);
        CHECK(c.step_fail.empty());
    }
}

/* Streamed KV (docs/KV-STREAMING.md, WP14): the sparse latent group streams like GLM's MLA. A pass
 * on the ring stores through kv_store_streamed (mirror and ring), selects even at or below the dense
 * limit, and resolves between qsa_select and mla_attn_sel_sink with qsa_select's entry count; a
 * staged pass stores twice (mirror, then the stage at identity) and never resolves; the sliding
 * window and the indexer's caches never stream. */
TEST(a_streamed_sparse_group_stores_to_its_mirror_and_resolves_between_select_and_attention) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    RadBuilder& b = d.b;
    const auto& m = *hyv4flash::models[0];
    CHECK_EQ(count_op(b, "kv_store_streamed"), 2);   /* the sparse layers only */
    CHECK_EQ(count_op(b, "kv_resolve"), 2);
    const RecOp* rs = find_op(b, "kv_resolve");
    REQUIRE(rs != nullptr);
    CHECK_EQ(pint(rs, "sel_pitch"), 4);              /* the selection row, group_topk */
    CHECK_EQ(pint(rs, "block_bytes"), 320);          /* (32 + 8) bf16 a slot, 4 slots */
    /* The probe states the capability. */
    RadArchProbe pr{};
    REQUIRE_EQ(hyv4flash::probe(d.mk.meta(), &pr), RAD_OK);
    CHECK_EQ(pr.kv_stream_ok, 1);

    const int64_t blocks = 64, slots = 16, bbytes = 320;
    std::vector<uint8_t> host((size_t)(blocks * bbytes * 3));
    std::vector<int32_t> res((size_t)(blocks * 3)), slot_block((size_t)(slots * 3)),
        slot_state((size_t)(slots * 6)), ctl(16);
    RadKVStream st{};
    st.host = host.data();
    st.res = res.data();
    st.slot_block = slot_block.data();
    st.slot_state = slot_state.data();
    st.ctl = ctl.data();
    st.block_bytes = bbytes;
    st.host_layer_stride = bbytes * blocks;
    st.slots = slots;
    st.host_bytes = (int64_t)host.size();
    auto find_issue = [&](const RadCtx& c, rad_op op, size_t from = 0) -> size_t {
        for (size_t i = from; i < c.issues.size(); ++i)
            if (c.issues[i].op == op) return i;
        return SIZE_MAX;
    };
    for (int staged = 0; staged < 2; ++staged) {
        /* Two decode rows at contexts 6 and 9: below the dense limit of 16. */
        auto k = kit(b, { 1, 1 }, {}, 0);
        k->pos[0] = 5; k->pos[1] = 8;     /* in place: the batch points into these */
        k->used[0] = 6; k->used[1] = 9;
        k->ctxl[0] = 5; k->ctxl[1] = 8;
        k->b.positions_h = k->pos.data();
        k->b.max_ctx_len = 9;
        k->kv[(size_t)m.kv_sparse - 1].stream = &st;
        RadCtx c{ .b = &b, .batch = &k->b };
        c.stage_on = staged != 0;
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(b, &c);
        if (!c.step_fail.empty()) fprintf(stderr, "%s\n", c.step_fail.c_str());
        CHECK(c.step_fail.empty());
        for (const auto& l : m.layers) {
            if (l.sliding) {
                /* the window group never streams: its row front is the one launch */
                CHECK(find_issue(c, l.attn.op_store) != SIZE_MAX ||
                      find_issue(c, l.attn.op_row_front) != SIZE_MAX);
                continue;
            }
            const size_t store = find_issue(c, l.attn.op_store_stream);
            const size_t ident = find_issue(c, l.attn.op_store);
            const size_t sel = find_issue(c, l.ix.op_select, store == SIZE_MAX ? 0 : store);
            const size_t res_i = find_issue(c, l.attn.op_resolve);
            const size_t attn = find_issue(c, l.attn.op_sel, store == SIZE_MAX ? 0 : store);
            REQUIRE(store != SIZE_MAX);
            if (staged) {
                /* The stage reads block ids: the identity store follows the mirror's, nothing
                 * resolves, and below the limit the pass attends densely as unstreamed. */
                CHECK(ident != SIZE_MAX && store < ident);
                CHECK_EQ(res_i, SIZE_MAX);
                CHECK_EQ(attn, SIZE_MAX);
                continue;
            }
            CHECK_EQ(ident, SIZE_MAX);
            REQUIRE(sel != SIZE_MAX);
            REQUIRE(res_i != SIZE_MAX);
            REQUIRE(attn != SIZE_MAX);
            CHECK(store < sel);
            CHECK(sel < res_i);
            CHECK(res_i < attn);
            CHECK_EQ(c.issues[res_i].opd[0].handle, m.iw.sel);
            CHECK_EQ(c.issues[res_i].opd[1].handle, m.iw.nsel);   /* the entry count */
            CHECK_EQ(c.issues[res_i].n, (int64_t)2);
        }
    }
}

/* RADIANCE_HY4_INDEXER=fp8 / int8: the key cache becomes the packed page and the score dsa_score
 * over the query codes (E4M3 through dsa_query, int8 through dsa_query_i8); anything else, and an
 * indexer head other than 128, is refused. */
TEST(the_quantised_indexers_swap_the_key_cache_and_the_score_for_the_dsa_forms) {
    auto wide_ix = [](MetaKV& k) { k.set("index_head_dim", "128"); };
    for (const char* mode : { "fp8", "int8" }) {
        setenv("RADIANCE_HY4_INDEXER", mode, 1);
        Declared small;
        CHECK_EQ(small.rc, RAD_E_UNSUPPORTED);              /* mini's indexer head is 16 */
        unsetenv("RADIANCE_HY4_INDEXER");
    }
    for (int m = 0; m < 3; ++m) {   /* exact, fp8, int8 */
        const bool q = m > 0, i8 = m == 2;
        if (q) setenv("RADIANCE_HY4_INDEXER", i8 ? "int8" : "fp8", 1);
        MetaKV mk = mini_meta();
        wide_ix(mk);
        RadBuilder b;
        bf16_encs(&b);
        RadBuildCtx ctx = build_ctx();
        int rc = 0;
        const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        unsetenv("RADIANCE_HY4_INDEXER");
        if (rc != RAD_OK) fputs(err.c_str(), stderr);
        REQUIRE_EQ(rc, RAD_OK);
        CHECK_EQ(count_op(b, "idx_pool_gated_fp8"), m == 1 ? 2 : 0);
        CHECK_EQ(count_op(b, "idx_pool_gated_i8"), i8 ? 2 : 0);
        CHECK_EQ(count_op(b, "idx_pool_gated"), q ? 0 : 2);
        CHECK_EQ(count_op(b, "dsa_query"), m == 1 ? 2 : 0);
        CHECK_EQ(count_op(b, "dsa_query_i8"), i8 ? 2 : 0);
        CHECK_EQ(count_op(b, "dsa_score"), q ? 2 : 0);
        CHECK_EQ(count_op(b, "idx_score"), q ? 0 : 2);
        const RecKV* ixk = kv_named(b, "kv_index_keys");
        REQUIRE(ixk != nullptr);
        CHECK_EQ(ixk->d.dtype, q ? RAD_U8 : RAD_BF16);
        CHECK_EQ(ixk->d.head_dim, q ? 17 : 16);
        CHECK_EQ(block_of(b, "kv_index_keys"), 4);
        if (q) {
            const RecOp* sc = find_op(b, "dsa_score");
            CHECK_EQ(pint(sc, "heads"), 2);
            CHECK_EQ(pint(sc, "n"), 128);
        }
        auto k = kit(b, { 1, 1 }, {}, 0);
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        CHECK_EQ(count_issued(b, c, q ? "dsa_score" : "idx_score"), 2);
        CHECK_EQ(count_issued(b, c, "mla_attn_sel_sink"), 2);
    }
    setenv("RADIANCE_HY4_INDEXER", "bf16", 1);
    Declared bad;
    unsetenv("RADIANCE_HY4_INDEXER");
    CHECK_EQ(bad.rc, RAD_E_UNSUPPORTED);
}

/* WP15: the tower and the 3-D positions. With room for an encoder pass (--mm-max-patches) and a
 * config that states the tower, the M-RoPE sections and the media ids, the program declares the
 * encoder (pictures and video, 4 patches a row, the mini tower's 4 x 4 position grid), every
 * rotary of the trunk becomes imrope_gptj over the layer kind's sections, the indexer's groups
 * rotate at three components ([work, 3], qsa_work's), and a media step scatters the encoder's rows
 * over the lookups. An encoder pass of two segments (4 x 6 and 4 x 4 patches: 12 + 8 rows with
 * their newlines) and the media prefill after it are lint clean. */
TEST(the_tower_and_the_three_component_positions) {
    struct Vis : Declared {
        explicit Vis(bool sections) : Declared(bf16_encs) {
            auto& k = mk;
            k.set("vision_config.num_hidden_layers", "1").set("vision_config.hidden_size", "32")
                .set("vision_config.intermediate_size", "64").set("vision_config.patch_size", "4")
                .set("vision_config.num_channels", "3").set("vision_config.spatial_merge_size", "2")
                .set("vision_config.num_attention_heads", "2")
                .set("vision_config.max_image_size", "16")
                .set("vision_config.max_vit_seq_len", "16")
                .set("vision_config.rms_norm_eps", "1e-05");
            if (sections)
                k.set("rope_parameters.sliding_attention.mrope_section", "2 1 1")
                    .set("rope_parameters.deepseek_sparse_attention.mrope_section", "2 1 1");
            b = RadBuilder();
            bf16_encs(&b);
            ctx = build_ctx();
            ctx.max_enc_patches = 64;
            rc = hyv4flash::declare(&b, mk.meta(), &ctx);
        }
    };
    Vis d(true);
    REQUIRE_EQ(d.rc, RAD_OK);
    RadBuilder& b = d.b;
    REQUIRE(b.encoder.out != 0);
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.patch_dim, 48);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 64);
    CHECK_EQ(count_op(b, "grid_embed_sf"), 1);
    CHECK_EQ(pint(find_op(b, "grid_embed_sf"), "side"), 4);
    CHECK(std::fabs(pflt(find_op(b, "grid_embed_sf"), "offset") - 0.1) < 1e-12);
    CHECK_EQ(count_op(b, "attn_dense"), 1);
    CHECK_EQ(count_op(b, "conv2d_unfold"), 1);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);
    for (const auto& o : b.ops)
        if (o.op == "rope" || o.op == "rope_rows") {   /* both names of one op (K-49) */
            CHECK_EQ(pstr(&o, "mode"), std::string("imrope_gptj"));
            CHECK_EQ(pstr(&o, "sections"), std::string("2 1 1"));
        }
    CHECK_EQ(pstr(find_op(b, "idx_pool_gated"), "mode"), std::string("imrope_gptj"));
    const RecBuf* bp = find_b(b, "ix.bpos");
    REQUIRE(bp != nullptr);
    CHECK_EQ(bp->d.rank, 2u);
    CHECK_EQ(bp->d.shape[1], 3);
    CHECK(find_w(b, "v.blk.0.attn_qkv.weight") != nullptr);
    CHECK_EQ(find_w(b, "v.position.weight")->d.shape[0], 16);

    /* The encoder pass. */
    {
        const int segs[2][2] = { { 4, 6 }, { 4, 4 } };
        const int64_t n = 40;
        std::vector<uint16_t> pix((size_t)(n * 48), 0);
        std::vector<int32_t> coord((size_t)(4 * n), 0), cu = { 0, 24, 40 };
        std::vector<int32_t> sh = { segs[0][0], segs[1][0] }, sw = { segs[0][1], segs[1][1] };
        RadBatch e{};
        e.enc = RAD_MM_IMAGE;
        e.enc_n_patch = n;
        e.enc_pixels = pix.data();
        e.enc_coord = coord.data();
        e.enc_n_seg = 2;
        e.enc_cu = cu.data();
        e.enc_max_seg = 24;
        e.enc_seg_h = sh.data();
        e.enc_seg_w = sw.data();
        RadCtx c{ .b = &b, .batch = &e };
        hyv4flash::step(&c, &e);
        CHECK_LINT(b, &c);
        if (!c.step_fail.empty()) fprintf(stderr, "encoder pass: %s\n", c.step_fail.c_str());
        CHECK(c.step_fail.empty());
        CHECK_EQ(count_issued(b, c, "gemm_nt_bias"), 7);   /* patch, 4 a block, 2 merger */
    }
    /* A prefill with media rows at three-component positions. */
    {
        auto k = kit(b, {}, { 12 }, 0);
        k->b.positions_h = k->pos.data();
        std::vector<int32_t> rp((size_t)(3 * 12));
        for (int c = 0; c < 3; ++c)
            for (int t = 0; t < 12; ++t) rp[(size_t)(c * 12 + t)] = t < 2 ? t : 2 + c;
        std::vector<int32_t> rows = { 2, 3, 4, 5 };
        std::vector<uint16_t> embd((size_t)(4 * 128), 0);
        k->b.rope_pos = rp.data();
        k->b.rope_mixed = 1;
        k->b.n_mm_rows = 4;
        k->b.mm_rows = rows.data();
        k->b.mm_embd = embd.data();
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK_LINT(b, &c);
        if (!c.step_fail.empty()) fprintf(stderr, "media prefill: %s\n", c.step_fail.c_str());
        CHECK(c.step_fail.empty());
        CHECK_EQ(count_issued(b, c, "scatter_rows"), 1);
    }
    /* No sections: the tower's weights are declared (a container carries them), nothing served,
     * and the trunk stays gptj. */
    Vis n(false);
    REQUIRE_EQ(n.rc, RAD_OK);
    CHECK_EQ(n.b.encoder.out, (rad_buf)0);
    CHECK(find_w(n.b, "v.blk.0.attn_qkv.weight") != nullptr);
    CHECK_EQ(pstr(find_op(n.b, "rope"), "mode"), std::string("gptj"));       /* the key's */
    CHECK_EQ(pstr(find_op(n.b, "rope_rows"), "mode"), std::string("gptj"));  /* the query's */
}

/* What this build does not serve is refused by name at declare: a quantised kv_b (DECISIONS U2),
 * a recipe over a parameter the model applies in f32, a quantised token embedding, and a trunk
 * linear in a form that reads E4M3 codes no producer of this graph writes (MXFP4). */
TEST(unserved_forms_are_refused_at_declare) {
    auto kvb = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "attn_kv_b", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
    };
    CHECK_EQ(Declared(kvb).rc, RAD_E_UNSUPPORTED);
    auto experts = [](RadBuilder* b) {
        bf16_encs(b);
        RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        rad_enc_copy_str(i8.transform, "fwht128");
        b->encs[0].second = i8;
        b->encs[1].second = i8;
    };
    /* The case this test used to refuse: quantised routed experts declare and are served (the
     * refused forms above are the ones no producer of this graph writes). */
    {
        Declared q(experts);
        REQUIRE_EQ(q.rc, RAD_OK);
        CHECK(find_w(q.b, "blk.2.ffn_gate_up_exps.3.weight") != nullptr);
        CHECK(find_w(q.b, "blk.2.glm_router.weight") != nullptr);
    }
    auto hc = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "hc_attn.fn", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
    };
    CHECK(Declared(hc).rc < 0);
    auto tok = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { "token_embd", enc_i8(false) });
    };
    CHECK_EQ(Declared(tok).rc, RAD_E_UNSUPPORTED);
    auto mx = [](RadBuilder* b) {
        bf16_encs(b);
        RadEncoding e = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
        b->encs.insert(b->encs.begin(), { "attn_q_a", e });
    };
    CHECK_EQ(Declared(mx).rc, RAD_E_DTYPE);
}


/* ------------------------------------------------------------------ tensor parallelism (PLAN §1.5) */
namespace {

/* What a weight's name says it is: an attention part split by whole heads, a width split, the
 * vocabulary-split head, a routed expert (whole on one rank), or replicated. */
enum Part { kReplicated, kRows, kCols, kExpert };
Part part_of(const std::string& n) {
    auto has = [&](const char* x) { return n.find(x) != std::string::npos; };
    if (has("_exps.")) return kExpert;
    if (n.rfind("dfly.", 0) == 0 || n.rfind("v.", 0) == 0) return kReplicated;   /* not served */
    for (const char* r : { ".attn_q_nope.", ".attn_q_rope.", ".attn_kv_b.", ".attn_gate.",
                           ".attn_sinks.", ".ffn_gate_up.", ".ffn_gate_up_shexp." })
        if (has(r)) return kRows;
    if (n == "output.weight") return kRows;
    for (const char* c : { ".attn_output.", ".ffn_down.", ".ffn_down_shexp." })
        if (has(c)) return kCols;
    return kReplicated;
}

rad_buf buf_named(const RadBuilder& b, const char* n) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == n) return (rad_buf)(i + 1);
    return 0;
}

void steps_clean_at(RadBuilder& b, int rank, int world) {
    struct Case { std::vector<int> dq, pq; int32_t ctx; };
    const Case cases[] = { { { 1, 1, 1 }, {}, 0 }, { {}, { 12 }, 0 }, { {}, { 40 }, 0 },
                           { { 1 }, { 10 }, 0 }, { {}, { 16 }, 16 } };
    for (const Case& cs : cases) {
        auto k = kit(b, cs.dq, cs.pq, cs.ctx);
        k->b.positions_h = k->pos.data();
        auto v = run_passes(b, hyv4flash::step, { k.get() }, rank, world);
        for (auto& c : v)
            if (!c->step_fail.empty()) fprintf(stderr, "rank %d/%d: %s\n", rank, world,
                                               c->step_fail.c_str());
        CHECK_PASSES(b, v);
    }
}

}  // namespace

/* Every weight of the one-rank graph is covered exactly once by the ranks of two and four: a head
 * or width split is an even share along its split dimension on every rank (each stacked half of
 * a fused gate_up on its own), a routed expert is whole on exactly one rank (bf16 experts,
 * `expert_base = rank * E / world`), and everything else (iHC, norms, q_a, kv_a, the indexer,
 * the router, the latent pair, the Engram, the embedding) is the whole weight on every rank. */
TEST(the_ranks_cover_every_weight_exactly_once) {
    for (bool wide : { false, true }) {
        Declared one(bf16_encs, 1, 0, wide);
        REQUIRE_EQ(one.rc, RAD_OK);
        std::map<std::string, const RecWeight*> whole;
        for (const auto& w : one.b.weights) whole[w.name] = &w;
        for (int world : { 2, 4 }) {
            if (!wide && world == 4) continue;   /* mini's 2 sparse heads: refused, below */
            std::vector<std::unique_ptr<Declared>> ranks;
            std::map<std::string, int> seen;
            for (int r = 0; r < world; ++r) {
                ranks.push_back(std::make_unique<Declared>(bf16_encs, world, r, wide));
                REQUIRE_EQ(ranks.back()->rc, RAD_OK);
                for (const auto& w : ranks.back()->b.weights) {
                    ++seen[w.name];
                    auto it = whole.find(w.name);
                    REQUIRE(it != whole.end());
                    const RadWeightDecl& o = it->second->d;
                    const Part pt = part_of(w.name);
                    CHECK_EQ(w.d.rank, o.rank);
                    CHECK(!w.span_set);   /* every split here is even */
                    const int split = pt == kRows ? 0 : pt == kCols ? 1 : -1;
                    CHECK_EQ(w.d.shard, pt == kRows ? (int)RAD_SHARD_ROW
                                        : pt == kCols ? (int)RAD_SHARD_COL
                                                      : (int)RAD_SHARD_NONE);
                    for (uint32_t k = 0; k < o.rank; ++k) {
                        const int64_t want = (int)k == split ? o.shape[k] / world : o.shape[k];
                        if (w.d.shape[k] != want)
                            fprintf(stderr, "%s dim %u: %lld, want %lld (world %d)\n",
                                    w.name.c_str(), k, (long long)w.d.shape[k], (long long)want,
                                    world);
                        CHECK_EQ(w.d.shape[k], want);
                    }
                    /* A fused gate_up splits each half (an expert is whole: no parts). */
                    if (pt == kExpert) continue;
                    CHECK_EQ(w.d.n_row_parts, o.n_row_parts);
                    for (int q = 0; q < o.n_row_parts; ++q)
                        CHECK_EQ(w.d.row_parts[q], o.row_parts[q] / (pt == kRows ? world : 1));
                }
            }
            for (const auto& [name, w] : whole) {
                const int want = part_of(name) == kExpert ? 1 : world;
                if (seen[name] != want)
                    fprintf(stderr, "%s on %d of %d ranks, want %d\n", name.c_str(), seen[name],
                            world, want);
                CHECK_EQ(seen[name], want);
            }
            /* Experts [r E / W, (r + 1) E / W) on rank r. */
            const int E = wide ? 8 : 4;
            for (int r = 0; r < world; ++r)
                for (int e = 0; e < E; ++e) {
                    char n[64];
                    std::snprintf(n, sizeof n, "blk.2.ffn_down_exps.%d.weight", e);
                    CHECK_EQ(find_w(ranks[(size_t)r]->b, n) != nullptr, e / (E / world) == r);
                }
        }
    }
}

/* A rank's graph: its heads in every attention op, two exact all-reduces a layer (behind o_proj and
 * behind the feed-forward's sum, each before its sandwich norm, in place), its experts in the
 * routed arm, its vocabulary in the head and the logits buffer; every step kind lint clean on every
 * rank. */
TEST(each_rank_declares_its_share_and_reduces_twice_a_layer) {
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            Declared d(bf16_encs, world, rank, true);
            REQUIRE_EQ(d.rc, RAD_OK);
            RadBuilder& b = d.b;
            const int Hs = 4 / world, Hw = 8 / world, E = 8 / world;
            for (const char* op : { "mla_attn_paged_sink", "mla_project_kv", "attn_kv_sink",
                                    "mla_kv_split" })
                for (int i = 0; i < count_op(b, op); ++i) {
                    const int64_t h = pint(find_op(b, op, i), "heads");
                    CHECK(h == Hs || h == Hw);
                }
            CHECK_EQ(pint(find_op(b, "mla_attn_paged_sink", 0), "heads"), Hs);   /* layer 0 sparse */
            CHECK_EQ(pint(find_op(b, "mla_attn_paged_sink", 2), "heads"), Hw);   /* layer 1 sliding */
            CHECK_EQ(pint(find_op(b, "mla_attn_sel_sink", 0), "heads"), Hs);
            CHECK_EQ(pint(find_op(b, "gate_mul", 1), "n"), Hw * 32);
            /* Replicated: the cache row, the indexer, the router over every expert. */
            CHECK_EQ(kv_named(b, "kv_latent_sparse")->d.head_dim, 20);
            CHECK_EQ(pint(find_op(b, "idx_score"), "heads"), 2);
            CHECK_EQ(pint(find_op(b, "router_sigmoid_topk"), "n_expert"), 8);
            int packed_projections = 0;
            for (const auto& o : b.ops) {
                if (o.op != "gemm_nt_f32" || o.w.empty()) continue;
                const auto& weight = b.weights.at(o.w[0] - 1);
                if (weight.name.find(".glm_router.weight") == std::string::npos) continue;
                CHECK_EQ(pint(&o, "weight_bf16") == 1, weight.d.dtype == RAD_BF16);
                packed_projections += weight.d.dtype == RAD_BF16;
            }
            /* Eight experts split into four f32 logits per rank at TP2. */
            CHECK_EQ(packed_projections, world == 2 ? 2 : 1);
            int based = 0;
            for (const auto& o : b.ops)
                if (param(&o, "expert_base")) {
                    ++based;
                    CHECK_EQ(pint(&o, "expert_base"), rank * E);
                }
            CHECK(based >= 2);
            int mg = 0;
            for (const auto& o : b.ops)
                if (o.op == "moe_gemm") {
                    ++mg;
                    CHECK_EQ(pint(&o, "n_expert"), E);
                }
            CHECK_EQ(mg, 4);
            CHECK_EQ(pint(find_op(b, "logits_gemm"), "n_vocab"), 320 / world);
            CHECK_EQ(find_b(b, "logits")->d.shape[1], 320 / world);

            /* Six reductions at more than one rank, and each has its narrow twin beside it
             * (`overlap` 1): the twin is declared whether or not the pipeline is on, because a
             * graph whose op list depends on `max_tok` cannot be re-declared at a smaller step
             * for the arena's levels (a_sizing_declare_...). */
            const int nar = count_op(b, "all_reduce");
            CHECK_EQ(nar, world == 1 ? 0 : 12);
            int narrow = 0;
            for (int i = 0; i < nar; ++i) narrow += pint(find_op(b, "all_reduce", i), "overlap") == 1;
            CHECK_EQ(narrow, world == 1 ? 0 : 6);
            for (int i = 0; i < nar; ++i) {
                const RecOp* ar = find_op(b, "all_reduce", i);
                CHECK_EQ(pint(ar, "exact"), 1);
                CHECK_EQ(pint(ar, "world_size"), world);
                CHECK_EQ(pstr(ar, "dtype"), std::string("bf16"));
                REQUIRE_EQ(ar->reads.size(), (size_t)1);
                CHECK(ar->writes == ar->reads);
                /* Its buffer is what the next rmsnorm reads: o_proj's output or the block's sum. */
                const rad_buf x = ar->reads[0];
                bool normed = false;
                for (const auto& o : b.ops)
                    if (o.op == "rmsnorm" && in(o.reads, x)) normed = true;
                CHECK(normed);
            }
            if (world > 1) {
                const rad_buf o = buf_named(b, "sparse.o_proj"), sum = buf_named(b, "moe.routed"),
                              dense = buf_named(b, "dense_out");
                int on_o = 0, on_sum = 0, on_dense = 0;
                for (int i = 0; i < nar; ++i) {
                    const rad_buf x = find_op(b, "all_reduce", i)->reads[0];
                    on_o += x == o;
                    on_sum += x == sum;
                    on_dense += x == dense;
                }
                CHECK_EQ(on_o, 4);       /* the two sparse layers, wide and narrow */
                CHECK_EQ(on_sum, 4);     /* the two MoE layers */
                CHECK_EQ(on_dense, 2);   /* layer 0 */
            }
            steps_clean_at(b, rank, world);
        }
}

/* P-02: the routing counts are the model's routing table (rad_route_table): an i32 PERSIST row a
 * layer at the model's expert count on every rank, each MoE layer's router writing and reporting
 * its own row. The core then copies nothing a report, and no wide prefill chunk can alias a
 * transient histogram (qwen4exp_fp8's note). */
TEST(the_routing_counts_are_the_models_routing_table) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            Declared d(bf16_encs, world, rank, true);
            REQUIRE_EQ(d.rc, RAD_OK);
            RadBuilder& b = d.b;
            const rad_buf tab = buf_named(b, "moe.expert_counts");
            REQUIRE(tab != 0);
            CHECK_EQ(b.route_table, tab);
            const RecBuf& t = b.bufs[(size_t)tab - 1];
            CHECK_EQ(t.d.kind, RAD_BUF_PERSIST);
            CHECK_EQ(t.d.dtype, (uint32_t)RAD_I32);
            CHECK_EQ(t.d.rank, 2u);
            CHECK_EQ(t.d.shape[0], 3);   /* mini4: a row for each of the three layers */
            CHECK_EQ(t.d.shape[1], 8);   /* the model's experts, whatever share this rank holds */
            int writers = 0;   /* the routers write the table's rows through the declared buffer */
            for (const auto& o : b.ops)
                if (in(o.writes, tab)) ++writers;
            CHECK(writers > 0);
            const std::vector<int> dq[2] = { { 1, 1, 1 }, {} }, pq[2] = { {}, { 12 } };
            for (int i = 0; i < 2; ++i) {   /* a decode step and a fresh prefill */
                auto k = kit(b, dq[i], pq[i], 0);
                k->b.positions_h = k->pos.data();
                RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = world };
                hyv4flash::step(&c, &k->b);
                CHECK_LINT(b, &c);
                CHECK(c.step_fail.empty());
                CHECK(c.routed_layers == std::vector<int>({ 1, 2 }));   /* layer 0 is dense */
                for (const auto& cp : c.copies) CHECK(cp.second != tab);   /* nothing is copied into it */
                REQUIRE_EQ(c.routed_counts.size(), c.routed_layers.size());
                for (size_t j = 0; j < c.routed_counts.size(); ++j) {
                    CHECK_EQ(c.routed_counts[j].first, tab);
                    CHECK_EQ(c.routed_counts[j].second,
                             (int64_t)c.routed_layers[j] * 8 * (int64_t)sizeof(int32_t));
                }
            }
        }
}

/* The taps the harness compares are the replicated values at every world; a rank's partial sums
 * (the gate logits of its heads, the arms' outputs) are tapped at one rank only, o_proj after its
 * reduction. */
TEST(taps_at_more_ranks_are_the_replicated_values) {
    setenv("RADIANCE_HY4_TAPS", "1", 1);
    Declared d(bf16_encs, 2, 1);
    unsetenv("RADIANCE_HY4_TAPS");
    REQUIRE_EQ(d.rc, RAD_OK);
    for (const char* n : { "dbg.resid.0", "dbg.attn_out.1", "dbg.mlp_out.2", "dbg.attn_o_proj.0",
                           "dbg.router_topk_idx.1", "dbg.index_sel.0", "dbg.engram_out.1" })
        CHECK(find_b(d.b, n) != nullptr);
    for (const char* n : { "dbg.attn_gate_logits.0", "dbg.moe_experts_out.1",
                           "dbg.moe_routed_out.1", "dbg.moe_shared_out.2" })
        CHECK(find_b(d.b, n) == nullptr);
    steps_clean_at(d.b, 1, 2);
}

/* What does not split is refused by name at declare: a world other than 1, 2 or 4, heads that are
 * not whole a rank (mini's two sparse heads over four), quantised experts whose width is not whole
 * 128-column groups a rank (mini's 128 at two), and a library without an exact all-reduce. */
TEST(tensor_parallel_refusals) {
    CHECK_EQ(Declared(bf16_encs, 3, 0, true).rc, RAD_E_UNSUPPORTED);
    CHECK_EQ(Declared(bf16_encs, 8, 0, true).rc, RAD_E_UNSUPPORTED);
    CHECK_EQ(Declared(bf16_encs, 4, 0, false).rc, RAD_E_UNSUPPORTED);
    auto experts = [](RadBuilder* b) {
        bf16_encs(b);
        RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        rad_enc_copy_str(i8.transform, "fwht128");
        b->encs[0].second = i8;
        b->encs[1].second = i8;
    };
    CHECK_EQ(Declared(experts, 2).rc, RAD_E_UNSUPPORTED);
    auto no_ar = [](RadBuilder* b) {
        bf16_encs(b);
        b->refuse.insert("all_reduce");
    };
    CHECK_EQ(Declared(no_ar, 2).rc, RAD_E_UNSUPPORTED);
    CHECK_EQ(Declared(no_ar, 1).rc, RAD_OK);   /* one rank reduces nothing */
}

/* ------------------------------------------------------------------ quantised formats (PLAN §4.3) */
namespace {

/* Every declared weight of the served graph (all but the drafter and the tower, declared and not
 * served) is a weight operand of some op: no plane is declared and left unread. */
void every_weight_read(const RadBuilder& b) {
    std::vector<bool> read(b.weights.size() + 1, false);
    for (const auto& o : b.ops)
        for (rad_weight w : o.w)
            if (w > 0 && (size_t)w <= b.weights.size()) read[(size_t)w] = true;
    for (size_t i = 0; i < b.weights.size(); ++i) {
        const std::string& n = b.weights[i].name;
        if (n.rfind("dfly.", 0) == 0 || n.rfind("v.", 0) == 0) continue;
        if (!read[i + 1]) fprintf(stderr, "declared and not read: %s\n", n.c_str());
        CHECK(read[i + 1]);
    }
}

/* Layer 1's routed arm in the form `gu` / `dn`: the GEMMs' dtypes, and the clamped product's
 * chain (swiglu_clamped, then the quantiser the down projection reads). */
void check_routed(const RadBuilder& b, const char* gu_dt, const char* dn_dt, const char* quant) {
    int n = 0;
    for (const auto& o : b.ops)
        if (o.op == "moe_gemm_q") {
            CHECK_EQ(pstr(&o, "dtype"), std::string(n % 2 == 0 ? gu_dt : dn_dt));
            ++n;
        }
    CHECK_EQ(n, 4);   /* two MoE layers, gate_up and down */
    CHECK_EQ(count_op(b, "moe_gemm"), 0);
    CHECK_EQ(count_op(b, "swiglu_clamped"), 2);
    const RecOp* cl = find_op(b, "swiglu_clamped");
    REQUIRE(cl != nullptr);
    CHECK_EQ(pflt(cl, "limit"), 10.0);
    REQUIRE_EQ(cl->writes.size(), (size_t)1);
    const rad_buf prod = cl->writes[0];
    bool quantised = false;
    for (const auto& o : b.ops)
        if (o.op == quant && in(o.reads, prod)) quantised = true;
    CHECK(quantised);
}

}  // namespace

/* Every routed-expert form of PLAN §4.3 at K 1536's shape class (here 128), clamped at 10 in every
 * layer: the codebook forms on the rotated E4M3 arm (clamp, then had_quant_act_fp8 into the down
 * projection's codes; the latent's rotated copy by had_quant_act_fp8), int8 and the IQR trellis on
 * the rotated int8 arm (had_quant_act_i8g), and a trellis gate_up with a six-bit codebook down,
 * whose clamped product is the rotated E4M3 pair beside the int8 gate_up input. Every step kind is
 * lint clean. */
TEST(quantised_routed_experts_are_served_clamped) {
    struct Form {
        const char* what;
        void (*encs)(RadBuilder*);
        const char* gu, *dn, *quant, *hq;
    };
    const Form forms[] = {
        { "w4nl64", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_nl64(4); },
          "w4nl64a8h", "w4nl64a8h", "had_quant_act_fp8", "had_quant_act_fp8" },
        { "w5nl64", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_nl64(5); },
          "w5nl64a8h", "w5nl64a8h", "had_quant_act_fp8", "had_quant_act_fp8" },
        { "w6nl64", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_nl64(6); },
          "w6nl64a8h", "w6nl64a8h", "had_quant_act_fp8", "had_quant_act_fp8" },
        { "i8", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_i8(true); },
          "i8a8h", "i8a8h", "had_quant_act_i8g", "had_quant_act_i8g" },
        { "iqr", [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_trellis(RAD_U4, true); },
          "i8a8h", "i8a8h", "had_quant_act_i8g", "had_quant_act_i8g" },
        { "iqr+w6", [](RadBuilder* b) {
              bf16_encs(b);
              b->encs[0].second = enc_trellis(RAD_U4, true);
              b->encs[1].second = enc_nl64(6);
          }, "i8a8h", "w6nl64a8h", "had_quant_act_fp8", "had_quant_act_i8g" },
    };
    for (const Form& f : forms) {
        fprintf(stderr, "form %s\n", f.what);
        Declared d(f.encs);
        REQUIRE_EQ(d.rc, RAD_OK);
        check_routed(d.b, f.gu, f.dn, f.quant);
        /* The latent's rotated codes for gate_up, from latent_in's bf16 output. */
        const rad_buf lat = buf_named(d.b, "moe.latent");
        bool hq = false;
        for (const auto& o : d.b.ops)
            if (o.op == f.hq && in(o.reads, lat)) hq = true;
        CHECK(hq);
        every_weight_read(d.b);
        steps_clean_at(d.b, 0, 1);
    }
}

/* K-18: the block declares the sort's inverse table, so the fused selection or the separate
 * `moe_scatter_inv` writes `moe.expert_slot` and `moe_gather_inv` reads it instead of rebuilding
 * it; without the device rows the pair falls back to `moe_scatter` / `moe_gather`, which scan the
 * sort and add the two prepass launches above 512 slots. */
std::vector<std::string> g_undecl;
void undeclaring_encs(RadBuilder* b) {
    bf16_encs(b);
    for (const std::string& op : g_undecl) b->undeclared.insert(op);
}
TEST(the_moe_gather_reads_the_sorts_inverse_table) {
    struct Case {
        const char* what;
        std::vector<std::string> undecl;
        int invg;    /* per MoE layer: moe_gather_inv (1) or moe_gather (0) */
        int writers; /* ops that write the inverse table: the fused selection and the separate
                      * scatter above it (the block declares both, each for a size of pass) */
    };
    const Case cases[] = {
        { "the fused selection", {}, 1, 4 },
        { "the separate sort", { "router_sigmoid_topk_scatter" }, 1, 2 },
        { "no device rows", { "router_sigmoid_topk_scatter", "moe_scatter_inv", "moe_gather_inv" },
          0, 0 },
    };
    for (const Case& c : cases) {
        fprintf(stderr, "case %s\n", c.what);
        g_undecl = c.undecl;
        Declared d(undeclaring_encs);
        g_undecl.clear();
        REQUIRE_EQ(d.rc, RAD_OK);
        const rad_buf inv = buf_named(d.b, "moe.expert_slot");
        const rad_buf sorted = buf_named(d.b, "moe.expert_sorted");
        REQUIRE(inv != 0);
        REQUIRE(sorted != 0);
        int writers = 0;
        for (const auto& o : d.b.ops)
            if (in(o.writes, inv)) ++writers;
        CHECK_EQ(count_op(d.b, "moe_gather_inv"), c.invg * 2);
        CHECK_EQ(count_op(d.b, "moe_gather"), (1 - c.invg) * 2);
        CHECK_EQ(writers, c.writers);
        for (int i = 0; i < c.invg * 2; ++i) {
            const RecOp* o = find_op(d.b, "moe_gather_inv", i);
            REQUIRE(o != nullptr);
            REQUIRE(o->reads.size() >= 3);
            CHECK_EQ(o->reads[2], inv);
        }
        for (int i = 0; i < (1 - c.invg) * 2; ++i) {
            const RecOp* o = find_op(d.b, "moe_gather", i);
            REQUIRE(o != nullptr);
            REQUIRE(o->reads.size() >= 3);
            CHECK_EQ(o->reads[2], sorted);
        }
        every_weight_read(d.b);
        steps_clean_at(d.b, 0, 1);
    }
}

/* Trunk linears a recipe made int8 (rotated or not) or the u5 trellis read int8 codes from their
 * input's bf16 plane; the others stay bf16 GEMMs, and an int8 lm_head is read by logits_gemm at its
 * codes' dtype. Linears of one input and one code form share one quantiser (Lin::feed: q_a with
 * the gate, the latent with the shared expert's gate_up; the u5 trellis and the rotated int8 read
 * the same had_quant_act_i8g codes), and where the library serves it the shared expert's gate writes
 * its down projection's codes itself (gated_had_quant_i8g). Without the device rows (a CPU
 * engine), or with RADIANCE_HY4_SHARE_CODES=0 / RADIANCE_HY4_FUSED_QUANT=0, each quantises its own. */
void int8_trunk_encs(RadBuilder* b) {
    bf16_encs(b);
    b->encs.insert(b->encs.begin(), { { "attn_output", rad_enc_plain(RAD_BF16) },
                                      { "attn_q_a.", enc_trellis(RAD_U5, true) },
                                      { "attn_gate.", enc_i8(true) },
                                      { "ffn_gate_up_shexp", enc_i8(true) },
                                      { "ffn_down_shexp", enc_i8(true) },
                                      { "ffn_gate_up.", enc_i8(true) },
                                      { "ffn_down.", enc_i8(false) },
                                      { "ffn_latent_in", enc_i8(true) },
                                      { "ffn_latent_out", enc_i8(true) },
                                      { "output.weight", enc_i8(false) } });
}
/* The same on a CPU engine: no device-only fused rows. */
void int8_trunk_encs_host(RadBuilder* b) {
    int8_trunk_encs(b);
    for (const char* op : { "gated_had_quant_i8g", "rmsnorm_had_quant_i8g", "router_sigmoid_topk_scatter" })
        b->undeclared.insert(op);
}

/* K-19: the gather writes the routed sum's rotated int8 codes and scales in the launch that sums
 * it (`moe_gather_inv_i8g`), so `ffn_latent_out` declares no `had_quant_act_i8g` of its own -- one
 * launch fewer on every MoE layer, the same codes. It needs the slot table, no shared arm in the
 * gather, `top_k` 16 (the codes' 256-column wave is written from the sixteen-slot selection), a
 * latent that is a whole number of those waves and one the weight's scale group divides; where any
 * of those fails the plain gather serves and the linear quantises for itself (and the codes' pair
 * is only declared, never written, when the switch is off). */
TEST(the_moe_gather_writes_the_routed_sums_codes) {
    struct Run {
        const char* what;
        bool off;              /* RADIANCE_HY4_FUSE_GATHER_Q8=0 */
        int64_t latent, top_k;
        int q8g;               /* moe_gather_inv_i8g launches, one on each of the two MoE layers */
        int had;               /* had_quant_act_i8g: the latent_out pair gone where fused */
    };
    const Run runs[] = {
        { "latent 256, sixteen slots", false, 256, 16, 2, 8 - 2 },
        { "the switch off", true, 256, 16, 0, 8 },
        { "a latent the group does not divide", false, 128, 16, 0, 8 },
        { "eight slots, not the sixteen the codes need", false, 256, 8, 0, 8 },
    };
    for (const Run& r : runs) {
        fprintf(stderr, "run %s\n", r.what);
        if (r.off) setenv("RADIANCE_HY4_FUSE_GATHER_Q8", "0", 1);
        Declared d(int8_trunk_encs, 1, 0, false, 128, r.latent, r.top_k, 16);
        unsetenv("RADIANCE_HY4_FUSE_GATHER_Q8");
        REQUIRE_EQ(d.rc, RAD_OK);
        const RadBuilder& b = d.b;
        CHECK_EQ(count_op(b, "moe_gather_inv_i8g"), r.q8g);
        CHECK_EQ(count_op(b, "moe_gather_inv"), r.q8g ? 0 : 2);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), r.had);
        /* The routed sum's pair: a role of its own (`sum`), the trunk's for that width and group,
         * the codes one int8 a column; lat_out reads them only where the gather wrote them. */
        const RecBuf* q8 = find_b(b, "trunk_codes.sum.256.128.q8");
        CHECK_EQ(q8 != nullptr, r.q8g > 0);
        if (q8) CHECK_EQ(q8->d.dtype, RAD_I8);
        for (int i = 0; i < r.q8g; ++i) {
            const RecOp* o = find_op(b, "moe_gather_inv_i8g", i);
            REQUIRE(o != nullptr);
            /* y_expert, expert_w and the slot table read; the codes, their scales and the sum
             * written -- the sum being the bf16 latent ffn_latent_out reads. */
            REQUIRE_EQ(o->reads.size(), (size_t)3);
            REQUIRE_EQ(o->writes.size(), (size_t)3);
            CHECK_EQ(o->writes[0], buf_named(b, "trunk_codes.sum.256.128.q8"));
            CHECK_EQ(o->writes[1], buf_named(b, "trunk_codes.sum.256.128.s8"));
            CHECK_EQ(o->writes[2], buf_named(b, "moe.latent"));
            CHECK_EQ(pint(o, "group"), 128);
            CHECK_EQ(pint(o, "n"), 256);
            CHECK_EQ(pint(o, "top_k"), 16);
        }
        every_weight_read(b);
        steps_clean_at(d.b, 0, 1);
    }
}

TEST(quantised_trunk_linears_share_one_quantiser_an_input) {
    struct Run { const char* what; bool host, share, fuse; int had, gated; };
    /* 15 rotated GEMMs: q_a and gate (3 layers each), the dense gate_up, the shared pair and the
     * latent pair (2 MoE layers each). Their quantisers: q_a (3, the gate fed), the dense gate_up
     * (1), the latent (2, the shared gate_up fed), the latent_out (2) and, unless the gate writes
     * them, the shared down's (2). */
    const Run runs[] = {
        { "device", false, true, true, 3 + 1 + 2 + 2, 2 },
        { "host", true, true, true, 3 + 1 + 2 + 2 + 2, 0 },
        { "no fusion", false, true, false, 3 + 1 + 2 + 2 + 2, 0 },
        { "no sharing", false, false, false, 15, 0 },
    };
    for (const Run& r : runs) {
        fprintf(stderr, "run %s\n", r.what);
        if (!r.share) setenv("RADIANCE_HY4_SHARE_CODES", "0", 1);
        if (!r.fuse) setenv("RADIANCE_HY4_FUSED_QUANT", "0", 1);
        Declared d(r.host ? int8_trunk_encs_host : int8_trunk_encs);
        unsetenv("RADIANCE_HY4_SHARE_CODES");
        unsetenv("RADIANCE_HY4_FUSED_QUANT");
        REQUIRE_EQ(d.rc, RAD_OK);
        const RadBuilder& b = d.b;
        /* A pair a width (the 128-wide inputs and the dense down's 256), and a pair of its own for
         * each shared input and for the gate's codes. */
        for (const char* n : { "trunk_codes.128.128.q8", "trunk_codes.256.128.q8" }) {
            const RecBuf* q8 = find_b(b, n);
            REQUIRE(q8 != nullptr);
            CHECK_EQ(q8->d.dtype, RAD_I8);
        }
        CHECK(find_b(b, "trunk_codes.64.128.q8") == nullptr);
        CHECK_EQ(find_b(b, "trunk_codes.attn_in.128.128.q8") != nullptr, r.share);
        CHECK_EQ(find_b(b, "trunk_codes.ffn_in.128.128.q8") != nullptr, r.share);
        CHECK_EQ(find_b(b, "trunk_codes.gated.128.128.q8") != nullptr, r.gated > 0);
        int i8h = 0, i8 = 0;
        for (const auto& o : b.ops)
            if (o.op == "gemm_nt_q") (pstr(&o, "dtype") == "i8a8h" ? i8h : i8) += 1;
        CHECK_EQ(i8h, 3 + 3 + 1 + 2 * 2 + 2 * 2);
        CHECK_EQ(i8, 1);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), r.had);
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), r.gated);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 1);
        /* The routers' selection and sort in one launch where a library serves it (MoeFP8's
         * router_fuse), the pair beside it for wider passes. */
        CHECK_EQ(count_op(b, "router_sigmoid_topk_scatter"), r.host ? 0 : 2);
        CHECK_EQ(count_op(b, "router_sigmoid_topk"), 2);
        /* Left bf16: q_nope / q_rope, kv_a, attn_output, the indexer. */
        CHECK(count_op(b, "gemm_nt") > 0);
        const RecWeight* head = find_w(b, "output.weight");
        REQUIRE(head != nullptr);
        CHECK_EQ(head->d.dtype, RAD_I8);
        every_weight_read(d.b);
        /* Every fed GEMM's codes are written before it in each step kind. */
        steps_clean_at(d.b, 0, 1);
    }
}

/* K-23: the iqr-gptq recipe's FFN is a u4 trellis gate_up and a rotated int8 down, so the gate
 * and up rows' GEMM carries the product's rotated int8 quantiser in its own epilogue
 * (decl_gated_fold_iqr = gemm_nt_q_gated at i8a8h) for the rows libr4d's gemm_iqr_gated serves
 * (M <= 16): layer 0's dense MLP and both MoE layers' shared expert. The gate_up + gated_had_quant
 * pair stays declared beside it for the longer rows, and it writes the same two code buffers, so a
 * decode step is one launch where the pair is two and the codes are the pair's own (docs/OPS.md,
 * "gemm_nt_q_gated": byte-identical). Where the library serves no such row, the pair runs. */
void iqr_ffn_encs(RadBuilder* b) {
    bf16_encs(b);
    b->encs.insert(b->encs.begin(), { { "ffn_down_shexp", enc_i8(true) },
                                      { "ffn_gate_up_shexp", enc_trellis(RAD_U4, true) },
                                      { "ffn_down.", enc_i8(true) },
                                      { "ffn_gate_up.", enc_trellis(RAD_U4, true) } });
}
TEST(the_gate_and_up_rows_fold_the_quantiser_into_the_gemm) {
    for (int host = 0; host < 2; ++host) {
        /* n_embd 256: the row's K and N are whole 256s, and the mini preset's 128 wide trunk
         * leaves none to declare (decl_gated_fold_iqr refuses that geometry). */
        MetaKV mk = mini_meta(false, 128, 256);
        RadBuilder b;
        iqr_ffn_encs(&b);
        if (host) b.undeclared.insert("gemm_nt_q_gated");
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
        const int folds = host ? 0 : 3;
        CHECK_EQ(count_op(b, "gemm_nt_q_gated"), folds);
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 3);   /* the pair, for the rows past 16 */
        for (int i = 0; i < folds; ++i) {
            const RecOp* f = find_op(b, "gemm_nt_q_gated", i);
            REQUIRE(f != nullptr);
            const RecParam* m = param(f, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->kind, (int)RAD_P_RANGE);
            CHECK_EQ(m->ival, 1);
            CHECK_EQ(m->ihi, 16);
            CHECK_EQ(pstr(f, "dtype"), std::string("i8a8h"));
            CHECK_EQ(pstr(f, "act"), std::string("silu"));
            CHECK_EQ(pint(f, "group"), 128);
            CHECK_EQ(pint(f, "K"), 256);                     /* the trunk width */
            CHECK_EQ(pint(f, "N") % 256, 0);
            /* the weight's two halves, the input's codes and the product's: no bf16 plane */
            REQUIRE_EQ(f->w.size(), (size_t)2);
            REQUIRE_EQ(f->reads.size(), (size_t)2);
            REQUIRE_EQ(f->writes.size(), (size_t)2);
            const RecOp* a = find_op(b, "gated_had_quant_i8g", i);
            REQUIRE(a != nullptr);
            CHECK(a->writes == f->writes);
        }
        /* A decode-class step issues the fold, not the pair; the pair stays for the longer ones. */
        std::vector<std::unique_ptr<Kit>> dec;
        dec.push_back(kit(b, { 1 }, {}));
        dec.push_back(kit(b, { 16 }, {}));
        for (auto& k : dec) {
            k->b.positions_h = k->pos.data();
            RadCtx c{ .b = &b, .batch = &k->b };
            hyv4flash::step(&c, &k->b);
            CHECK(c.step_fail.empty());
            CHECK_EQ(count_issued(b, c, "gemm_nt_q_gated"), folds);
            CHECK_EQ(count_issued(b, c, "gated_had_quant_i8g"), host ? 3 : 0);
            /* the codes still go through the gate_up's own quantiser where they are not fed */
            CHECK_EQ(count_issued(b, c, "had_quant_act_i8g"), 3);
        }
        auto pf = kit(b, {}, { 17 });
        pf->b.positions_h = pf->pos.data();
        RadCtx pc{ .b = &b, .batch = &pf->b };
        hyv4flash::step(&pc, &pf->b);
        CHECK(pc.step_fail.empty());
        CHECK_EQ(count_issued(b, pc, "gemm_nt_q_gated"), 0);
        CHECK_EQ(count_issued(b, pc, "gated_had_quant_i8g"), 3);
        every_weight_read(b);
        steps_clean_at(b, 0, 1);
    }
}

/* The fold over a row-split gate_up: it takes the shard's own share of the weight and half of the
 * gate|up rows, so a 2-rank split halves the dense MLP's 512 columns and the shared expert's 512
 * to a whole 256 each. (The shared widths are set to 512 for the split: the mini preset's 256 would
 * halve to 128 gate|up columns a rank, which the shared expert's int8 down refuses.) */
TEST(the_fold_follows_the_gate_ups_own_row_share) {
    MetaKV mk = mini_meta(false, 128, 256);
    mk.set("moe_intermediate_size", "256");
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            iqr_ffn_encs(&b);
            RadBuildCtx ctx = build_ctx(rank, world);
            G3Env g3env;
            REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 3);
            CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 3);
            const RecOp* f = find_op(b, "gemm_nt_q_gated");
            REQUIRE(f != nullptr);
            CHECK_EQ(pint(f, "N"), world == 1 ? 512 : 256);
            CHECK_EQ(pint(f, "K"), 256);
            every_weight_read(b);
            steps_clean_at(b, rank, world);
        }
    }
}

/* ---- P-01 (#71): the shared expert beside the routed GEMMs (MoeFP8's `side`, SharedSide). */
namespace {

rad_buf buf_h(const RadBuilder& b, const char* name) {
    const RecBuf* r = find_b(b, name);
    return r ? (rad_buf)(r - b.bufs.data() + 1) : 0;
}
bool names_w(const RadBuilder& b, const RecOp& o, const std::string& name) {
    for (rad_weight w : o.w)
        if (w && w <= b.weights.size() && b.weights[w - 1].name == name) return true;
    return false;
}
bool has_buf(const std::vector<rad_buf>& v, rad_buf h) {
    return h && std::find(v.begin(), v.end(), h) != v.end();
}

/* Where one MoE layer's launches sit in a pass: the routed gate_up, its quantiser, the routed
 * down, the gather, latent_out, and the shared arm's gate_up (or the fold, which names the same
 * weight), gate and down. -1 when absent: no gate behind the fold (K-23), which writes the codes
 * down reads and no gate_up product. */
struct MoeOrder {
    int rgu = -1, rq = -1, rdn = -1, gather = -1, lat_out = -1, sgu = -1, sact = -1, sdn = -1;
};
MoeOrder moe_order(const RadBuilder& b, const RadCtx& c, int l) {
    MoeOrder m;
    char lin[64], lout[64], sgu[64], sdn[64];
    std::snprintf(lin, sizeof lin, "blk.%d.ffn_latent_in.weight", l);
    std::snprintf(lout, sizeof lout, "blk.%d.ffn_latent_out.weight", l);
    std::snprintf(sgu, sizeof sgu, "blk.%d.ffn_gate_up_shexp.weight", l);
    std::snprintf(sdn, sizeof sdn, "blk.%d.ffn_down_shexp.weight", l);
    const rad_buf egu = buf_h(b, "moe.expert_gate_up"), edn = buf_h(b, "moe.expert_down"),
                  sgu_b = buf_h(b, "moe.shared_gate_up");
    int start = -1;
    for (int i = 0; i < (int)c.issues.size(); ++i) {
        const RecOp* o = b.op_of(c.issues[(size_t)i].op);
        if (!o) continue;
        if (start < 0) {
            if (names_w(b, *o, lin)) start = i;
            continue;
        }
        if (m.rgu < 0 && has_buf(o->writes, egu)) m.rgu = i;
        else if (m.rgu >= 0 && m.rq < 0 && has_buf(o->reads, egu)) m.rq = i;
        if (m.rdn < 0 && has_buf(o->writes, edn)) m.rdn = i;
        else if (m.rdn >= 0 && m.gather < 0 && has_buf(o->reads, edn)) m.gather = i;
        if (m.sgu < 0 && names_w(b, *o, sgu)) m.sgu = i;
        else if (m.sgu >= 0 && m.sact < 0 && has_buf(o->reads, sgu_b)) m.sact = i;
        if (m.sdn < 0 && names_w(b, *o, sdn)) m.sdn = i;
        if (m.lat_out < 0 && names_w(b, *o, lout)) m.lat_out = i;
        if (m.lat_out >= 0 && m.sdn >= 0) break;
    }
    return m;
}

/* The side's places in one MoE layer's pass (SharedSide): stage 0 -- the fold where it serves the
 * rows, gate_up otherwise -- behind the routed gate_up (in front of it where MoeFP8 puts its own
 * first stage, `first`); the gate behind the routed quantiser, none after the fold (stage 1 then
 * issues nothing); down behind the routed down; all before the gather. */
void side_order(const MoeOrder& m, bool fold, bool first = false) {
    REQUIRE(m.rgu >= 0 && m.rq > m.rgu && m.rdn > m.rq && m.gather > m.rdn);
    REQUIRE(m.sgu >= 0 && m.sdn > m.sgu);
    if (first) CHECK(m.sgu < m.rgu);
    else CHECK(m.sgu > m.rgu && m.sgu < m.rq);
    if (fold) {
        CHECK_EQ(m.sact, -1);
    } else {
        REQUIRE(m.sact > m.sgu && m.sdn > m.sact);
        CHECK(m.sact > m.rq && m.sact < m.rdn);
    }
    CHECK(m.sdn > m.rdn && m.sdn < m.gather);
}

/* One launch as its op, rows, lane and operands (names, offsets, extents): two steps issuing the
 * same launches in another order give the same sorted list. */
std::vector<std::string> launches(const RadBuilder& b, const RadCtx& c) {
    std::vector<std::string> v;
    for (const RecIssue& r : c.issues) {
        const RecOp* o = b.op_of(r.op);
        std::string s = o ? o->op : "<null>";
        s += "|" + std::to_string(r.n) + "|" + std::to_string(r.lane);
        for (const RadOperand& d : r.opd) {
            s += "|" + std::to_string(d.kind) + ":" + std::to_string(d.offset) + ":" +
                 std::to_string(d.rows) + ":" + std::to_string(d.cols) + ":";
            if (d.kind == RAD_OPK_BUF && d.handle && d.handle <= b.bufs.size())
                s += b.bufs[d.handle - 1].name;
            else if ((d.kind == RAD_OPK_WEIGHT || d.kind == RAD_OPK_WTAB) && d.handle &&
                     d.handle <= b.weights.size())
                s += b.weights[d.handle - 1].name;
        }
        v.push_back(s);
    }
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

/* With the side on, each MoE layer's shared arm is issued inside MoeFP8's pass where MoeFP8 puts
 * its own: gate_up behind the routed gate_up, the gate behind the routed quantiser, down behind
 * the routed down, all before the gather and latent_out; its written buffers are concurrent. Off
 * (RADIANCE_HY4_SHARED_SIDE=0), or where a shared linear quantises for itself (a CPU engine: no
 * gated_had_quant_i8g), the arm follows latent_out as before and only the pipeline's marks stand.
 * Either way the same launches with the same operands (order only), every step kind lint clean,
 * at one, two and four ranks. */
TEST(the_shared_expert_rides_the_routed_pass) {
    struct Run { const char* what; void (*encs)(RadBuilder*); bool off, on; std::vector<int> worlds; };
    const Run runs[] = {
        { "bf16", bf16_encs, false, true, { 1, 2, 4 } },
        { "int8 device", int8_trunk_encs, false, true, { 1 } },
        { "int8 host", int8_trunk_encs_host, false, false, { 1 } },
        { "bf16, side off", bf16_encs, true, false, { 1, 2, 4 } },
    };
    struct Case { std::vector<int> dq, pq; int32_t ctx; };
    const Case cases[] = { { { 1, 1, 1 }, {}, 0 }, { {}, { 12 }, 0 }, { {}, { 40 }, 0 },
                           { { 1 }, { 10 }, 0 }, { {}, { 16 }, 16 } };
    for (const Run& r : runs)
        for (int world : r.worlds)
            for (int rank = 0; rank < world; ++rank) {
                fprintf(stderr, "run %s, rank %d of %d\n", r.what, rank, world);
                if (r.off) setenv("RADIANCE_HY4_SHARED_SIDE", "0", 1);
                Declared d(r.encs, world, rank, world > 1);   /* four ranks need the wide heads */
                unsetenv("RADIANCE_HY4_SHARED_SIDE");
                REQUIRE_EQ(d.rc, RAD_OK);
                RadBuilder& b = d.b;
                const rad_buf sgu = buf_h(b, "moe.shared_gate_up");
                REQUIRE(sgu != 0);
                CHECK_EQ(has_buf(b.concurrent, sgu), r.on);
                if (r.on) {
                    CHECK(has_buf(b.concurrent, buf_h(b, "moe.shared_out")));
                    const rad_buf gq = buf_h(b, "trunk_codes.gated.128.128.q8");
                    if (gq) CHECK(has_buf(b.concurrent, gq));
                }
                for (const Case& cs : cases) {
                    auto k = kit(b, cs.dq, cs.pq, cs.ctx);
                    k->b.positions_h = k->pos.data();
                    auto v = run_passes(b, hyv4flash::step, { k.get() }, rank, world);
                    CHECK_PASSES(b, v);
                    REQUIRE(!v.empty());
                    for (int l : { 1, 2 }) {
                        const MoeOrder m = moe_order(b, *v[0], l);
                        REQUIRE(m.rgu >= 0 && m.rq > m.rgu && m.rdn > m.rq && m.gather > m.rdn &&
                                m.lat_out > m.gather);
                        REQUIRE(m.sgu >= 0 && m.sact > m.sgu && m.sdn > m.sact);
                        if (r.on) side_order(m, false);
                        else CHECK(m.sgu > m.lat_out);
                    }
                }
            }
    /* Order only: the side on and off issue the same launches, operand for operand, on every
     * step kind and rank. */
    struct Got { std::vector<std::vector<std::string>> sorted, order; };
    /* The steps of a declare, run before the next declare (the step serves the last model). */
    auto steps_of = [&](bool side, int world, int rank) {
        if (!side) setenv("RADIANCE_HY4_SHARED_SIDE", "0", 1);
        Declared d(bf16_encs, world, rank, world > 1);
        unsetenv("RADIANCE_HY4_SHARED_SIDE");
        Got g;
        CHECK_EQ(d.rc, RAD_OK);
        if (d.rc != RAD_OK) return g;
        for (const Case& cs : cases) {
            auto k = kit(d.b, cs.dq, cs.pq, cs.ctx);
            k->b.positions_h = k->pos.data();
            auto v = run_passes(d.b, hyv4flash::step, { k.get() }, rank, world);
            for (const auto& c : v) {
                g.sorted.push_back(launches(d.b, *c));
                g.order.push_back(issued_ops(d.b, *c));
            }
        }
        return g;
    };
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            const Got on = steps_of(true, world, rank), off = steps_of(false, world, rank);
            REQUIRE_EQ(on.sorted.size(), off.sorted.size());
            int moved = 0;
            for (size_t p = 0; p < on.sorted.size(); ++p) {
                CHECK(on.sorted[p] == off.sorted[p]);
                moved += on.order[p] != off.order[p];
            }
            /* ...in another order: every step kind's trunk pass. */
            CHECK_EQ(moved, (int)(sizeof cases / sizeof cases[0]));
        }
}

/* Behind a trellis routed gate_up from MoeFP8::kSharedFirstRows routed rows, MoeFP8 issues its own
 * shared arm's first stage behind the router, in front of that GEMM (compute-bound there); the
 * side's stage 0 goes there too, and below that height behind the GEMM. The gate and down keep
 * their places. */
TEST(the_shared_sides_first_stage_moves_with_moefp8s_own) {
    auto iqr = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = b->encs[1].second = enc_trellis(RAD_U4, true);
    };
    Declared d(+iqr, 1, 0);
    REQUIRE_EQ(d.rc, RAD_OK);
    /* top_k 2: 12 rows are 24 routed rows, 64 are 128. */
    for (int rows : { 12, 64 }) {
        auto k = kit(d.b, {}, { rows }, 0);
        k->b.positions_h = k->pos.data();
        auto v = run_passes(d.b, hyv4flash::step, { k.get() }, 0, 1);
        CHECK_PASSES(d.b, v);
        REQUIRE(!v.empty());
        for (int l : { 1, 2 }) {
            const MoeOrder m = moe_order(d.b, *v[0], l);
            side_order(m, false, rows * 2 >= 100);
        }
    }
}

/* K-23 under the side: where the shared gate_up is an IQR trellis fed by the latent linear's
 * rotated int8 codes and its down is rotated int8, the arm's gate_up and gate are one launch for
 * the rows the fold serves (M <= 16): the side's stage 0 issues the fold, stage 1 nothing, stage 2
 * down -- two launches where the pair's arm is three. The fold writes the gate's code pair (concurrent with
 * the rest of the arm's buffers) and quantises nothing (its input codes are fed), so the arm still
 * writes only its own buffers and rides the pass. Past 16 rows the pair runs as three stages. On
 * and off (RADIANCE_HY4_SHARED_SIDE=0) issue the same launches with the same operands, in another
 * order, at decode, verify and prefill heights, and every step kind is lint clean. */
void iqr_side_encs(RadBuilder* b) {
    iqr_ffn_encs(b);
    b->encs.insert(b->encs.begin(), { { "ffn_latent_in", enc_i8(true) } });
}
TEST(the_shared_side_issues_the_fold_as_its_first_stage) {
    struct Case { std::vector<int> dq, pq; int32_t ctx; bool fold; };
    /* three decode rows; one decode beside a 10-row verify; a 12-row prefill; 40 rows (the pair). */
    const Case cases[] = { { { 1 }, {}, 0, true }, { { 1, 1, 1 }, {}, 0, true },
                           { { 1 }, { 10 }, 0, true }, { {}, { 12 }, 0, true },
                           { {}, { 16 }, 16, true }, { {}, { 40 }, 0, false } };
    struct Got { std::vector<std::vector<std::string>> sorted, order; };
    /* REQUIRE returns from the lambda: it fills `g` rather than returning it. */
    auto run = [&](bool side, Got& g) {
        MetaKV mk = mini_meta(false, 128, 256);
        RadBuilder b;
        iqr_side_encs(&b);
        RadBuildCtx ctx = build_ctx();
        G3Env g3env;
        if (!side) setenv("RADIANCE_HY4_SHARED_SIDE", "0", 1);
        const int rc = hyv4flash::declare(&b, mk.meta(), &ctx);
        unsetenv("RADIANCE_HY4_SHARED_SIDE");
        REQUIRE_EQ(rc, RAD_OK);
        CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 3);   /* layer 0's MLP and two shared experts */
        const rad_buf sgu = buf_h(b, "moe.shared_gate_up");
        REQUIRE(sgu != 0);
        CHECK_EQ(has_buf(b.concurrent, sgu), side);
        /* The shared experts' folds (layers 1 and 2) write only buffers the side marked. */
        for (int i = 1; side && i < 3; ++i) {
            const RecOp* f = find_op(b, "gemm_nt_q_gated", i);
            REQUIRE(f != nullptr);
            for (rad_buf w : f->writes) CHECK(has_buf(b.concurrent, w));
        }
        for (const Case& cs : cases) {
            auto k = kit(b, cs.dq, cs.pq, cs.ctx);
            k->b.positions_h = k->pos.data();
            auto v = run_passes(b, hyv4flash::step, { k.get() }, 0, 1);
            CHECK_PASSES(b, v);
            REQUIRE(!v.empty());
            const RadCtx& c = *v[0];
            if (cs.fold) {
                CHECK_EQ(count_issued(b, c, "gemm_nt_q_gated"), 3);
                CHECK_EQ(count_issued(b, c, "gated_had_quant_i8g"), 0);
            }
            for (int l : { 1, 2 }) {
                const MoeOrder m = moe_order(b, c, l);
                REQUIRE(m.sgu >= 0);
                const RecOp* o = b.op_of(c.issues[(size_t)m.sgu].op);
                REQUIRE(o != nullptr);
                CHECK_EQ(o->op == "gemm_nt_q_gated", cs.fold);
                if (side) side_order(m, cs.fold);
                else CHECK(m.sgu > m.lat_out);
            }
            for (const auto& p : v) {
                g.sorted.push_back(launches(b, *p));
                g.order.push_back(issued_ops(b, *p));
            }
        }
        every_weight_read(b);
        steps_clean_at(b, 0, 1);
    };
    Got on, off;
    run(true, on);
    run(false, off);
    REQUIRE_EQ(on.sorted.size(), off.sorted.size());
    int moved = 0;
    for (size_t p = 0; p < on.sorted.size(); ++p) {
        CHECK(on.sorted[p] == off.sorted[p]);
        moved += on.order[p] != off.order[p];
    }
    CHECK_EQ(moved, (int)(sizeof cases / sizeof cases[0]));
}

/* K-02 (P-14's switch): q_nope and q_rope read one q residual's codes, so where both are one IQR
 * form the plugin gives the pair one gemm_nt_q_dual a layer, at a decode step's GEMV only (M == 1,
 * kDualMaxM) -- the two launches' blocks in one grid, each part's blocks the blocks its own launch
 * gives it, so both products are byte-identical (the row's oracle). Off with
 * RADIANCE_HY4_FUSE_QPROJ=0, and never for a pair of two forms nor for two runs of the trellis --
 * the run is the fold's length and the pair folds once. The pair's two Ns differ (the mini model's
 * q_nope is H * qk_nope_head_dim wide, its q_rope H * qk_rope_head_dim), which is what the row's
 * N2 is for. On an engine whose library does not declare the row the pair keeps its two launches,
 * as on the switch's off arm: the fallback is the plain kernel's launches, not a refused declare.
 *
 * Which parts a block holds is the row's (`dual_block_rows`): a part is declared only where its own
 * launch's blocks cover its N whole -- sixteen rows below 4096 columns, thirty-two from there up.
 * The runs below reach each answer from both sides, since a part declared in blocks too wide is a
 * fusion lost (the two launches, silently) and one declared in blocks too narrow is a launch the
 * row refuses mid-chain: `q_nope` 48 and 4080 are whole in sixteen rows and not in thirty-two,
 * `q_nope` 4112 is whole in sixteen and not in thirty-two with the wide launch's block, and
 * `q_nope` 4096 is the edge itself. */
TEST(the_q_projections_go_out_as_one_launch) {
    /* `nope` and `swa_nope` are the two layer kinds' qk_nope_head_dim; the sparse layers carry four
     * heads and the sliding ones eight, so a part's N is heads * that. */
    struct Run { const char* what; bool same, off, host; uint32_t nope_run, rope_run;
                 int64_t nope, swa_nope; int duals; };
    const Run runs[] = {
        { "one form, one run", true, false, false, 16, 16, 24, 24, 3 },
        { "one form, no dual row", true, false, true, 16, 16, 24, 24, 0 },
        { "one form, fused off", true, true, false, 16, 16, 24, 24, 0 },
        { "two forms", false, false, false, 16, 16, 24, 24, 0 },
        { "two runs", true, false, false, 16, 32, 24, 24, 0 },
        { "a part of three sixteen-row blocks", true, false, false, 16, 16, 12, 12, 3 },
        { "a part four thousand and eighty wide", true, false, false, 16, 16, 512, 510, 3 },
        { "a part the wide block leaves a strip of", true, false, false, 16, 16, 520, 514, 2 },
        { "a part at the wide launch's edge", true, false, false, 16, 16, 512, 512, 3 },
    };
    for (const Run& r : runs) {
        fprintf(stderr, "run %s\n", r.what);
        /* G3Env leaves a switch alone when the test has already set it: the on arm asks for the pair
         * itself, the off arm for the two launches. */
        setenv("RADIANCE_HY4_FUSE_QPROJ", r.off ? "0" : "1", 1);
        RadBuilder b;
        bf16_encs(&b);
        b.encs.insert(b.encs.begin(),
                      { { "attn_q_nope", enc_trellis(RAD_U5, true, r.nope_run) },
                        { "attn_q_rope", r.same ? enc_trellis(RAD_U5, true, r.rope_run)
                                                : enc_i8(true) } });
        /* A CPU engine: no device-only fused row, this file's idiom for one. */
        if (r.host) b.undeclared.insert("gemm_nt_q_dual");
        /* K is the q_lora rank: a trellis weight takes K in whole 128-column runs of the table. */
        MetaKV mk = mini_meta(/* wide */ true);
        mk.set("q_lora_rank", "128");
        mk.set("qk_nope_head_dim", std::to_string(r.nope).c_str());
        mk.set("swa_qk_nope_head_dim", std::to_string(r.swa_nope).c_str());
        /* The pair's widths a layer kind, as the plugin reads them off the heads: four sparse, eight
         * sliding, and q_rope is H * qk_rope_head_dim (8) either way. */
        const int64_t nope_n[2] = { 4 * r.nope, 8 * r.swa_nope };
        const int64_t rope_n[2] = { 32, 64 };
        auto is_pair_op = [&](const RecOp* o) {
            if (o->op != "gemm_nt_q" || pint(o, "K") != 128) return false;
            for (int i = 0; i < 2; ++i)
                if (pint(o, "N") == nope_n[i] || pint(o, "N") == rope_n[i]) return true;
            return false;
        };
        RadBuildCtx ctx = build_ctx();
        G3Env g3env;
        int rc = 0;
        const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        unsetenv("RADIANCE_HY4_FUSE_QPROJ");
        if (rc != RAD_OK) fputs(err.c_str(), stderr);
        REQUIRE_EQ(rc, RAD_OK);
        /* One dual a layer where the pair is one form and one run, and none otherwise. */
        CHECK_EQ(count_op(b, "gemm_nt_q_dual"), r.duals);
        /* No op reaches rad_decl_op that the library does not carry: the pair without the row
         * declares its two Lins and nothing else, where asking would fail the declare. */
        CHECK_EQ(join(b.asked_undeclared), std::string());
        /* Both own GEMMs stay declared: the pair serves the GEMV's band and every wider pass keeps
         * the Lins' launches, so neither weight loses its launch. */
        int own = 0;
        for (const auto& o : b.ops)
            if (is_pair_op(&o)) {
                ++own;
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK(m->ihi > 1);
            }
        CHECK_EQ(own, 6);
        /* The dual carries one of the two layer kinds' widths and the one K, and is declared at the
         * GEMV's height; each declared pair is one whose parts are whole in the blocks it takes. */
        for (const auto& o : b.ops)
            if (o.op == "gemm_nt_q_dual") {
                const int64_t N = pint(&o, "N"), N2 = pint(&o, "N2");
                int kind = -1;
                for (int i = 0; i < 2; ++i)
                    if (N == nope_n[i] && N2 == rope_n[i]) kind = i;
                CHECK(kind >= 0);
                const int64_t rows = N >= 4096 ? 32 : 16;
                CHECK_EQ(N % rows, 0);
                CHECK_EQ(N2 % rows, 0);
                CHECK_EQ(pint(&o, "K"), 128);
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, 1);
            }
        every_weight_read(b);
        steps_clean_at(b, 0, 1);
        /* A one-token decode step: the pair's launch where it is declared, and the two Lins' own
         * launches (both parts, three layers) everywhere else -- the switch off, a form or a run
         * apart, a library with no such row, a part whose N the row's blocks do not cover. */
        {
            auto k = kit(b, { 1 }, {}, 0);
            k->b.positions_h = k->pos.data();
            auto v = run_passes(b, hyv4flash::step, { k.get() }, 0, 1);
            CHECK_PASSES(b, v);
            auto issued = [&](bool dual) {
                int n = 0;
                for (const auto& i : v[0]->issues) {
                    const RecOp* o = b.op_of(i.op);
                    if (!o) continue;
                    if (dual) n += o->op == "gemm_nt_q_dual";
                    else n += is_pair_op(o);
                }
                return n;
            };
            CHECK_EQ(issued(true), r.duals);
            /* Two parts a layer, three layers (sparse, sliding, sparse). */
            CHECK_EQ(issued(false), 2 * (3 - r.duals));
        }
    }
}

/* K-03: q_a and the gate read one set of codes -- the attention input's, which q_a owns and the
 * gate is fed -- and both are GEMVs at a decode step, which is what a pair's launch needs. The
 * gate is off the critical path until `mla_expand_gate` reads it, so where their encodings agree
 * the pair goes out in one `gemm_nt_q_dual` a layer and each Lin keeps its own launch for the wider
 * passes. Off with RADIANCE_HY4_FUSE_QA_GATE=0; a library with no such row keeps the two launches,
 * as the off arm does. Never for two forms (`tcq_code=mad` and `tcq=16` are the fold's), nor for
 * two runs of the trellis (K-01's latent_in / gate_up_shexp pair differs exactly there: the run is
 * the fold's length and the pair folds once). The pair's widths are the model's: q_a is the q_lora
 * rank (replicated, 1024) and the gate its layer kind's heads * v (4096 on a sparse layer at TP2,
 * 8192 on a sliding one), so the pair is an eight-wave part and a four-wave one -- the mix the
 * row's block rule covers (sixteen rows below 4096 columns, thirty-two from there up). */
TEST(the_gate_and_q_a_go_out_as_one_launch) {
    struct Run { const char* what; bool same, off, host, swa_i8; uint32_t qa_run, gate_run;
                 int64_t q_lora, v; int duals; };
    const Run runs[] = {
        /* q_a 1024 (eight waves), the gate 4096 / 8192 (four): one form, one run. */
        { "one form, one run", true, false, false, false, 256, 256, 1024, 1024, 3 },
        { "one form, no dual row", true, false, true, false, 256, 256, 1024, 1024, 0 },
        { "one form, fused off", true, true, false, false, 256, 256, 1024, 1024, 0 },
        { "two forms", false, false, false, false, 256, 256, 1024, 1024, 0 },
        { "two runs", true, false, false, false, 128, 256, 1024, 1024, 0 },
        { "the sliding layer's gate of another form", true, false, false, true, 256, 256, 1024, 1024,
          2 },
        { "a gate below the wide launch", true, false, false, false, 256, 256, 1024, 512, 3 },
        /* The sparse kind's gate 16-aligned but not 32 (4112): the row's wide block cannot take it,
         * so that kind keeps its two launches and the sliding layer's pair goes out. */
        { "a gate part the wide block leaves a strip of", true, false, false, false, 256, 256, 1024,
          1028, 1 },
    };
    for (const Run& r : runs) {
        fprintf(stderr, "run %s\n", r.what);
        /* G3Env leaves a switch alone when the test has already set it: the on arm asks for the pair
         * itself, the off arm for the two launches. */
        setenv("RADIANCE_HY4_FUSE_QA_GATE", r.off ? "0" : "1", 1);
        RadBuilder b;
        bf16_encs(&b);
        b.encs.insert(b.encs.begin(),
                      { { "attn_q_a.weight", enc_trellis(RAD_U5, true, r.qa_run) },
                        { "attn_gate.weight", r.same ? enc_trellis(RAD_U5, true, r.gate_run)
                                                     : enc_i8(true) } });
        /* One layer's gate of another form, in front of the whole-model entry. */
        if (r.swa_i8)
            b.encs.insert(b.encs.begin(), { { "blk.1.attn_gate.weight", enc_i8(true) } });
        /* A CPU engine: no device-only fused row, this file's idiom for one. */
        if (r.host) b.undeclared.insert("gemm_nt_q_dual");
        /* K is the embedding width for both parts: a trellis weight takes K in whole 128-column
         * runs of the table, and the gate's K is the attention input's. */
        MetaKV mk = mini_meta(/* wide */ true);
        mk.set("q_lora_rank", std::to_string(r.q_lora).c_str());
        mk.set("v_head_dim", std::to_string(r.v).c_str());
        mk.set("swa_v_head_dim", std::to_string(r.v).c_str());
        /* q_a is one width on every layer kind (replicated); the gate is its kind's heads * v:
         * four sparse heads, eight sliding. */
        const int64_t gate_n[2] = { 4 * r.v, 8 * r.v };
        auto is_pair_op = [&](const RecOp* o) {
            if (o->op != "gemm_nt_q" || pint(o, "K") != 128) return false;
            if (pint(o, "N") == r.q_lora) return true;
            for (int i = 0; i < 2; ++i)
                if (pint(o, "N") == gate_n[i]) return true;
            return false;
        };
        RadBuildCtx ctx = build_ctx();
        G3Env g3env;
        int rc = 0;
        const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        unsetenv("RADIANCE_HY4_FUSE_QA_GATE");
        if (rc != RAD_OK) fputs(err.c_str(), stderr);
        REQUIRE_EQ(rc, RAD_OK);
        /* One dual a layer where the pair is one form and one run, and none otherwise. */
        CHECK_EQ(count_op(b, "gemm_nt_q_dual"), r.duals);
        /* No op reaches rad_decl_op that the library does not carry: the pair without the row
         * declares its two Lins and nothing else, where asking would fail the declare. */
        CHECK_EQ(join(b.asked_undeclared), std::string());
        /* Both own GEMMs stay declared: the pair serves the GEMV's band and every wider pass keeps
         * the Lins' launches, so neither weight loses its launch. */
        int own = 0;
        for (const auto& o : b.ops)
            if (is_pair_op(&o)) {
                ++own;
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK(m->ihi > 1);
            }
        CHECK_EQ(own, 6);
        /* The dual carries q_a's width first (it owns the codes and its launch is the earlier one)
         * and one of the two layer kinds' gates second, at the GEMV's height; each declared pair is
         * one whose parts are whole in the blocks the row takes. */
        for (const auto& o : b.ops)
            if (o.op == "gemm_nt_q_dual") {
                const int64_t N = pint(&o, "N"), N2 = pint(&o, "N2");
                CHECK_EQ(N, r.q_lora);
                CHECK(N2 == gate_n[0] || N2 == gate_n[1]);
                const int64_t rows = N >= 4096 ? 32 : 16, rows2 = N2 >= 4096 ? 32 : 16;
                CHECK_EQ(N % rows, 0);
                CHECK_EQ(N2 % rows2, 0);
                CHECK_EQ(pint(&o, "K"), 128);
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, 1);
            }
        every_weight_read(b);
        steps_clean_at(b, 0, 1);
        /* A one-token decode step: the pair's launch where it is declared, and the two Lins' own
         * launches (both parts, three layers) everywhere else -- the switch off, a form or a run
         * apart, a library with no such row, a part whose N the row's blocks do not cover. */
        {
            auto k = kit(b, { 1 }, {}, 0);
            k->b.positions_h = k->pos.data();
            auto v = run_passes(b, hyv4flash::step, { k.get() }, 0, 1);
            CHECK_PASSES(b, v);
            auto issued = [&](bool dual) {
                int n = 0;
                for (const auto& i : v[0]->issues) {
                    const RecOp* o = b.op_of(i.op);
                    if (!o) continue;
                    if (dual) n += o->op == "gemm_nt_q_dual";
                    else n += is_pair_op(o);
                }
                return n;
            };
            CHECK_EQ(issued(true), r.duals);
            /* Two parts a layer, three layers (sparse, sliding, sparse). */
            CHECK_EQ(issued(false), 2 * (3 - r.duals));
        }
        /* A two-token pass: the pair's height is the single row, so both Lins keep their launches
         * whatever the switch says, and the fused arms step exactly as the off arm does. */
        {
            auto k = kit(b, { 2 }, {}, 0);
            k->b.positions_h = k->pos.data();
            auto v = run_passes(b, hyv4flash::step, { k.get() }, 0, 1);
            CHECK_PASSES(b, v);
            int dual = 0, own2 = 0;
            for (const auto& i : v[0]->issues) {
                const RecOp* o = b.op_of(i.op);
                if (!o) continue;
                if (o->op == "gemm_nt_q_dual") ++dual;
                else if (is_pair_op(o)) ++own2;
            }
            CHECK_EQ(dual, 0);
            CHECK_EQ(own2, 6);
        }
    }
}

/* The handle a declared buffer was given, for a check on which of two pairs an op took: a rad_buf
 * is the buffer's index in RadBuilder::bufs, from 1. */
static rad_buf buf_handle(const RadBuilder& b, const std::string& name) {
    for (size_t k = 0; k < b.bufs.size(); ++k)
        if (b.bufs[k].name == name) return (rad_buf)(k + 1);
    return 0;
}

/* K-23 where the gate_up's input codes are fed: a recipe whose latent linear is rotated int8 opens
 * the MoE layer's input, so the trunk's junction quantises the feed-forward's input once (G-3b) and
 * the shared expert's gate_up reads that pair (Lin::feed) with no quantiser of its own -- the shape
 * a served u3/IQR build takes, layer 0's dense MLP being the only gate_up with nothing in front of
 * it. The fold reads the pair the layer was given, and nothing writes that pair a second time. */
void iqr_fed_ffn_encs(RadBuilder* b) {
    iqr_ffn_encs(b);
    b->encs.insert(b->encs.begin(), { { "ffn_latent_in", enc_i8(true) } });
}
TEST(the_gate_fold_reads_the_codes_a_fed_gate_up_is_given) {
    MetaKV mk = mini_meta(false, 128, 256);
    RadBuilder b;
    iqr_fed_ffn_encs(&b);
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 3);
    CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 3);        /* the pair, for the rows past 16 */
    /* Layer 0's dense MLP quantises its own input into the untagged trunk pair; the MoE layers'
     * feed-forward input is the junction's pair, which their latent linear and shared gate_up share. */
    const rad_buf own = buf_handle(b, "trunk_codes.256.128.q8");
    const rad_buf own_s = buf_handle(b, "trunk_codes.256.128.s8");
    const rad_buf fed = buf_handle(b, "trunk_codes.junc_ffn_in.256.128.q8");
    const rad_buf fed_s = buf_handle(b, "trunk_codes.junc_ffn_in.256.128.s8");
    REQUIRE(own != 0);
    REQUIRE(own_s != 0);
    REQUIRE(fed != 0);
    REQUIRE(fed_s != 0);
    for (int i = 0; i < 3; ++i) {
        const RecOp* f = find_op(b, "gemm_nt_q_gated", i);
        REQUIRE(f != nullptr);
        REQUIRE_EQ(f->reads.size(), (size_t)2);
        REQUIRE_EQ(f->writes.size(), (size_t)2);
        CHECK_EQ(pstr(f, "dtype"), std::string("i8a8h"));
        /* the pair writes the same two code buffers the fold does */
        const RecOp* a = find_op(b, "gated_had_quant_i8g", i);
        REQUIRE(a != nullptr);
        CHECK(a->writes == f->writes);
        /* layer 0's fold reads the codes its own quantiser writes; a shared expert's fold reads the
         * fed pair, and neither of the two ever reads the other's. */
        CHECK_EQ(f->reads[0], i == 0 ? own : fed);
        CHECK_EQ(f->reads[1], i == 0 ? own_s : fed_s);
        CHECK(!in(f->reads, i == 0 ? fed : own));
    }
    /* The fed pair's writers are the trunk's junction quantisers, one a MoE layer: no gate_up
     * quantises it a second time for the fold to read instead. */
    int writers = 0;
    for (const auto& o : b.ops)
        if (in(o.writes, fed)) { ++writers; CHECK_EQ(o.op, std::string("rmsnorm_ihc_junction")); }
    CHECK_EQ(writers, 2);
    CHECK_EQ(count_op(b, "had_quant_act_i8g"), 1);          /* layer 0's gate_up, the only one */
    every_weight_read(b);
    steps_clean_at(b, 0, 1);
    /* A decode-class step issues the fold and no pair, and the only input quantiser it issues is
     * layer 0's gate_up: a fed gate_up leaves the codes to the trunk. */
    for (int rows : { 1, 16 }) {
        std::unique_ptr<Kit> k = kit(b, { rows }, {});
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_EQ(count_issued(b, c, "gemm_nt_q_gated"), 3);
        CHECK_EQ(count_issued(b, c, "gated_had_quant_i8g"), 0);
        CHECK_EQ(count_issued(b, c, "had_quant_act_i8g"), 1);
    }
    /* Past the rows the fold serves, the pair runs and writes the same codes. */
    std::unique_ptr<Kit> pf = kit(b, {}, { 17 });
    pf->b.positions_h = pf->pos.data();
    RadCtx pc{ .b = &b, .batch = &pf->b };
    hyv4flash::step(&pc, &pf->b);
    CHECK(pc.step_fail.empty());
    CHECK_EQ(count_issued(b, pc, "gemm_nt_q_gated"), 0);
    CHECK_EQ(count_issued(b, pc, "gated_had_quant_i8g"), 3);
    every_weight_read(b);
    steps_clean_at(b, 0, 1);
}

/* q_nope and q_rope rotated int8 of one form at a q_lora of whole 128 groups: the q residual's norm
 * writes their codes in its pass (rmsnorm_had_quant_i8g) from q_a's raw output, so neither has a
 * quantiser; on a CPU engine q_nope quantises the normed residual once and q_rope reads its codes. */
void q_b_i8_encs(RadBuilder* b) {
    bf16_encs(b);
    b->encs.insert(b->encs.begin(), { { "attn_q_nope", enc_i8(true) }, { "attn_q_rope", enc_i8(true) } });
}
void q_b_i8_encs_host(RadBuilder* b) {
    q_b_i8_encs(b);
    b->undeclared.insert("rmsnorm_had_quant_i8g");
}
TEST(the_q_residuals_norm_writes_q_bs_rotated_codes) {
    for (int host = 0; host < 2; ++host) {
        MetaKV mk = mini_meta();
        mk.set("q_lora_rank", "128");
        RadBuilder b;
        (host ? q_b_i8_encs_host : q_b_i8_encs)(&b);
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), host ? 0 : 3);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), host ? 3 : 0);
        CHECK_EQ(find_b(b, "sparse.q_a_raw") != nullptr, !host);
        CHECK_EQ(find_b(b, "sliding.q_a_raw") != nullptr, !host);
        int i8h = 0;
        for (const auto& o : b.ops) i8h += o.op == "gemm_nt_q" && pstr(&o, "dtype") == "i8a8h";
        CHECK_EQ(i8h, 6);
        every_weight_read(b);
        steps_clean_at(b, 0, 1);
    }
}

/* A sizing declare (RadBuildCtx::shape_probe) at a smaller step declares the same ops and buffers,
 * every buffer at most as tall and no other extent changed, and leaves the served model alone:
 * the real declare's steps still issue its own handles. The probe says so (shape_probe_ok). */
TEST(a_sizing_declare_matches_the_real_one_and_leaves_its_model_alone) {
    for (int i8 = 0; i8 < 2; ++i8) {
        Declared d(i8 ? int8_trunk_encs : bf16_encs);
        REQUIRE_EQ(d.rc, RAD_OK);
        RadBuilder pb;
        (i8 ? int8_trunk_encs : bf16_encs)(&pb);
        RadBuildCtx pc = build_ctx();
        pc.max_tok = 64;
        pc.shape_probe = 1;
        G3Env g3env;
        REQUIRE_EQ(hyv4flash::declare(&pb, d.mk.meta(), &pc), RAD_OK);
        REQUIRE_EQ(pb.ops.size(), d.b.ops.size());
        for (size_t k = 0; k < pb.ops.size(); ++k) CHECK_EQ(pb.ops[k].op, d.b.ops[k].op);
        REQUIRE_EQ(pb.bufs.size(), d.b.bufs.size());
        for (size_t k = 0; k < pb.bufs.size(); ++k) {
            const auto &q = pb.bufs[k].d, &f = d.b.bufs[k].d;
            CHECK_EQ(pb.bufs[k].name, d.b.bufs[k].name);
            CHECK_EQ(q.rank, f.rank);
            CHECK(q.shape[0] <= f.shape[0]);
            for (uint32_t x = 1; x < f.rank; ++x) CHECK_EQ(q.shape[x], f.shape[x]);
        }
        steps_clean_at(d.b, 0, 1);
    }
    RadArchProbe p{};
    CHECK_OK(hyv4flash::probe(mini_meta().meta(), &p));
    CHECK_EQ(p.shape_probe_ok, 1);
}

/* The same at more ranks, where the pipeline lives. With `RADIANCE_HY4_PIPE` on, a junction
 * declares the narrow twin its second lane issues, and every sizing declare at a smaller step has
 * to declare exactly the same ops -- that is how the arena sizes the levels it lends, and a level
 * whose declare differs is dropped ("the sizing declare failed"), which is what happened while the
 * twin was declared only above the pipeline's 1024-row floor: an 8192-token deployment lost its
 * 256- and 1024-token levels, so small steps could not lend their arena back to the slab and boot
 * held 1.9 GiB more than it needed. The row floor is the step's, not the graph's. */
TEST(a_sizing_declare_at_four_ranks_matches_with_the_pipeline_on_and_off) {
    for (int pipe = 0; pipe < 2; ++pipe) {
        if (pipe) setenv("RADIANCE_HY4_PIPE", "1", 1);
        else unsetenv("RADIANCE_HY4_PIPE");
        for (int world : { 1, 2, 4 }) {
            Declared d(bf16_encs, world, 0, true);
            REQUIRE_EQ(d.rc, RAD_OK);
            for (int64_t mt : { (int64_t)64, (int64_t)512, (int64_t)2048 }) {
                RadBuilder pb;
                bf16_encs(&pb);
                RadBuildCtx pc = build_ctx(0, world);
                pc.max_tok = mt;
                pc.shape_probe = 1;
                G3Env g3env;
                REQUIRE_EQ(hyv4flash::declare(&pb, d.mk.meta(), &pc), RAD_OK);
                REQUIRE_EQ(pb.ops.size(), d.b.ops.size());
                for (size_t k = 0; k < pb.ops.size(); ++k) CHECK_EQ(pb.ops[k].op, d.b.ops[k].op);
                for (size_t k = 0; k < pb.bufs.size(); ++k)
                    CHECK_EQ(pb.bufs[k].name, d.b.bufs[k].name);
            }
        }
        unsetenv("RADIANCE_HY4_PIPE");
    }
}

/* The Engram table as E4M3 rows under a bf16 or an f32 scale a row (hy4-flash-bf16.recipe,
 * hy4-flash-bf16-f32.recipe): the scale view at the stored dtype, no runtime round trip. */
TEST(the_engram_table_takes_a_bf16_or_an_f32_row_scale) {
    auto bf = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { { "engram_table", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 1, 0) },
                                          { "engram_mult", rad_enc_plain(RAD_I64) } });
    };
    auto f32 = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(), { { "engram_table", rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 0) },
                                          { "engram_mult", rad_enc_plain(RAD_I64) } });
    };
    for (auto encs : { +bf, +f32 }) {
        Declared d(encs);
        REQUIRE_EQ(d.rc, RAD_OK);
        const RecWeight* sc = find_w(d.b, "blk.1.engram_table.scale");
        REQUIRE(sc != nullptr);
        CHECK_EQ(sc->d.dtype, encs == +f32 ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16);
        CHECK_EQ(find_w(d.b, "blk.1.engram_table.weight")->d.group.layer, -1);
        CHECK_EQ(sc->d.group.layer, -1);
        CHECK_EQ(count_op(d.b, "dequant"), 0);
        every_weight_read(d.b);
        steps_clean_at(d.b, 0, 1);
    }
}

/* Quantised experts at two and four ranks are a slice of every expert's width (PLAN §1.5, 768 /
 * 384 at the real size; 256 / 128 here at a 512-wide expert): every rank declares all eight
 * experts, gate_up split by rows in its two parts and down by columns, the routed arm reduced
 * with the shared expert's; every step kind lint clean on every rank. */
TEST(quantised_experts_are_sliced_at_more_ranks) {
    auto nl = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = b->encs[1].second = enc_nl64(4);
    };
    auto i8 = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = b->encs[1].second = enc_i8(true);
    };
    for (auto encs : { +nl, +i8 })
        for (int world : { 2, 4 })
            for (int rank = 0; rank < world; ++rank) {
                Declared d(encs, world, rank, true, 512);
                REQUIRE_EQ(d.rc, RAD_OK);
                const RadBuilder& b = d.b;
                const int64_t F = 512 / world;
                for (int e = 0; e < 8; ++e) {
                    char n[64];
                    std::snprintf(n, sizeof n, "blk.2.ffn_gate_up_exps.%d.weight", e);
                    const RecWeight* gu = find_w(b, n);
                    REQUIRE(gu != nullptr);
                    CHECK_EQ(gu->d.shape[0], 2 * F);
                    CHECK_EQ(gu->d.shard, (int)RAD_SHARD_ROW);
                    CHECK_EQ(gu->d.n_row_parts, 2);
                    std::snprintf(n, sizeof n, "blk.2.ffn_down_exps.%d.weight", e);
                    const RecWeight* dn = find_w(b, n);
                    REQUIRE(dn != nullptr);
                    CHECK_EQ(dn->d.shape[1], F);
                    CHECK_EQ(dn->d.shard, (int)RAD_SHARD_COL);
                }
                for (const auto& o : b.ops)
                    if (o.op == "moe_gemm_q") CHECK_EQ(pint(&o, "n_expert"), 8);
                /* Six reductions and the narrow twin of each (see
             * each_rank_declares_its_share_and_reduces_twice_a_layer). */
            CHECK_EQ(count_op(b, "all_reduce"), 12);
                every_weight_read(d.b);
                steps_clean_at(d.b, rank, world);
            }
    /* mini's 128-wide expert does not split into whole 128-column groups. */
    CHECK_EQ(Declared(nl, 2, 0, true).rc, RAD_E_UNSUPPORTED);
}

/* A recipe that keeps a few experts bf16 beside quantised ones (the friend's hy4-flash-iqr-v4: 50
 * whole bf16 experts, the rest u4 trellis / w4nl64). The container stores every bf16 expert whole,
 * so at more ranks each is declared sharded at its share's extents like the quantised experts, for
 * the loader to cut (it was declared as an unsharded slice, which the loader refused as "the file
 * stores the whole expert"). */
TEST(bf16_experts_beside_quantised_ones_are_sliced_with_them) {
    auto mixed = [](RadBuilder* b) {
        bf16_encs(b);
        b->encs.clear();
        for (const char* e : { "3", "6" })
            for (const char* p : { "ffn_gate_up_exps", "ffn_down_exps" })
                b->encs.push_back({ std::string("blk.2.") + p + "." + e + ".",
                                    rad_enc_plain(RAD_BF16) });
        b->encs.push_back({ "ffn_gate_up_exps", enc_nl64(4) });
        b->encs.push_back({ "ffn_down_exps", enc_nl64(4) });
        b->kv_block_by_consumer = true;
    };
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            Declared d(+mixed, world, rank, true, 512);
            REQUIRE_EQ(d.rc, RAD_OK);
            const RadBuilder& b = d.b;
            const int64_t F = 512 / world;
            for (int e = 0; e < 8; ++e) {
                char n[64];
                std::snprintf(n, sizeof n, "blk.2.ffn_gate_up_exps.%d.weight", e);
                const RecWeight* gu = find_w(b, n);
                REQUIRE(gu != nullptr);
                CHECK_EQ(gu->d.shape[0], 2 * F);
                if (world > 1) {
                    CHECK_EQ(gu->d.shard, (int)RAD_SHARD_ROW);
                    CHECK_EQ(gu->d.n_row_parts, 2);
                }
                std::snprintf(n, sizeof n, "blk.2.ffn_down_exps.%d.weight", e);
                const RecWeight* dn = find_w(b, n);
                REQUIRE(dn != nullptr);
                CHECK_EQ(dn->d.shape[1], F);
                if (world > 1) CHECK_EQ(dn->d.shard, (int)RAD_SHARD_COL);
            }
            every_weight_read(d.b);
            steps_clean_at(d.b, rank, world);
        }
}

/* ================================================================== DFly2 (WP13.3) */
namespace {
/* mini with its drafter configured as ref/minicfg.py writes it: one draft layer over trunk layer
 * 2 (sparse) for both the cache and the selection, mask id V - 1, the selection width
 * group_topk * ratio. `two` is mini6d2's shape on mini's trunk: two draft layers, both over layer
 * 2 for the selection and layers 2 and 0 for the cache (0 is refused, so `two` uses 2 twice for
 * the selection and the test that wants distinct caches edits the keys). */
MetaKV dfly_meta(int layers = 1, bool wide = false, int64_t n_embd = 128) {
    MetaKV k = mini_meta(wide, 128, n_embd);
    k.set("dfly_num_hidden_layers", std::to_string(layers).c_str());
    k.set("dfly_target_kv_layers", layers == 1 ? "2" : "2 2");
    k.set("dfly_target_index_layer", layers == 1 ? "2" : "2 2");
    k.set("dfly_mask_token_id", "319");
    k.set("dfly_target_index_topk", "16");
    return k;
}
struct DeclDfly {
    MetaKV mk;
    RadBuilder b;
    RadBuildCtx ctx;
    int rc = 0;
    std::string err;
    explicit DeclDfly(int depth = 5, MetaKV m = dfly_meta(), int world = 1, int rank = 0,
                      void (*encs)(RadBuilder*) = bf16_encs)
        : mk(std::move(m)) {
        encs(&b);
        ctx = spec_ctx(rank, world, depth);
        err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        if (rc != RAD_OK && std::getenv("HY4_TEST_STDERR")) fputs(err.c_str(), stderr);
    }
    const hyv4flash::Dfly& df() const { return hyv4flash::models[0]->df; }
};
/* The core's query pass for `n` sequences at `block` rows each, anchored past contexts 40, 47, ..
 * (kit's decode contexts), the anchor rows the trunk step's rows of the committed ends. */
struct Query {
    std::unique_ptr<Kit> k;
    std::vector<int32_t> anchors;
    Query(const RadBuilder& b, int64_t n, int64_t block, int64_t depth) {
        k = query_pass(b, n, block, depth);
        for (int64_t s = 0; s < n; ++s) anchors.push_back((int32_t)s);
        k->b.anchor_rows = anchors.data();
        k->b.positions_h = k->pos.data();
    }
};
const char* kTrunkOnly[] = { "qsa_work", "qsa_tail_store", "idx_pool_gated", "idx_score",
                             "qsa_select", "engram_ids", "ihc_pre", "ihc_post", "idx_pool_gated_fp8",
                             "dsa_score", "attn_kv_sink", "mla_attn_paged_sink" };
}  // namespace

/* kernels/dfly2.md §3.1: a block drafter of block depth + 1 (the anchor predicts nothing), no cache
 * of its own and window 1 (§3.5 point 1), its proposals at the depth's pitch, the trunk's mask id. */
TEST(dfly2_is_a_block_drafter_of_block_depth_plus_one_with_window_1) {
    for (int depth : { 5, 3, 1 }) {
        DeclDfly d(depth);
        REQUIRE_EQ(d.rc, RAD_OK);
        const RadDrafterDecl& dd = d.b.drafter;
        CHECK_EQ(d.b.drafter_name, std::string("dfly2"));
        CHECK_EQ(dd.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(dd.depth, depth);
        CHECK_EQ(dd.block, (int64_t)depth + 1);
        CHECK_EQ(dd.window, (int64_t)1);
        CHECK_EQ(dd.mask_token, 319);
        CHECK_EQ(dd.proposal_pitch, (int64_t)depth);
        /* D-06: a short block of 4 rows (3 proposals) where the block is longer */
        CHECK_EQ(d.b.drafter_short, depth >= 4 ? (int64_t)4 : (int64_t)0);
        CHECK(dd.proposal != 0);
        CHECK(!d.b.drafter_shared && d.b.drafter_parents == 0);
        /* No KV group of its own; the snapshots and the proposal persist. P-09: the selections
         * live in two layer-major planes named without the layer. With the drafter served,
         * final_hidden persists too: the query pass reads anchor rows of an earlier trunk step. */
        for (const auto& g : d.b.kv_groups) CHECK(g.name.find("dfly") == std::string::npos);
        for (const char* n : { "dfly.sel_snap", "dfly.nsel_snap", "dfly.tokens" }) {
            const RecBuf* bb = find_b(d.b, n);
            REQUIRE(bb != nullptr);
            CHECK_EQ(bb->d.kind, (int)RAD_BUF_PERSIST);
        }
        CHECK_EQ(find_b(d.b, "final_hidden")->d.kind, (int)RAD_BUF_PERSIST);
        /* The served weights are read every step; nothing is renamed or fused anew. */
        const RecWeight* qb = find_w(d.b, "dfly.blk.0.attn_q_b.weight");
        REQUIRE(qb != nullptr);
        CHECK_EQ(qb->d.access, (int)RAD_ACCESS_PER_TOKEN);
        CHECK(find_w(d.b, "dfly.blk.0.attn_q_nope.weight") == nullptr);
        CHECK(find_w(d.b, "dfly.correct_gate.weight") != nullptr);
        CHECK(find_w(d.b, "dfly.correct_up.weight") != nullptr);
    }
    /* Without the drafter's config keys (the convert fixture's case) the manifest is declared and
     * nothing is served. */
    MetaKV m = mini_meta();
    m.set("dfly_num_hidden_layers", "1");
    DeclDfly d(5, m);
    REQUIRE_EQ(d.rc, RAD_OK);
    CHECK_EQ(d.b.drafter.kind, (int)RAD_DRAFT_NONE);
    const RecWeight* qb = find_w(d.b, "dfly.blk.0.attn_q_b.weight");
    REQUIRE(qb != nullptr);
    CHECK_EQ(qb->d.access, (int)RAD_ACCESS_RARE);
    CHECK(d.err.find("DFly2 is declared and not served") != std::string::npos);
    CHECK_EQ(count_op(d.b, "dfly_sel"), 0);
    /* P-09: with no drafter nothing reads final_hidden across steps: it is a step plane. */
    CHECK(find_b(d.b, "dfly.sel_snap") == nullptr);
    CHECK_EQ(find_b(d.b, "final_hidden")->d.kind, (int)RAD_BUF_TRANSIENT);
}

/* --draft-suffix beside DFly2 (WP13.8b): with the drafter in the container the probe says its
 * depth is fixed, so the core cuts a suffix chain to num-speculative-tokens instead of widening the
 * verify window past the block (DFly2 would decline the wider max_spec). No depth is stated
 * (auto keeps speculation off), and a container without the drafter states nothing. */
TEST(the_probe_keeps_a_suffix_chain_inside_dfly2s_block) {
    MetaKV m = dfly_meta();
    RadArchProbe p{};
    CHECK_OK(hyv4flash::probe(m.meta(), &p));
    CHECK_EQ(p.draft_depth_fixed, 1);
    CHECK_EQ(p.draft_depth, 0);
    MetaKV plain = mini_meta();
    RadArchProbe q{};
    CHECK_OK(hyv4flash::probe(plain.meta(), &q));
    CHECK_EQ(q.draft_depth_fixed, 0);
    CHECK_EQ(q.draft_depth, 0);
}

/* §3.3 / §6: one engine step -- the trunk step (with its snapshots), the empty context pass, the
 * query pass at depth 5, 3 and 1 -- is lint clean as a whole; the context pass issues nothing; the
 * query pass none of the trunk's key-side, selection, Engram or hyper-connection ops; and it
 * stores every block row before its attention. */
TEST(an_engine_step_with_dfly2_is_lint_clean_and_the_query_pass_touches_no_trunk_state) {
    for (int depth : { 5, 3, 1 }) {
        DeclDfly d(depth);
        REQUIRE_EQ(d.rc, RAD_OK);
        RadBuilder& b = d.b;
        const int64_t block = depth + 1;
        auto trunk = kit(b, std::vector<int>(2, (int)block), {});
        trunk->b.positions_h = trunk->pos.data();
        auto ctxp = context_pass(b, *trunk);
        Query qp(b, 2, block, depth);
        auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() });
        for (const auto& c : ps) {
            if (!c->step_fail.empty()) fprintf(stderr, "%s\n", c->step_fail.c_str());
            CHECK(c->step_fail.empty());
        }
        CHECK_LINT(b, ps[0].get(), ps[1].get(), ps[2].get());
        CHECK_EQ(ps[1]->issues.size(), (size_t)0);
        const auto q = issued_ops(b, *ps[2]);
        for (const char* op : kTrunkOnly) CHECK_EQ(std::count(q.begin(), q.end(), std::string(op)), 0);
        CHECK_EQ(std::count(q.begin(), q.end(), std::string("dfly_sel")), 1);
        CHECK_EQ(std::count(q.begin(), q.end(), std::string("mla_attn_sel_sink")), 1);
        CHECK_EQ(std::count(q.begin(), q.end(), std::string("logits_gemm")), depth);
        /* the f32 logits' top-k in one op where a library serves it (D-04) */
        CHECK_EQ(std::count(q.begin(), q.end(), std::string("cast_row_topk")), depth);
        CHECK_EQ(std::count(q.begin(), q.end(), std::string("row_topk")), 0);
        const auto& df = d.df();
        size_t store = SIZE_MAX, attn = SIZE_MAX;
        for (size_t i = 0; i < ps[2]->issues.size(); ++i) {
            if (ps[2]->issues[i].op == df.lay[0].op_store && store == SIZE_MAX) store = i;
            if (ps[2]->issues[i].op == df.lay[0].op_attn) attn = i;
        }
        REQUIRE(store != SIZE_MAX && attn != SIZE_MAX);
        CHECK(store < attn);
        /* The trunk step (contexts 40+ pass mini's dense limit of 16) writes its selection
         * straight into the snapshot plane (P-09): no copy launch, the selection's own op. */
        const auto& m = *hyv4flash::models[0];
        const rad_op ix_sel = m.layers[df.snap[0].layer].ix.op_select;
        int snap = 0, sels = 0;
        for (const auto& is : ps[0]->issues) {
            snap += is.op == df.op_sel;
            sels += is.op == ix_sel;
        }
        CHECK_EQ(snap, 0);
        CHECK_EQ(sels, 1);
    }
}

/* D-04: each chain round's top-k reads the f32 logits through cast_row_topk, which rounds them as
 * `cast` does (the same bytes, one op); without a library that declares it, or with one that
 * refuses it, the round keeps the pair, `cast` into logits_bf16 then row_topk. Whole and sharded. */
void no_cast_topk(RadBuilder* b) { bf16_encs(b); b->undeclared.insert("cast_row_topk"); }
void refused_cast_topk(RadBuilder* b) { bf16_encs(b); b->refuse.insert("cast_row_topk"); }
TEST(dfly2_takes_cast_row_topk_where_served_and_the_cast_pair_otherwise) {
    const int depth = 5;
    for (int world : { 1, 2 }) for (int form = 0; form < 3; ++form) {
        DeclDfly d(depth, dfly_meta(), world, 0,
                   form == 0 ? bf16_encs : form == 1 ? no_cast_topk : refused_cast_topk);
        REQUIRE_EQ(d.rc, RAD_OK);
        RadBuilder& b = d.b;
        const auto& df = d.df();
        const bool fused = form == 0;
        for (const auto& S : df.steps) {
            CHECK_EQ(S.op_ctopk != 0, fused);
            CHECK_EQ(S.op_lcast != 0, !fused);
            CHECK_EQ(S.op_topk != 0, !fused);
            const RecOp* o = b.op_of(fused ? S.op_ctopk : S.op_topk);
            REQUIRE(o);
            CHECK_EQ(pstr(o, "dtype"), std::string(fused ? "f32" : "bf16"));
            CHECK_EQ(pint(o, "N"), 320 / world);
            REQUIRE_EQ(o->reads.size(), (size_t)1);
            CHECK_EQ(b.bufs[o->reads[0] - 1].name.find(fused ? "dfly.logits" : "logits_bf16") !=
                         std::string::npos, true);
            CHECK_EQ(o->writes.size(), (size_t)(world > 1 ? 3 : 2));
            if (world > 1) CHECK_EQ(pint(o, "vocab_off"), 0);
        }
        if (world > 1) continue;   /* the passes below are the whole-vocabulary engine's */
        auto trunk = kit(b, std::vector<int>(2, depth + 1), {});
        trunk->b.positions_h = trunk->pos.data();
        auto ctxp = context_pass(b, *trunk);
        Query qp(b, 2, depth + 1, depth);
        auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() });
        for (const auto& c : ps) {
            if (!c->step_fail.empty()) fprintf(stderr, "%s\n", c->step_fail.c_str());
            CHECK(c->step_fail.empty());
        }
        CHECK_LINT(b, ps[0].get(), ps[1].get(), ps[2].get());
        const auto q = issued_ops(b, *ps[2]);
        auto count = [&](const char* op) { return (int64_t)std::count(q.begin(), q.end(), std::string(op)); };
        CHECK_EQ(count("cast_row_topk"), fused ? depth : 0);
        CHECK_EQ(count("row_topk"), fused ? 0 : depth);
        CHECK_EQ(count("cast") >= (fused ? 0 : depth), true);
        /* each round: the head, then (the cast and) the top-k */
        for (const auto& S : df.steps) {
            size_t at_head = SIZE_MAX, at_cast = SIZE_MAX, at_top = SIZE_MAX;
            const auto& is = ps[2]->issues;
            for (size_t i = 0; i < is.size(); ++i) {
                if (is[i].op == S.op_head && at_head == SIZE_MAX) at_head = i;
                if (S.op_lcast && is[i].op == S.op_lcast && at_cast == SIZE_MAX) at_cast = i;
                if (is[i].op == (fused ? S.op_ctopk : S.op_topk) && at_top == SIZE_MAX) at_top = i;
            }
            REQUIRE(at_head != SIZE_MAX && at_top != SIZE_MAX);
            CHECK(at_head < at_top);
            if (!fused) { REQUIRE(at_cast != SIZE_MAX); CHECK(at_head < at_cast && at_cast < at_top); }
        }
    }
}

/* D-06 (#91): DFly2 also serves a short block (rad_drafter_short_block) of 4 rows, the anchor and 3
 * masks, which the core stages under load (--draft-short-from): 3 proposals, two chain rounds fewer
 * at depth 5. The draft is causal (dfly_sel gives a row the keys up to its own position), so the
 * short pass is the full pass's first rows. Declared only where the drafter's block is longer; the
 * query pass takes either block, finds each anchor row and each round's row at its block's own
 * stride, and runs one round a proposal (coupled: one draw a round); any other block is refused. */
TEST(dfly2_serves_a_short_block_of_four_rows) {
    for (int depth : { 5, 4, 3 }) {
        DeclDfly d(depth);
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK_EQ(d.b.drafter_short, depth >= 4 ? (int64_t)4 : (int64_t)0);
    }
    for (int world : { 1, 2 })
        for (int coupled : { 0, 1 }) {
            MetaKV mk = dfly_meta();
            RadBuilder b;
            bf16_encs(&b);
            RadBuildCtx ctx = spec_ctx(0, world, 5);
            ctx.draft_coupling = coupled;
            int rc = 0;
            const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
            if (rc != RAD_OK) fputs(err.c_str(), stderr);
            REQUIRE_EQ(rc, RAD_OK);
            const auto& df = hyv4flash::models[0]->df;
            REQUIRE(df.served);
            CHECK_EQ(b.drafter_short, (int64_t)4);
            REQUIRE(df.op_row0_lo != 0);
            CHECK_EQ(pint(b.op_of(df.op_row0), "step"), 6);
            CHECK_EQ(pint(b.op_of(df.op_row0_lo), "step"), 4);
            for (int64_t blk : { (int64_t)6, (int64_t)4 }) {
                const int64_t rounds = blk - 1;
                auto trunk = kit(b, std::vector<int>(2, 6), {});
                trunk->b.positions_h = trunk->pos.data();
                auto ctxp = context_pass(b, *trunk);
                Query qp(b, 2, blk, rounds);
                auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() }, 0,
                                     world);
                for (const auto& c : ps) {
                    if (!c->step_fail.empty()) fprintf(stderr, "%s\n", c->step_fail.c_str());
                    CHECK(c->step_fail.empty());
                }
                CHECK_LINT(b, ps[0].get(), ps[1].get(), ps[2].get());
                const auto q = issued_ops(b, *ps[2]);
                auto count = [&](const char* op) {
                    return (int64_t)std::count(q.begin(), q.end(), std::string(op));
                };
                CHECK_EQ(count("logits_gemm"), rounds);
                CHECK_EQ(count("cast_row_topk"), rounds);
                CHECK_EQ(count("mtp_couple_propose"), coupled ? rounds : 0);
                /* each anchor and each round's row found at this block's stride */
                int row0 = 0, row0_lo = 0, rowk = 0, rowk_lo = 0;
                for (const auto& is : ps[2]->issues) {
                    row0 += is.op == df.op_row0;
                    row0_lo += is.op == df.op_row0_lo;
                    for (const auto& st : df.steps) {
                        rowk += is.op == st.op_rowk;
                        rowk_lo += st.op_rowk_lo && is.op == st.op_rowk_lo;
                    }
                }
                CHECK_EQ(row0, blk == 6 ? 1 : 0);
                CHECK_EQ(row0_lo, blk == 4 ? 1 : 0);
                CHECK_EQ(rowk, blk == 6 ? 5 : 0);
                CHECK_EQ(rowk_lo, blk == 4 ? 3 : 0);
            }
            /* a block the drafter declared neither way is refused, by name */
            auto trunk = kit(b, std::vector<int>(2, 6), {});
            trunk->b.positions_h = trunk->pos.data();
            auto ctxp = context_pass(b, *trunk);
            Query bad(b, 2, 5, 4);
            auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), bad.k.get() }, 0,
                                 world);
            CHECK(!ps[2]->step_fail.empty());
        }
}

/* WP13.8 (kernels/dfly2.md §8.2): under --draft-coupling the chain declares the planes the engine
 * binds by name (`mtp_couple_draws` [seqs, draw], `mtp_couple_state` [depth, seqs, state]) and one
 * mtp_couple_propose a step (round k + 1 of max_rounds = depth, the WHOLE vocabulary at TP, over the
 * merged shortlist with its values widened to f32), issued after the step's top-8 (and merge) and
 * before the proposal is copied out; lint clean. Uncoupled, none of it. Sabotaged and coupled, the
 * shortlist's ids are replaced (iota) before the draw instead of the proposals after it. */
TEST(dfly2_coupled_draws_each_chain_step_from_its_shortlist_law) {
    for (int world : { 1, 2 })
        for (int depth : { 5, 2 }) {
            DeclDfly plain(depth, dfly_meta(), world, 0);
            REQUIRE_EQ(plain.rc, RAD_OK);
            CHECK_EQ(count_op(plain.b, "mtp_couple_propose"), 0);
            CHECK(find_b(plain.b, "mtp_couple_state") == nullptr);
            /* row_topk / row_topk_merge write `val` at f32 (the op's contract; libr4d's kernels
             * write f32 whatever the plane is declared as) */
            REQUIRE(find_b(plain.b, "dfly.cand_val") != nullptr);
            CHECK_EQ(find_b(plain.b, "dfly.cand_val")->d.dtype, (uint32_t)RAD_F32);
            MetaKV mk = dfly_meta();
            RadBuilder b;
            bf16_encs(&b);
            RadBuildCtx ctx = spec_ctx(0, world, depth);
            ctx.draft_coupling = 1;
            int rc = 0;
            const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
            if (rc != RAD_OK) fputs(err.c_str(), stderr);
            REQUIRE_EQ(rc, RAD_OK);
            const auto& df = hyv4flash::models[0]->df;
            REQUIRE(df.served && df.coupled);
            const RecBuf* dr = find_b(b, "mtp_couple_draws");
            const RecBuf* st = find_b(b, "mtp_couple_state");
            REQUIRE(dr != nullptr && st != nullptr);
            CHECK_EQ(st->d.rank, 3u);
            CHECK_EQ(st->d.shape[0], depth);
            CHECK_EQ(st->d.shape[2], (int64_t)RAD_MTP_COUPLE_STATE_WORDS);
            CHECK_EQ(dr->d.shape[1], (int64_t)RAD_MTP_COUPLE_DRAW_WORDS);
            CHECK_EQ(st->d.shape[1], dr->d.shape[0]);
            CHECK_EQ(st->d.kind, (int)RAD_BUF_PERSIST);
            CHECK_EQ(count_op(b, "mtp_couple_propose"), depth);
            for (int k = 0; k < depth; ++k) {
                const auto& o = b.op_of(df.steps[(size_t)k].op_couple);
                CHECK_EQ(pint(o, "round"), k + 1);
                CHECK_EQ(pint(o, "max_rounds"), depth);
                CHECK_EQ(pint(o, "n_vocab"), 320);
            }
            /* the engine's query pass: 2 sequences */
            const int64_t block = depth + 1;
            auto trunk = kit(b, std::vector<int>(2, (int)block), {});
            trunk->b.positions_h = trunk->pos.data();
            auto ctxp = context_pass(b, *trunk);
            Query qp(b, 2, block, depth);
            auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() });
            for (const auto& c : ps) {
                if (!c->step_fail.empty()) fprintf(stderr, "%s\n", c->step_fail.c_str());
                CHECK(c->step_fail.empty());
            }
            CHECK_LINT(b, ps[0].get(), ps[1].get(), ps[2].get());
            const auto& is = ps[2]->issues;
            for (int k = 0; k < depth; ++k) {
                const auto& S = df.steps[(size_t)k];
                size_t at_top = SIZE_MAX, at_c = SIZE_MAX, at_tok = SIZE_MAX, at_merge = SIZE_MAX;
                for (size_t i = 0; i < is.size(); ++i) {
                    if (is[i].op == (S.op_ctopk ? S.op_ctopk : S.op_topk)) at_top = i;
                    if (is[i].op == S.op_merge && S.op_merge) at_merge = i;
                    if (is[i].op == S.op_couple) at_c = i;
                    if (is[i].op == S.op_tok) at_tok = i;
                }
                REQUIRE(at_top != SIZE_MAX && at_c != SIZE_MAX && at_tok != SIZE_MAX);
                CHECK(at_top < at_c && at_c < at_tok);
                if (world > 1) { REQUIRE(at_merge != SIZE_MAX); CHECK(at_merge < at_c); }
                CHECK_EQ(is[at_c].n, 2);
            }
        }
    /* sabotaged: the ids before the draw, no fill of the proposals after it */
    setenv("RADIANCE_HY4_DFLY_SABOTAGE", "1", 1);
    {
        MetaKV mk = dfly_meta();
        RadBuilder b;
        bf16_encs(&b);
        RadBuildCtx ctx = spec_ctx(0, 1, 5);
        ctx.draft_coupling = 1;
        int rc = 0;
        (void)stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        REQUIRE_EQ(rc, RAD_OK);
        const auto& df = hyv4flash::models[0]->df;
        CHECK(df.op_sabotage == 0);
        for (const auto& S : df.steps) CHECK(S.op_csab != 0);
    }
    unsetenv("RADIANCE_HY4_DFLY_SABOTAGE");
}

/* WP13.9: the drafter's linears stored quantised (IQ8R, rotated int8, ...) are read as the trunk's
 * are (Lin): each its own input quantised into the rank's code pair, then gemm_nt_q; this rank's
 * slice at TP; every other tensor as before. A form no Lin reads, or a quantised kv_b / embedding,
 * declines the drafter by name and leaves the manifest whole. */
namespace {
void dfly_i8_encs(RadBuilder* b) {
    bf16_encs(b);
    std::vector<std::pair<std::string, RadEncoding>> q;
    /* mini's widths take int8 in these (q_b's K is 64, kv_a's 40 rows, and the output projection's
     * and the MLP down's K split below 128 at TP2); the rest stay bf16, a mix being served too. */
    for (const char* n : { "attn_q_a", "attn_gate", "ffn_gate_up" })
        q.push_back({ std::string("dfly.blk.0.") + n + ".weight", enc_i8(true) });
    for (const char* n : { "fusion", "correct_gate", "correct_up", "correct_down" })
        q.push_back({ std::string("dfly.") + n + ".weight", enc_i8(n[0] == 'f' ? false : true) });
    b->encs.insert(b->encs.begin(), q.begin(), q.end());
}
void dfly_kvb_encs(RadBuilder* b) {
    dfly_i8_encs(b);
    b->encs.insert(b->encs.begin(), { "dfly.blk.0.attn_kv_b.weight", enc_i8(true) });
}
void dfly_trellis_encs(RadBuilder* b) {
    dfly_i8_encs(b);
    b->encs.insert(b->encs.begin(), { "dfly.blk.0.attn_q_a.weight", enc_nl64(4) });
}
}  // namespace
TEST(dfly2_reads_quantised_linears_as_the_trunk_does) {
    const char* lin[] = { "attn_q_a", "attn_gate", "ffn_gate_up" };
    for (const int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            /* 256 wide, so that a rank's correction K (128 at two ranks) is a whole scale group:
             * the down is a K split, and mini's 128 does not divide into 128-wide groups. */
            DeclDfly d(5, dfly_meta(1, true, 256), world, rank, dfly_i8_encs);
            REQUIRE_EQ(d.rc, RAD_OK);
            if (!d.df().served) fprintf(stderr, "%s\n", d.df().why.c_str());
            REQUIRE(d.df().served);
            CHECK_EQ(d.b.drafter.kind, (int)RAD_DRAFT_BLOCK);
            CHECK_EQ(d.df().n_quant, 3 + 4);
            RadBuilder& b = d.b;
            /* Seven linears: int8 GEMMs over rotated codes; the rest, the gains and the head as before. */
            int q = 0;
            for (const auto& o : b.ops)
                if (o.op == "gemm_nt_q" && o.w.size() >= 1) ++q;
            CHECK(q >= 3 + 4);
            for (const char* n : lin) {
                const std::string nm = std::string("dfly.blk.0.") + n;
                const RecWeight* w = find_w(b, (nm + ".weight").c_str());
                REQUIRE(w != nullptr);
                CHECK_EQ(w->d.dtype, RAD_I8);
                CHECK(find_w(b, (nm + ".scale").c_str()) != nullptr);
            }
            /* This rank's slice at TP (the same shapes the bf16 manifest slices). */
            const RecWeight* ga = find_w(b, "dfly.blk.0.attn_gate.weight");
            const RecWeight* gu = find_w(b, "dfly.blk.0.ffn_gate_up.weight");
            CHECK_EQ(ga->d.shard, world > 1 ? (int)RAD_SHARD_ROW : (int)RAD_SHARD_NONE);
            CHECK_EQ(ga->d.shape[0], (int64_t)(4 * 32 / world));
            if (world > 1) {
                CHECK_EQ(gu->d.n_row_parts, 2);
                CHECK_EQ(gu->d.row_parts[0], d.df().ff / world);
            }
            CHECK(find_w(b, "dfly.blk.0.attn_kv_b.weight")->d.dtype == RAD_BF16);
            CHECK(find_w(b, "dfly.blk.0.attn_q_a.weight") != nullptr);
            /* The correction slices with it, its down as a K split in whole scale groups. */
            CHECK_EQ(find_w(b, "dfly.correct_gate.weight")->d.shard,
                     world > 1 ? (int)RAD_SHARD_ROW : (int)RAD_SHARD_NONE);
            CHECK_EQ(find_w(b, "dfly.correct_down.weight")->d.shard,
                     world > 1 ? (int)RAD_SHARD_COL : (int)RAD_SHARD_NONE);
            CHECK(d.df().c_g.q && d.df().c_u.q && d.df().c_d.q);
            CHECK_EQ(pint(b.op_of(d.df().c_g.lin.op), "N"), d.df().E / world);
            CHECK_EQ(pint(b.op_of(d.df().c_d.lin.op), "K"), d.df().E / world);
            CHECK(d.df().op_ar_c != 0 || world == 1);
            /* One engine step, lint clean, the query pass issuing the int8 chain. */
            auto trunk = kit(b, std::vector<int>(2, 6), {});
            trunk->b.positions_h = trunk->pos.data();
            auto ctxp = context_pass(b, *trunk);
            Query qp(b, 2, 6, 5);
            auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() }, rank,
                                 world);
            CHECK_PASSES(b, ps);
            const auto is = issued_ops(b, *ps[2]);
            CHECK(std::count(is.begin(), is.end(), std::string("gemm_nt_q")) >= 3 + 3 * 5 + 1);
        }
    }
    /* A form no draft linear reads, and a quantised kv_b: declined by name, the manifest whole. */
    for (auto encs : { +dfly_trellis_encs, +dfly_kvb_encs }) {
        DeclDfly d(5, dfly_meta(), 1, 0, encs);
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK(!d.df().served);
        CHECK_EQ(d.b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK_EQ(count_op(d.b, "dfly_sel"), 0);
        CHECK(d.err.find("DFly2 is declared and not served") != std::string::npos);
        CHECK(d.err.find("quantised") != std::string::npos || d.err.find("does not read") != std::string::npos);
        const RecWeight* w = find_w(d.b, "dfly.blk.0.attn_q_a.weight");
        REQUIRE(w != nullptr);   /* kept as its planes' views */
        CHECK(find_w(d.b, "dfly.blk.0.attn_q_a.scale") != nullptr);
        CHECK(find_w(d.b, "dfly.blk.0.attn_gate.scale") != nullptr);
    }
}

/* §3.2 / P-09: a dense trunk step clears the snapshot's counts with the one fill over the
 * plane, and a selecting single-span step writes the selection into the plane itself: the
 * attention reads it from the layer's window, and no copy launch runs in between. */
TEST(the_trunk_writes_its_selection_into_the_snapshot_and_clears_it_on_a_dense_step) {
    DeclDfly d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    const auto& sn = m.df.snap[0];
    for (int dense = 0; dense < 2; ++dense) {
        auto k = kit(d.b, { 1 }, {});
        if (dense) { k->pos[0] = 5; k->used[0] = 6; k->ctxl[0] = 5; k->b.max_ctx_len = 6; }
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(d.b, &c);
        size_t sel = SIZE_MAX, cp = SIZE_MAX, zero = SIZE_MAX, attn = SIZE_MAX;
        int cps = 0, zeros = 0;
        for (size_t i = 0; i < c.issues.size(); ++i) {
            const rad_op op = c.issues[i].op;
            if (op == m.layers[2].ix.op_select && sel == SIZE_MAX) sel = i;
            if (op == m.df.op_sel) { cp = i; ++cps; }
            if (op == m.df.op_zero) { zero = i; ++zeros; }
            if ((op == m.layers[2].attn.op_sel || op == m.layers[2].attn.op_paged) &&
                attn == SIZE_MAX)
                attn = i;
        }
        REQUIRE(attn != SIZE_MAX);
        /* The copy is never issued on a step that writes in place, and the fill runs once a
         * step for the whole plane, never once a layer. */
        CHECK_EQ(cps, 0);
        if (dense) {
            CHECK(zero != SIZE_MAX && zero < attn);
            CHECK_EQ(zeros, 1);
            CHECK_EQ(sel, SIZE_MAX);
        } else {
            CHECK(sel != SIZE_MAX && sel < attn);
            CHECK_EQ(zero, SIZE_MAX);
            /* qsa_select wrote the plane's window for this layer, and the attention reads the
             * selection from there. */
            CHECK_EQ((rad_buf)c.issues[sel].opd[5].handle, m.df.snap_sel);
            CHECK_EQ((rad_buf)c.issues[sel].opd[6].handle, m.df.snap_nsel);
            CHECK_EQ(c.issues[sel].opd[5].offset, sn.base * m.s.ix_group_topk);
            CHECK_EQ(c.issues[sel].opd[6].offset, sn.base);
            CHECK_EQ((rad_buf)c.issues[attn].opd[3].handle, m.df.snap_sel);
            CHECK_EQ(c.issues[attn].opd[3].offset, sn.base * m.s.ix_group_topk);
        }
    }
}

/* P-09: with taps on, the in-place step still refreshes the tap's group ids in the snapshot (the
 * drafter's tap table reads them): only the ids are copied, the selection itself is not. */
TEST(the_in_place_snapshot_step_still_copies_the_tap_group_ids) {
    setenv("RADIANCE_HY4_TAPS", "1", 1);
    {
        DeclDfly d;
        REQUIRE_EQ(d.rc, RAD_OK);
        const auto& m = *hyv4flash::models[0];
        const auto& sn = m.df.snap[0];
        REQUIRE(m.df.op_ids != 0);
        auto k = kit(d.b, { 1 }, {});
        k->b.positions_h = k->pos.data();
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(d.b, &c);
        size_t tap = SIZE_MAX, ids = SIZE_MAX;
        int cps = 0;
        for (size_t i = 0; i < c.issues.size(); ++i) {
            const rad_op op = c.issues[i].op;
            if (op == m.layers[2].ix.op_tap_select && tap == SIZE_MAX) tap = i;
            if (op == m.df.op_sel) ++cps;
            if (op == m.df.op_ids && ids == SIZE_MAX) ids = i;
        }
        REQUIRE(tap != SIZE_MAX && ids != SIZE_MAX);
        CHECK(tap < ids);
        CHECK_EQ(cps, 0);
        CHECK_EQ((rad_buf)c.issues[ids].opd[1].handle, m.df.snap_ids);
        CHECK_EQ(c.issues[ids].opd[1].offset, sn.base * m.s.ix_group_topk);
    }
    unsetenv("RADIANCE_HY4_TAPS");
}

/* P-09: a streamed group keeps the copy: its kv_resolve rewrites the wire selection in place,
 * so the snapshot is still taken between the selection and the attention — now at the layer's
 * window in the plane, and the attention keeps reading the wire. */
TEST(a_streamed_sparse_group_keeps_the_snapshot_copy_at_the_layer_window) {
    DeclDfly d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    const auto& sn = m.df.snap[0];
    const int64_t blocks = 64, slots = 16, bbytes = 320;
    std::vector<uint8_t> host((size_t)(blocks * bbytes * 3));
    std::vector<int32_t> res((size_t)(blocks * 3)), slot_block((size_t)(slots * 3)),
        slot_state((size_t)(slots * 6)), ctl(16);
    RadKVStream st{};
    st.host = host.data(); st.res = res.data(); st.slot_block = slot_block.data();
    st.slot_state = slot_state.data(); st.ctl = ctl.data(); st.block_bytes = bbytes;
    st.host_layer_stride = bbytes * blocks; st.slots = slots; st.host_bytes = (int64_t)host.size();
    auto k = kit(d.b, { 1 }, {});
    k->b.positions_h = k->pos.data();
    k->kv[(size_t)m.kv_sparse - 1].stream = &st;
    RadCtx c{ .b = &d.b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    if (!c.step_fail.empty()) fprintf(stderr, "%s\n", c.step_fail.c_str());
    CHECK(c.step_fail.empty());
    CHECK_LINT(d.b, &c);
    size_t sel = SIZE_MAX, cp = SIZE_MAX, attn = SIZE_MAX;
    for (size_t i = 0; i < c.issues.size(); ++i) {
        const rad_op op = c.issues[i].op;
        if (op == m.layers[2].ix.op_select && sel == SIZE_MAX) sel = i;
        if (op == m.df.op_sel) cp = i;
        if (op == m.layers[2].attn.op_sel && attn == SIZE_MAX) attn = i;
    }
    REQUIRE(sel != SIZE_MAX && cp != SIZE_MAX && attn != SIZE_MAX);
    CHECK(sel < cp && cp < attn);
    CHECK_EQ((rad_buf)c.issues[cp].opd[0].handle, m.iw.sel);        /* src: the wire */
    CHECK_EQ((rad_buf)c.issues[cp].opd[1].handle, m.df.snap_sel);   /* dst: the layer's window */
    CHECK_EQ(c.issues[cp].opd[1].offset, sn.base * m.s.ix_group_topk);
    CHECK_EQ((rad_buf)c.issues[attn].opd[3].handle, m.iw.sel);      /* the attention: the wire */
}

/* Under streaming (WP14) the query pass resolves its table between the draft rows' store and the
 * attention, with dfly_sel's entry count, as the trunk resolves its selection. */
TEST(a_streamed_sparse_group_resolves_the_draft_table_too) {
    DeclDfly d;
    REQUIRE_EQ(d.rc, RAD_OK);
    const auto& m = *hyv4flash::models[0];
    const auto& L = m.df.lay[0];
    REQUIRE(L.op_resolve != 0);
    REQUIRE(L.op_store_stream != 0);
    const int64_t blocks = 64, slots = 16, bbytes = 320;
    std::vector<uint8_t> host((size_t)(blocks * bbytes * 3));
    std::vector<int32_t> res((size_t)(blocks * 3)), slot_block((size_t)(slots * 3)),
        slot_state((size_t)(slots * 6)), ctl(16);
    RadKVStream st{};
    st.host = host.data(); st.res = res.data(); st.slot_block = slot_block.data();
    st.slot_state = slot_state.data(); st.ctl = ctl.data(); st.block_bytes = bbytes;
    st.host_layer_stride = bbytes * blocks; st.slots = slots; st.host_bytes = (int64_t)host.size();
    Query qp(d.b, 1, 6, 5);
    qp.k->kv[(size_t)m.kv_sparse - 1].stream = &st;
    RadCtx c{ .b = &d.b, .batch = &qp.k->b };
    hyv4flash::step(&c, &qp.k->b);
    if (!c.step_fail.empty()) fprintf(stderr, "%s\n", c.step_fail.c_str());
    CHECK(c.step_fail.empty());
    CHECK_LINT(d.b, &c);
    size_t store = SIZE_MAX, res_i = SIZE_MAX, attn = SIZE_MAX, plain = SIZE_MAX;
    for (size_t i = 0; i < c.issues.size(); ++i) {
        const rad_op op = c.issues[i].op;
        if (op == L.op_store_stream) store = i;
        if (op == L.op_store) plain = i;
        if (op == L.op_resolve) res_i = i;
        if (op == L.op_attn) attn = i;
    }
    REQUIRE(store != SIZE_MAX && res_i != SIZE_MAX && attn != SIZE_MAX);
    CHECK(store < res_i && res_i < attn);
    CHECK_EQ(plain, SIZE_MAX);
    CHECK_EQ(c.issues[res_i].opd[0].handle, L.tab);
    CHECK_EQ(c.issues[res_i].opd[1].handle, L.cnt);
}

/* §2: what the drafter cannot serve declares the manifest, names the reason, and declares no
 * drafter (prompt lookup remains). */
TEST(dfly2_configurations_it_cannot_serve_are_refused_by_name) {
    struct Case { const char* key; const char* val; const char* says; int depth; int world; };
    const Case cases[] = {
        { "dfly_mask_token_id", "320", "outside the vocabulary", 5, 1 },
        { "dfly_mask_token_id", "-1", "outside the vocabulary", 5, 1 },
        { "dfly_target_kv_layers", "1", "not a sparse layer", 5, 1 },          /* sliding */
        { "dfly_target_index_layer", "0", "not a sparse layer", 5, 1 },
        { "dfly_target_index_topk", "2048", "group_topk * ratio", 5, 1 },
        { "dfly_context_local_window", "4", "local window", 5, 1 },
        { nullptr, nullptr, "trained at 5 proposals", 6, 1 },
    };
    for (const Case& cs : cases) {
        MetaKV m = dfly_meta();
        if (cs.key) m.set(cs.key, cs.val);
        DeclDfly d(cs.depth, m, cs.world);
        REQUIRE_EQ(d.rc, RAD_OK);
        if (d.err.find(cs.says) == std::string::npos) fprintf(stderr, "%s: %s\n", cs.says, d.err.c_str());
        CHECK(d.err.find(cs.says) != std::string::npos);
        CHECK_EQ(d.b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK_EQ(count_op(d.b, "dfly_sel"), 0);
        CHECK(find_w(d.b, "dfly.fusion.weight") != nullptr);
    }
    /* TP: a draft MLP width that does not split over the ranks (the checkpoint's, 191 here). */
    {
        MetaKV m = dfly_meta();
        RadBuilder b;
        bf16_encs(&b);
        b.encs.push_back({ "dfly.blk.0.ffn_down", rad_enc_plain(RAD_BF16) });
        b.shapes["dfly.blk.0.ffn_down.weight"] = { 128, 191 };
        RadBuildCtx ctx = spec_ctx(0, 2, 5);
        int rc = 0;
        const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, m.meta(), &ctx); });
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(err.find("does not split evenly over the ranks") != std::string::npos);
        CHECK_EQ(b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK_EQ(count_op(b, "dfly_sel"), 0);
        const RecWeight* dn = find_w(b, "dfly.blk.0.ffn_down.weight");
        REQUIRE(dn != nullptr);
        CHECK_EQ(dn->d.shard, (int)RAD_SHARD_NONE);   /* declared whole */
        CHECK_EQ(dn->d.shape[1], (int64_t)191);
    }
    /* Two draft layers over one cache layer: duplicate Lkv. */
    MetaKV two = dfly_meta(2);
    DeclDfly d(5, two);
    REQUIRE_EQ(d.rc, RAD_OK);
    CHECK(d.err.find("names layer 2 twice") != std::string::npos);
    /* RADIANCE_HY4_DFLY=0: prompt lookup alone, as --spec-ngram needs. */
    setenv("RADIANCE_HY4_DFLY", "0", 1);
    DeclDfly off;
    unsetenv("RADIANCE_HY4_DFLY");
    CHECK_EQ(off.b.drafter.kind, (int)RAD_DRAFT_NONE);
}

/* §3.6 (WP13.6): at 2 and 4 ranks the draft splits as the trunk's sparse layers -- whole heads a
 * rank in q_b, kv_b, the gate and the sinks (rows) and the output projection (columns), the MLP's
 * width (gate_up's rows of each half, down's columns), the correction's gate and up by their rows
 * and its down by columns (one exact reduction of the chain's delta a step, before the residual),
 * everything else whole -- its ops run this rank's heads and width, each layer reduces twice
 * (o_proj, the MLP) exactly before the norm or residual that reads the sum, and each chain step
 * ranks this rank's vocabulary shard (global ids, the pair plane), gathers every rank's pairs and
 * merges; one engine step lint clean on every rank. At one rank none of the three TP ops is
 * declared and the weights are whole and unsharded. */
TEST(dfly2_at_tp_splits_heads_and_width_and_merges_the_vocabulary_shards) {
    struct Want { const char* name; int shard; int dim; };
    const Want wants[] = {
        { "dfly.blk.0.attn_q_b.weight", RAD_SHARD_ROW, 0 },
        { "dfly.blk.0.attn_kv_b.weight", RAD_SHARD_ROW, 0 },
        { "dfly.blk.0.attn_gate.weight", RAD_SHARD_ROW, 0 },
        { "dfly.blk.0.attn_sinks.weight", RAD_SHARD_ROW, 0 },
        { "dfly.blk.0.attn_output.weight", RAD_SHARD_COL, 1 },
        { "dfly.blk.0.ffn_gate_up.weight", RAD_SHARD_ROW, 0 },
        { "dfly.blk.0.ffn_down.weight", RAD_SHARD_COL, 1 },
        { "dfly.blk.0.attn_q_a.weight", RAD_SHARD_NONE, -1 },
        { "dfly.blk.0.attn_kv_a.weight", RAD_SHARD_NONE, -1 },
        { "dfly.blk.0.attn_norm.weight", RAD_SHARD_NONE, -1 },
        { "dfly.fusion.weight", RAD_SHARD_NONE, -1 },
        { "dfly.correct_gate.weight", RAD_SHARD_ROW, 0 },
        { "dfly.correct_up.weight", RAD_SHARD_ROW, 0 },
        { "dfly.correct_down.weight", RAD_SHARD_COL, 1 },
    };
    for (const bool wide : { false, true }) {
        DeclDfly one(5, dfly_meta(1, wide));
        REQUIRE_EQ(one.rc, RAD_OK);
        REQUIRE_EQ(one.b.drafter.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(count_op(one.b, "all_gather"), 0);
        CHECK_EQ(count_op(one.b, "row_topk_merge"), 0);
        std::map<std::string, RadWeightDecl> whole;
        for (const auto& w : one.b.weights)
            if (w.name.rfind("dfly.", 0) == 0) {
                whole[w.name] = w.d;
                CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
                CHECK_EQ(w.d.n_row_parts, 0);
            }
        const int64_t H = wide ? 4 : 2, ff = one.df().ff;
        CHECK(one.df().op_ar_o == 0 && one.df().op_ar_y == 0 && one.df().op_ar_c == 0);
        for (int world : { 2, 4 }) {
            if (!wide && world == 4) continue;   /* mini's 2 sparse heads */
            for (int rank = 0; rank < world; ++rank) {
                DeclDfly d(5, dfly_meta(1, wide), world, rank);
                REQUIRE_EQ(d.rc, RAD_OK);
                if (!d.df().served) fprintf(stderr, "world %d: %s\n", world, d.df().why.c_str());
                REQUIRE_EQ(d.b.drafter.kind, (int)RAD_DRAFT_BLOCK);
                RadBuilder& b = d.b;
                for (const Want& wt : wants) {
                    const RecWeight* w = find_w(b, wt.name);
                    REQUIRE(w != nullptr);
                    const RadWeightDecl& o = whole[wt.name];
                    CHECK_EQ(w->d.shard, wt.shard);
                    CHECK_EQ(w->d.access, (int)RAD_ACCESS_PER_TOKEN);
                    for (uint32_t k = 0; k < o.rank; ++k)
                        CHECK_EQ(w->d.shape[k], (int)k == wt.dim ? o.shape[k] / world : o.shape[k]);
                }
                const RecWeight* gu = find_w(b, "dfly.blk.0.ffn_gate_up.weight");
                CHECK_EQ(gu->d.n_row_parts, 2);
                CHECK_EQ(gu->d.row_parts[0], ff / world);
                CHECK_EQ(gu->d.row_parts[1], ff / world);
                const auto& df = d.df();
                const auto& L = df.lay[0];
                CHECK_EQ(pint(b.op_of(L.op_attn), "heads"), H / world);
                CHECK_EQ(pint(b.op_of(L.p_qb.op), "N"), H / world * 32);
                CHECK_EQ(pint(b.op_of(L.p_gu.op), "N"), 2 * ff / world);
                CHECK_EQ(pint(b.op_of(L.p_down.op), "K"), ff / world);
                CHECK_EQ(pint(b.op_of(df.steps[0].op_head), "n_vocab"), 320 / world);
                CHECK_EQ(pint(b.op_of(df.steps[0].op_ctopk), "vocab_off"), 320 / world * rank);
                /* The correction's three linears are this rank's width too, and its down's partial
                 * sum is reduced exactly before the residual and the proposal law read it. */
                CHECK_EQ(pint(b.op_of(df.c_g.op), "N"), df.E / world);
                CHECK_EQ(pint(b.op_of(df.c_g.op), "K"), 2 * df.E);
                CHECK_EQ(pint(b.op_of(df.c_u.op), "N"), df.E / world);
                CHECK_EQ(pint(b.op_of(df.c_u.op), "K"), 2 * df.E);
                CHECK_EQ(pint(b.op_of(df.c_d.op), "N"), df.E);
                CHECK_EQ(pint(b.op_of(df.c_d.op), "K"), df.E / world);
                CHECK_EQ(pint(b.op_of(df.steps[0].op_silu), "n"), df.E / world);
                /* The chain's buffers are this rank's slice too: gate and up rows, the down's K. */
                const RecBuf* bg = find_b(b, "dfly.chain_gate");
                const RecBuf* bu = find_b(b, "dfly.chain_up");
                const RecBuf* bgu = find_b(b, "dfly.chain_gate_up");
                const RecBuf* ba = find_b(b, "dfly.chain_act");
                REQUIRE(bg && bu && bgu && ba);
                CHECK_EQ(bg->d.shape[1], df.E / world);
                CHECK_EQ(bu->d.shape[1], df.E / world);
                CHECK_EQ(bgu->d.shape[1], 2 * df.E / world);
                CHECK_EQ(ba->d.shape[1], df.E / world);
                CHECK(df.op_ar_c != 0);
                CHECK_EQ(pint(b.op_of(df.op_ar_c), "exact"), 1);
                for (const auto& st : df.steps) CHECK(st.op_gather != 0 && st.op_merge != 0);
                CHECK(df.op_ar_o != 0 && df.op_ar_y != 0 && df.op_ar_o != df.op_ar_y);
                /* One engine step: the trunk step, the empty context pass, the query pass. */
                auto trunk = kit(b, std::vector<int>(2, 6), {});
                trunk->b.positions_h = trunk->pos.data();
                auto ctxp = context_pass(b, *trunk);
                Query qp(b, 2, 6, 5);
                auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() }, rank,
                                     world);
                CHECK_PASSES(b, ps);
                CHECK_EQ(ps[1]->issues.size(), (size_t)0);
                int ar = 0, ga = 0, me = 0, tk = 0, arc = 0;
                size_t i_out = SIZE_MAX, i_ar = SIZE_MAX, i_post = SIZE_MAX, i_down = SIZE_MAX,
                       i_ar2 = SIZE_MAX, i_add2 = SIZE_MAX;
                std::vector<size_t> chain;
                const auto& is = ps[2]->issues;
                for (size_t i = 0; i < is.size(); ++i) {
                    const rad_op op = is[i].op;
                    if (op == df.op_ar_o || op == df.op_ar_y) ++ar;
                    if (op == df.op_ar_c) ++arc;
                    bool add = false;
                    for (const auto& st : df.steps) {
                        ga += op == st.op_gather;
                        me += op == st.op_merge;
                        add = add || op == st.op_add;
                    }
                    if (op == df.c_d.op || op == df.op_ar_c || add) chain.push_back(i);
                    if (op == df.steps[0].op_ctopk) ++tk;
                    if (op == L.p_out.op) i_out = i;
                    if (op == df.op_ar_o) i_ar = i;
                    if (op == L.op_post || (op == L.op_add1 && i_post == SIZE_MAX)) i_post = std::min(i_post, i);
                    if (op == L.p_down.op) i_down = i;
                    if (op == df.op_ar_y) i_ar2 = i;
                    if (op == L.op_add2) i_add2 = i;
                }
                CHECK_EQ(ar, 2);
                CHECK_EQ(arc, 5);
                REQUIRE_EQ(chain.size(), (size_t)15);
                for (int r = 0; r < 5; ++r) {
                    CHECK_EQ(is[chain[3 * r]].op, df.c_d.op);
                    CHECK_EQ(is[chain[3 * r + 1]].op, df.op_ar_c);
                    CHECK_EQ(is[chain[3 * r + 2]].op, df.steps[r].op_add);
                    CHECK_EQ(is[chain[3 * r + 1]].n, 2 * df.E);
                }
                CHECK_EQ(ga, 5);
                CHECK_EQ(me, 5);
                CHECK_EQ(tk, 1);
                CHECK(i_out < i_ar && i_ar < i_post);
                CHECK(i_down < i_ar2 && i_ar2 < i_add2);
                /* Exact reductions, in place over the draft's own buffers. */
                for (rad_op op : { df.op_ar_o, df.op_ar_y, df.op_ar_c })
                    CHECK_EQ(pint(b.op_of(op), "exact"), 1);
            }
        }
    }
}

/* The core records draft passes too: their ops do not depend on the context, so the key is fixed
 * a kind (the context pass and the query pass apart). */
TEST(the_draft_passes_carry_their_own_pass_keys) {
    DeclDfly d;
    REQUIRE_EQ(d.rc, RAD_OK);
    auto trunk = kit(d.b, { 1 }, {});
    auto ctxp = context_pass(d.b, *trunk);
    Query qp(d.b, 1, 6, 5);
    RadCtx c1{ .b = &d.b, .batch = &ctxp->b }, c2{ .b = &d.b, .batch = &qp.k->b };
    const uint64_t kc = hyv4flash::pass_key(&c1, &ctxp->b, nullptr);
    const uint64_t kq = hyv4flash::pass_key(&c2, &qp.k->b, nullptr);
    CHECK(kc != 0 && kq != 0 && kc != kq);
    qp.k->b.max_ctx_len = 4000;
    CHECK_EQ(hyv4flash::pass_key(&c2, &qp.k->b, nullptr), kq);
}

/* The debug taps of a query pass: the reference's dump keys (ref/dfly_ref.py), PERSIST, and the
 * trunk's rotated key part (dbg.attn_kv_rope.<l>, every layer). */
TEST(the_dfly2_taps_are_the_references_keys) {
    setenv("RADIANCE_HY4_TAPS", "1", 1);
    DeclDfly d;
    unsetenv("RADIANCE_HY4_TAPS");
    REQUIRE_EQ(d.rc, RAD_OK);
    const char* names[] = { "dbg.dfly.embed", "dbg.dfly.fused", "dbg.dfly.final", "dbg.dfly.tokens",
                            "dbg.dfly.attn_in_norm.0", "dbg.dfly.attn_q_resid.0",
                            "dbg.dfly.attn_kv_row.0", "dbg.dfly.attn_gate_logits.0",
                            "dbg.dfly.attn_o_proj.0", "dbg.dfly.attn_out.0", "dbg.dfly.resid_mid.0",
                            "dbg.dfly.ffn_in_norm.0", "dbg.dfly.mlp_out.0", "dbg.dfly.resid.0",
                            "dbg.dfly.sel_ids.0", "dbg.dfly.len.0", "dbg.dfly.corrected.4",
                            "dbg.dfly.logits.0", "dbg.dfly.cand_idx.4", "dbg.dfly.cand_val.0",
                            "dbg.dfly.slot", "dbg.dfly.pos", "dbg.dfly.anchor",
                            "dbg.attn_kv_rope.0", "dbg.attn_kv_rope.1", "dbg.attn_kv_rope.2" };
    for (const char* n : names) {
        const RecBuf* bb = find_b(d.b, n);
        if (!bb) fprintf(stderr, "missing %s\n", n);
        REQUIRE(bb != nullptr);
        CHECK_EQ(bb->d.kind, (int)RAD_BUF_PERSIST);
    }
    auto trunk = kit(d.b, { 6, 6 }, {});
    trunk->b.positions_h = trunk->pos.data();
    auto ctxp = context_pass(d.b, *trunk);
    Query qp(d.b, 2, 6, 5);
    auto ps = run_passes(d.b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() });
    for (const auto& c : ps) CHECK(c->step_fail.empty());
    CHECK_LINT(d.b, ps[0].get(), ps[1].get(), ps[2].get());
}

/* The four-rank twin ar_rmsnorm_ihc_junction fuses the reduction into the norm kernel, so it holds
 * the CUs while it waits on the link: the plugin declares it up to kjRows4 (8) rows, the height
 * measured to pay, and leaves the taller verify heights to the chain (standalone all_reduce +
 * plain rmsnorm_ihc_junction), which stays declared either way. RADIANCE_HY4_JUNCTION_TP4=1
 * widens the twin to the whole verify band, =0 turns it off at four ranks; world 2 declares it at
 * every height. */
TEST(the_four_rank_junction_twin_takes_the_launch_bound_heights) {
    unsetenv("RADIANCE_HY4_JUNCTION_TP4");
    for (int world : { 1, 2, 4 }) {
        Declared d(bf16_encs, world, 0, true);
        REQUIRE_EQ(d.rc, RAD_OK);
        CHECK(count_op(d.b, "rmsnorm_ihc_junction") > 0);
        CHECK_EQ(count_op(d.b, "ar_rmsnorm_ihc_junction") > 0, world == 2 || world == 4);
        /* Four ranks keep the chain declared for the heights above the fused band. */
        if (world == 4) CHECK(count_op(d.b, "all_reduce") > 0);
    }
    setenv("RADIANCE_HY4_JUNCTION_TP4", "1", 1);
    Declared d(bf16_encs, 4, 0, true);
    unsetenv("RADIANCE_HY4_JUNCTION_TP4");
    REQUIRE_EQ(d.rc, RAD_OK);
    CHECK(count_op(d.b, "ar_rmsnorm_ihc_junction") > 0);

    /* =0 is the chain at every height: no twin, the standalone reduction only. */
    setenv("RADIANCE_HY4_JUNCTION_TP4", "0", 1);
    Declared off(bf16_encs, 4, 0, true);
    unsetenv("RADIANCE_HY4_JUNCTION_TP4");
    REQUIRE_EQ(off.rc, RAD_OK);
    CHECK_EQ(count_op(off.b, "ar_rmsnorm_ihc_junction"), 0);
    CHECK(count_op(off.b, "all_reduce") > 0);
}

/* A per-stream control vector: rank counts the rows of a layer, rank / hc of them are groups of
 * one direction per stream, and each group is one chained hc_project with vsep = n_embd on every
 * steered layer. Without a vector the graph carries no projection and no switch buffer at all, so
 * a deployment that does not ask for one runs the graph it ran before. */
TEST(a_per_stream_control_vector_declares_one_projection_per_group) {
    const int64_t E = 128, hc = 2, groups = 2, rank = groups * hc, layers = 3;
    std::vector<float> dir((size_t) (layers * rank * E), 0.0f);
    float scales[3] = { 0.0f, 1.0f, 1.0f };
    RadControlVector cv{ dir.data(), scales, E, layers, rank, 1 };

    auto declared = [&](const RadControlVector* v, int* rc) {
        MetaKV mk = mini_meta();
        auto b = std::make_unique<RadBuilder>();
        bf16_encs(b.get());
        RadBuildCtx ctx = build_ctx();
        ctx.cvec = v;
        G3Env g3env;
        stderr_of([&] { *rc = hyv4flash::declare(b.get(), mk.meta(), &ctx); });
        return b;
    };
    int rc = 0;
    auto b = declared(&cv, &rc);
    REQUIRE_EQ(rc, RAD_OK);
    CHECK_EQ(count_op(*b, "hc_project"), (int) ((layers - 1) * groups));
    for (const RecOp& o : b->ops) {
        if (o.op != "hc_project") continue;
        bool vsep = false;
        for (const RecParam& p : o.p)
            if (p.key == "vsep") { vsep = true; CHECK_EQ(p.ival, (long long) E); }
        CHECK(vsep);
    }

    /* The shared single direction: one projection a layer, vsep 0. */
    RadControlVector one{ dir.data(), scales, E, layers, 1, 0 };
    auto b1 = declared(&one, &rc);
    REQUIRE_EQ(rc, RAD_OK);
    CHECK_EQ(count_op(*b1, "hc_project"), (int) (layers - 1));
    for (const RecOp& o : b1->ops)
        if (o.op == "hc_project")
            for (const RecParam& p : o.p)
                if (p.key == "vsep") CHECK_EQ(p.ival, 0LL);

    /* Rows that do not split into the model's streams are refused, not misread. */
    RadControlVector odd{ dir.data(), scales, E, layers, 3, 1 };
    auto b2 = declared(&odd, &rc);
    CHECK(rc != RAD_OK);

    /* No vector: no projection and no switch. */
    auto b0 = declared(nullptr, &rc);
    REQUIRE_EQ(rc, RAD_OK);
    CHECK_EQ(count_op(*b0, "hc_project"), 0);
    for (const RecBuf& x : b0->bufs) CHECK(x.name.find("projection_on") == std::string::npos);
}

RAD_TEST_MAIN()

/* G-3: with the fusions on, the junctions write the codes of the next norm (their `group` 128 and the
 * pair's buffers), the readers of that norm declare no quantiser of their own, and with each switch
 * off the unfused chain is back. */
TEST(g3_junctions_write_the_codes_their_readers_take) {
    /* the step reads the model the last declare built, so each graph is stepped before the next */
    g3_fusions = true;
    Declared on(int8_trunk_encs);
    g3_fusions = false;
    REQUIRE_EQ(on.rc, RAD_OK);
    auto with_codes = [](const RadBuilder& b) {
        int n = 0;
        for (const auto& o : b.ops)
            if (o.op == "rmsnorm_ihc_junction" || o.op == "ar_rmsnorm_ihc_junction")
                for (const auto& p : o.p) n += p.key == "group";
        return n;
    };
    CHECK(with_codes(on.b) > 0);
    every_weight_read(on.b);
    steps_clean_at(on.b, 0, 1);
    const int had_on = count_op(on.b, "had_quant_act_i8g");
    Declared off(int8_trunk_encs);
    REQUIRE_EQ(off.rc, RAD_OK);
    CHECK_EQ(with_codes(off.b), 0);
    /* every junction that writes codes removes the quantiser behind it */
    CHECK(had_on < count_op(off.b, "had_quant_act_i8g"));
}
