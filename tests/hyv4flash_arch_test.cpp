/* Hy4-flash's declared graph and step (arch/hyv4flash), without a GPU or checkpoint: the harness's
 * `mini` geometry (ref/minicfg.py, the convert test's fixture) at one and two ranks, and its
 * `mini4` twin (twice the heads and experts) at one, two and four ranks, bf16. */
#define RAD_ARCH_NO_EXPORTS 1
#include "rad_w2.h"
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
/* Two Engram layers (in front of layers 0 and 2): the case P-06 is about, where the group is
 * shared -- one window, one bound layer, and only the last of the two layers commits a step. */
void two_engram_layers(MetaKV& k) {
    k.set("engram_layer_ids", "1 3");
    k.set("engram_layer_head_vocab_sizes.1", "11 13 17 19 23 29 31 37 41 43");
    k.set("engram_layer_head_vocab_sizes.3", "11 13 17 19 23 29 31 37 41 43");
}

struct Declared {
    MetaKV mk;
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    int rc = 0;
    explicit Declared(void (*encs)(RadBuilder*) = bf16_encs, int world = 1, int rank = 0,
                      bool wide = false, int64_t ff_exp = 128, int64_t latent = 0,
                      int64_t top_k = 0, int64_t n_exp = 0, void (*kvset)(MetaKV&) = nullptr)
        : mk(mini_meta(wide, ff_exp, 128, latent, top_k, n_exp)) {
        if (kvset) kvset(mk);
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
    CHECK_EQ(count_op(b, "qsa_work_pool_tail"), 2);     /* K-24: the key side, fused */
    CHECK_EQ(count_op(b, "qsa_work"), 0);
    CHECK_EQ(count_op(b, "idx_pool_gated"), 0);
    CHECK_EQ(count_op(b, "qsa_tail_store"), 2);         /* the long chunks' rows */
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

TEST(the_engram_layers_share_one_window) {
    Declared d(bf16_encs, 1, 0, false, 128, 0, 0, 0, two_engram_layers);
    REQUIRE_EQ(d.rc, RAD_OK);
    const RadBuilder& b = d.b;
    const RecKV* eg = kv_named(b, "kv_engram_tokens");
    REQUIRE(eg != nullptr);
    /* One window for the two layers: ngram - 1 ids a sequence (plus the speculative entries),
     * i32, and one bound layer, so a sequence holds one CONV state per checkpoint. */
    REQUIRE(eg->d.kind == RAD_KV_CONV);
    CHECK_EQ(eg->d.conv_width, 3);      /* the config's ngram */
    CHECK_EQ(eg->d.n_head_kv, 1);
    CHECK_EQ(eg->d.head_dim, 1);
    const rad_kvgroup g = (rad_kvgroup)(eg - b.kv_groups.data() + 1);
    /* Declared and bound once, to the first of the two layers: a sequence owns one CONV state per
     * checkpoint and a rollback copies one, not one a layer. */
    int nb = 0;
    for (const auto& bd : b.binds)
        if (bd.second == g) {
            ++nb;
            CHECK_EQ(bd.first, 0);
        }
    CHECK_EQ(nb, 1);
    /* A step with both layouts (a decode sequence and a prefill chunk): every Engram layer issues
     * its own `engram_ids` (the multipliers, sizes and offsets differ a layer), every issue reads
     * the window through that one bound layer, and only the last Engram layer's issue of a layout
     * advances it -- so all of them hash the window the step opened with. */
    auto k = kit(b, { 1 }, { 12 }, 0);
    RadCtx c{ .b = &d.b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    CHECK(c.step_fail.empty());
    CHECK_LINT(b, &c);
    int n_ids = 0, n_adv = 0;
    for (const RecIssue& is : c.issues) {
        REQUIRE(is.op != 0);
        const RecOp& o = b.ops[(size_t)is.op - 1];
        if (o.op != "engram_ids") continue;
        ++n_ids;
        bool kv = false;
        std::string mult;
        for (const RadOperand& od : is.opd) {
            if (od.kind == RAD_OPK_KV) {
                kv = true;
                CHECK_EQ((int64_t)od.handle, (int64_t)g);
                CHECK_EQ((int64_t)od.offset, 0);   /* the bound layer, both layers of the pair */
            }
            if (od.kind == RAD_OPK_WEIGHT && od.handle > 0 &&
                (size_t)od.handle <= b.weights.size()) {
                const std::string& n = b.weights[(size_t)od.handle - 1].name;
                if (n.size() > 4 && n.compare(n.size() - 4, 4, "mult") == 0) mult = n;
            }
        }
        CHECK(kv);
        REQUIRE(!mult.empty());
        const int adv = (int)pint(&o, "advance");
        CHECK_EQ(adv, mult == "blk.2.engram_mult" ? 1 : 0);   /* layer 2 is the last of the two */
        n_adv += adv;
    }
    CHECK_EQ(n_ids, 4);         /* one a layer, in each of the two layouts */
    CHECK_EQ(n_adv, 2);         /* the window moves on once a layout, not once a layer */
}

/* M-18b: the step-start prefetch. A disk-placed Engram layer's row reads (~3 ms a layer at decode)
 * must not serialize in front of the layers that wait for them: the first Engram layer's step
 * hashes each later disk-placed layer's ids into that layer's own planes and issues a gather whose
 * ahead operand names those ids themselves -- the host gather's prefetch signal (avx_ngram.cpp's
 * prefetch mode: hand the ids to the reader thread, read nothing). The layer's own ids and gather
 * follow at its position, unchanged: the step's outputs are the parent's byte for byte, only the
 * cache's timing moves. A device-placed table declares the planes and issues nothing. */
TEST(the_later_engram_layers_prefetch_at_the_step_start) {
    Declared d(bf16_encs, 1, 0, false, 128, 0, 0, 0, two_engram_layers);
    REQUIRE_EQ(d.rc, RAD_OK);
    const RadBuilder& b = d.b;
    /* The second of the two layers declares its own planes and the two ops that fill them; the
     * first, which opens the step, declares none. */
    const RecBuf* pf_ids = find_b(b, "eg.pf_ids.2");
    const RecBuf* pf_mask = find_b(b, "eg.pf_mask.2");
    const RecBuf* pf_rows = find_b(b, "eg.pf_rows.2");
    REQUIRE(pf_ids != nullptr);
    REQUIRE(pf_mask != nullptr);
    REQUIRE(pf_rows != nullptr);
    CHECK(find_b(b, "eg.pf_ids.0") == nullptr);
    CHECK_EQ(count_op(b, "engram_ids"), 5);        /* two layouts a layer and the second's early hash */
    CHECK_EQ(count_op(b, "embed_lookup_q"), 3);    /* a gather a layer and the second's prefetch */
    /* The prefetch ops are declared last (after the layer's own vocabulary), so they are the last
     * `engram_ids` op and the last `embed_lookup_q` op. */
    std::vector<rad_op> ids_ops, gathers;
    for (size_t i = 0; i < b.ops.size(); ++i) {
        if (b.ops[i].op == "engram_ids") ids_ops.push_back((rad_op)(i + 1));
        if (b.ops[i].op == "embed_lookup_q") gathers.push_back((rad_op)(i + 1));
    }
    REQUIRE_EQ((int)ids_ops.size(), 5);
    REQUIRE_EQ((int)gathers.size(), 3);
    const rad_op pf_ids_op = ids_ops.back(), pf_gather = gathers.back();
    auto k = kit(b, { 1 }, { 12 }, 0);
    k->b.positions_h = k->pos.data();
    const int64_t T = k->b.n_tok;
    const rad_buf pf_h = (rad_buf)(pf_ids - b.bufs.data() + 1);
    const rad_buf pm_h = (rad_buf)(pf_mask - b.bufs.data() + 1);
    const rad_buf pr_h = (rad_buf)(pf_rows - b.bufs.data() + 1);
    /* The prefetch is off by default and gated on the RESOLVED placement: every case below
     * says which way RADIANCE_HY4_ENGRAM_PREFETCH and rad_ngram_layer_resolved stand. Layer 2
     * is the one with prefetch planes; the fake's unset default is Config's own: global disk. */
    setenv("RADIANCE_HY4_ENGRAM_PREFETCH", "1", 1);
    radtest_set_ngram_resolved(2, RAD_TABLE_PLACE_DISK);
    /* A device-placed table: the step is the parent's, nothing hashed or read early. */
    {
        RadCtx c{ .b = &d.b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 4);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 2);
    }
    /* A host-placed (disk) table. */
    {
        RadCtx c{ .b = &d.b, .batch = &k->b };
        for (rad_op h : gathers) c.host_ops.push_back(h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 6);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 3);
        int pf_hash = 0, first_pf = -1, i = 0;
        for (const RecIssue& is : c.issues) {
            if (is.op == pf_ids_op) {
                ++pf_hash;
                if (first_pf < 0) first_pf = i;
                CHECK_EQ((int)pint(b.op_of(is.op), "advance"), 0);   /* the window is not moved */
                /* The step's own tokens, no ahead tail; the ids land in this layer's plane. */
                CHECK_EQ(is.opd[0].rows, T);
                CHECK_EQ(is.opd.back().kind, (uint8_t)RAD_OPK_BUF);
                CHECK_EQ((int64_t)is.opd.back().handle, (int64_t)pm_h);      /* pf_mask */
                CHECK_EQ((int64_t)is.opd[is.opd.size() - 2].handle, (int64_t)pf_h);   /* pf_ids */
                std::string mult;
                for (const RadOperand& od : is.opd)
                    if (od.kind == RAD_OPK_WEIGHT && od.handle > 0 &&
                        (size_t)od.handle <= b.weights.size()) {
                        const std::string& n = b.weights[(size_t)od.handle - 1].name;
                        if (n.size() > 4 && n.compare(n.size() - 4, 4, "mult") == 0) mult = n;
                    }
                CHECK(mult == "blk.2.engram_mult");   /* the SECOND layer's table */
            }
            ++i;
        }
        CHECK_EQ(pf_hash, 2);   /* both layouts, as the step's own ids are */
        /* The prefetch gather: its ahead operand names its own ids (the alias is the signal), its
         * output is the plane nothing reads, and it rides the step's start -- before the second
         * layer's own gather. */
        int ng = 0;
        const RecIssue* pfg = nullptr;
        int own2 = -1, pfg_at = -1;
        for (i = 0, ng = 0; i < (int)c.issues.size(); ++i) {
            const RecIssue& is = c.issues[(size_t)i];
            if (b.ops[(size_t)is.op - 1].op != "embed_lookup_q") continue;
            if (is.op == pf_gather) { pfg = &is; pfg_at = i; }
            else if (is.op == gathers[1]) own2 = i;   /* declared second: layer 2's own gather */
            ++ng;
        }
        REQUIRE(pfg != nullptr);
        CHECK_EQ(ng, 3);
        CHECK_EQ((int64_t)pfg->opd[0].handle, (int64_t)pf_h);          /* ids */
        CHECK_EQ((int64_t)pfg->opd.back().handle, (int64_t)pf_h);      /* ahead: the same plane */
        CHECK_EQ(pfg->opd.back().rows, T);
        bool saw_rows = false;
        for (const RadOperand& od : pfg->opd)
            saw_rows = saw_rows || (od.kind == (uint8_t)RAD_OPK_BUF &&
                                    (int64_t)od.handle == (int64_t)pr_h);
        CHECK(saw_rows);                                               /* out: pf_rows */
        CHECK(first_pf >= 0 && first_pf < pfg_at);                     /* hash before the read */
        CHECK(pfg_at < own2);   /* the step's start, ahead of the layer this read belongs to */
    }
    /* The gate is placement, not naming. A layer NAMED 2=ram whose locked copy is host-gathered
     * passes the domain test and would take a second full gather into pf_rows every step: the
     * resolved view refuses it, and the step is the parent's byte for byte -- zero prefetch
     * work issued for the named-ram layer. */
    {
        radtest_set_ngram_resolved(2, RAD_TABLE_PLACE_RAM);
        RadCtx c{ .b = &d.b, .batch = &k->b };
        for (rad_op h : gathers) c.host_ops.push_back(h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 4);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 2);
        radtest_set_ngram_resolved(2, RAD_TABLE_PLACE_DISK);
    }
    /* A global ram placement says the same for the layer no spec named: the resolved view is
     * ram for it too, and the host gather of its locked copy prefetches nothing. */
    {
        radtest_set_ngram_resolved(2, -1);              /* layer 2 unnamed */
        radtest_set_ngram_resolved(-1, RAD_TABLE_PLACE_RAM);
        RadCtx c{ .b = &d.b, .batch = &k->b };
        for (rad_op h : gathers) c.host_ops.push_back(h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 4);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 2);
    }
    /* And the other way: the global all-disk step (nothing named at all) IS a disk placement --
     * the gate reads the placement, so the unnamed layer prefetches exactly as a named-disk
     * one does. */
    {
        radtest_set_ngram_resolved(-1, -1);             /* back to the Config default: disk */
        RadCtx c{ .b = &d.b, .batch = &k->b };
        for (rad_op h : gathers) c.host_ops.push_back(h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 6);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 3);
        radtest_set_ngram_resolved(2, RAD_TABLE_PLACE_DISK);
    }
    /* Off by default: PREFETCH unset leaves the parent's step whatever the knob says. */
    {
        unsetenv("RADIANCE_HY4_ENGRAM_PREFETCH");
        RadCtx c{ .b = &d.b, .batch = &k->b };
        for (rad_op h : gathers) c.host_ops.push_back(h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 4);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 2);
    }
    /* Only =1 turns it on: =0 (the sync-hybrid A/B arm) and any other value leave the parent's
     * step. */
    for (const char* v : {"0", "2"}) {
        setenv("RADIANCE_HY4_ENGRAM_PREFETCH", v, 1);
        RadCtx c{ .b = &d.b, .batch = &k->b };
        for (rad_op h : gathers) c.host_ops.push_back(h);
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        CHECK_EQ(count_issued(b, c, "engram_ids"), 4);
        CHECK_EQ(count_issued(b, c, "embed_lookup_q"), 2);
    }
    unsetenv("RADIANCE_HY4_ENGRAM_PREFETCH");
    radtest_set_ngram_resolved(2, -1);    /* the fake back to Config default: global disk */
    radtest_set_ngram_resolved(-1, -1);
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
        CHECK_EQ(height(last.ix.op_front), 12);
        CHECK_EQ(height(last.ix.op_tail), 12);
        CHECK_EQ(height(last.attn.q_a.op), outputs);
        CHECK_EQ(height(last.ix.op_select), outputs);
        CHECK_EQ(height(last.attn.op_sel_pf), outputs);
        CHECK_EQ(height(last.attn.out.op), outputs);
        CHECK_EQ(height(hyv4flash::models[0]->op_head), outputs);
    }
}

/* K-24: the key side is one op where the three were. `qsa_work_pool_tail` publishes the work list
 * (`page`, `bpos`, `nc`) and pools it in one launch, so a step whose chunks cannot outgrow the ring
 * needs no `qsa_tail_store` either: the kernel stores a chunk's own rows when
 * `rows <= ring - ratio + 1`, and `T - S + 1` bounds the longest chunk of a step of T tokens over
 * S sequences. Above that bound the store stays, after every gather of the step. */
TEST(k24_the_key_side_is_one_op_and_its_tail_store_only_past_the_ring) {
    RadBuilder b;
    bf16_encs(&b);
    auto mk = mini_meta();
    auto ctx = build_ctx();
    REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto& last = hyv4flash::models[0]->layers.back();
    REQUIRE(last.ix.op_front != 0);          /* the exact form fuses */
    REQUIRE_EQ(last.ix.op_work, (rad_op)0);
    REQUIRE_EQ(last.ix.op_pool, (rad_op)0);
    auto has = [](const RecIssue* i, rad_buf buf) {
        for (const RadOperand& o : i->opd)
            if (o.kind == (uint8_t)RAD_OPK_BUF && o.handle == buf) return true;
        return false;
    };
    struct Case { int64_t q, tail; };
    for (const Case cs : { Case{ 1, 0 }, Case{ 2, 1 }, Case{ 12, 1 } }) {
        auto k = kit(b, {}, { (int)cs.q });
        RadCtx c{ .b = &b, .batch = &k->b };
        hyv4flash::step(&c, &k->b);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        /* the three ops the item fuses are never issued while the fused one is */
        CHECK(issue_of(b, c, "qsa_work") == nullptr);
        CHECK(issue_of(b, c, "idx_pool_gated") == nullptr);
        const RecIssue* f = issue_of(b, c, "qsa_work_pool_tail");
        REQUIRE(f != nullptr);
        CHECK_EQ(f->n, cs.q);
        CHECK(has(f, last.ix.w.page));                 /* the work list travels with the pool */
        CHECK(has(f, last.ix.w.bpos));
        CHECK(has(f, last.ix.w.nc));
        CHECK(has(f, last.ix.w.tk));                   /* the keys it reads come from the step */
        CHECK_EQ(issue_of(b, c, "qsa_tail_store") != nullptr, cs.tail != 0);
    }
}

/* K-24, the other side of the same switch: the fused key side is one library's row, not a property
 * of the model, so a library that does not carry it (libref, libavx, a libr4d older than the
 * kernel) must still get a hyv4flash that declares -- and it declares the three ops, which are the
 * reference path (docs/OPS.md). The fused op is asked for only where one serves it
 * (rad_op_declared), which is why hyv4flash_convert_test declares on the host backend at all. */
TEST(k24_a_library_without_the_fused_key_side_keeps_the_three_ops) {
    RadBuilder b;
    bf16_encs(&b);
    b.undeclared.insert("qsa_work_pool_tail");
    auto mk = mini_meta();
    auto ctx = build_ctx();
    REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto& last = hyv4flash::models[0]->layers.back();
    CHECK_EQ(last.ix.op_front, (rad_op)0);
    REQUIRE(last.ix.op_work != 0);
    REQUIRE(last.ix.op_pool != 0);
    REQUIRE(last.ix.op_tail != 0);
    auto k = kit(b, {}, { 1 });
    RadCtx c{ .b = &b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    CHECK(c.step_fail.empty());
    CHECK_LINT(b, &c);
    CHECK_EQ(issue_of(b, c, "qsa_work_pool_tail"), nullptr);
    const RecIssue* w = issue_of(b, c, "qsa_work");
    const RecIssue* p = issue_of(b, c, "idx_pool_gated");
    REQUIRE(w != nullptr);
    REQUIRE(p != nullptr);
    /* the pool is issued over the work list qsa_work filled in, which is what the fused op does in
     * shared memory instead */
    CHECK(w->opd[7].handle == p->opd[0].handle);
    CHECK(p->opd[5].handle == w->opd[9].handle);   /* its positions are qsa_work's bpos */
    CHECK(p->opd[6].handle == w->opd[8].handle);   /* and its page, qsa_work's page */
    /* and the tail's rows are copied beside them, as before K-24 */
    CHECK(issue_of(b, c, "qsa_tail_store") != nullptr);
}

/* And the third case between them: a library that *declares* the fused key side -- libr4d does --
 * need not serve every geometry of it. Its kernel is compiled for the indexer the model has (n
 * 128, the row's constraint), where the conversion tests build 16-wide ones, so `mk` returns a
 * null handle there; that is the three-op path's cue, not a missing library, and declaring it is
 * not an error (`rad_op_resolved`, abi/rad_builder.h). A `HY4_NEED` on that handle would fail
 * every narrow declare instead. */
TEST(k24_a_declared_but_unserved_fused_key_side_still_declares) {
    RadBuilder b;
    bf16_encs(&b);
    b.refuse.insert("qsa_work_pool_tail");
    auto mk = mini_meta();
    auto ctx = build_ctx();
    REQUIRE_EQ(hyv4flash::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto& last = hyv4flash::models[0]->layers.back();
    CHECK_EQ(last.ix.op_front, (rad_op)0);
    REQUIRE(last.ix.op_work != 0);
    REQUIRE(last.ix.op_pool != 0);
    REQUIRE(last.ix.op_tail != 0);
    auto k = kit(b, {}, { 1 });
    RadCtx c{ .b = &b, .batch = &k->b };
    hyv4flash::step(&c, &k->b);
    CHECK(c.step_fail.empty());
    CHECK_LINT(b, &c);
    CHECK_EQ(issue_of(b, c, "qsa_work_pool_tail"), nullptr);
    CHECK(issue_of(b, c, "qsa_work") != nullptr);
    CHECK(issue_of(b, c, "idx_pool_gated") != nullptr);
    CHECK(issue_of(b, c, "qsa_tail_store") != nullptr);
    /* and every unserved declaration of it is marked dropped (rad_op_dropped), so rad-convert's
     * dead-op count and the engine's holes leave it out: the three ops run in its place */
    int declared = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "qsa_work_pool_tail") {
            ++declared;
            CHECK(o.dropped);
        }
    CHECK(declared > 0);
    for (const RecOp& o : b.ops)
        if (o.op != "qsa_work_pool_tail") CHECK(!o.dropped);
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

/* X-05's Change line, as a guard rather than as prose. The backlog's X-05 entry asks that `hyv4flash`
 * set `kv_stream_ok` and that `mla_attn_sel_sink` read a selection of the sparse latent; both shipped
 * ahead of this card -- the flag at `arch/hyv4flash/hyv4flash.cpp:1603` (`489fd4f4`), the op in
 * `libavx/avx_hy4.cpp:520` -- and the rows X-05a adds to GUIDE §4.4.1 and ARCHITECTURES state which
 * of the three latent/index groups streams and which is read whole. That claim now fails a test
 * instead of a reader's trust: the probe claims the capability, the selection-reading attention is
 * declared, the indexer's pooled keys are dense-read (the scorer reads every group of the history
 * every step, so they are never a streamed group), and the two latent groups are not. */
TEST(x05_which_group_streams_and_which_is_read_whole) {
    Declared d;
    REQUIRE_EQ(d.rc, RAD_OK);
    RadBuilder& b = d.b;

    RadArchProbe pr{};
    REQUIRE_EQ(hyv4flash::probe(d.mk.meta(), &pr), RAD_OK);
    CHECK_EQ(pr.kv_stream_ok, 1);                  /* the item's first clause */
    CHECK(count_op(b, "mla_attn_sel_sink") > 0);    /* and its second: a selection is read */

    for (const char* n : { "kv_latent_sparse", "kv_latent_sliding" }) {
        const RecKV* k = kv_named(b, n);
        REQUIRE(k != nullptr);
        CHECK(!k->dense);                          /* paged; the sparse one is the streamed one */
    }
    const RecKV* ix = kv_named(b, "kv_index_keys");
    REQUIRE(ix != nullptr);
    CHECK(ix->dense);                              /* `rad_kv_group_dense_read`, never streamed */
    CHECK_EQ(count_op(b, "kv_store_streamed"), 2);  /* the sparse layers, and nothing else */
    CHECK_EQ(count_op(b, "kv_resolve"), 2);
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
        CHECK_EQ(count_op(b, "idx_pool_gated"), 0);
        CHECK_EQ(count_op(b, "qsa_work_pool_tail"), q ? 0 : 2);   /* K-24: exact only */
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
    CHECK_EQ(pstr(find_op(b, "qsa_work_pool_tail"), "mode"), std::string("imrope_gptj"));
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
/* W2 (abi/rad_w2.h, issue #309): the routed layer of a decode-phase step issued as one pass.
 *
 * With the core's switch off (rad_expert_w2 returns null) the step issues what it always did: no
 * routed GEMM and no router carries a W2 operand. With it on, each routed layer's router carries
 * the control block as its last operand (it plans there), its two routed GEMMs carry it as `w2`
 * (the resident experts), and behind the first down GEMM comes the tail: gate_up, the gated
 * quantiser's issues again, down, each GEMM with the block as `w2_tail` and at the pass height of the first
 * launch (the same tile geometry and sums). Nothing else of the step changes: with the tail's
 * three issues and the W2 operands taken out, the issues are the switch-off step's. A pass below
 * the core's row threshold, and a prefill step, issue the switch-off step. */
TEST(w2_splits_a_decode_passes_routed_layer_into_the_resident_launches_and_a_tail) {
    /* Hy4-flash's served forms: an IQR gate_up and a six-bit codebook down. */
    Declared d(+[](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = enc_trellis(RAD_U4, true);
        b->encs[1].second = enc_nl64(6);
    });
    REQUIRE_EQ(d.rc, RAD_OK);
    static char block[sizeof(RadW2Layer)];
    auto opname = [&](const RecIssue& i) {
        const RecOp* o = d.b.op_of(i.op);
        return o ? o->op : std::string("<null>");
    };
    auto same = [](const RadOperand& a, const RadOperand& b) {
        return a.kind == b.kind && a.dtype == b.dtype && a.handle == b.handle && a.raw == b.raw &&
               a.offset == b.offset && a.rows == b.rows && a.cols == b.cols;
    };
    auto is_w2 = [&](const RadOperand& o) {
        return o.kind == RAD_OPK_RAW && o.raw == (void*)block && o.dtype == RAD_U8 &&
               o.rows == (int64_t)sizeof(RadW2Layer);
    };
    auto k = kit(d.b, { 1, 1, 1 }, {}, 0);
    REQUIRE_EQ(k->b.phase, (int)RAD_PHASE_DECODE);

    RadCtx off{ .b = &d.b, .batch = &k->b };
    hyv4flash::step(&off, &k->b);
    CHECK(off.step_fail.empty());
    CHECK(off.w2_asked >= 2);          /* every routed layer asked, and was told no */
    CHECK(off.w2_tails.empty());
    int gemms_off = 0;
    for (const RecIssue& i : off.issues) {
        for (const RadOperand& o : i.opd) CHECK(!is_w2(o));
        if (opname(i) == "moe_gemm_q") { ++gemms_off; CHECK_EQ(i.opd.size(), (size_t)9); }
        if (opname(i) == "router_sigmoid_topk_scatter") CHECK(i.opd.size() <= (size_t)9);
    }
    CHECK_EQ(gemms_off, 4);            /* two MoE layers, gate_up and down */

    RadCtx on{ .b = &d.b, .batch = &k->b };
    on.w2 = block;
    hyv4flash::step(&on, &k->b);
    CHECK(on.step_fail.empty());
    /* The routers plan, the first GEMMs carry `w2`, the tail's `w2_tail`. */
    std::vector<size_t> gemm, tail_issue;
    for (size_t x = 0; x < on.issues.size(); ++x) {
        const RecIssue& i = on.issues[x];
        if (opname(i) == "router_sigmoid_topk_scatter") {
            REQUIRE_EQ(i.opd.size(), (size_t)10);
            CHECK(is_w2(i.opd[9]));
        }
        if (opname(i) == "moe_gemm_q") gemm.push_back(x);
    }
    REQUIRE_EQ(gemm.size(), (size_t)8);
    for (size_t l = 0; l < 2; ++l) {
        const RecIssue &gu = on.issues[gemm[4 * l]], &dn = on.issues[gemm[4 * l + 1]];
        const RecIssue &tgu = on.issues[gemm[4 * l + 2]], &tdn = on.issues[gemm[4 * l + 3]];
        for (const RecIssue* f : { &gu, &dn }) {
            REQUIRE_EQ(f->opd.size(), (size_t)11);
            CHECK_EQ((int)f->opd[9].kind, (int)RAD_OPK_NONE);      /* split */
            CHECK(is_w2(f->opd[10]));
        }
        for (const RecIssue* t : { &tgu, &tdn }) {
            REQUIRE_EQ(t->opd.size(), (size_t)12);
            CHECK_EQ((int)t->opd[9].kind, (int)RAD_OPK_NONE);
            CHECK_EQ((int)t->opd[10].kind, (int)RAD_OPK_NONE);     /* w2: the tail's is the next */
            CHECK(is_w2(t->opd[11]));
        }
        /* The tail is the first launches' ops over the same operands at the same height. */
        CHECK_EQ(tgu.op, gu.op);
        CHECK_EQ(tdn.op, dn.op);
        CHECK_EQ(tgu.n, gu.n);
        CHECK_EQ(tdn.n, dn.n);
        for (size_t o = 0; o < 9; ++o) {
            CHECK(same(tgu.opd[o], gu.opd[o]));
            CHECK(same(tdn.opd[o], dn.opd[o]));
        }
        /* It sits right behind the first down GEMM: gate_up, the expert quantiser's issues again
         * (in their order, the ones between the first GEMMs that read gate_up's output), down. */
        REQUIRE_EQ(gemm[4 * l + 2], gemm[4 * l + 1] + 1);
        REQUIRE(gemm[4 * l + 3] > gemm[4 * l + 2] + 1);
        size_t at = gemm[4 * l] + 1;
        for (size_t x = gemm[4 * l + 2] + 1; x < gemm[4 * l + 3]; ++x) {
            const RecIssue& tq = on.issues[x];
            bool found = false;
            for (; at < gemm[4 * l + 1] && !found; ++at) {
                const RecIssue& q = on.issues[at];
                if (q.op != tq.op || q.n != tq.n || q.opd.size() != tq.opd.size()) continue;
                found = true;
                for (size_t o = 0; o < q.opd.size(); ++o) found = found && same(q.opd[o], tq.opd[o]);
            }
            CHECK(found);
        }
        /* the core is told right in front of the tail's first launch (where its ring is cached
         * memory, that launch acquires) */
        REQUIRE_EQ(on.w2_tails.size(), (size_t)2);
        CHECK_EQ(on.w2_tails[l], gemm[4 * l + 2]);
        for (size_t x = gemm[4 * l + 2]; x <= gemm[4 * l + 3]; ++x) tail_issue.push_back(x);
    }
    /* Without the tail and the W2 operands the step is the switch-off step. */
    std::vector<RecIssue> rest;
    for (size_t x = 0; x < on.issues.size(); ++x) {
        if (std::find(tail_issue.begin(), tail_issue.end(), x) != tail_issue.end()) continue;
        RecIssue i = on.issues[x];
        while (!i.opd.empty() && (is_w2(i.opd.back()) || (i.opd.back().kind == RAD_OPK_NONE &&
                                                           opname(i) == "moe_gemm_q")))
            i.opd.pop_back();
        rest.push_back(i);
    }
    REQUIRE_EQ(rest.size(), off.issues.size());
    for (size_t x = 0; x < rest.size(); ++x) {
        CHECK_EQ(rest[x].op, off.issues[x].op);
        CHECK_EQ(rest[x].n, off.issues[x].n);
        CHECK_EQ(rest[x].lane, off.issues[x].lane);
        REQUIRE_EQ(rest[x].opd.size(), off.issues[x].opd.size());
        for (size_t o = 0; o < rest[x].opd.size(); ++o) CHECK(same(rest[x].opd[o], off.issues[x].opd[o]));
    }

    /* A pass under the core's threshold is told no and issues the switch-off step. */
    RadCtx narrow{ .b = &d.b, .batch = &k->b };
    narrow.w2 = block;
    narrow.w2_min_rows = 4;
    hyv4flash::step(&narrow, &k->b);
    REQUIRE_EQ(narrow.issues.size(), off.issues.size());
    for (size_t x = 0; x < narrow.issues.size(); ++x) {
        CHECK_EQ(narrow.issues[x].op, off.issues[x].op);
        CHECK_EQ(narrow.issues[x].opd.size(), off.issues[x].opd.size());
    }
    /* A prefill step does not ask at all. */
    auto pk = kit(d.b, {}, { 12 }, 0);
    RadCtx pre{ .b = &d.b, .batch = &pk->b };
    pre.w2 = block;
    hyv4flash::step(&pre, &pk->b);
    CHECK_EQ(pre.w2_asked, 0);
    for (const RecIssue& i : pre.issues)
        for (const RadOperand& o : i.opd) CHECK(!is_w2(o));
}

/* Issue #324: W2 only where the routed GEMMs read its operands. Above their decode band
 * (rad_w2_in_band: 2 x rows x top_k <= 3 x n_expert; the preset's 4 experts at top-2 hold 3 rows)
 * the GEMMs take a prefill form, which runs every expert on the first launch and again on the tail:
 * the whole routed layer twice, which cost a DFly2 c16 verify pass (64-96 rows of Hy4-flash's 512
 * experts at top-16, a band of 48) a fifth of its rate. Such a pass does not ask for the form, and
 * issues the switch-off step; the band's last row still takes it. */
TEST(w2_leaves_a_pass_above_the_routed_gemms_decode_band_to_the_one_launch_form) {
    Declared d(+[](RadBuilder* b) {
        bf16_encs(b);
        b->encs[0].second = enc_trellis(RAD_U4, true);
        b->encs[1].second = enc_nl64(6);
    });
    REQUIRE_EQ(d.rc, RAD_OK);
    static char block[sizeof(RadW2Layer)];
    auto is_w2 = [&](const RadOperand& o) {
        return o.kind == RAD_OPK_RAW && o.raw == (void*)block && o.dtype == RAD_U8 &&
               o.rows == (int64_t)sizeof(RadW2Layer);
    };
    CHECK(rad_w2_in_band(3 * 2, 4));
    CHECK(!rad_w2_in_band(4 * 2, 4));

    /* the band's last row: the form, a tail a routed layer */
    auto k3 = kit(d.b, { 1, 1, 1 }, {}, 0);
    RadCtx edge{ .b = &d.b, .batch = &k3->b };
    edge.w2 = block;
    hyv4flash::step(&edge, &k3->b);
    CHECK(edge.step_fail.empty());
    CHECK(edge.w2_asked >= 2);
    CHECK_EQ(edge.w2_tails.size(), (size_t)2);

    /* a row past it: not asked, no tail, no W2 operand, the switch-off step's issues */
    auto k4 = kit(d.b, { 1, 1, 1, 1 }, {}, 0);
    REQUIRE_EQ(k4->b.phase, (int)RAD_PHASE_DECODE);
    RadCtx off{ .b = &d.b, .batch = &k4->b };
    hyv4flash::step(&off, &k4->b);
    CHECK(off.step_fail.empty());
    RadCtx wide{ .b = &d.b, .batch = &k4->b };
    wide.w2 = block;
    hyv4flash::step(&wide, &k4->b);
    CHECK(wide.step_fail.empty());
    CHECK_EQ(wide.w2_asked, 0);
    CHECK(wide.w2_tails.empty());
    REQUIRE_EQ(wide.issues.size(), off.issues.size());
    for (size_t x = 0; x < wide.issues.size(); ++x) {
        CHECK_EQ(wide.issues[x].op, off.issues[x].op);
        REQUIRE_EQ(wide.issues[x].opd.size(), off.issues[x].opd.size());
        for (const RadOperand& o : wide.issues[x].opd) CHECK(!is_w2(o));
    }
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

/* #282: the pipelined prefill, run card-less. With the floor and the cut moved to the GEMMs' 64-row
 * grain (RADIANCE_HY4_PIPE_MIN / _ROWS), a 128-row fresh prefill cuts 64 + 64, the shape a served
 * 4096-row chunk takes at 2048 + 2048. The int8 recipe makes the junctions write the G-3b codes the
 * feed-forward's and the attention's input linears read. Checked against the same graph declared
 * with the pipeline off (the same ops, the sizing test above):
 *  - every junction piece carries every buffer the whole step's issue of that op carries: a piece
 *    without the codes left the linears behind it reading stale ones (on-card: long prompts off
 *    topic, and different from run to run);
 *  - every buffer an issue on the second lane touches is named concurrent (lane1_unnamed): the last
 *    junction's pieces wrote `final_hidden` there unnamed, into bytes lane 0's arms could hold;
 *  - the piped step is lint clean;
 *  - a cut whose second piece is under the floor does not pipe (96 rows: 64 + 32), the gate's
 *    both-pieces rule (it used to check the first piece only, so 1984 rows piped as 1024 + 960). */
void pipe_i8_encs(RadBuilder* b) {
    bf16_encs(b);   /* only the replicated linears int8: a rank's shard of the others is too narrow */
    b->encs.insert(b->encs.begin(), { { "ffn_latent_in", enc_i8(true) }, { "attn_q_a.", enc_i8(true) } });
}
TEST(the_pipelined_prefill_pieces_are_the_whole_steps_junctions) {
    const bool g3 = g3_fusions;
    g3_fusions = true;
    setenv("RADIANCE_HY4_PIPE_MIN", "64", 1);
    setenv("RADIANCE_HY4_PIPE_ROWS", "64", 1);
    auto declare = [](RadBuilder& b, MetaKV& mk, int world, bool pipe) {
        if (pipe) setenv("RADIANCE_HY4_PIPE", "1", 1);
        else unsetenv("RADIANCE_HY4_PIPE");
        pipe_i8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        G3Env g3env;
        const int rc = hyv4flash::declare(&b, mk.meta(), &ctx);
        unsetenv("RADIANCE_HY4_PIPE");
        return rc;
    };
    /* The buffers each issue of `op` names, unioned over its issues. */
    auto bufs_of = [](const RadCtx& c, rad_op op) {
        std::set<rad_buf> v;
        for (const RecIssue& is : c.issues)
            if (is.op == op)
                for (const RadOperand& od : is.opd)
                    if (od.kind == RAD_OPK_BUF && od.handle) v.insert(od.handle);
        return v;
    };
    for (int world : { 2, 4 }) {
        MetaKV mk = mini_meta(true, 128, 256);
        RadBuilder whole, piped;
        REQUIRE_EQ(declare(whole, mk, world, false), RAD_OK);
        REQUIRE_EQ(declare(piped, mk, world, true), RAD_OK);
        REQUIRE_EQ(whole.ops.size(), piped.ops.size());
        REQUIRE(buf_handle(piped, "trunk_codes.junc_ffn_in.256.128.q8") != 0);
        auto kw = kit(whole, {}, { 128 });
        auto kp = kit(piped, {}, { 128 });
        kw->b.positions_h = kw->pos.data();
        kp->b.positions_h = kp->pos.data();
        auto vw = run_passes(whole, hyv4flash::step, { kw.get() }, 0, world);
        auto vp = run_passes(piped, hyv4flash::step, { kp.get() }, 0, world);
        CHECK(steps_ok(vw));
        CHECK_PASSES(piped, vp);
        const RadCtx& cw = *vw[0];
        const RadCtx& cp = *vp[0];
        /* It piped: the junctions ran as 64-row pieces, their reductions on the second lane. */
        int lane1 = 0, pieces = 0, junctions = 0;
        for (const RecIssue& is : cp.issues) {
            lane1 += is.lane == 1;
            const RecOp* o = piped.op_of(is.op);
            if (o && o->op == "rmsnorm_ihc_junction") pieces += is.n == 64;
        }
        CHECK(lane1 > 0);
        CHECK(pieces > 0);
        /* Every buffer a junction declares it writes is an operand of every piece's issue (the
         * whole step's issue passes them all; the reads may differ, the chain's add having taken
         * `b`). At four ranks the whole step runs the same op, so its operands are compared too
         * (at two the whole step takes the fused twin, another op). */
        for (const RecIssue& is : cp.issues) {
            const RecOp* o = piped.op_of(is.op);
            if (!o || o->op != "rmsnorm_ihc_junction") continue;
            ++junctions;
            std::set<rad_buf> got;
            for (const RadOperand& od : is.opd)
                if (od.kind == RAD_OPK_BUF) got.insert(od.handle);
            for (rad_buf h : o->writes)
                if (!got.count(h)) {
                    fprintf(stderr, "    world %d: a %lld-row junction issue leaves out '%s'\n",
                            world, (long long)is.n, piped.bufs[h - 1].name.c_str());
                    CHECK(got.count(h) != 0);
                }
        }
        for (size_t k = 0; k < whole.ops.size(); ++k) {
            if (whole.ops[k].op != "rmsnorm_ihc_junction") continue;
            const std::set<rad_buf> w = bufs_of(cw, (rad_op)(k + 1)), p = bufs_of(cp, (rad_op)(k + 1));
            if (!w.empty()) CHECK(w == p);
        }
        CHECK(junctions > 0);
        CHECK_EQ(report_lint(lane1_unnamed(piped, view(vp))), (size_t)0);
        /* 96 rows cut 64 + 32: the second piece is under the floor, so the step runs whole. */
        auto k96 = kit(piped, {}, { 96 });
        k96->b.positions_h = k96->pos.data();
        auto v96 = run_passes(piped, hyv4flash::step, { k96.get() }, 0, world);
        CHECK_PASSES(piped, v96);
        int l96 = 0, short96 = 0;
        for (const RecIssue& is : v96[0]->issues) {
            l96 += is.lane == 1;
            const RecOp* o = piped.op_of(is.op);
            if (o && o->op == "rmsnorm_ihc_junction") short96 += is.n != 96;
        }
        CHECK_EQ(l96, 0);
        CHECK_EQ(short96, 0);
    }
    unsetenv("RADIANCE_HY4_PIPE_MIN");
    unsetenv("RADIANCE_HY4_PIPE_ROWS");
    g3_fusions = g3;
}

/* P-17 (#301): the attention core in the pipeline's two pieces (RADIANCE_HY4_PIPE_ATTN=1), run
 * card-less at 128 rows cut 64 + 64, fresh (the sliding layer's dense form, the sparse layers'
 * selection) and behind a 96-token cached prefix (the paged form), at two and four ranks:
 *  - with the switch the declare is the same with the pipeline on and off, op for op (sizing), and
 *    without it the bounds and their two launches are not declared at all;
 *  - every attention launch of the step is a 64-row piece on lane 0, and lane 1 runs exactly what it
 *    runs without the split (only junction pieces);
 *  - each piece's bounds, replayed from the step's own integer launches (IntReplay), put its first
 *    query at its own position: the paged form's `used - qlen` at positions[r0], the dense form's
 *    `n - m` at key r0 over keys from row 0, the selection form's tables at row r0;
 *  - no two issues the lanes may run at once touch the same elements with one writing (lane_races),
 *    with the split and without; every buffer lane 1 touches is named (lane1_unnamed); lint clean;
 *  - with the switch off the attention runs whole, as before. */
TEST(the_attention_pieces_bound_each_piece_at_its_own_rows) {
    const bool g3 = g3_fusions;
    g3_fusions = true;
    setenv("RADIANCE_HY4_PIPE_MIN", "64", 1);
    setenv("RADIANCE_HY4_PIPE_ROWS", "64", 1);
    auto declare = [](RadBuilder& b, MetaKV& mk, int world, bool pipe) {
        if (pipe) setenv("RADIANCE_HY4_PIPE", "1", 1);
        else unsetenv("RADIANCE_HY4_PIPE");
        setenv("RADIANCE_HY4_PIPE_ATTN", "1", 1);   /* the bounds plane is declared with the switch */
        pipe_i8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        G3Env g3env;
        const int rc = hyv4flash::declare(&b, mk.meta(), &ctx);
        unsetenv("RADIANCE_HY4_PIPE");
        unsetenv("RADIANCE_HY4_PIPE_ATTN");
        return rc;
    };
    /* the rebased bounds plane's extent: the pieces' eight bounds ride behind its entries */
    auto cu_rows = [](const RadBuilder& b) {
        for (const RecBuf& r : b.bufs)
            if (r.name.size() >= 10 && r.name.compare(r.name.size() - 10, 10, "cu_prefill") == 0)
                return r.d.shape[0];
        return (int64_t)0;
    };
    auto is_attn = [](const RecOp* o) {
        return o && (o->op == "mla_attn_paged_sink" || o->op == "mla_attn_sel_sink" ||
                     o->op == "attn_kv_sink");
    };
    auto pitch_of = [](const RadBuilder& b, rad_buf h) {
        const RadBufDecl& d = b.bufs[h - 1].d;
        int64_t p = 1;
        for (uint32_t k = 1; k < d.rank; ++k) p *= d.shape[k];
        return p;
    };
    auto lane1_ops = [](const RadCtx& c) {
        std::vector<std::pair<rad_op, int64_t>> v;
        for (const RecIssue& is : c.issues)
            if (is.lane == 1) v.emplace_back(is.op, is.n);
        return v;
    };
    for (int world : { 2, 4 })
        for (int32_t prefix : { 0, 96 }) {
            MetaKV mk = mini_meta(true, 128, 256);
            RadBuilder whole, piped;
            REQUIRE_EQ(declare(whole, mk, world, false), RAD_OK);
            REQUIRE_EQ(declare(piped, mk, world, true), RAD_OK);
            REQUIRE_EQ(whole.ops.size(), piped.ops.size());
            REQUIRE_EQ(whole.bufs.size(), piped.bufs.size());
            REQUIRE(cu_rows(piped) > 8);
            auto run = [&](bool split) {
                if (split) setenv("RADIANCE_HY4_PIPE_ATTN", "1", 1);
                else unsetenv("RADIANCE_HY4_PIPE_ATTN");
                auto k = kit(piped, {}, { 128 }, prefix);
                k->b.positions_h = k->pos.data();
                auto v = run_passes(piped, hyv4flash::step, { k.get() }, 0, world);
                unsetenv("RADIANCE_HY4_PIPE_ATTN");
                return std::make_pair(std::move(k), std::move(v));
            };
            auto off = run(false);
            auto on = run(true);
            CHECK_PASSES(piped, off.second);
            CHECK_PASSES(piped, on.second);
            const RadCtx& c0 = *off.second[0];
            const RadCtx& c1 = *on.second[0];
            const Kit& kt = *on.first;
            /* the attention launches: whole without the switch, 64-row pieces on lane 0 with it */
            int whole_attn = 0, pieces = 0, wrong = 0;
            for (const RecIssue& is : c0.issues)
                if (is_attn(piped.op_of(is.op))) whole_attn += is.n == 128;
            CHECK(whole_attn > 0);
            IntReplay rep;
            rep.run(piped, c1, c1.issues.size());
            std::map<rad_op, int64_t> sel_base;
            for (const RecIssue& is : c1.issues) {
                const RecOp* o = piped.op_of(is.op);
                if (!is_attn(o)) continue;
                if (is.n != 64 || is.lane != 0) {
                    ++wrong;
                    continue;
                }
                ++pieces;
                const RadOperand& out = is.opd.back();
                const int64_t r0 = out.offset / pitch_of(piped, out.handle);
                CHECK(r0 == 0 || r0 == 64);
                const RadOperand& q = is.opd[0];
                CHECK_EQ(q.offset / pitch_of(piped, q.handle), r0);
                if (o->op == "mla_attn_paged_sink") {
                    const int64_t qlen = rep.at(piped, is.opd[5], 1) - rep.at(piped, is.opd[5], 0);
                    const int64_t first = rep.at(piped, is.opd[4], 0) - qlen;
                    CHECK_EQ(qlen, (int64_t)64);
                    if (first != kt.pos[(size_t)r0])
                        fprintf(stderr, "    world %d prefix %d: paged piece at row %lld sees position %lld, "
                                "not %d\n", world, prefix, (long long)r0, (long long)first, kt.pos[(size_t)r0]);
                    CHECK_EQ(first, (int64_t)kt.pos[(size_t)r0]);
                } else if (o->op == "attn_kv_sink") {
                    const int64_t m = rep.at(piped, is.opd[3], 1) - rep.at(piped, is.opd[3], 0);
                    const int64_t nk = rep.at(piped, is.opd[4], 1) - rep.at(piped, is.opd[4], 0);
                    CHECK_EQ(prefix, 0);
                    CHECK_EQ(m, (int64_t)64);
                    CHECK_EQ(nk - m, r0);
                    CHECK_EQ(is.opd[1].offset, (int64_t)0);
                    CHECK(is.opd[1].rows >= nk);
                } else {
                    /* the selection's tables at the piece's rows: `len` a row each, `sel` at its
                     * window's base (P-09) plus r0 rows */
                    CHECK_EQ(is.opd[4].offset, r0);
                    const int64_t K = pitch_of(piped, is.opd[3].handle);
                    if (!r0) sel_base[is.op] = is.opd[3].offset;
                    else CHECK_EQ(is.opd[3].offset, sel_base[is.op] + r0 * K);
                }
            }
            CHECK_EQ(wrong, 0);
            CHECK_EQ(pieces, 2 * whole_attn);
            CHECK(lane1_ops(c1) == lane1_ops(c0));
            CHECK(!lane1_ops(c1).empty());
            CHECK_EQ(report_lint(lane_races(piped, view(off.second))), (size_t)0);
            CHECK_EQ(report_lint(lane_races(piped, view(on.second))), (size_t)0);
            CHECK_EQ(report_lint(lane1_unnamed(piped, view(on.second))), (size_t)0);
            /* last: a declare replaces the model the step runs */
            {
                RadBuilder plain;
                setenv("RADIANCE_HY4_PIPE", "1", 1);
                pipe_i8_encs(&plain);
                RadBuildCtx ctx = build_ctx(0, world);
                G3Env g3env;
                REQUIRE_EQ(hyv4flash::declare(&plain, mk.meta(), &ctx), RAD_OK);
                unsetenv("RADIANCE_HY4_PIPE");
                CHECK_EQ(cu_rows(plain) + 8, cu_rows(piped));
                CHECK_EQ(plain.bufs.size(), piped.bufs.size());
                CHECK_EQ(plain.ops.size() + 2, piped.ops.size());
            }
        }
    unsetenv("RADIANCE_HY4_PIPE_MIN");
    unsetenv("RADIANCE_HY4_PIPE_ROWS");
    g3_fusions = g3;
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
    /* `refuse`: ops this build's kernels do not serve, so the declare must fall back (D-01). */
    explicit DeclDfly(int depth = 5, MetaKV m = dfly_meta(), int world = 1, int rank = 0,
                      void (*encs)(RadBuilder*) = bf16_encs,
                      const std::vector<const char*>& refuse = {}, int max_seqs = 8)
        : mk(std::move(m)) {
        encs(&b);
        for (const char* r : refuse) b.refuse.insert(r);
        ctx = spec_ctx(rank, world, depth);
        ctx.max_seqs = max_seqs;
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
const char* kTrunkOnly[] = { "qsa_work", "qsa_work_pool_tail", "qsa_tail_store", "idx_pool_gated",
                             "idx_score", "qsa_select", "engram_ids", "ihc_pre", "ihc_post",
                             "idx_pool_gated_fp8", "dsa_score", "attn_kv_sink",
                             "mla_attn_paged_sink" };
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
            /* D-01: the row's norm, key rotary and store are one launch where declared */
            const rad_op h = ps[2]->issues[i].op;
            if ((h == df.lay[0].op_store || h == df.lay[0].op_row_q) && store == SIZE_MAX) store = i;
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

/* D-03 (#89): the drafter's two reductions a layer take the fused twin where a library serves it --
 * ar_add_rmsnorm: the exact all-reduce of the block's partial sum with the residual add and the RMS
 * norm beside it, one launch, in either order. The output projection's chain is all_reduce,
 * post-norm, add (mode 0, reading o_proj and x, writing y); the MLP's is all_reduce, add, norm
 * (mode 1, reading y and x, writing x and the norm), that norm being the next layer's input norm
 * or -- on the only draft layer this model declares, the last one -- the final norm, so the add and
 * the final norm are then issued nowhere else. At one rank there is no reduction to fuse and the
 * pair is kept, as it is by a library that does not declare the op or refuses it. */
void no_ar_add_rmsnorm(RadBuilder* b) { bf16_encs(b); b->undeclared.insert("ar_add_rmsnorm"); }
void refused_ar_add_rmsnorm(RadBuilder* b) { bf16_encs(b); b->refuse.insert("ar_add_rmsnorm"); }
TEST(dfly2_takes_the_fused_reduction_add_norm_where_served_and_the_chain_otherwise) {
    const int depth = 5;
    for (int world : { 1, 2 }) for (int form = 0; form < 3; ++form) {
        DeclDfly d(depth, dfly_meta(), world, 0,
                   form == 0 ? bf16_encs : form == 1 ? no_ar_add_rmsnorm : refused_ar_add_rmsnorm);
        REQUIRE_EQ(d.rc, RAD_OK);
        RadBuilder& b = d.b;
        const auto& df = d.df();
        const bool fused = world == 2 && form == 0;
        REQUIRE_EQ(df.lay.size(), (size_t)1);
        const auto& L = df.lay[0];
        REQUIRE(L.post);   /* the served model's draft layer has the output projection's post-norm */
        CHECK_EQ(L.op_ar_add != 0, fused);
        CHECK_EQ(L.op_ar_next != 0, fused);
        CHECK_EQ(L.op_post != 0, !fused);
        CHECK_EQ(L.op_add1 != 0, !fused);
        /* the last layer's fused twin takes the final norm, so nothing else issues it */
        CHECK_EQ(df.op_final != 0 || df.op_final_add != 0, !fused);
        CHECK_EQ(L.op_add2 != 0, false);   /* D-01's add_rmsnorm: the MLP's sum is not its shape */
        if (fused) {
            const RecOp* s0 = b.op_of(L.op_ar_add);
            REQUIRE(s0);
            CHECK_EQ(pstr(s0, "dtype"), std::string("bf16"));
            CHECK_EQ(pint(s0, "world_size"), 2);
            CHECK_EQ(pint(s0, "mode"), 0);
            REQUIRE_EQ(s0->reads.size(), (size_t)2);
            CHECK_EQ(b.bufs[s0->reads[0] - 1].name, std::string("dfly.o_proj"));
            CHECK_EQ(b.bufs[s0->reads[1] - 1].name, std::string("dfly.x"));
            REQUIRE_EQ(s0->writes.size(), (size_t)3);
            CHECK_EQ(b.bufs[s0->writes[2] - 1].name, std::string("dfly.y"));
            REQUIRE_EQ(s0->w.size(), (size_t)1);
            CHECK_EQ(s0->w[0], L.post_norm);
            const RecOp* s1 = b.op_of(L.op_ar_next);
            REQUIRE(s1);
            CHECK_EQ(pint(s1, "world_size"), 2);
            CHECK_EQ(pint(s1, "mode"), 1);
            REQUIRE_EQ(s1->reads.size(), (size_t)2);
            CHECK_EQ(b.bufs[s1->reads[0] - 1].name, std::string("dfly.y"));
            CHECK_EQ(b.bufs[s1->reads[1] - 1].name, std::string("dfly.x"));
            REQUIRE_EQ(s1->writes.size(), (size_t)3);
            CHECK_EQ(b.bufs[s1->writes[2] - 1].name, std::string("dfly.final"));
            REQUIRE_EQ(s1->w.size(), (size_t)1);
            CHECK_EQ(s1->w[0], L.ar_next_w);
            CHECK_EQ(L.ar_next_out, s1->writes[2]);
        }
        auto trunk = kit(b, std::vector<int>(2, depth + 1), {});
        trunk->b.positions_h = trunk->pos.data();
        auto ctxp = context_pass(b, *trunk);
        Query qp(b, 2, depth + 1, depth);
        auto ps = run_passes(b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() }, 0, world);
        for (const auto& c : ps) {
            if (!c->step_fail.empty()) fprintf(stderr, "%s\n", c->step_fail.c_str());
            CHECK(c->step_fail.empty());
        }
        CHECK_LINT(b, ps[0].get(), ps[1].get(), ps[2].get());
        const auto q = issued_ops(b, *ps[2]);
        auto count = [&](const char* op) { return (int64_t)std::count(q.begin(), q.end(), std::string(op)); };
        CHECK_EQ(count("ar_add_rmsnorm"), fused ? 2 : 0);
        /* the drafter's own reductions: one a chain round, plus the layer's pair unless fused (the
         * pass's other all_reduces are the trunk's, which this does not count) */
        int64_t ars = 0, attn_issues = 0, ffn_issues = 0, fins = 0;
        for (const auto& is : ps[2]->issues) {
            const bool draft_ar = !fused && (is.op == df.op_ar_o || is.op == df.op_ar_y);
            ars += draft_ar || (df.op_ar_c && is.op == df.op_ar_c);
            attn_issues += is.op == L.op_attn_norm || (L.op_attn_norm_q && is.op == L.op_attn_norm_q);
            ffn_issues += is.op == L.op_ffn_norm || (L.op_ffn_norm_q && is.op == L.op_ffn_norm_q);
            fins += is.op == df.op_final || (df.op_final_add && is.op == df.op_final_add);
        }
        CHECK_EQ(ars, world == 1 ? 0 : (int64_t)df.steps.size() + (fused ? 0 : 2));
        /* each of the layer's input norms is issued once -- after the fused twin of the layer
         * before it as much as after the chain -- and the final norm only where nothing fused it */
        CHECK_EQ(attn_issues, 1);
        CHECK_EQ(ffn_issues, 1);
        CHECK_EQ(fins, fused ? 0 : 1);
        /* mode 0's launch stands where the output projection's reduction, post-norm and add stood:
         * after the projection, before the MLP's input norm. Mode 1's closes the layer, after the
         * down projection, and the final norm is not issued beside it. */
        size_t at_oproj = SIZE_MAX, at_s0 = SIZE_MAX, at_ffn = SIZE_MAX, at_down = SIZE_MAX,
               at_s1 = SIZE_MAX, at_fin = SIZE_MAX, at_attn = SIZE_MAX;
        for (size_t i = 0; i < ps[2]->issues.size(); ++i) {
            const rad_op h = ps[2]->issues[i].op;
            if (h == L.p_out.op && at_oproj == SIZE_MAX) at_oproj = i;
            if (fused && h == L.op_ar_add && at_s0 == SIZE_MAX) at_s0 = i;
            if (h == L.op_ffn_norm && at_ffn == SIZE_MAX) at_ffn = i;
            if (h == L.p_down.op && at_down == SIZE_MAX) at_down = i;
            if (fused && h == L.op_ar_next && at_s1 == SIZE_MAX) at_s1 = i;
            if ((h == df.op_final || h == df.op_final_add) && at_fin == SIZE_MAX) at_fin = i;
            if (h == L.op_attn_norm && at_attn == SIZE_MAX) at_attn = i;
        }
        REQUIRE(at_oproj != SIZE_MAX && at_ffn != SIZE_MAX && at_attn != SIZE_MAX);
        if (fused) {
            REQUIRE(at_s0 != SIZE_MAX && at_s1 != SIZE_MAX);
            CHECK(at_oproj < at_s0 && at_s0 < at_ffn);
            CHECK(at_down < at_s1);
            CHECK_EQ(at_fin, SIZE_MAX);
        } else {
            CHECK_EQ(at_s0, SIZE_MAX);
            CHECK_EQ(at_s1, SIZE_MAX);
            CHECK(at_oproj < at_ffn);
            if (world == 2) CHECK(at_fin != SIZE_MAX);
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

/* ------------------------------------------------------------------ D-02 (#88): the chain replayed
 *
 * The tests above read a tape's ops and counts. D-02 moves operands (every round's rows of z
 * gathered and normalised in one launch, rounds reading row blocks of shared planes, the anchors'
 * embeddings kept across the layers), so these replay a query pass's tape over symbolic values: a
 * cell is a 64-bit name of the expression that produced it, operands resolved as the core resolves
 * them (offset, rows, cols at the declared pitch). The ops the chain is made of are modelled by row
 * (a copy keeps its cells, so a moved or merged launch gives the same cells exactly when it gives
 * the same values); every other op is opaque, a function of all it reads. */
namespace {
struct Replay {
    static uint64_t mix(uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdull;
        return h ^ (h >> 33);
    }
    static uint64_t tag(const std::string& s) {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char ch : s) h = (h ^ ch) * 1099511628211ull;
        return h;
    }
    static uint64_t fold(const std::vector<uint64_t>& v) {
        uint64_t h = tag("row");
        for (uint64_t x : v) h = mix(h, x);
        return h;
    }
    /* The modelled ops' values, which the test's own statement of the chain is written in. */
    static uint64_t embed(uint64_t tok, int64_t c) { return mix(mix(tag("embed"), tok), (uint64_t)c); }
    static uint64_t linear(const char* op, uint64_t w, uint64_t row, int64_t j) {
        return mix(mix(mix(tag(op), w), row), (uint64_t)j);
    }
    static uint64_t silu_mul(uint64_t g, uint64_t u) { return mix(mix(tag("silu_mul"), g), u); }
    static uint64_t add(uint64_t a, uint64_t b2) { return mix(mix(tag("add"), a), b2); }

    /* An operand's cells, row major. */
    struct View {
        int64_t rows = 0, cols = 0;
        std::vector<uint64_t*> at;       /* a buffer's cells */
        std::vector<uint64_t>  own;      /* a raw operand's, a weight's or a cache's */
        bool written = false, buf = false;
        uint64_t get(int64_t r, int64_t c) const {
            const size_t i = (size_t)(r * cols + c);
            return buf ? *at[i] : own[i];
        }
        std::vector<uint64_t> row(int64_t r) const {
            std::vector<uint64_t> v;
            for (int64_t c = 0; c < cols; ++c) v.push_back(get(r, c));
            return v;
        }
        std::vector<uint64_t> all() const {
            std::vector<uint64_t> v;
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; ++c) v.push_back(get(r, c));
            return v;
        }
    };

    const RadBuilder& b;
    std::vector<std::vector<uint64_t>> mem;   /* by buffer handle */
    std::vector<std::string> bad;
    uint64_t spill = 0;
    explicit Replay(const RadBuilder& bld) : b(bld), mem(bld.bufs.size() + 1) {
        for (size_t h = 1; h <= b.bufs.size(); ++h) {
            const RadBufDecl& d = b.bufs[h - 1].d;
            int64_t n = 1;
            for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
            /* what an earlier pass left there: a name of its own a cell */
            const uint64_t t = tag("left in " + b.bufs[h - 1].name);
            for (int64_t i = 0; i < n; ++i) mem[h].push_back(mix(t, (uint64_t)i));
        }
    }
    View view(const RecOp& o, const RadOperand& od) {
        View v;
        if (od.kind == RAD_OPK_BUF) {
            const RadBufDecl& d = b.bufs[od.handle - 1].d;
            int64_t pitch = 1;
            for (uint32_t i = 1; i < d.rank; ++i) pitch *= d.shape[i];
            v.buf = true;
            v.rows = od.rows > 0 ? od.rows : d.shape[0];
            v.cols = od.cols > 0 ? od.cols : pitch;
            v.written = in(o.writes, od.handle);
            std::vector<uint64_t>& m = mem[od.handle];
            for (int64_t r = 0; r < v.rows; ++r)
                for (int64_t c = 0; c < v.cols; ++c) {
                    const int64_t i = od.offset + r * pitch + c;
                    if (i < 0 || (size_t)i >= m.size()) {
                        bad.push_back(o.op + ": an operand past '" + b.bufs[od.handle - 1].name + "'");
                        v.at.push_back(&spill);   /* keeps the view's size; `bad` fails the case */
                        continue;
                    }
                    v.at.push_back(&m[(size_t)i]);
                }
        } else if (od.kind == RAD_OPK_RAW) {
            v.rows = od.rows > 0 ? od.rows : 1;
            v.cols = od.cols > 0 ? od.cols : 1;
            for (int64_t i = 0; i < v.rows * v.cols; ++i)
                v.own.push_back(od.dtype == RAD_I32 && od.raw
                                    ? (uint64_t)(int64_t)((const int32_t*)od.raw)[i]
                                    : mix(tag("raw"), (uint64_t)i));
        } else if (od.kind == RAD_OPK_WEIGHT) {
            v.rows = v.cols = 1;
            v.own.push_back(tag("weight " + b.weights[od.handle - 1].name));
        } else if (od.kind != RAD_OPK_NONE) {
            v.rows = v.cols = 1;
            v.own.push_back(mix(mix(tag("handle"), od.kind), mix((uint64_t)od.offset, (uint64_t)od.rows)));
        }
        return v;
    }
    void put(View& v, int64_t r, int64_t c, uint64_t x) { *v.at[(size_t)(r * v.cols + c)] = x; }

    /* One issue. `after` sees it once its outputs are written. */
    void issue(const RecIssue& is) {
        const RecOp* op = b.op_of(is.op);
        if (!op) { bad.push_back("a null op issued"); return; }
        const RecOp& o = *op;
        std::vector<View> v;
        for (const RadOperand& od : is.opd) v.push_back(view(o, od));
        auto need = [&](size_t n) {
            if (v.size() >= n) return true;
            bad.push_back(o.op + ": too few operands");
            return false;
        };
        if (o.op == "iota") {
            if (!need(1)) return;
            const long long start = pint(&o, "start"), step = pint(&o, "step");
            const RecParam* per = param(&o, "period");
            for (int64_t i = 0; i < v[0].rows * v[0].cols; ++i)
                *v[0].at[(size_t)i] = (uint64_t)(start + (per ? i % per->ival : i) * step);
        } else if (o.op == "cast" && pstr(&o, "from") == pstr(&o, "to") && v.size() == 2 &&
                   v[0].rows * v[0].cols == v[1].rows * v[1].cols) {   /* a copy, cell for cell */
            const std::vector<uint64_t> x = v[0].all();
            for (size_t i = 0; i < x.size(); ++i) *v[1].at[i] = x[i];
        } else if (o.op == "embed_lookup") {
            if (!need(3)) return;
            for (int64_t r = 0; r < v[2].rows; ++r)
                for (int64_t c = 0; c < v[2].cols; ++c) put(v[2], r, c, embed(v[0].get(r, 0), c));
        } else if (o.op == "gather_rows") {
            if (!need(3)) return;
            for (int64_t r = 0; r < v[2].rows; ++r) {
                const int64_t i = (int64_t)v[1].get(r, 0);
                if (i < 0 || i >= v[0].rows || v[0].cols != v[2].cols) {
                    bad.push_back("gather_rows: an index that is not a row of its source");
                    return;
                }
                const std::vector<uint64_t> x = v[0].row(i);
                for (int64_t c = 0; c < v[2].cols; ++c) put(v[2], r, c, x[(size_t)c]);
            }
        } else if (o.op == "scatter_rows") {
            if (!need(3)) return;
            for (int64_t r = 0; r < v[0].rows; ++r) {
                const int64_t i = (int64_t)v[1].get(r, 0);
                if (i < 0 || i >= v[2].rows) { bad.push_back("scatter_rows: a row out of range"); return; }
                for (int64_t c = 0; c < v[0].cols; ++c) put(v[2], i, c, v[0].get(r, c));
            }
        } else if (o.op == "rmsnorm" || o.op == "gemm_nt" || o.op == "logits_gemm") {
            if (!need(3)) return;
            for (int64_t r = 0; r < v[2].rows; ++r) {
                const uint64_t h = fold(v[0].row(r));   /* before the row is written: may be in place */
                for (int64_t c = 0; c < v[2].cols; ++c)
                    put(v[2], r, c, linear(o.op.c_str(), v[1].own[0], h, c));
            }
        } else if (o.op == "silu_mul") {
            if (!need(2)) return;
            for (int64_t r = 0; r < v[1].rows; ++r)
                for (int64_t c = 0; c < v[1].cols; ++c)
                    put(v[1], r, c, silu_mul(v[0].get(r, c), v[0].get(r, v[1].cols + c)));
        } else if (o.op == "add") {
            if (!need(3)) return;
            for (int64_t r = 0; r < v[2].rows; ++r)
                for (int64_t c = 0; c < v[2].cols; ++c)
                    put(v[2], r, c, add(v[0].get(r, c), v[1].get(r, c)));
        } else {
            /* Opaque: each output cell a function of the op, its fixed parameters and everything
             * it reads (an operand it only writes is not read). */
            uint64_t h = tag(o.op);
            for (const auto& p : o.p)
                if (p.key != "M") h = mix(mix(h, tag(p.key + "=" + p.sval)), (uint64_t)p.ival);
            for (size_t k = 0; k < v.size(); ++k)
                if (!v[k].written || in(o.reads, is.opd[k].handle)) h = mix(h, fold(v[k].all()));
            for (size_t k = 0; k < v.size(); ++k)
                if (v[k].buf && v[k].written)
                    for (size_t i = 0; i < v[k].at.size(); ++i)
                        *v[k].at[i] = mix(mix(h, (uint64_t)k), (uint64_t)i);
        }
    }
    /* A buffer's first `rows` rows. */
    std::vector<uint64_t> rows_of(rad_buf h, int64_t rows) {
        RecOp none;
        return view(none, hyv4flash::brows(h, rows)).all();
    }
};

/* A drafter declared as the server's: `seqs` drafting sequences at most. */
struct DflyAt {
    MetaKV mk;
    RadBuilder b;
    int rc = 0;
    DflyAt(int depth, int world, int coupled, int64_t seqs, void (*encs)(RadBuilder*) = bf16_encs,
           MetaKV m = dfly_meta())
        : mk(std::move(m)) {
        encs(&b);
        RadBuildCtx ctx = spec_ctx(0, world, depth);
        ctx.max_seqs = seqs;
        ctx.draft_coupling = coupled;
        const std::string err = stderr_of([&] { rc = hyv4flash::declare(&b, mk.meta(), &ctx); });
        if (rc != RAD_OK) fputs(err.c_str(), stderr);
    }
    const hyv4flash::Dfly& df() const { return hyv4flash::models[0]->df; }
    /* The query pass of `n` sequences at `blk` rows each, alone: what it reads of the trunk step
     * is whatever that left. */
    std::unique_ptr<RadCtx> query(int64_t n, int64_t blk, int world, Query** keep) {
        *keep = new Query(b, n, blk, blk - 1);
        auto c = std::make_unique<RadCtx>();
        c->b = &b;
        c->batch = &(*keep)->k->b;
        c->world = world;
        hyv4flash::step(c.get(), c->batch);
        return c;
    }
};
}  // namespace

/* D-02: at every stream count the server runs (1, 2, 8, 16) and every depth to the configured one,
 * full block and short, greedy and coupled, each chain round's corrected state is the header's
 *
 *   zc = z[k+1] + down(silu(gate(c)) * up(c)),   c = [hnorm(z[k+1]) | enorm(embed(prev))]
 *
 * over the rows the final norm wrote, prev the anchor's token and then each proposal, and proposal
 * k is what the round's shortlist put first. Stated here from the modelled ops alone, so it holds
 * the hoisted gather and norm, the rounds' row blocks and the kept anchor embeddings to the values
 * the per-round launches gave. RADIANCE_HY4_DFLY_TEST_BREAK=chain_rows (every round reading round
 * 0's rows) must fail it from the second round on. HY4_TEST_D02_DIGEST=1 prints a digest of every
 * round's logits, shortlist and proposals a shape, to compare two trees (TP and int8 included,
 * where the chain's linears are opaque here). */
TEST(dfly2_chain_rounds_compute_the_headers_chain_at_every_served_shape) {
    const bool digest = std::getenv("HY4_TEST_D02_DIGEST") != nullptr;
    int64_t wrong = -1;   /* cells that differ from the header's; -1 until a run completes */
    auto run = [&](int depth, int world, int coupled, int64_t n, int64_t blk, bool i8, bool broken) {
        wrong = -1;
        DflyAt d(depth, world, coupled, 16, i8 ? dfly_i8_encs : bf16_encs,
                 i8 ? dfly_meta(1, true, 256) : dfly_meta());
        REQUIRE_EQ(d.rc, RAD_OK);
        const auto& df = d.df();
        REQUIRE(df.served);
        Query* qp = nullptr;
        auto c = d.query(n, blk, world, &qp);
        std::unique_ptr<Query> hold(qp);
        if (!c->step_fail.empty()) fprintf(stderr, "%s\n", c->step_fail.c_str());
        REQUIRE(c->step_fail.empty());
        const int64_t rounds = blk - 1, E = df.E;
        const RadBuilder& b = d.b;
        Replay rp(b);
        std::vector<uint64_t> Z, prev;   /* the final norm's rows; each sequence's previous token */
        for (int64_t s = 0; s < n; ++s) prev.push_back((uint64_t)qp->k->tok[(size_t)(s * blk)]);
        const uint64_t wh = Replay::tag("weight dfly.correct_hidden_norm.weight"),
                       we = Replay::tag("weight dfly.correct_embed_norm.weight"),
                       wg = Replay::tag("weight dfly.correct_gate.weight"),
                       wu = Replay::tag("weight dfly.correct_up.weight"),
                       wd = Replay::tag("weight dfly.correct_down.weight");
        uint64_t dg = Replay::tag("digest");
        int64_t heads = 0;
        wrong = 0;
        for (const RecIssue& is : c->issues) {
            /* a round's head: its input against the header, before the head runs */
            int64_t k = -1;
            for (int64_t i = 0; i < rounds; ++i)
                if (is.op == df.steps[(size_t)i].op_head) k = i;
            if (k >= 0 && world == 1 && !i8) {
                ++heads;
                REQUIRE(!Z.empty());
                const Replay::View zc = rp.view(*b.op_of(is.op), is.opd[0]);
                REQUIRE_EQ(zc.rows, n);
                REQUIRE_EQ(zc.cols, E);
                for (int64_t s = 0; s < n; ++s) {
                    const size_t row = (size_t)((s * blk + k + 1) * E);
                    const std::vector<uint64_t> z(Z.begin() + (long)row, Z.begin() + (long)(row + E));
                    std::vector<uint64_t> cc, pe, gu, act;
                    for (int64_t j = 0; j < E; ++j) pe.push_back(Replay::embed(prev[(size_t)s], j));
                    for (int64_t j = 0; j < E; ++j)
                        cc.push_back(Replay::linear("rmsnorm", wh, Replay::fold(z), j));
                    for (int64_t j = 0; j < E; ++j)
                        cc.push_back(Replay::linear("rmsnorm", we, Replay::fold(pe), j));
                    for (int64_t j = 0; j < df.Ew; ++j)
                        act.push_back(Replay::silu_mul(Replay::linear("gemm_nt", wg, Replay::fold(cc), j),
                                                       Replay::linear("gemm_nt", wu, Replay::fold(cc), j)));
                    for (int64_t j = 0; j < E; ++j)
                        wrong += zc.get(s, j) != Replay::add(z[(size_t)j],
                                                             Replay::linear("gemm_nt", wd, Replay::fold(act), j));
                }
            }
            rp.issue(is);
            /* D-01 (#87) folds the last add into the final norm where a library serves add_rmsnorm */
            if (is.op == (df.op_final_add ? df.op_final_add : df.op_final)) Z = rp.rows_of(df.z, n * blk);
            for (int64_t i = 0; i < rounds; ++i) {
                const auto& st = df.steps[(size_t)i];
                if (is.op == st.op_head) dg = Replay::mix(dg, Replay::fold(rp.rows_of(df.logits, n)));
                if (is.op == st.op_tok) {   /* the round's proposal, as it left the shortlist */
                    const Replay::View t = rp.view(*b.op_of(is.op), is.opd[0]);
                    dg = Replay::mix(dg, Replay::fold(rp.rows_of(df.cidx, n)));
                    dg = Replay::mix(dg, Replay::fold(rp.rows_of(df.cval, n)));
                    for (int64_t s = 0; s < n; ++s) {
                        prev[(size_t)s] = t.get(s, 0);
                        /* ... and where the core reads it: column i of the proposals */
                        const Replay::View o = rp.view(*b.op_of(is.op), is.opd[1]);
                        wrong += o.get(s, 0) != rp.rows_of(df.tokens, n)[(size_t)(s * depth + i)];
                    }
                }
            }
        }
        for (size_t i = 0; i < rp.bad.size() && i < 4; ++i) fprintf(stderr, "    replay: %s\n", rp.bad[i].c_str());
        CHECK_EQ(rp.bad.size(), (size_t)0);
        if (world == 1 && !i8) CHECK_EQ(heads, rounds);
        dg = Replay::mix(dg, Replay::fold(rp.rows_of(df.tokens, n)));
        if (digest && !broken)
            printf("    d02 digest depth %d world %d coupled %d n %lld block %lld %s: %016llx, %zu issues\n",
                   depth, world, coupled, (long long)n, (long long)blk, i8 ? "int8" : "bf16",
                   (unsigned long long)dg, c->issues.size());
    };
    for (int depth : { 1, 2, 3, 4, 5 })
        for (int coupled : { 0, 1 })
            for (int64_t n : { (int64_t)1, (int64_t)2, (int64_t)8, (int64_t)16 }) {
                run(depth, 1, coupled, n, depth + 1, false, false);
                CHECK_EQ(wrong, (int64_t)0);
                if (depth >= 4) {
                    run(depth, 1, coupled, n, 4, false, false);
                    CHECK_EQ(wrong, (int64_t)0);
                }
                if (digest) {
                    run(depth, 2, coupled, n, depth + 1, false, false);
                    run(depth, 1, coupled, n, depth + 1, true, false);
                    run(depth, 2, coupled, n, depth + 1, true, false);
                }
            }
    /* the mutant: caught wherever there is a second round, harmless at depth 1 */
    setenv("RADIANCE_HY4_DFLY_TEST_BREAK", "chain_rows", 1);
    for (int64_t n : { (int64_t)1, (int64_t)2, (int64_t)16 }) {
        run(5, 1, 0, n, 6, false, true);
        CHECK(wrong > 0);
        run(5, 1, 1, n, 4, false, true);
        CHECK(wrong > 0);
        run(1, 1, 0, n, 2, false, true);
        CHECK_EQ(wrong, (int64_t)0);
    }
    unsetenv("RADIANCE_HY4_DFLY_TEST_BREAK");
}

/* D-02: what a query pass no longer launches. One gather and one norm for all the rounds' rows of
 * z, the rounds' row indices in one plane; the block's embeddings written straight into the
 * residual and the anchors' gathered once; no copy of the last round's proposal that nothing reads;
 * and, where gate and up are int8 linears of one form, one quantiser a round for the two (up's
 * GEMM reading the codes gate's wrote, at the round's rows). Per pass of r rounds that is
 * 3 + 2 (r - 1) launches fewer than one launch a round and a block copy, and r more at int8. */
TEST(dfly2_chain_launches_each_shared_step_once_a_pass) {
    for (const bool i8 : { false, true })
        for (int world : { 1, 2 })
            for (int coupled : { 0, 1 })
                for (int64_t n : { (int64_t)1, (int64_t)2, (int64_t)8, (int64_t)16 })
                    for (int64_t blk : { (int64_t)6, (int64_t)4 }) {
                        DflyAt d(5, world, coupled, 16, i8 ? dfly_i8_encs : bf16_encs,
                                 i8 ? dfly_meta(1, true, 256) : dfly_meta());
                        REQUIRE_EQ(d.rc, RAD_OK);
                        const auto& df = d.df();
                        REQUIRE(df.served);
                        const RadBuilder& b = d.b;
                        Query* qp = nullptr;
                        auto c = d.query(n, blk, world, &qp);
                        std::unique_ptr<Query> hold(qp);
                        REQUIRE(c->step_fail.empty());
                        const int64_t rounds = blk - 1, E = df.E;
                        auto count = [&](rad_op h) {
                            int64_t k = 0;
                            for (const auto& is : c->issues) k += h && is.op == h;
                            return k;
                        };
                        const auto q = issued_ops(b, *c);
                        auto named = [&](const char* op) {
                            return (int64_t)std::count(q.begin(), q.end(), std::string(op));
                        };
                        /* the hoisted pair, over every round's rows */
                        CHECK_EQ(count(df.op_zk), (int64_t)1);
                        CHECK_EQ(count(df.op_hn), (int64_t)1);
                        CHECK_EQ(named("gather_rows"), (int64_t)3);   /* anchors, target hidden, chain */
                        const RecIssue* zk = nullptr;
                        const RecIssue* hn = nullptr;
                        for (const auto& is : c->issues) {
                            if (is.op == df.op_zk) zk = &is;
                            if (is.op == df.op_hn) hn = &is;
                        }
                        REQUIRE(zk != nullptr && hn != nullptr);
                        CHECK_EQ(zk->n, rounds * n);
                        CHECK_EQ(zk->opd[1].rows, rounds * n);
                        CHECK_EQ(hn->n, rounds * n);
                        CHECK_EQ(hn->opd[2].cols, E);
                        CHECK_EQ(hn->opd[2].offset, (int64_t)0);
                        /* one row-index launch a round, each into its own rows of the plane */
                        int64_t r = 0;
                        for (const auto& is : c->issues)
                            for (int64_t k = 0; k < rounds; ++k) {
                                const auto& st = df.steps[(size_t)k];
                                if (is.op != (blk == 6 ? st.op_rowk : st.op_rowk_lo)) continue;
                                CHECK_EQ(is.opd[0].handle, df.rowk);
                                CHECK_EQ(is.opd[0].offset, k * n);
                                CHECK_EQ(is.opd[0].rows, n);
                                ++r;
                            }
                        CHECK_EQ(r, rounds);
                        /* the embeddings: the block's once, then one a round after the first */
                        CHECK_EQ(named("embed_lookup"), rounds);
                        /* the proposal copies: tokens every round, prev where a round follows */
                        int64_t tok = 0, prev = 0;
                        for (int64_t k = 0; k < rounds; ++k) {
                            tok += count(df.steps[(size_t)k].op_tok);
                            prev += count(df.steps[(size_t)k].op_prev);
                        }
                        CHECK_EQ(tok, rounds);
                        CHECK_EQ(prev, rounds - 1);
                        /* gate and up of a round read that round's rows of the shared input */
                        if (!i8) {
                            int64_t g = 0;
                            for (const auto& is : c->issues)
                                if (is.op == df.c_g.op || is.op == df.c_u.op) {
                                    CHECK_EQ(is.opd[0].handle, df.ccat);
                                    CHECK_EQ(is.opd[0].offset, (g / 2) * n * 2 * E);
                                    CHECK_EQ(is.opd[0].rows, n);
                                    ++g;
                                }
                            CHECK_EQ(g, 2 * rounds);
                        } else {
                            /* int8: one quantiser of the shared input a round, written and read at
                             * the round's rows by both GEMMs */
                            REQUIRE(df.c_g.q && df.c_u.q);
                            CHECK_EQ(df.c_g.q8, df.c_u.q8);
                            REQUIRE(df.c_g.lin.op_q != 0 && df.c_u.lin.op_q != 0);
                            CHECK_EQ(count(df.c_g.lin.op_q), rounds);
                            CHECK_EQ(count(df.c_u.lin.op_q), (int64_t)0);
                            const int64_t K = 2 * E;
                            int64_t kq = 0, kg = 0, ku = 0;
                            for (const auto& is : c->issues) {
                                if (is.op == df.c_g.lin.op_q) {
                                    CHECK_EQ(is.opd[0].handle, df.ccat);
                                    CHECK_EQ(is.opd[0].offset, kq * n * K);
                                    CHECK_EQ(is.opd[1].handle, df.c_g.q8);
                                    CHECK_EQ(is.opd[1].offset, kq * n * K);
                                    ++kq;
                                }
                                if (is.op == df.c_g.lin.op) {
                                    CHECK_EQ(is.opd[0].offset, kg * n * K);
                                    ++kg;
                                }
                                if (is.op == df.c_u.lin.op) {
                                    CHECK_EQ(kq, ku + 1);   /* behind this round's quantiser */
                                    CHECK_EQ(is.opd[0].handle, df.c_g.q8);
                                    CHECK_EQ(is.opd[0].offset, ku * n * K);
                                    CHECK_EQ(is.opd[1].handle, df.c_g.s8);
                                    ++ku;
                                }
                            }
                            CHECK_EQ(kg, rounds);
                            CHECK_EQ(ku, rounds);
                            const RecBuf* q8 = find_b(b, b.bufs[df.c_g.q8 - 1].name);
                            REQUIRE(q8 != nullptr);
                            CHECK_EQ(q8->d.shape[0], (int64_t)(5 * 16));
                        }
                        /* the pass's launches: a fixed part and a part a round */
                        const int64_t tp = world > 1 ? 1 : 0;
                        /* D-04: no cast of the logits where the fused shortlist is served */
                        const int64_t lcast = df.steps[0].op_ctopk ? 0 : 1;
                        const int64_t a_round = 14 + lcast + (i8 ? 2 : 0) + 3 * tp + coupled;
                        const int64_t fixed = (int64_t)c->issues.size() - rounds * a_round;
                        if (blk == 6) {
                            Query* q4 = nullptr;
                            auto c4 = d.query(n, 4, world, &q4);
                            std::unique_ptr<Query> hold4(q4);
                            CHECK_EQ((int64_t)c4->issues.size(), fixed + 3 * a_round);
                        }
                    }
}

/* D-01 (#87): a draft layer takes the trunk's fused producers wherever a loaded kernel library
 * serves them and the draft's linears read rotated int8 codes of the 128 group -- the norm of the
 * attention's input and the feed-forward's write their readers' codes, the q residual's norm writes
 * q_b's, the MLP's activation writes down's, the expansion keeps its gate and o_proj's quantiser,
 * the cache row's norm, key rotary and store are one launch, and the pass's last residual add folds
 * into the final norm -- so the layer issues nine fewer launches for the same bytes and the same
 * codes (docs/OPS.md states each byte-identical to the chain it replaces). A switch off, or a build
 * without an op, keeps the plain chain and the pass is lint clean either way. */
namespace {
/* D-01's container: every draft linear a fused producer feeds reads rotated int8 codes of the 128
 * group, q_lora and o_proj's input whole groups (the default 64 and 64 are not). */
void dfly_fused_encs(RadBuilder* b) {
    dfly_i8_encs(b);
    std::vector<std::pair<std::string, RadEncoding>> q;
    for (const char* n : { "attn_kv_a", "attn_q_b", "attn_output", "ffn_down" })
        q.push_back({ std::string("dfly.blk.0.") + n + ".weight", enc_i8(true) });
    b->encs.insert(b->encs.begin(), q.begin(), q.end());
}
MetaKV dfly_fused_meta() {
    MetaKV k = dfly_meta(1, true, 256);
    k.set("q_lora_rank", "128");       /* q_b's K is whole 128 groups */
    k.set("kv_lora_rank", "40");       /* so kv_a's 40 + 8 columns are int8 rows of 16s */
    return k;
}
/* The switches a case turns off, restored when it ends. */
struct DflyOff {
    std::vector<const char*> set;
    explicit DflyOff(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const auto& p : kv) {
            setenv(p.first, p.second, 1);
            set.push_back(p.first);
        }
    }
    ~DflyOff() { for (const char* k : set) unsetenv(k); }
};
/* One query pass over D-01's container: every draft op's issue count, and the layer's launch count
 * (its first norm through its last add/norm, so the trunk's launches do not enter it). */
struct DflyIssues {
    std::map<std::string, int> n;
    int64_t launches = 0;
};
}  // namespace
TEST(dfly2_draft_layers_use_the_trunks_fused_producers) {
    g3_fusions = true;
    auto run = [](int off, std::vector<const char*> refuse, DflyIssues& got) {
        std::unique_ptr<DflyOff> env;
        if (off == 1)   /* the fused producers off: the plain chain, and sharing still on */
            env = std::make_unique<DflyOff>(std::initializer_list<std::pair<const char*, const char*>>{
                { "RADIANCE_HY4_FUSED_QUANT", "0" }, { "RADIANCE_HY4_FUSE_EXPAND_QUANT", "0" },
                { "RADIANCE_HY4_CACHE_ROW", "0" } });
        DeclDfly d(5, dfly_fused_meta(), 1, 0, dfly_fused_encs, refuse);
        CHECK_EQ(d.rc, RAD_OK);
        if (!d.df().served) fprintf(stderr, "%s\n", d.df().why.c_str());
        CHECK(d.df().served);
        const auto& df = d.df();
        const auto& L = df.lay[0];
        if (off == 0)
            CHECK(L.op_attn_norm_q != 0 && L.op_qa_norm_q != 0 && L.op_ffn_norm_q != 0 &&
                  L.op_row_q != 0 && L.op_expand_q != 0 && L.op_gated != 0 && df.op_final_add != 0);
        if (off == 2)   /* a build whose kernels serve none of them */
            CHECK(L.op_attn_norm_q == 0 && L.op_qa_norm_q == 0 && L.op_ffn_norm_q == 0 &&
                  L.op_row_q == 0 && L.op_expand_q == 0 && L.op_gated == 0 && df.op_final_add == 0);
        /* the trunk's own fusions stay as the build has them, so only the draft layer is compared */
        auto trunk = kit(d.b, std::vector<int>(2, 6), {});
        trunk->b.positions_h = trunk->pos.data();
        auto ctxp = context_pass(d.b, *trunk);
        Query qp(d.b, 2, 6, 5);
        auto ps = run_passes(d.b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() }, 0, 1);
        CHECK_PASSES(d.b, ps);
        CHECK_LINT(d.b, ps[2].get());
        const auto& is = ps[2]->issues;
        auto n_of = [&](rad_op h) {
            int n = 0;
            for (const auto& s : is) n += h != 0 && s.op == h;
            return n;
        };
        const std::pair<const char*, rad_op> ops[] = {
            { "attn_norm", L.op_attn_norm },     { "attn_norm_q", L.op_attn_norm_q },
            { "qa_norm", L.op_qa_norm },         { "qa_norm_q", L.op_qa_norm_q },
            { "ffn_norm", L.op_ffn_norm },       { "ffn_norm_q", L.op_ffn_norm_q },
            { "kv_norm", L.op_kv_norm },         { "k_rope", L.op_k_rope },
            { "store", L.op_store },            { "row_q", L.op_row_q },
            { "expand", L.op_expand },           { "expand_q", L.op_expand_q },
            { "gmul", L.op_gmul },               { "silu", L.op_silu },
            { "gated", L.op_gated },             { "add2", L.op_add2 },
            { "final", df.op_final },           { "final_add", df.op_final_add },
        };
        for (const auto& [nm, h] : ops) got.n[nm] = n_of(h);
        size_t first = SIZE_MAX, last = SIZE_MAX;
        for (size_t i = 0; i < is.size(); ++i) {
            if ((is[i].op == L.op_attn_norm || is[i].op == L.op_attn_norm_q) && first == SIZE_MAX)
                first = i;
            if (is[i].op == df.op_final || is[i].op == df.op_final_add) last = i;
        }
        CHECK(first != SIZE_MAX && last != SIZE_MAX && first <= last);
        got.launches = first == SIZE_MAX || last < first ? -1 : (int64_t)(last - first + 1);
    };
    const char* fused_ops[] = { "attn_norm_q", "qa_norm_q", "ffn_norm_q", "row_q", "expand_q",
                               "gated", "final_add" };
    const char* plain_ops[] = { "attn_norm", "qa_norm", "ffn_norm", "kv_norm", "k_rope", "store",
                               "expand", "gmul", "silu", "add2", "final" };
    DflyIssues on, off_by_switch, off_by_library;
    run(0, {}, on);
    run(1, {}, off_by_switch);
    run(2, { "rmsnorm_had_quant_i8g", "mla_row_norm_rope_store", "mla_expand_gate_had_quant_i8g",
             "gated_had_quant_i8g", "add_rmsnorm" }, off_by_library);
    for (const char* nm : fused_ops) CHECK_EQ(on.n[nm], 1);
    for (const char* nm : plain_ops) CHECK_EQ(on.n[nm], 0);
    /* the switches off: the five switch-gated families fall back and the pass stays lint clean.
     * The last add folds whatever the switches say: it has no switch, only a serving library. */
    const char* switched[] = { "attn_norm", "qa_norm", "ffn_norm", "kv_norm", "k_rope", "store",
                               "expand", "gmul", "silu" };
    for (const char* nm : switched) CHECK_EQ(off_by_switch.n[nm], 1);
    for (const char* nm : { "attn_norm_q", "qa_norm_q", "ffn_norm_q", "row_q", "expand_q", "gated" })
        CHECK_EQ(off_by_switch.n[nm], 0);
    CHECK_EQ(off_by_switch.n["final_add"], 1);
    CHECK_EQ(off_by_switch.n["add2"], 0);
    CHECK_EQ(off_by_switch.n["final"], 0);
    /* a library that serves none of the seven: the plain chain from the first norm to the final
     * norm, and no launch of a fused op */
    for (const char* nm : plain_ops) CHECK_EQ(off_by_library.n[nm], 1);
    for (const char* nm : fused_ops) CHECK_EQ(off_by_library.n[nm], 0);
    /* nine launches a draft layer: three of the row and the expansion, one of each of the rest;
     * eight against the switches alone, which leave the last add folded */
    CHECK_EQ(off_by_library.launches - on.launches, 9);
    CHECK_EQ(off_by_switch.launches - on.launches, 8);
    g3_fusions = false;
}

/* The fused expansion takes at most kFuseRows rows (its range), which a server with many
 * sequences a step can exceed: the declare keeps it, and the pass that drafts more rows than that
 * issues the plain expansion, the gate's product and o_proj's own quantiser instead -- the same
 * graph, fused for a block of drafts that fits and plain for one that does not. */
TEST(dfly2_a_block_taller_than_the_fused_expansion_keeps_its_plain_chain) {
    g3_fusions = true;
    DeclDfly d(5, dfly_fused_meta(), 1, 0, dfly_fused_encs, {}, /*max_seqs=*/64);
    CHECK_EQ(d.rc, RAD_OK);
    const auto& df = d.df();
    if (!df.served) fprintf(stderr, "%s\n", df.why.c_str());
    REQUIRE(df.served);
    CHECK(df.Md > 64);   /* Hy4Attn::kFuseRows, the expansion's row range */
    const auto& L = df.lay[0];
    CHECK(L.op_expand_q != 0 && L.op_expand != 0 && L.op_gmul != 0);
    /* 8 sequences of 6 rows fit the expansion; 12 do not */
    struct Case { int64_t n, block, exp, gmul, exp_q; };
    for (const Case& k : { Case{ 8, 6, 0, 0, 1 }, Case{ 12, 6, 1, 1, 0 } }) {
        auto trunk = kit(d.b, std::vector<int>(2, 6), {});
        trunk->b.positions_h = trunk->pos.data();
        auto ctxp = context_pass(d.b, *trunk);
        Query qp(d.b, k.n, k.block, 5);
        auto ps = run_passes(d.b, hyv4flash::step, { trunk.get(), ctxp.get(), qp.k.get() }, 0, 1);
        CHECK_PASSES(d.b, ps);
        CHECK_LINT(d.b, ps[2].get());
        int exp = 0, gmul = 0, exp_q = 0, quant = 0;
        for (const auto& s : ps[2]->issues) {
            exp += s.op == L.op_expand;
            gmul += s.op == L.op_gmul;
            exp_q += s.op == L.op_expand_q;
            quant += s.op == L.p_out.lin.op_q;
        }
        CHECK_EQ(exp, k.exp);
        CHECK_EQ(gmul, k.gmul);
        CHECK_EQ(exp_q, k.exp_q);
        CHECK_EQ(quant, k.exp_q == 0 ? 1 : 0);   /* o_proj quantises where the expansion did not */
    }
    g3_fusions = false;
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
                /* D-03: at two ranks the layer's two reductions take the fused twin (ar_add_rmsnorm,
                 * mode 0 at the output projection, mode 1 at the MLP) where the library serves it,
                 * and the standalone all-reduces, the post-norm and the adds are then issued nowhere
                 * (dfly2_takes_the_fused_reduction_add_norm_where_served_and_the_chain_otherwise). */
                const bool fused = L.op_ar_add != 0;
                CHECK_EQ(L.op_ar_next != 0, fused);
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
                    if (L.op_ar_add && op == L.op_ar_add) i_ar = i;
                    if (op == L.op_post || (op == L.op_add1 && i_post == SIZE_MAX)) i_post = std::min(i_post, i);
                    if (op == L.p_down.op) i_down = i;
                    if (op == df.op_ar_y) i_ar2 = i;
                    if (L.op_ar_next && op == L.op_ar_next) i_ar2 = i;
                    if (op == L.op_add2) i_add2 = i;
                }
                CHECK_EQ(ar, fused ? 0 : 2);
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
                if (!fused) {
                    /* the chain's order: the projection, its reduction, the post-norm; the down
                     * projection, the MLP's reduction, its add */
                    REQUIRE(i_ar != SIZE_MAX && i_ar2 != SIZE_MAX);
                    CHECK(i_out < i_ar && i_ar < i_post);
                    CHECK(i_down < i_ar2 && i_ar2 < i_add2);
                } else {
                    /* Mode 0 stands where the output projection's three launches stood: after the
                     * projection, before the MLP's input norm. Mode 1 closes the layer where the
                     * MLP's reduction and the add stood, and the only draft layer is the last, so
                     * its norm is the final one (dfly.final). */
                    REQUIRE(i_ar != SIZE_MAX && i_ar2 != SIZE_MAX);
                    CHECK(i_out < i_ar && i_ar < i_down && i_down < i_ar2);
                    auto bname = [&](size_t at, int k) {
                        const int64_t h = (int64_t)is[at].opd[(size_t)k].handle;
                        return (h >= 1 && (size_t)h <= b.bufs.size()) ? b.bufs[(size_t)h - 1].name
                                                                     : std::string("?");
                    };
                    /* The twins read and write the draft's own stream buffers only: `o` and `x` in,
                     * the gain and `y` out (mode 0); `y` and `x` in, the gain and the norm out
                     * (mode 1). Nothing of the trunk's enters them. */
                    CHECK_EQ(bname(i_ar, 0), std::string("dfly.o_proj"));
                    CHECK_EQ(bname(i_ar, 1), std::string("dfly.x"));
                    CHECK_EQ((rad_weight)is[i_ar].opd[2].handle, L.post_norm);
                    CHECK_EQ(bname(i_ar, 3), std::string("dfly.y"));
                    CHECK_EQ(bname(i_ar2, 0), std::string("dfly.y"));
                    CHECK_EQ(bname(i_ar2, 1), std::string("dfly.x"));
                    CHECK_EQ((rad_weight)is[i_ar2].opd[2].handle, L.ar_next_w);
                    CHECK_EQ(L.ar_next_out, (rad_buf)is[i_ar2].opd[3].handle);
                    CHECK_EQ(bname(i_ar2, 3), std::string("dfly.final"));
                }
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

/* P-11b. The buffer plan charges a transient the whole program when no op of the declared list
 * names it (core/build/rad_bufplan.cpp; `radiance --debug-graph` prints the ones it counts, with
 * the bytes), so each of them is arena the engine holds and never reads. The merge base's plan
 * charged fifteen of them at TP2/2048; five are gone with this change -- the dense MLP's bf16
 * product plane and the shared expert's, both declared now by the first layer that takes the
 * unfused pair, and the Engram round trip's three, declared now by the first layer whose table is
 * bf16 -- and the list below is what is left, a reason each, so a new plane no op names fails this
 * test by its name.
 *   moe.expert_ff.q / .s -- the routed arm's E4M3 product codes. An int8 arm's fused SwiGLU writes
 *       the rotated int8 pair the down projection reads, so no op names them; MoeFP8's own rule
 *       does ask for them (down_codebook_fp8 / down_supported,
 *       arch/common/rad_block_moe_fp8.h), the mixed IQR-gate_up-behind-a-w6-down recipe with them.
 *   moe.expert_ff -- the routed arm's bf16 product. The rig's dump names it (48 MiB at TP2/2048, 96
 *       at TP4/8192): there the fused gated quantiser serves the expert gate and nothing reads the
 *       plane. No plan of this fixture reaches that form -- at this geometry `silu_mul` is still
 *       declared for the arm -- so the product is in none of the lists below. It stays in the arena
 *       either way: MoeFP8::declare refuses a wire without it (the two tests of
 *       quantised_experts_are_served_clamped), which is why this PR hands over the planes the arch
 *       can place and not this one.
 *   moe.latent.q / .s -- the same for the latent linear: the gather's output is read as the rotated
 *       int8 pair, and MoeFP8::declare refuses a wire whose latent carries no codes at all.
 *   ix.stage -- the indexer stages a closing group's raw keys through it, and the step issues that
 *       path only where the library does not serve the fused key side (qsa_work_pool_tail).
 * The sampler's four candidate planes are arena for the whole program for the same reason, but the
 * sampler declares its ops with no operand lists at all (core/sample/sampler.cpp), so this builder
 * never sees them; making the plan name them is a core change. */
namespace {

std::string dead_join(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& n : v) s += n + " ";
    return s;
}

/* The plan's own rule: a transient no op of the declared list reads or writes. */
std::vector<std::string> unused_transients(const RadBuilder& b) {
    std::vector<std::string> dead;
    for (size_t i = 0; i < b.bufs.size(); ++i) {
        if ((int)b.bufs[i].d.kind != (int)RAD_BUF_TRANSIENT) continue;
        const rad_buf h = (rad_buf)(i + 1);
        bool used = false;
        for (const auto& o : b.ops)
            if (in(o.reads, h) || in(o.writes, h)) { used = true; break; }
        if (!used) dead.push_back(b.bufs[i].name);
    }
    std::sort(dead.begin(), dead.end());
    return dead;
}

/* How many buffers of the plan carry this name. The builder refuses a duplicate, so the answer is
 * 0 or 1: the count is how the Engram test says two layers share one trio instead of each
 * declaring its own, and how the P-11b test says a plane is not in the plan at all. */
int count_buf(const RadBuilder& b, const char* name) {
    int n = 0;
    for (const auto& r : b.bufs) n += r.name == name;
    return n;
}

/* Whether some declared op reads or writes the buffer of this name: charged and used is a plane the
 * step runs through, charged and unnamed is the arena this card is about. */
bool named_by_op(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.bufs.size(); ++i) {
        if (b.bufs[i].name != name) continue;
        const rad_buf h = (rad_buf)(i + 1);
        for (const auto& o : b.ops)
            if (in(o.reads, h) || in(o.writes, h)) return true;
        return false;
    }
    return false;
}

/* The bytes the plan of this test charges for nothing: a plane the list allows must not grow into a
 * wider one behind the same name, so the lint reads the bytes too. */
int64_t dead_bytes(const RadBuilder& b) {
    int64_t n = 0;
    for (size_t i = 0; i < b.bufs.size(); ++i) {
        const RadBufDecl& d = b.bufs[i].d;
        if ((int)d.kind != (int)RAD_BUF_TRANSIENT) continue;
        const rad_buf h = (rad_buf)(i + 1);
        bool used = false;
        for (const auto& o : b.ops)
            if (in(o.reads, h) || in(o.writes, h)) { used = true; break; }
        if (used) continue;
        int64_t numel = 1;
        for (uint32_t r = 0; r < d.rank; ++r) numel *= d.shape[r];
        n += rad_dtype_bytes(d.dtype, numel);
    }
    return n;
}

/* The served u3's shape (the recipe of /models/hy4-flash-iqr-v4-u3.rad): the trunk's linears and both
 * experts' under the rotated computed trellis, and the Engram table stored as E4M3 rows so its rows
 * take no round trip. This is the plan the card-less dump of the served model was read from. */
void served_u3_encs(RadBuilder* b) {
    int8_trunk_encs(b);
    const RadEncoding iqr = enc_trellis(RAD_U4, true);
    for (auto& e : b->encs)
        if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps" ||
            e.first == "ffn_gate_up." || e.first == "ffn_down." ||
            e.first == "ffn_gate_up_shexp" || e.first == "ffn_down_shexp" ||
            e.first == "ffn_latent_in" || e.first == "ffn_latent_out")
            e.second = iqr;
    b->encs.insert(b->encs.begin(), { { "engram_table", rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 0) },
                                      { "engram_mult", rad_enc_plain(RAD_I64) } });
}

}  // namespace

TEST(no_transient_is_declared_that_no_op_reads) {
    struct Plan {
        const char* what;
        void (*encs)(RadBuilder*);
        int world;
        /* The planes this plan leaves without a use, sorted, each with its reason in the comment
         * above. The set is the same for every plan but the bf16 ones, where the MoE codes the arm
         * does not need are not declared at all. */
        const char* kept;
        /* And their bytes, so a plane the list allows cannot quietly become a wider one. */
        long long dead;
    };
    const Plan plans[] = {
        { "bf16", bf16_encs, 1, "ix.stage ", 9216 },
        { "int8 trunk", int8_trunk_encs, 1, "ix.stage ", 9216 },
        { "int8 experts",
          [](RadBuilder* b) { bf16_encs(b); b->encs[0].second = b->encs[1].second = enc_i8(true); },
          1, "ix.stage moe.expert_ff.q moe.expert_ff.s moe.latent.q moe.latent.s ", 110592 },
        { "iqr experts",
          [](RadBuilder* b) {
              bf16_encs(b);
              b->encs[0].second = b->encs[1].second = enc_trellis(RAD_U4, true);
          },
          1, "ix.stage moe.expert_ff.q moe.expert_ff.s moe.latent.q moe.latent.s ", 110592 },
        { "bf16, two ranks", bf16_encs, 2, "ix.stage ", 9216 },
        /* The served recipe as the rig's dump was read (the Engram table at E4M3, so the round trip
         * is off; the trunk's linears and the experts' under the rotated trellis). It pins the four
         * MoE code planes of the dump's list. The product itself, `moe.expert_ff`, is not dead here:
         * at this geometry the expert gate is not served the fused gated quantiser, so `silu_mul`
         * still writes the plane and the down projection's quantiser reads it. On the rig it is --
         * the dump names it, 48 MiB at TP2/2048 and 96 MiB at TP4/8192, the largest plane of the ten
         * that stay; see notes/backlog/measurements/P-11b.md. */
        { "the served u3", served_u3_encs, 1,
          "ix.stage moe.expert_ff.q moe.expert_ff.s moe.latent.q moe.latent.s ", 110592 },
    };
    for (const Plan& p : plans) {
        Declared d(p.encs, p.world);
        REQUIRE_EQ(d.rc, RAD_OK);
        steps_clean_at(d.b, 0, 1);   /* every step kind, each lint clean */
        const int64_t bytes = dead_bytes(d.b);
        fprintf(stderr, "plan %-16s dead %lld bytes: %s\n", p.what, (long long)bytes,
                dead_join(unused_transients(d.b)).c_str());
        CHECK_EQ(dead_join(unused_transients(d.b)), std::string(p.kept));
        CHECK_EQ(bytes, (int64_t)p.dead);
    }
}

/* P-11b. A product plane belongs to the gate that writes it, and the round trip belongs to the
 * table it is taken from. Each of these planes was charged for the whole program from the wire,
 * where the declare cannot see which form the layer takes; each is now declared where the form is
 * settled, so a model that runs the other form reserves nothing for it. */
TEST(the_product_planes_follow_the_gate_that_writes_them) {
    /* The rotated int8 trunk with the fused gate: the shared arm's gate writes the codes it was
     * asked for, so the model has no shared-expert product plane at all -- that is the plane this
     * round takes away. The dense MLP's gate_up is not a rotated int8 weight in this container, so
     * its pair is unfused, its plane is there, and `silu_mul` names it: charged and used, which is
     * not the arena this card is about. */
    {
        Declared d(int8_trunk_encs, 1);
        REQUIRE_EQ(d.rc, RAD_OK);
        steps_clean_at(d.b, 0, 1);
        CHECK_EQ(count_op(d.b, "gated_had_quant_i8g") > 0, true);
        CHECK_EQ(count_buf(d.b, "moe.shared_ff"), 0);
        CHECK_EQ(count_buf(d.b, "dense_ff"), 1);
        CHECK(named_by_op(d.b, "dense_ff"));
    }
    /* The switch off: the pair is back, the first MoE layer declares its plane into the model's slot
     * for every later one -- one buffer for the model's two MoE layers, not one a layer -- and the
     * activation op names it. */
    setenv("RADIANCE_HY4_FUSED_QUANT", "0", 1);
    Declared unfused(int8_trunk_encs, 1);
    unsetenv("RADIANCE_HY4_FUSED_QUANT");
    REQUIRE_EQ(unfused.rc, RAD_OK);
    steps_clean_at(unfused.b, 0, 1);
    CHECK_EQ(count_buf(unfused.b, "dense_ff"), 1);
    CHECK(named_by_op(unfused.b, "dense_ff"));
    CHECK_EQ(count_buf(unfused.b, "moe.shared_ff"), 1);
    CHECK(named_by_op(unfused.b, "moe.shared_ff"));
    CHECK_EQ(count_op(unfused.b, "silu_mul") > 0, true);
    /* Both are named, so neither is what the plan charges for nothing. */
    CHECK_EQ(dead_join(unused_transients(unfused.b)), std::string("ix.stage "));
}

/* P-11b. The Engram round trip's three planes follow the trip: the served recipe stores the table
 * as E4M3 rows under a scale a row, which is what the model reads, so the trip is not taken and the
 * plan carries none of them; a bf16 table with `engram_fp8_embedding` takes it, and the two Engram
 * layers of this model share one trio -- the handles are carried to the later layer on the wire. */
TEST(the_engram_round_trip_declares_its_planes_only_when_it_is_taken) {
    /* The table as the u3 ships it: fp8_e4m3*f32[1x*] (data/recipes/hy4-flash-bf16-f32.recipe). */
    Declared e4m3([](RadBuilder* b) {
        bf16_encs(b);
        b->encs.insert(b->encs.begin(),
                       { { "engram_table", rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 0) },
                         { "engram_mult", rad_enc_plain(RAD_I64) } });
    }, 1, 0, false, 128, 0, 0, 0, two_engram_layers);
    REQUIRE_EQ(e4m3.rc, RAD_OK);
    steps_clean_at(e4m3.b, 0, 1);
    /* two_engram_layers puts the Engram at blk.0 and blk.2: the scale view is what says the trip
     * really was not taken. */
    REQUIRE(find_w(e4m3.b, "blk.0.engram_table.scale") != nullptr);
    for (const char* n : { "eg.rows.q", "eg.rows.s", "eg.rows.rt" })
        CHECK_EQ(count_buf(e4m3.b, n), 0);

    /* The bf16 table as shipped: the round trip runs, so the trio is in the plan once, named by the
     * quantiser and the dequant that write and read it. */
    Declared bf16(bf16_encs, 1, 0, false, 128, 0, 0, 0, two_engram_layers);
    REQUIRE_EQ(bf16.rc, RAD_OK);
    steps_clean_at(bf16.b, 0, 1);
    REQUIRE_EQ(count_op(bf16.b, "quant_act_fp8") > 0, true);
    for (const char* n : { "eg.rows.q", "eg.rows.s", "eg.rows.rt" }) {
        CHECK_EQ(count_buf(bf16.b, n), 1);
        CHECK(named_by_op(bf16.b, n));
    }
}

/* P-11b, the guard itself and not only the arch that leans on it. `GateQuantFP8::declare` picks a
 * form, and every form that names the bf16 plane -- `relu2`, any clamp, the bf16 line, the rotated
 * chain -- demands it; what this change permits is one case, the gated quantiser that writes the
 * codes and nothing else. The first round's review could see the hunk and nothing else, so both
 * halves are asserted here: a caller that asked for the product and handed no plane is refused as it
 * was before the change, and the codes-only caller is accepted and its row names two buffers, so no
 * launch can write buffer 0. */
TEST(a_gate_that_asks_for_the_product_without_a_plane_is_refused) {
    RadBuilder b;
    hyv4flash::Geom g;
    g.max_tok = 64;
    const rad_buf gate_up = hyv4flash::decl_b(&b, "t.gate_up", RAD_BF16, { 64, 256 });
    const rad_buf q       = hyv4flash::decl_b(&b, "t.q", RAD_F8E4M3, { 64, 128 });
    const rad_buf s       = hyv4flash::decl_b(&b, "t.s", RAD_F32,    { 64, 1 });
    const rad_buf x       = hyv4flash::decl_b(&b, "t.x", RAD_BF16,   { 64, 128 });
    REQUIRE(gate_up && q && s && x);

    hyv4flash::ActFP8 codes_only;   /* the codes pair, no bf16 plane: what an arch that declares none hands */
    codes_only.q = q;
    codes_only.s = s;

    hyv4flash::GateQuantFP8 asked;
    CHECK_EQ(asked.declare(&b, g, gate_up, codes_only, 128, "silu", 0, /*bf16_out=*/true),
             RAD_E_INVAL);
    CHECK_EQ(asked.want_x, true);
    REQUIRE(!b.ops.empty());
    CHECK_EQ((int)b.ops.back().writes.size(), 0);   /* refused before it named an operand */

    hyv4flash::ActFP8 with_plane = codes_only;
    with_plane.x = x;
    hyv4flash::GateQuantFP8 full;
    CHECK_OK(full.declare(&b, g, gate_up, with_plane, 128, "silu", 0, /*bf16_out=*/true));
    REQUIRE(!b.ops.empty());
    CHECK_EQ((int)b.ops.back().writes.size(), 3);
    CHECK(in(b.ops.back().writes, x));
    CHECK(in(b.ops.back().writes, q));

    hyv4flash::GateQuantFP8 folded;
    CHECK_OK(folded.declare(&b, g, gate_up, codes_only, 128, "silu", 0, /*bf16_out=*/false));
    CHECK_EQ(folded.want_x, false);
    REQUIRE(!b.ops.empty());
    CHECK_EQ(b.ops.back().op, std::string("gated_quant_fp8"));
    CHECK_EQ((int)b.ops.back().writes.size(), 2);
    for (const rad_buf h : b.ops.back().writes) CHECK(h != 0);
}
