#!/usr/bin/env python3
"""probe.py <port> <out.json> <phases,comma>: sampled (T 0.9, seeded) phases for the general-levers note.
phases: c1, c8, think (c1, reasoning high), c4, pf6 (prefill ~6.7K tokens: 22,000 characters of tree source),
pf20 (prefill ~24.9K: 72,000), pf2 (~2.4K: 7,400), stall1 / stall4 (a ~5.7K arrival while 1 / 4 decoders run),
arr60 (a ~60K-token arrival 2 s into one decoder, RAD_ARRIVE_S: S-10's "TTFT of a 60K prompt after 2 s
of decode"), tailN (a prompt of ~N characters), c1s (64 tok), c8s (48 tok), greedy (three prompts and one 5.7K arrival
at temperature 0 with ignore_eos off: the text itself, as gen / chars / sha256, so two runs' answers
compare byte for byte -- the merge bar's first clause), idle (a finished conversation's KV
leaves VRAM for the experts and its prefix still resumes; RAD_IDLE_CHARS, RAD_IDLE_WAIT_S), liveN
(N decoders, /stats read while they all decode and again once they are done: M-11's gain, the
window slots no sequence occupies), long8
(the item's eight long conversations at once, then idle, demand and return; RAD_LONG8_CHARS), and
longctx (eight ~46K-token sessions: a prefill wave of one token each, then the same eight prompts
decoded 128 tokens, so all eight are at long context together: the shape notes/backlog/refs/notes/
longctx.md measured as N8 48K, where the stager overflows its ring -- longctx-pre / longctx-decode in
the JSON), and lctxN (N sessions of RAD_LCTX_CHARS characters -- 190,000 is the 58,903-token
conversation of notes/backlog/CARDS.md §1.3, so the default is a 60K session -- filled by a one-token
wave and then decoded for RAD_LCTX_NTOK tokens: the long-context decode stamps of T-03, at one
decoder as `lctx1` and at four as `lctx4`, with `lctxN-pre` / `lctxN-decode` in the JSON), longresumeN
(X-03: a ~N-character first turn answered with RAD_RESUME_REPLY tokens, then the same conversation with
the reply and ~RAD_RESUME_NEW new tokens: each turn's prompt, prefix hits, tokens actually prefilled and
its text's sha256, so a resumed and a cold arm compare byte for byte), and stress (issue #251: RAD_STRESS_WAVES waves
of eight requests at once, RAD_STRESS_CHARS characters of context and RAD_STRESS_NTOK tokens each, a
fresh wave and then the same prompts again; failed requests and a server that stops are counted, not
raised: `ok` / `err` / `server_dead`). The token
counts are the ones the backlog's own runs measured on these phases, so a lane's
ttft compares with the numbers in notes/backlog/."""
import hashlib, json, sys, threading, time, urllib.request, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import CODE_PROMPTS as P, GREEDY, SRC, long_prompt
PORT, OUT, PH = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
BASE = f"http://127.0.0.1:{PORT}"
def get(p): return urllib.request.urlopen(BASE + p, timeout=10).read().decode()
def find(d, k):
    if isinstance(d, dict):
        if k in d: return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None: return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, k)
            if r is not None: return r
    return None
def stats():
    o = json.loads(get("/stats")); e = find(o, "experts") or {}; l = find(o, "link") or {}
    d = {k: v for k, v in e.items() if isinstance(v, (int, float))}
    d.update({"link_" + k: v for k, v in l.items() if isinstance(v, (int, float))}); return d
def metrics():
    m = {}
    for ln in get("/metrics").splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        k, _, v = ln.rpartition(" "); n = k.split("{")[0]
        if n.startswith("radiance:"):
            try: m[n] = m.get(n, 0) + float(v)
            except ValueError: pass
    return m
SEED = [1000]
def temperature():
    # RAD_TEMP, else TEMP when it holds a number: a measuring host often sets TEMP to a directory
    # (it is a system variable, so a value that is not a number is ignored).
    want, v = os.environ.get("RAD_TEMP"), os.environ.get("RAD_TEMP") or os.environ.get("TEMP") or "0.9"
    try: return float(v)
    except ValueError:
        if want: sys.exit(f"probe.py: RAD_TEMP '{want}' is not a number")
        return 0.9
def request(content, n, effort="none"):
    SEED[0] += 1
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": temperature(), "top_p": 1.0, "seed": SEED[0], "reasoning_effort": effort, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; gen = 0; pt = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:"): continue
            d = ln[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d)
            if ev.get("usage"): gen = ev["usage"].get("completion_tokens", 0); pt = ev["usage"].get("prompt_tokens", 0)
            for ch in ev.get("choices", []):
                c = (ch.get("delta", {}).get("content") or "") + (ch.get("delta", {}).get("reasoning_content") or "")
                if c and tf is None: tf = time.time()
    t1 = time.time()
    return {"gen": gen, "prompt": pt, "ttft": (tf or t1) - t0, "dec": (gen - 1) / (t1 - tf) if tf and gen > 1 else 0, "t_first": tf or t1, "t1": t1}
def request_gaps(content, n):
    SEED[0] += 1
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0.9, "top_p": 1.0, "seed": SEED[0], "reasoning_effort": "none", "stream": True, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ts = []; t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if ln.startswith("data:") and ln[5:].strip() != "[DONE]": ts.append(time.time() - t0)
    return ts
def stage_sampler(period=0.04):
    """Poll /stats' expert staging while an arrival runs: (stop flag, samples, t0), a sample being
    [ms since start, stage_layers, staged_bytes, stage_copies]. The design's staging check (S-02 §5,
    §10 item 1) is per sub-step: the group's first slice should pay the whole chunk's copy and the
    later ones find the ring warm, so when the increments happen, against `arrival_ttft`, matters as
    much as their total."""
    stop = threading.Event(); samples = []; t0 = time.time()
    def run():
        while not stop.is_set():
            try:
                e = find(json.loads(get("/stats")), "experts") or {}
                samples.append([round((time.time() - t0) * 1000), int(e.get("stage_layers") or 0),
                                int(e.get("staged_bytes") or 0), int(e.get("stage_copies") or 0)])
            except Exception:
                pass
            stop.wait(period)
    threading.Thread(target=run, daemon=True).start()
    return stop, samples, t0

def stall(name, nbg, chars, ntok=420, after=1.5):
    out = {}; bg = [None] * nbg
    def w(i): bg[i] = request_gaps(P[i], ntok)
    th = [threading.Thread(target=w, args=(i,)) for i in range(nbg)]
    m0 = metrics(); t0 = time.time()
    stop, samples, _ = stage_sampler()
    for t in th: t.start()
    time.sleep(after); ta = time.time() - t0
    r = request(long_prompt(chars, 31 + nbg), 8); tb = time.time() - t0
    for t in th: t.join()
    stop.set()
    m1 = metrics(); md = {k.split(":")[1]: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
    gaps = []
    for ts in bg:
        g = sorted(((b - a) * 1000, a) for a, b in zip(ts, ts[1:]) if ta <= b <= tb + 0.5)
        gaps.append([round(x[0], 1) for x in g[-8:]])
    typ = sorted((b - a) * 1000 for ts in bg for a, b in zip(ts, ts[1:]) if b < ta)
    # Staging inside the arrival window: the samples' increments over the last one before it.
    base = [s for s in samples if s[0] / 1000 <= ta]
    win = [s for s in samples if ta <= s[0] / 1000 <= tb + 0.5]
    b0 = base[-1] if base else (samples[0] if samples else [0, 0, 0, 0])
    stage = {"base": [round(ta * 1000), b0[1], b0[2], b0[3]],
             "samples": [[s[0], s[1] - b0[1], s[2] - b0[2], s[3] - b0[3]] for s in win],
             "stage_layers": (win[-1][1] if win else b0[1]) - b0[1],
             "staged_bytes": (win[-1][2] if win else b0[2]) - b0[2],
             "stage_copies": (win[-1][3] if win else b0[3]) - b0[3]}
    o = {"name": name, "bg": nbg, "arrival_prompt": r["prompt"], "arrival_ttft": round(r["ttft"], 3), "arrive_at": round(ta, 2), "first_at": round(ta + r["ttft"], 2),
         "typical_gap_ms": round(typ[len(typ) // 2], 1) if typ else None, "worst_gaps_ms": gaps, "metrics": md, "clients": nbg, "mono_ns": [0, 0], "stage": stage}
    print(json.dumps({k: v for k, v in o.items() if k != "metrics"} | {"steps_sliced": md.get("engine_steps_sliced_total", 0), "substeps": md.get("engine_substeps_total", 0)}), flush=True); time.sleep(2); return o
def greedy_text(content, n=256):
    # Temperature 0 and ignore_eos off: no sampler noise and no cut-off, so the only thing that can
    # change the text is the step's chunking or a kernel's arithmetic.
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0.0, "top_p": 1.0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    text = []; gen = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:"): continue
            d = ln[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d)
            if ev.get("usage"): gen = ev["usage"].get("completion_tokens", 0)
            for ch in ev.get("choices", []):
                c = ch.get("delta", {}).get("content") or ""
                if c: text.append(c)
    s = "".join(text)
    return {"gen": gen, "chars": len(s), "sha256": hashlib.sha256(s.encode()).hexdigest()}

def greedy():
    # The three short prompts are one chunk either way; the 5.7K one is not, so it is the prompt whose
    # chunks a budget change re-tiles -- it shows whether two arms' chunk boundaries and row sets
    # agree where they matter.
    out = [greedy_text(p) for p in GREEDY]
    out.append(dict(greedy_text(long_prompt(22000, 5701, stamp=False), 64), name="arrival5.7K"))
    res = {"name": "greedy", "texts": out}
    print(json.dumps(res), flush=True)
    return res

def view(at):
    """What the views the host-tier check reads say, at one moment: the KV pool in VRAM and the expert
    slab that borrows its spare room (/stats), and the tier's bytes, counters and stored sessions
    (/sessions). `expert_units` is the residency per tier -- the item's gain is `vram` growing by the
    units the finished conversation's blocks were holding -- and `slab_mib` is the lent VRAM the
    experts hold, against `lend_mib`, the most the KV pool lets them borrow."""
    t = json.loads(get("/stats")); o = json.loads(get("/sessions"))
    p, mv, oc, st = o.get("pool", {}), o.get("moved", {}), o.get("occupancy", {}), o.get("stall", {})
    ex = find(t, "experts") or {}
    tiers = [r for r in (ex.get("tiers") or []) if isinstance(r, dict)]
    return {"at": at,
            "kv_mib": round(t.get("kv_bytes_used") or 0, 0) / 2**20, "kv_total_mib": round(t.get("kv_bytes_total") or 0, 0) / 2**20,
            "kv_tokens_cached": t.get("kv_tokens_cached"), "kv_tokens_used": t.get("kv_tokens_used"),
            "hit_cuts": t.get("hit_cut_tokens"),
            "expert_n_units": ex.get("n_units"),
            "expert_units": {r.get("name"): r.get("units") for r in tiers},
            "expert_mib": {r.get("name"): round((r.get("bytes") or 0) / 2**20, 1) for r in tiers},
            "slab_mib": round((ex.get("flex_bytes") or 0) / 2**20, 1),
            "lend_mib": round((ex.get("flex_capacity") or 0) / 2**20, 1),
            "routed": ex.get("routed"), "routed_res": ex.get("routed_res"),
            "promotions": ex.get("promotions"), "demotions": ex.get("demotions"),
            "starved_slab": ex.get("starved_slab"),
            "host_on": o.get("enabled"), "host_mib": round((oc.get("host_bytes") or 0) / 2**20, 1),
            "host_cap_mib": round((oc.get("host_cap_bytes") or 0) / 2**20, 1),
            "to_host": mv.get("to_host"), "promoted": mv.get("promoted"), "dropped": mv.get("dropped"),
            "held_back": mv.get("held_back"), "restores": st.get("restores"),
            "pool_mib": round((p.get("bytes_used") or 0) / 2**20, 1),
            "kv_loan": t.get("kv_loan") or {},
            "stored": [{k: s.get(k) for k in ("tokens", "idle_s", "on_device", "on_host", "resume_tokens")}
                       for s in o.get("sessions", [])]}
def hits(fn):
    """Run fn and report the prompt tokens it took from the prefix cache, from /metrics' own counters
    (`vllm:prefix_cache_hits_total`, and the attention blocks `radiance:attn_cached_tokens_total`)."""
    def c():
        m = {}
        for ln in get("/metrics").splitlines():
            if ln.startswith("#") or not ln.strip(): continue
            k, _, v = ln.rpartition(" "); n = k.split("{")[0]
            try: m[n] = m.get(n, 0) + float(v)
            except ValueError: pass
        return m
    m0 = c(); r = fn(); m1 = c()
    d = lambda k: m1.get(k, 0) - m0.get(k, 0)
    r["hit_tokens"] = int(d("vllm:prefix_cache_hits_total"))
    r["attn_cached_tokens"] = int(d("radiance:attn_cached_tokens_total"))
    return r
def delta(a, b, k):
    """A cumulative counter's change between two views."""
    return (b.get(k) or 0) - (a.get(k) or 0)
def idle(chars):
    """The host-tier check (X-01): a conversation that finishes and goes idle, a decode-only request
    that makes the experts ask for VRAM back, then the same prefix again (a resume) against a cold
    prefill of the same size. Read with `prefix-cache-mib: 0` and with it set: the difference is the
    tier's, in /stats' kv bytes and /sessions' host bytes, moved counters and resume tokens, and in
    the expert units the decode-only request can then make resident (expert_vram_units here, with
    slab_mib against lend_mib: the VRAM the experts hold against the most the KV pool lends them)."""
    wait = float(os.environ.get("RAD_IDLE_WAIT_S", 5))
    pA, pB = long_prompt(chars, 71), long_prompt(chars, 72, stamp=True)   # pB is a cold arrival
    v = [view("start")]
    a = hits(lambda: request(pA, 4)); v.append(view("conversation"))
    time.sleep(wait); v.append(view("idle"))
    d = request(P[0], 64); v.append(view("demanded"))
    r = hits(lambda: request(pA, 4)); v.append(view("resume"))
    c = hits(lambda: request(pB, 4)); v.append(view("cold"))
    o = {"name": "idle", "chars": chars, "wait_s": wait,
         "prompt": [a["prompt"], r["prompt"], c["prompt"]],
         "conv_ttft": round(a["ttft"], 3), "idle_d_ttft": round(d["ttft"], 3),
         "resume_ttft": round(r["ttft"], 3), "cold_ttft": round(c["ttft"], 3),
         "resume_hit_tokens": r["hit_tokens"], "resume_attn_cached": r["attn_cached_tokens"],
         "cold_hit_tokens": c["hit_tokens"], "cold_attn_cached": c["attn_cached_tokens"],
         "conv_decode_tps": round(a["dec"], 1),
         # the item's gain, at each step of the check: the expert units resident in VRAM and the slab
         # holding them; the demand's hit rate is the first effect of the units it promoted
         "expert_vram_units": {x["at"]: (x["expert_units"] or {}).get("vram") for x in v},
         "slab_mib": {x["at"]: x["slab_mib"] for x in v},
         "lend_mib": {x["at"]: x["lend_mib"] for x in v},
         "demand_routed": delta(v[2], v[3], "routed"), "demand_routed_res": delta(v[2], v[3], "routed_res"),
         "views": v}
    print(json.dumps({k: x for k, x in o.items() if k != "views"}), flush=True)
    for x in v: print(json.dumps(x), flush=True)
    time.sleep(2)
    return o
def loan_frag(v):
    """The KV loan's headroom in MiB, from the /stats `kv_loan` object (M-15): `holes_mib` is free
    blocks under the highest held one, which the loan may not cross and a compaction would recover;
    `buffer_mib` is what the cache keeps in hand; `headroom_mib` is what the arbiter has not lent
    yet. The blocks are the binding group's, so `block_bytes` turns them into VRAM."""
    l = (v or {}).get("kv_loan") or {}
    mib = (l.get("block_bytes") or 0) / 2**20
    held, top = l.get("held") or 0, l.get("held_top") or 0
    buf, live = l.get("buffer_blocks") or 0, l.get("live") or 0
    return {"group": l.get("group"), "held": held, "held_top": top,
            "holes_mib": round((top - held) * mib, 1),
            "buffer_mib": round(buf * mib, 1),
            "headroom_mib": round(max(0, live - top - buf) * mib, 1),
            "live_mib": round(live * mib, 1),
            "spare_tokens": l.get("spare_tokens"), "loan_tokens": l.get("loan_tokens")}

def live(n, ntok=256):
    """M-11: the loan at N live sequences -- the item's own point, "73-129 units at 1-4 live
    sequences". Starts N decoders, reads /stats while they all decode (the loan ticks every 100 ms
    and fills a slot only for a unit that is wanted, so a read 1.5 s in is settled), and again when
    they are done: `expert_vram_units` and `slab_mib` against `lend_mib` at each point, the units
    the window group's idle slots let the slab hold. `views` keeps the whole /stats and /sessions
    reading at each of the three points."""
    ps = [P[i] for i in (1, 3, 5, 7)][:n]
    res = [None] * n
    def w(i): res[i] = request(ps[i], ntok)
    v = [view("start")]
    t0 = time.time()
    th = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    for t in th: t.start()
    time.sleep(float(os.environ.get("RAD_LIVE_S", 1.5)))
    v.append(view("live"))
    for t in th: t.join()
    v.append(view("end"))
    wall = max(r["t1"] for r in res) - t0
    o = {"name": f"live{n}", "clients": n, "gen": sum(r["gen"] for r in res), "wall": round(wall, 3),
         "ttft": [round(r["ttft"], 3) for r in res],
         "agg_tps": round(sum(r["gen"] for r in res) / wall, 1),
         "sum_req_dec_tps": round(sum(r["dec"] for r in res), 1),
         "expert_vram_units": {x["at"]: (x["expert_units"] or {}).get("vram") for x in v},
         "slab_mib": {x["at"]: x["slab_mib"] for x in v},
         "lend_mib": {x["at"]: x["lend_mib"] for x in v},
         "starved_slab": {x["at"]: x["starved_slab"] for x in v}, "views": v}
    print(json.dumps({k: x for k, x in o.items() if k != "views"}), flush=True)
    for x in v: print(json.dumps(x), flush=True)
    time.sleep(2)
    return o

def long8(chars, gen=32):
    """The item's multi-session case (M longctx, 8 sessions of ~50K tokens: 391K cached tokens): eight
    long conversations at once, then idle, then a decode-only request that asks the experts for the
    VRAM they hold, then the same eight back. With `prefix-cache-mib: 0` the pool keeps their blocks
    and the expert loan cannot have them; with it set the blocks move to host, the slab grows
    (slab_mib against lend_mib) and the return pass still hits the prefix cache."""
    v = [view("start")]
    pr = [long_prompt(chars, 80 + i) for i in range(8)]
    a = phase("long8a", pr, gen); v.append(view("finished"))
    time.sleep(float(os.environ.get("RAD_IDLE_WAIT_S", 5))); v.append(view("idle"))
    dq = request(P[0], 64); v.append(view("demanded"))
    r = hits(lambda: phase("long8b", pr, gen)); v.append(view("resume"))
    o = {"name": "long8", "chars": chars, "gen": gen, "tokens": a["prompt"][0],
         "first_ttft": a["ttft"], "first_agg_tps": round(a["agg_tps"], 1), "first_hit": round(a["hit"], 4),
         "resume_ttft": r["ttft"], "resume_agg_tps": round(r["agg_tps"], 1),
         "resume_hit_tokens": r["hit_tokens"], "demand_ttft": round(dq["ttft"], 3),
         "demand_hit": round(delta(v[2], v[3], "routed_res") / max(1, delta(v[2], v[3], "routed")), 4),
         "kv_mib": {x["at"]: x["kv_mib"] for x in v}, "host_mib": {x["at"]: x["host_mib"] for x in v},
         "expert_vram_units": {x["at"]: (x["expert_units"] or {}).get("vram") for x in v},
         "slab_mib": {x["at"]: x["slab_mib"] for x in v}, "lend_mib": {x["at"]: x["lend_mib"] for x in v},
         "loan": {x["at"]: loan_frag(x) for x in v},
         "views": v, "waves": [a, r]}
    print(json.dumps({k: x for k, x in o.items() if k not in ("views", "waves")}), flush=True)
    for x in v: print(json.dumps(x), flush=True)
    time.sleep(2)
    return o
def phase(name, prompts, n=256, effort="none"):
    res = [None] * len(prompts)
    def w(i): res[i] = request(prompts[i], n, effort)
    s0, m0 = stats(), metrics(); ns0 = time.monotonic_ns(); t0 = time.time()
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(prompts))]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; ns1 = time.monotonic_ns(); s1, m1 = stats(), metrics()
    sd = {k: s1[k] - s0.get(k, 0) for k in s1}; md = {k.split(":")[1]: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
    lo = max(r["t_first"] for r in res); hi = min(r["t1"] for r in res)
    steps = md.get("engine_steps_decode_total", 0)
    o = {"name": name, "clients": len(prompts), "wall": wall, "gen": sum(r["gen"] for r in res), "prompt": [r["prompt"] for r in res],
         "ttft": [round(r["ttft"], 3) for r in res], "agg_tps": sum(r["gen"] for r in res) / wall,
         "sum_req_dec_tps": sum(r["dec"] for r in res), "decode_steps": steps,
         "ms_step": 1000 * (hi - lo) / max(1, steps) if len(prompts) == 1 else None,
         "tok_per_step": sum(r["gen"] for r in res) / max(1, steps),
         "mono_ns": [ns0, ns1], "all_decoding_s": [lo - t0, hi - t0], "stats": sd, "metrics": md,
         "stage_layers": s1.get("stage_layers"),
         "hit": sd.get("routed_res", 0) / max(1, sd.get("routed", 1)),
         "routed_per_step": sd.get("routed", 0) / max(1, steps),
         "miss_units_per_step": (sd.get("routed", 0) - sd.get("routed_res", 0)) / max(1, steps)}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k not in ("stats", "metrics")} | {"steps_sliced": md.get("engine_steps_sliced_total", 0), "substeps": md.get("engine_substeps_total", 0)}), flush=True)
    time.sleep(2)
    return o
def resume(chars, extra=900, turns=3, ntok=4):
    """A resumed turn (M-03 (b): the churn case, 213-450 tokens re-prefilled a turn, TTFT 0.69-0.72 s
    on the audit host). The session's first turn fills the KV; each later turn is the same prompt with
    this turn's rows appended, so the prefix cache covers all of it but those rows and the step that
    remains is one lone small prefill step -- the one the item stages. Reports each turn's TTFT, its
    prompt tokens, the rows it took from the prefix cache and the rows it had to prefill."""
    first = long_prompt(chars, 90)
    out = []
    def turn(p):
        s0 = stats()
        r = hits(lambda: request(p, ntok))
        s1 = stats()
        r["stage_layers"] = s1.get("stage_layers")
        return r
    a = turn(first)
    later = [turn(first + long_prompt(extra, 91 + i)) for i in range(turns)]
    o = {"name": "resume", "chars": chars, "extra_chars": extra, "conv_prompt": a["prompt"],
         "conv_ttft": round(a["ttft"], 3),
         "turn_prompt": [r["prompt"] for r in later], "turn_hit": [r["hit_tokens"] for r in later],
         "turn_prefill": [r["prompt"] - r["hit_tokens"] for r in later],
         "turn_ttft": [round(r["ttft"], 3) for r in later],
         "turn_attn_cached": [r["attn_cached_tokens"] for r in later],
         "stage_layers": [r["stage_layers"] for r in later]}
    print(json.dumps(o), flush=True); time.sleep(2)
    return o

def longresume(chars, n1=1024, n2=64, runs=3):
    """X-03 (issue #130): the next turn of an agentic conversation, after a long reply. Turn 1 is a
    ~`chars`-character code prompt answered with `n1` output tokens (greedy, ignore_eos), so the
    reply builds the decode rows a rolling decode snapshot is for; turn 2 is the same conversation
    -- that prompt, the reply as the assistant turn and ~`n2` tokens of new user text -- which the
    prefix cache can resume past the reply only where the first turn left a checkpoint: without a
    decode snapshot that is the prompt's end, so the whole reply is prefilled again. Per turn:
    prompt tokens, the ones the prefix cache covered, the ones the engine actually computed
    (`radiance:prefill_tokens_total`), the checkpoints written, TTFT, decode speed and the sha256
    of the text; turn 2's hash is what a resumed arm and a cold arm compare, byte for byte.
    `runs` repeats the conversation with a fresh prompt each time (the same seeds in every arm)."""
    def chat(msgs, n):
        SEED[0] += 1
        body = {"model": "hy4-flash", "messages": msgs, "max_tokens": n, "temperature": 0.0,
                "top_p": 1.0, "seed": SEED[0], "reasoning_effort": "none", "stream": True,
                "stream_options": {"include_usage": True}, "ignore_eos": True}
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time(); tf = None; gen = pt = 0; text = []
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                ln = raw.decode(errors="replace").strip()
                if not ln.startswith("data:"): continue
                d = ln[5:].strip()
                if d == "[DONE]": break
                ev = json.loads(d)
                if ev.get("usage"):
                    gen = ev["usage"].get("completion_tokens", 0); pt = ev["usage"].get("prompt_tokens", 0)
                for ch in ev.get("choices", []):
                    c = ch.get("delta", {}).get("content") or ""
                    if c:
                        text.append(c)
                        if tf is None: tf = time.time()
        t1 = time.time()
        return {"gen": gen, "prompt": pt, "ttft": (tf or t1) - t0,
                "dec": (gen - 1) / (t1 - tf) if tf and gen > 1 else 0, "t1": t1,
                "text": "".join(text)}
    def turn(msgs, n, name):
        m0 = metrics(); r = chat(msgs, n); m1 = metrics()
        d = lambda k: int(m1.get(k, 0) - m0.get(k, 0))
        o = {"name": name, "gen": r["gen"], "prompt": r["prompt"],
             "hit_tokens": d("vllm:prefix_cache_hits_total"),
             "prompt_tokens_metric": d("vllm:prompt_tokens_total"),
             "prefill_tokens": d("radiance:prefill_tokens_total"),
             "decode_tokens": d("radiance:decode_tokens_total"),
             "checkpoints_written": d("radiance:linear_checkpoints_written_total"),
             "decode_steps": d("radiance:engine_steps_decode_total"),
             "ttft": round(r["ttft"], 3), "dec_tps": round(r["dec"], 2),
             "chars": len(r["text"]), "sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
        print(json.dumps({k: v for k, v in o.items() if k != "sha256"} | {"sha256": o["sha256"][:16]}), flush=True)
        return o, r["text"]
    new = ("Continue from the code you just wrote. Add a --verbose flag, a --seed option and a test "
           "that runs the program twice with the same seed and checks that the output is identical. "
           "Show only the changed parts, then explain in one short paragraph why the test is worth "
           "keeping and what could still make it flaky on a machine with a different clock.")
    out = []
    for i in range(runs):
        p1 = long_prompt(chars, 130 + 100 * i)
        t1o, reply = turn([{"role": "user", "content": p1}], n1, f"longresume{chars}-r{i}-turn1")
        t2o, _ = turn([{"role": "user", "content": p1}, {"role": "assistant", "content": reply},
                       {"role": "user", "content": new}], n2, f"longresume{chars}-r{i}-turn2")
        out.append({"run": i, "turn1": t1o, "turn2": t2o})
    o = {"name": "longresume", "chars": chars, "reply_tokens": n1, "new_tokens": n2, "runs": out}
    print(json.dumps({"name": "longresume", "prompt1": [x["turn1"]["prompt"] for x in out],
                      "turn2_prompt": [x["turn2"]["prompt"] for x in out],
                      "turn2_hit": [x["turn2"]["hit_tokens"] for x in out],
                      "turn2_prefill": [x["turn2"]["prefill_tokens"] for x in out],
                      "turn2_sha256": [x["turn2"]["sha256"] for x in out],
                      "turn1_tps": [x["turn1"]["dec_tps"] for x in out]}), flush=True)
    return o

def longctx(chars=140000, n=8, ntok=128):
    # Eight sessions arrive and fill. One prompt each, built once, so both waves carry the same
    # strings -- what a prefix-cache hit needs (at 46K they can still be evicted: check the log's
    # `cache` figure, which a re-prefill sets to 0).
    ps = [long_prompt(chars, 40 + i, stamp=True) for i in range(n)]
    pre = phase("longctx-pre", ps, 1)
    dec = phase("longctx-decode", ps, ntok)
    return {"name": "longctx", "prefill": pre, "decode": dec}

def lctx(n, chars, ntok):
    """T-03, the long-context decode stamps: N conversations of about `chars` characters fill with a
    one-token wave, then the same strings decode, so the decode passes carry the whole context. The
    default 190,000 characters is the 58,903-token conversation notes/backlog/CARDS.md §1.3 measured,
    so `lctx1` and `lctx4` are that session at one and at four decoders -- the shape K-25's
    `qsa_select` and `idx_score` terms have no served number for (notes/backlog/refs/notes/idxint8.md
    has only `--perfidx` ones); the run's short `c1` / `c4` phases are the baseline they compare to.
    Both waves are recorded as ordinary phases (`lctxN-pre`, `lctxN-decode`), so agg.py --match, attr.py
    and layer.py window the decode at 60K. `hit` near 1 and a non-zero `cache` in the server log say the
    decode wave resumed the fill rather than prefilling it again -- at four sessions the whole 236K
    tokens are in the pool, so read both before trusting the decode step's context."""
    ps = [long_prompt(chars, 120 + i, stamp=True) for i in range(n)]
    return [phase(f"lctx{n}-pre", ps, 1), phase(f"lctx{n}-decode", ps, ntok)]

R = {"phases": []}
def stress(waves, chars, ntok, clients=8):
    """Sustained concurrent load (issue #251): `waves` waves of `clients` requests at once, each a
    ~`chars`-character context decoded for `ntok` tokens. A wave's prompts are fresh (cold prefill
    beside the others' decode) and the wave after it sends the same ones again (prefix hit, decode
    at that context). A request that fails or a server that stops answering is counted,
    not raised: `ok` / `err` and whether /health still answers are the result."""
    ok = err = slow = 0; pr = []; t0 = time.time(); dead = False; wv = []; m0 = metrics(); m1 = {}
    for wi in range(waves):
        if wi % 2 == 0: pr = [long_prompt(chars, 300 + wi * clients + i, stamp=True) for i in range(clients)]
        res = [None] * clients
        def w(i):
            try: res[i] = request(pr[i], ntok)
            except Exception as e: res[i] = {"error": repr(e)[:120]}
        th = [threading.Thread(target=w, args=(i,)) for i in range(clients)]
        tw = time.time()
        for t in th: t.start()
        for t in th: t.join()
        good = [r for r in res if "error" not in r and r["gen"] > 0]
        ok += len(good); err += clients - len(good); slow += sum(1 for r in good if r["ttft"] > 30)
        wv.append({"wave": wi, "ok": len(good), "wall": round(time.time() - tw, 2), "gen": sum(r["gen"] for r in good),
                   "errors": [r["error"] for r in res if "error" in r][:2]})
        print(json.dumps(wv[-1]), flush=True)
        try: get("/health"); m1 = metrics()
        except Exception: dead = True; break
    o = {"name": "stress", "waves": len(wv), "of": waves, "clients": clients, "chars": chars, "ntok": ntok,
         "ok": ok, "err": err, "slow": slow, "server_dead": dead, "wall": round(time.time() - t0, 1),
         "steps": {k.split(":")[1]: m1[k] - m0.get(k, 0) for k in m1 if "engine_steps" in k and m1[k] != m0.get(k, 0)},
         "per_wave": wv}
    print(json.dumps({k: v for k, v in o.items() if k != "per_wave"}), flush=True)
    return o
def record(o):
    """One phase's result into the JSON. A composite phase (`longctx`, `long8`) keeps its summary entry
    as it always did, and now also records the waves it ran as phases of their own: the stamp readers
    (`agg.py`, `attr.py`, `layer.py`) work per phase window and a summary entry has none, so without
    this a stamped long-context run cannot be read at all -- the readers name and skip what has no
    window, and read the waves this puts in."""
    R["phases"].append(o)
    if isinstance(o, dict):
        for v in o.values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, dict) and x.get("mono_ns") and "decode_steps" in x: R["phases"].append(x)
if "nowarm" not in PH: request(P[11], 48); request(P[10], 48)
for ph in PH:
    if ph == "c1": R["phases"].append(phase("c1", [P[0]]))
    elif ph == "c1b": R["phases"].append(phase("c1b", [P[5]]))
    elif ph == "c8": R["phases"].append(phase("c8", [P[i] for i in range(8)]))
    elif ph == "c4": R["phases"].append(phase("c4", [P[i] for i in (1, 3, 5, 7)]))
    elif ph == "c2": R["phases"].append(phase("c2", [P[2], P[6]]))
    elif ph == "think": R["phases"].append(phase("think", [P[4]], 384, "high"))
    elif ph == "c1s": R["phases"].append(phase("c1s", [P[0]], 64))
    elif ph == "c8s": R["phases"].append(phase("c8s", [P[i] for i in range(8)], 48))
    elif ph == "pf6": R["phases"].append(phase("pf6", [long_prompt(22000, 7, stamp=True)], 2))
    elif ph == "pf20": R["phases"].append(phase("pf20", [long_prompt(72000, 9, stamp=True)], 2))
    elif ph == "pf2": R["phases"].append(phase("pf2", [long_prompt(7400, 5)], 2))
    elif ph == "pf8": R["phases"].append(phase("pf8", [long_prompt(22000, 61 + i) for i in range(8)], 2))
    elif ph == "stall1": R["phases"].append(stall("stall1", 1, 22000))
    elif ph == "stall4": R["phases"].append(stall("stall4", 4, 22000))
    elif ph == "arr60":  # S-10: the whole prompt's loan recalled once, read as TTFT under decode
        R["phases"].append(stall("arr60", 1, int(os.environ.get("RAD_ARR60_CHARS", 174000)),
                                 after=float(os.environ.get("RAD_ARRIVE_S", 2.0))))
    elif ph.startswith("tail"):  # tailN: prompt of ~N chars lone, to size small prefill steps
        R["phases"].append(phase(ph, [long_prompt(int(ph[4:]), 3)], 2))
    elif ph == "greedy": record(greedy())
    elif ph == "longctx": record(longctx())
    elif ph.startswith("lctx"):  # lctxN: N sessions of RAD_LCTX_CHARS fill, then the same strings decode
        if not ph[4:].isdigit():
            sys.exit(f"probe.py: lctx wants a session count (lctx1, lctx4); got '{ph}'")
        R["phases"].extend(lctx(int(ph[4:]), int(os.environ.get("RAD_LCTX_CHARS", 190000)),
                                int(os.environ.get("RAD_LCTX_NTOK", 128))))
    elif ph == "pf6c": R["phases"].append(phase("pf6c", [long_prompt(22000, 7, stamp=True)], 2))
    elif ph.startswith("resume"):  # resumeN: a ~N-character session, then N-row turns resumed
        record(resume(int(ph[6:]), int(os.environ.get("RAD_RESUME_EXTRA", 900))))
    elif ph.startswith("longresume"):  # X-03 (#130): a long reply, then the conversation's next turn
        R["phases"].append(longresume(int(ph[10:]), int(os.environ.get("RAD_RESUME_REPLY", 1024)),
                                      int(os.environ.get("RAD_RESUME_NEW", 64)),
                                      int(os.environ.get("RAD_RESUME_RUNS", 3))))
    elif ph == "idle": record(idle(int(os.environ.get("RAD_IDLE_CHARS", 72000))))
    elif ph == "nowarm": pass  # read above: no warmup requests
    elif ph == "stress":  # issue #251: RAD_STRESS_WAVES waves of eight ~12K-token requests
        R["phases"].append(stress(int(os.environ.get("RAD_STRESS_WAVES", 40)), int(os.environ.get("RAD_STRESS_CHARS", 39500)),
                                  int(os.environ.get("RAD_STRESS_NTOK", 96))))
    elif ph.startswith("live"): R["phases"].append(live(int(ph[4:])))
    elif ph == "long8": record(long8(int(os.environ.get("RAD_LONG8_CHARS", 140000))))
    else: sys.exit(f"probe.py: no phase '{ph}'; phases: c1 c1b c1s c2 c4 c8 c8s think pf2 pf6 pf6c pf8 pf20 "
                  "greedy longctx lctxN stall1 stall4 arr60 tailN idle liveN long8 resumeN longresumeN stress nowarm")
json.dump(R, open(OUT, "w"), indent=1)
