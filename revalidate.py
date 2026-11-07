"""Regenerate every benchmark number the report cites, from one consistent code state.

Why this file exists: the generator had two defects (reactions dropped when a leader had two or
more followers whose coins both landed, and a shared base seed across grid cells) and the evaluator
reported precision only over instances that fired. Every earlier result file predates those fixes,
so the report's benchmark numbers had to be recomputed rather than trusted.

Everything here uses the SPARSITY-PRESERVING permutation surrogate, because the shift surrogate is
the thing this project shows to be invalid. "pairwise" and "conditional" therefore differ only in
whether the channel's recent state is conditioned on, which is the comparison the claim is about.
"""
import sys, json, time, collections
sys.path.insert(0, r"C:\dev\swarm-prep"); sys.path.insert(0, r"C:\dev\swarm-sprint")
sys.path.insert(0, r"C:\dev\swarm-bench")
import numpy as np, swarm_te
from swarm_load import TopicSymboliser
import fastte, bench_gen

SEEDS, N_SUR, K = 3, 100, 8
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def prep(events, k=K, seed=0):
    """What a real analyst has: speaker, time, text. The detector symbolises the text itself."""
    ev = sorted(events, key=lambda e: e[1])
    texts = [e[3] for e in ev]
    sym = TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
    cnt = collections.Counter(e[0] for e in ev)
    agents = [a for a, n in cnt.most_common() if n >= 20][:15]
    keep = set(agents)
    stream = [(e[0], s) for e, s in zip(ev, sym) if e[0] in keep]
    n = len(stream)
    series = {a: [swarm_te.SILENT] * n for a in agents}
    for t, (sp, s) in enumerate(stream):
        series[sp][t] = s
    coded, A = fastte.code_series(series)
    alpha = sorted({x for v in series.values() for x in v})
    sil = alpha.index(swarm_te.SILENT)
    return coded, A, sil, [alpha.index(s) for _, s in stream]


def modal_ctx(chan, sil, window=12):
    out = np.zeros(len(chan), dtype=np.int64)
    for t in range(len(chan)):
        lo = max(0, t - window)
        c = [x for x in chan[lo:max(lo, t - 1)] if x != sil]
        out[t] = collections.Counter(c).most_common(1)[0][0] if c else sil
    return out


def run_pair(inst, seed):
    """Both estimators on the same instance, same surrogate, same symbolisation."""
    coded, A, sil, chan = prep(inst["events"], seed=seed)
    ctx = modal_ctx(chan, sil)
    pw = fastte.directed_tests_permute(coded, A, sil, n_perm=N_SUR, seed=seed)
    cd = fastte.directed_tests_conditional(coded, ctx, A, A, sil, n_perm=N_SUR, seed=seed)
    sel = lambda t: {k for k, v in t.items() if v["p"] < 0.05}
    return sel(pw), sel(cd)


FAMILIES = (
    ("pure_null", dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.0)),
    ("common_cause_null", dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.55)),
    ("structure", dict(n_agents=12, n_messages=8000, shared_topic=0.55,
                       edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4)))),
)

out = {"seeds": SEEDS, "n_surrogates": N_SUR, "k": K, "surrogate": "symbol permutation",
       "families": {}}

for fam_i, (fam, spec) in enumerate(FAMILIES):
    acc = {m: {"edges": [], "tp": []} for m in ("pairwise", "conditional")}
    realised, gts = [], []
    for s in range(SEEDS):
        inst = bench_gen.generate_instance(seed=900 + s + 1000 * fam_i, **spec)
        gt = inst["ground_truth"]
        gts.append(len(gt))
        realised.append(inst["realised_coupling"])
        pw, cd = run_pair(inst, s)
        for m, e in (("pairwise", pw), ("conditional", cd)):
            acc[m]["edges"].append(len(e))
            acc[m]["tp"].append(len(e & gt))
        log(f"{fam} s{s}: pairwise {len(pw)} edges ({len(pw & gt)} tp) | "
            f"conditional {len(cd)} edges ({len(cd & gt)} tp) | planted {len(gt)}")
    assert len(set(gts)) == 1, f"{fam}: planted count varies across seeds: {gts}"
    rec = {"planted": gts[0]}
    for m in acc:
        ed = float(np.mean(acc[m]["edges"])); tp = float(np.mean(acc[m]["tp"]))
        rec[m] = {"mean_edges": round(ed, 2), "mean_true_positives": round(tp, 2),
                  "fired_in": int(sum(1 for x in acc[m]["edges"] if x > 0)), "of": SEEDS,
                  # pooled over every instance, so staying silent cannot flatter the number
                  "pooled_precision": round(sum(acc[m]["tp"]) / sum(acc[m]["edges"]), 4)
                                      if sum(acc[m]["edges"]) else None,
                  "mean_recall": round(tp / gts[0], 4) if gts[0] else None}
    if gts[0]:
        rec["realised_coupling"] = realised
    out["families"][fam] = rec
    json.dump(out, open("bench_validation.json", "w", encoding="utf-8"), indent=2)

# power: does conditioning cost sensitivity? One planted edge, common cause switched on.
# Each coupling gets its OWN seed stream. Sharing `950 + s` across couplings made the vocabulary,
# the opening topic and every agent's home topic byte-identical at all five points, so "3 of 3
# everywhere" was partly three shared instances rather than five independent measurements.
rows = []
for i, c in enumerate((0.05, 0.1, 0.2, 0.4, 0.8)):
    acc = {m: {"det": 0, "edges": []} for m in ("pairwise", "conditional")}
    realised = []
    for s in range(SEEDS):
        inst = bench_gen.generate_instance(n_agents=12, n_messages=8000, shared_topic=0.55,
                                           edges=((0, 5, c),), seed=950 + s + 1000 * i)
        gt = inst["ground_truth"]
        realised.append(inst["realised_coupling"].get("0->5"))
        pw, cd = run_pair(inst, s)
        for m, e in (("pairwise", pw), ("conditional", cd)):
            acc[m]["det"] += len(e & gt) > 0
            acc[m]["edges"].append(len(e))
    row = {"coupling": c,
           "realised_hit_rate_mean": round(float(np.mean(realised)), 4),
           "realised_per_seed": realised,
           **{m: {"detected": acc[m]["det"], "of": SEEDS,
                  "mean_edges": round(float(np.mean(acc[m]["edges"])), 2)} for m in acc}}
    rows.append(row)
    log(f"coupling {c} (realised {row['realised_hit_rate_mean']}): "
        f"pairwise {acc['pairwise']['det']}/{SEEDS} @ {np.mean(acc['pairwise']['edges']):.1f} edges | "
        f"conditional {acc['conditional']['det']}/{SEEDS} @ {np.mean(acc['conditional']['edges']):.1f}")
out["power"] = rows
json.dump(out, open("bench_validation.json", "w", encoding="utf-8"), indent=2)
log("wrote bench_validation.json")
