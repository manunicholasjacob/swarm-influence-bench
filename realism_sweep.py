"""Do the benchmark's conclusions survive imperfect symbolisation?

The benchmark's text layer is synthetic, and the standing objection to that is fair: with disjoint
per-topic vocabularies a topic model recovers the generator's topic almost exactly, so every number
measured on it assumes a symboliser that does not exist. Real messages are not like that. Most words
are common to every subject and only a few are distinctive, so a real symboliser is lossy.

An LLM renderer is one way to get realistic text. This is the cheaper and more controlled way: sweep
`vocab_overlap`, the fraction of each topic's vocabulary shared with every other topic, and watch
both how much the symboliser degrades (adjusted Rand index against the true labels) and whether the
pairwise-versus-conditional conclusion still holds. A conclusion that survives a symboliser this
degraded does not depend on the text layer being easy.

Writes realism_sweep.json.
"""
import collections
import json
import pathlib
import sys
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, r"C:\dev\swarm-prep")
sys.path.insert(0, r"C:\dev\swarm-sprint")
sys.path.insert(0, str(HERE))
import swarm_te
from swarm_load import TopicSymboliser
import fastte
import bench_gen

SEEDS, N_SUR, K = 3, 100, 8
OVERLAPS = (0.0, 0.5, 0.8, 0.9)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def prep(events, seed):
    """Symbolise, build the turn-ordered grid, and report how well symbolisation recovered truth."""
    ev = sorted(events, key=lambda e: e[1])
    texts = [e[3] for e in ev]
    true_topics = [e[2] for e in ev]
    sym = TopicSymboliser(k=K, seed=seed).fit(texts).transform(texts)
    ari = float(adjusted_rand_score(true_topics, list(sym)))
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
    return coded, A, sil, [alpha.index(s) for _, s in stream], ari


def modal_ctx(chan, sil, window=12):
    out = np.zeros(len(chan), dtype=np.int64)
    for t in range(len(chan)):
        lo = max(0, t - window)
        c = [x for x in chan[lo:max(lo, t - 1)] if x != sil]
        out[t] = collections.Counter(c).most_common(1)[0][0] if c else sil
    return out


FAMILIES = (
    ("common_cause_null", dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.55)),
    ("structure", dict(n_agents=12, n_messages=8000, shared_topic=0.55,
                       edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4)))),
)

out = {"seeds": SEEDS, "n_surrogates": N_SUR, "k": K, "overlaps": list(OVERLAPS), "rows": []}

for ov in OVERLAPS:
    rec = {"vocab_overlap": ov, "families": {}}
    aris = []
    for fam_i, (fam, spec) in enumerate(FAMILIES):
        acc = {m: {"edges": [], "tp": []} for m in ("pairwise", "conditional")}
        for s in range(SEEDS):
            inst = bench_gen.generate_instance(seed=900 + s + 1000 * fam_i,
                                               vocab_overlap=ov, **spec)
            gt = inst["ground_truth"]
            coded, A, sil, chan, ari = prep(inst["events"], s)
            aris.append(ari)
            ctx = modal_ctx(chan, sil)
            pw = {k for k, v in fastte.directed_tests_permute(
                coded, A, sil, n_perm=N_SUR, seed=s).items() if v["p"] < 0.05}
            cd = {k for k, v in fastte.directed_tests_conditional(
                coded, ctx, A, A, sil, n_perm=N_SUR, seed=s).items() if v["p"] < 0.05}
            for m, e in (("pairwise", pw), ("conditional", cd)):
                acc[m]["edges"].append(len(e))
                acc[m]["tp"].append(len(e & gt))
        f = {"planted": len(gt)}
        for m in acc:
            tot_e, tot_tp = sum(acc[m]["edges"]), sum(acc[m]["tp"])
            f[m] = {"mean_edges": round(float(np.mean(acc[m]["edges"])), 2),
                    "mean_true_positives": round(float(np.mean(acc[m]["tp"])), 2),
                    "pooled_precision": round(tot_tp / tot_e, 4) if tot_e else None,
                    "mean_recall": round(tot_tp / (SEEDS * len(gt)), 4) if gt else None}
        rec["families"][fam] = f
        log(f"overlap {ov} {fam}: pairwise {f['pairwise']['mean_edges']} edges "
            f"(prec {f['pairwise']['pooled_precision']}, rec {f['pairwise']['mean_recall']}) | "
            f"conditional {f['conditional']['mean_edges']} "
            f"(prec {f['conditional']['pooled_precision']}, rec {f['conditional']['mean_recall']})")
    rec["symboliser_ari_mean"] = round(float(np.mean(aris)), 4)
    rec["symboliser_ari_min"] = round(float(np.min(aris)), 4)
    log(f"overlap {ov}: symboliser ARI {rec['symboliser_ari_mean']:.3f} "
        f"(min {rec['symboliser_ari_min']:.3f})")
    out["rows"].append(rec)
    json.dump(out, open(HERE / "realism_sweep.json", "w", encoding="utf-8"), indent=2)

log("wrote realism_sweep.json")
