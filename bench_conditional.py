"""Does conditioning on the channel fix the common-cause problem?

The benchmark's sharpest criticism of our own detector is precision, not calibration: on multi-edge
instances it recovered all 4 planted edges while certifying 17, because pairwise transfer entropy
cannot separate direct influence from a shared drifting channel topic. The benchmark can measure
that directly, because it has COMMON-CAUSE null instances: no influence at all, but everyone tracks
the channel.

This compares plain pairwise TE against conditional TE, both with the permutation surrogate, on
(a) common-cause nulls, where any edge is a false positive caused by shared context,
(b) pure nulls, to check conditioning does not break calibration,
(c) multi-edge structure instances, to check it does not destroy power.
"""
import sys, json, time, collections
sys.path.insert(0, r"C:\dev\swarm-prep")
sys.path.insert(0, r"C:\dev\swarm-sprint")
sys.path.insert(0, r"C:\dev\swarm-bench")
import numpy as np
import swarm_te
from swarm_load import TopicSymboliser
import fastte, bench_gen

K, N_SUR, SEEDS = 8, 100, 3
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def detect(events, conditional, k=K, n_sur=N_SUR, seed=0, min_msgs=20, top_k=15):
    ev = sorted(events, key=lambda e: e[1])
    texts = [e[3] for e in ev]
    sym = TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
    counts = collections.Counter(e[0] for e in ev)
    agents = [a for a, n in counts.most_common() if n >= min_msgs][:top_k]
    keep = set(agents)
    stream = [(e[0], s) for e, s in zip(ev, sym) if e[0] in keep]
    if len(agents) < 3 or len(stream) < 200:
        return set()
    n = len(stream)
    series = {a: [swarm_te.SILENT] * n for a in agents}
    for t, (sp, s) in enumerate(stream):
        series[sp][t] = s
    coded, A = fastte.code_series(series)
    alpha = sorted({x for v in series.values() for x in v})
    sil = alpha.index(swarm_te.SILENT)
    if conditional:
        chan_syms = [alpha.index(s) for _, s in stream]
        ctx = fastte.channel_context(chan_syms, sil, window=12, n_levels=3)
        tests = fastte.directed_tests_conditional(coded, ctx, A, 3, sil, n_perm=n_sur, seed=seed)
    else:
        tests = fastte.directed_tests_permute(coded, A, sil, n_perm=n_sur, seed=seed)
    return {e for e, v in tests.items() if v["p"] < 0.05}


FAMILIES = {
    "common_cause_null": dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.55),
    "pure_null":         dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.0),
    "structure":         dict(n_agents=12, n_messages=8000, shared_topic=0.55,
                              edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4))),
}

out = {"k": K, "n_surrogates": N_SUR, "seeds": SEEDS, "families": {}}
for fam, spec in FAMILIES.items():
    rec = {}
    for mode, cond in (("pairwise", False), ("conditional", True)):
        fired, found_n, tp_n, planted = 0, [], [], 0
        for s in range(SEEDS):
            inst = bench_gen.generate_instance(seed=900 + s, **spec)
            gt = inst["ground_truth"]; planted = len(gt)
            found = detect(inst["events"], conditional=cond, seed=s)
            fired += len(found) > 0
            found_n.append(len(found)); tp_n.append(len(found & gt))
        rec[mode] = {"fired_in": fired, "of": SEEDS,
                     "mean_edges": round(float(np.mean(found_n)), 2),
                     "mean_true_positives": round(float(np.mean(tp_n)), 2),
                     "planted": planted}
        log(f"{fam:20s} {mode:12s}: fired {fired}/{SEEDS} mean edges {np.mean(found_n):.2f} "
            f"tp {np.mean(tp_n):.2f}/{planted}")
    out["families"][fam] = rec
    json.dump(out, open("bench_conditional.json", "w", encoding="utf-8"), indent=2)
log("wrote bench_conditional.json")
