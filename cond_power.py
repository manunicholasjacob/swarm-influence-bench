"""Does conditioning cost power at weaker couplings?

Conditional TE removed every common-cause false edge while keeping all four planted edges, but those
were couplings of 0.3 to 0.5. A detector that only sees strong influence is less useful than it
looks, so this sweeps the coupling with a single planted edge and compares pairwise against
conditional, both with the permutation surrogate.
"""
import sys, time, json, collections, pathlib
# Paths resolved from this file, never hardcoded. These were absolute C:\dev paths, so the
# script could only ever run on one machine. The README already requires the three
# repositories to sit side by side, which is what this assumes.
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "swarm-prep", _HERE.parent / "swarm-sprint"):
    sys.path.insert(0, str(_p))
import numpy as np, swarm_te
from swarm_load import TopicSymboliser
import fastte, bench_gen

COUPLINGS = [0.05, 0.1, 0.2, 0.4, 0.8]
SEEDS, N_SUR = 3, 100
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def prep(events, k=8, seed=0):
    ev = sorted(events, key=lambda e: e[1]); texts = [e[3] for e in ev]
    sym = TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
    cnt = collections.Counter(e[0] for e in ev)
    agents = [a for a, n in cnt.most_common() if n >= 20][:15]; keep = set(agents)
    stream = [(e[0], s) for e, s in zip(ev, sym) if e[0] in keep]
    n = len(stream); series = {a: [swarm_te.SILENT] * n for a in agents}
    for t, (sp, s) in enumerate(stream):
        series[sp][t] = s
    coded, A = fastte.code_series(series)
    alpha = sorted({x for v in series.values() for x in v}); sil = alpha.index(swarm_te.SILENT)
    return coded, A, sil, [alpha.index(s) for _, s in stream]


def modal_ctx(chan, sil, window=12):
    out = np.zeros(len(chan), dtype=np.int64)
    for t in range(len(chan)):
        lo = max(0, t - window)
        c = [x for x in chan[lo:max(lo, t - 1)] if x != sil]
        out[t] = collections.Counter(c).most_common(1)[0][0] if c else sil
    return out


rows = []
for c in COUPLINGS:
    rec = {"coupling": c}
    for mode in ("pairwise", "conditional"):
        hits, edges = 0, []
        for s in range(SEEDS):
            inst = bench_gen.generate_instance(n_agents=12, n_messages=8000, shared_topic=0.55,
                                               edges=((0, 5, c),), seed=950 + s)
            gt = inst["ground_truth"]
            coded, A, sil, chan = prep(inst["events"], seed=s)
            if mode == "conditional":
                t_ = fastte.directed_tests_conditional(coded, modal_ctx(chan, sil), A, A, sil,
                                                       n_perm=N_SUR, seed=s)
            else:
                t_ = fastte.directed_tests_permute(coded, A, sil, n_perm=N_SUR, seed=s)
            e = {k for k, v in t_.items() if v["p"] < 0.05}
            hits += len(e & gt) > 0; edges.append(len(e))
        rec[mode] = {"detected": hits, "of": SEEDS, "mean_edges": round(float(np.mean(edges)), 2)}
        log(f"c={c} {mode:12s}: planted edge found {hits}/{SEEDS}, mean edges {np.mean(edges):.2f}")
    rows.append(rec)
    json.dump({"seeds": SEEDS, "n_surrogates": N_SUR, "rows": rows},
              open("cond_power.json", "w", encoding="utf-8"), indent=2)
log("wrote cond_power.json")
