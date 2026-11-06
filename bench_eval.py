"""Evaluate a coordination detector against SwarmInfluence-Bench.

A detector receives only what a researcher really has: who spoke, when, and the message text. It must
symbolise the text itself, so it cannot read the generator's topic labels. It returns a set of
certified directed edges. The harness returns the three numbers any influence claim should carry:

  precision / recall   against the planted edge set
  family-wise error    the share of NULL instances where anything at all is certified
  detection floor      the coupling and channel volume at which power reaches 80%
"""
from __future__ import annotations
import sys, json, time, collections
import numpy as np

sys.path.insert(0, r"C:\dev\swarm-prep")
sys.path.insert(0, r"C:\dev\swarm-sprint")
import swarm_te
from swarm_load import TopicSymboliser
import fastte
import bench_gen


def te_detector(events, null="permute", k=8, n_surrogates=200, seed=0, alpha=0.05,
                min_msgs=20, top_k=15):
    """The detector this project recommends: turn-ordered lag-1 TE with a chosen surrogate.

    Symbolises the raw text, builds the one-message-per-tick grid, and certifies edges with the
    max-T family-wise test. `null` selects the circular-shift surrogate or the permutation repair.
    """
    ev = sorted(events, key=lambda e: e[1])
    texts = [e[3] for e in ev]
    sym = TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
    counts = collections.Counter(e[0] for e in ev)
    agents = [a for a, n in counts.most_common() if n >= min_msgs][:top_k]
    keep = set(agents)
    stream = [(e[0], s) for e, s in zip(ev, sym) if e[0] in keep]
    if len(agents) < 3 or len(stream) < 100:
        return set()
    n = len(stream)
    series = {a: [swarm_te.SILENT] * n for a in agents}
    for t, (sp, s) in enumerate(stream):
        series[sp][t] = s
    coded, A = fastte.code_series(series)
    if null == "permute":
        alpha_syms = sorted({x for v in series.values() for x in v})
        tests = fastte.directed_tests_permute(
            coded, A, alpha_syms.index(swarm_te.SILENT), n_perm=n_surrogates, seed=seed)
    else:
        tests = fastte.directed_tests_fast(coded, A, n_shifts=n_surrogates, seed=seed)
    return {e for e, v in tests.items() if v["p"] < alpha}


def evaluate(detector, instances=None, out_path=None, log=print):
    """Run a detector over the grid and return the three numbers."""
    specs = instances if instances is not None else bench_gen.grid()
    rows = []
    t0 = time.time()
    for i, spec in enumerate(specs):
        name = spec.pop("name")
        inst = bench_gen.generate_instance(**spec)
        spec["name"] = name
        found = detector(inst["events"])
        gt = inst["ground_truth"]
        tp = len(found & gt)
        rows.append({
            "name": name, "n_agents": inst["config"]["n_agents"],
            "n_messages": inst["config"]["n_messages"],
            "edges_planted": len(gt), "edges_found": len(found), "true_positives": tp,
            "precision": round(tp / len(found), 4) if found else None,
            "recall": round(tp / len(gt), 4) if gt else None,
            "is_null": len(gt) == 0, "any_certified": len(found) > 0,
            "couplings": [c for *_, c in inst["config"]["edges"]],
        })
        if out_path and (i + 1) % 5 == 0:
            json.dump({"rows": rows}, open(out_path, "w", encoding="utf-8"), indent=2)
        log(f"[{time.time()-t0:5.0f}s] {i+1}/{len(specs)} {name}: planted {len(gt)} "
            f"found {len(found)} tp {tp}")
    return summarise(rows, out_path)


def summarise(rows, out_path=None):
    nulls = [r for r in rows if r["is_null"]]
    pure = [r for r in nulls if r["name"].startswith("nullpure")]
    common = [r for r in nulls if r["name"].startswith("nullcommon")]
    signal = [r for r in rows if not r["is_null"]]
    fwer = float(np.mean([r["any_certified"] for r in pure])) if pure else None
    fwer_common = float(np.mean([r["any_certified"] for r in common])) if common else None
    edges_common = float(np.mean([r["edges_found"] for r in common])) if common else None
    prec = [r["precision"] for r in signal if r["precision"] is not None]
    rec = [r["recall"] for r in signal if r["recall"] is not None]
    by_coupling = collections.defaultdict(list)
    for r in signal:
        if r["name"].startswith("coupling"):
            by_coupling[r["couplings"][0]].append(r["recall"])
    by_volume = collections.defaultdict(list)
    for r in signal:
        if r["name"].startswith("volume"):
            by_volume[r["n_messages"]].append(r["recall"])
    power_c = {k: round(float(np.mean(v)), 3) for k, v in sorted(by_coupling.items())}
    power_v = {k: round(float(np.mean(v)), 3) for k, v in sorted(by_volume.items())}
    floor_c = next((k for k, v in sorted(power_c.items()) if v >= 0.8), None)
    floor_v = next((k for k, v in sorted(power_v.items()) if v >= 0.8), None)
    summary = {
        "n_instances": len(rows), "n_null": len(nulls), "n_signal": len(signal),
        "familywise_error_pure_null": round(fwer, 4) if fwer is not None else None,
        "familywise_error_common_cause_null": round(fwer_common, 4) if fwer_common is not None else None,
        "mean_edges_on_common_cause_null": round(edges_common, 2) if edges_common is not None else None,
        "mean_precision": round(float(np.mean(prec)), 4) if prec else None,
        "mean_recall": round(float(np.mean(rec)), 4) if rec else None,
        "power_by_coupling": power_c, "power_by_channel_volume": power_v,
        "detection_floor_coupling": floor_c, "detection_floor_volume": floor_v,
    }
    if out_path:
        json.dump({"summary": summary, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=2)
    return summary
