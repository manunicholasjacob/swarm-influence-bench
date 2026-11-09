"""Evaluate a coordination detector against SwarmInfluence-Bench.

A detector receives only what a researcher really has: who spoke, when, and the message text. It must
symbolise the text itself, so it cannot read the generator's topic labels. It returns a set of
certified directed edges. The harness returns the three numbers any influence claim should carry:

  precision / recall   against the planted edge set
  family-wise error    the share of NULL instances where anything at all is certified
  detection floor      the coupling and channel volume at which power reaches 80%
"""
from __future__ import annotations
import sys, json, time, collections, math, warnings, pathlib
import numpy as np

# Resolved from this file, not hardcoded. These were absolute C:\dev paths, so the harness ran on
# exactly one machine; the README already requires the three repositories to sit side by side.
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (_HERE.parent / "swarm-prep", _HERE.parent / "swarm-sprint"):
    sys.path.insert(0, str(_p))
import swarm_te
from swarm_load import TopicSymboliser
import fastte
import bench_gen


def te_detector(events, null="permute", k=8, n_surrogates=200, seed=0, alpha=0.05,
                min_msgs=20, top_k=15):
    """The detector this project recommends: turn-ordered lag-1 TE with a chosen surrogate.

    Symbolises the raw text, builds the one-message-per-tick grid, and certifies edges with the
    max-T family-wise test. `null` selects the circular-shift surrogate or the permutation repair.

    Records which agents it actually tested in `te_detector.last_tested`. The family size matters
    for reading any family-wise error rate, and `top_k` silently drops agents past the cap: on a
    25-agent channel all 25 can clear `min_msgs` while only 15 are tested, so a planted edge may
    not be inside the tested set at all.
    """
    ev = sorted(events, key=lambda e: e[1])
    texts = [e[3] for e in ev]
    sym = TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
    counts = collections.Counter(e[0] for e in ev)
    eligible = [a for a, n in counts.most_common() if n >= min_msgs]
    agents = eligible[:top_k]
    te_detector.last_tested = {"eligible": len(eligible), "tested": len(agents),
                               "dropped_by_top_k": len(eligible) - len(agents),
                               "agents": list(agents),
                               "family_size": len(agents) * (len(agents) - 1)}
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
        try:
            inst = bench_gen.generate_instance(**spec)
        finally:
            spec["name"] = name
        found = detector(inst["events"])
        tested = getattr(te_detector, "last_tested", None)
        gt = inst["ground_truth"]
        tp = len(found & gt)
        rows.append({
            "name": name, "n_agents": inst["config"]["n_agents"],
            "shared_topic": inst["config"].get("shared_topic", 0.0),
            "n_agents_tested": tested["tested"] if tested else None,
            "family_size": tested["family_size"] if tested else None,
            "dropped_by_top_k": tested["dropped_by_top_k"] if tested else None,
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


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials, rounded to three places.

    Every error rate in this repo is a proportion from a few dozen or a few hundred replicates,
    where the normal approximation is wrong in the direction that flatters: it gives intervals that
    cross zero at small k and are too narrow near the ends. An 8-replicate sweep once reported a
    permutation-surrogate error rate of 0.375 that turned out to be noise, which is why nothing
    here reports a rate without one.

    Lived in stationarity_big.py; moved here so the sweeps share one implementation rather than
    each carrying a copy.
    """
    if n == 0:
        return (None, None)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, c - m), 3), round(min(1.0, c + m), 3))


def floor_of(curve, n_seeds):
    """The smallest swept value at which power reaches 80%, with an explicit status.

    Returning a bare number conflated four different situations, and the worst of them were
    silent: an empty curve and a curve that never reached 80% both came back None, and a
    non-monotone curve returned its first crossing as though power stayed above it. Power is
    also not estimable to 80% from a handful of seeds, since with n seeds the only reachable
    values are multiples of 1/n, so the status says so rather than implying a precision the
    grid cannot support.
    """
    if not curve:
        return {"floor": None, "status": "no_data"}
    items = sorted(curve.items())
    status_note = None
    if n_seeds < 10:
        status_note = f"coarse: {n_seeds} seeds resolve power only to 1/{n_seeds}"
    above = [k for k, v in items if v is not None and v >= 0.8]
    if not above:
        return {"floor": None, "status": "never_reached", "note": status_note}
    # monotone only if everything at or above the first crossing also passes
    first = above[0]
    tail = [v for k, v in items if k >= first]
    monotone = all(v is not None and v >= 0.8 for v in tail)
    if items[0][1] is not None and items[0][1] >= 0.8:
        return {"floor": items[0][0], "status": "below_grid", "note": status_note,
                "monotone": monotone}
    return {"floor": first, "status": "bracketed" if monotone else "non_monotone",
            "note": status_note, "monotone": monotone}


def summarise(rows, out_path=None):
    nulls = [r for r in rows if r["is_null"]]
    # classify on the generator's actual setting, not on a name prefix
    pure = [r for r in nulls if r.get("shared_topic", 0.0) == 0]
    common = [r for r in nulls if r.get("shared_topic", 0.0) > 0]
    unclassified = [r for r in nulls if "shared_topic" not in r]
    if unclassified:
        raise ValueError(f"{len(unclassified)} null rows carry no shared_topic and cannot be classified")
    signal = [r for r in rows if not r["is_null"]]
    fwer = float(np.mean([r["any_certified"] for r in pure])) if pure else None
    fwer_common = float(np.mean([r["any_certified"] for r in common])) if common else None
    edges_common = float(np.mean([r["edges_found"] for r in common])) if common else None
    prec = [r["precision"] for r in signal if r["precision"] is not None]
    rec = [r["recall"] for r in signal if r["recall"] is not None]
    # the per-instance mean above only averages instances that fired, which flatters a detector
    # that stays silent. The pooled value counts every instance.
    tp_sum = sum(r["true_positives"] for r in signal)
    found_sum = sum(r["edges_found"] for r in signal)
    fired_rate = float(np.mean([r["any_certified"] for r in signal])) if signal else None
    # Select the power curves on the generator's actual settings, not on a name prefix. Keying off
    # `name.startswith("coupling")` meant anyone supplying their own instance list with their own
    # names got an empty curve and a None floor, indistinguishable from "power never reached 80%".
    single = [r for r in signal if len(r["couplings"]) == 1 and r["recall"] is not None]
    volumes = collections.Counter(r["n_messages"] for r in single)
    fixed_volume = volumes.most_common(1)[0][0] if volumes else None
    couplings = collections.Counter(r["couplings"][0] for r in single)
    fixed_coupling = couplings.most_common(1)[0][0] if couplings else None
    by_coupling, by_volume = collections.defaultdict(list), collections.defaultdict(list)
    for r in single:
        # realised coupling is what the stream contains; the requested value is only a label
        if r["n_messages"] == fixed_volume:
            by_coupling[r["couplings"][0]].append(r["recall"])
        if r["couplings"][0] == fixed_coupling:
            by_volume[r["n_messages"]].append(r["recall"])
    power_c = {k: round(float(np.mean(v)), 3) for k, v in sorted(by_coupling.items())}
    power_v = {k: round(float(np.mean(v)), 3) for k, v in sorted(by_volume.items())}
    if signal and not power_c:
        warnings.warn("no single-edge instances at a common channel volume: "
                      "power_by_coupling is empty, which is not the same as zero power")
    n_seeds_c = max((len(v) for v in by_coupling.values()), default=0)
    n_seeds_v = max((len(v) for v in by_volume.values()), default=0)
    floor_c = floor_of(power_c, n_seeds_c)
    floor_v = floor_of(power_v, n_seeds_v)
    summary = {
        "n_instances": len(rows), "n_null": len(nulls), "n_signal": len(signal),
        "familywise_error_pure_null": round(fwer, 4) if fwer is not None else None,
        "familywise_error_common_cause_null": round(fwer_common, 4) if fwer_common is not None else None,
        "mean_edges_on_common_cause_null": round(edges_common, 2) if edges_common is not None else None,
        "mean_precision_when_fired": round(float(np.mean(prec)), 4) if prec else None,
        "pooled_precision": round(tp_sum / found_sum, 4) if found_sum else None,
        "fired_rate_on_signal": round(fired_rate, 4) if fired_rate is not None else None,
        "mean_recall": round(float(np.mean(rec)), 4) if rec else None,
        "power_by_coupling": power_c, "power_by_channel_volume": power_v,
        "detection_floor_coupling": floor_c, "detection_floor_volume": floor_v,
    }
    if out_path:
        json.dump({"summary": summary, "rows": rows}, open(out_path, "w", encoding="utf-8"), indent=2)
    return summary
