"""Does any label-free statistic place a symboliser on the right side of the symbolisation floor?

`realism_sweep.py` sweeps `vocab_overlap` and reports, at each level, the true adjusted Rand index
between symboliser output and the generator's topics, together with whether conditional detection
still works. It finds the floor: detection holds down to ARI about 0.44 and collapses below it.
The number it needs and cannot have for the real channels is that ARI, because ARI needs labels.

This script runs the same harness over more levels, and at every configuration computes BOTH the
true ARI and every label-free proxy in `floor_proxy.py`. That makes the proxies measurable against
a known answer, and it makes the real question answerable: is there a cut on some proxy that sorts
above-floor from below-floor configurations?

The four levels `realism_sweep.py` used are included, with the same instance seeds and the same
symboliser seeds, so its ARI and its detection counts are reproduced rather than assumed. Five
further levels are added, three of them between 0.8 and 0.95 where the collapse happens.

Writes floor_proxy_sweep.json.

    python floor_proxy_sweep.py [--quick]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

import floor_proxy as fx          # also puts the sibling swarm checkouts on sys.path

import bench_gen                  # noqa: E402
import fastte                     # noqa: E402
import swarm_te                   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

K = 8
N_SUR = 100
SEEDS = 3
# the four levels realism_sweep.py used, plus five more, weighted towards the collapse region
OVERLAPS = (0.0, 0.3, 0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95)
STATED_FLOOR = 0.44        # the ARI the report names as the floor

# Same two families, same specs, as realism_sweep.py.
FAMILIES = (
    ("common_cause_null", dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.55)),
    ("structure", dict(n_agents=12, n_messages=8000, shared_topic=0.55,
                       edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4)))),
)

t0 = time.time()


def log(m):
    print("[%5.0fs] %s" % (time.time() - t0, m), flush=True)


# `turn_grid` and `modal_ctx` are the same code as realism_sweep.prep and realism_sweep.modal_ctx.
# They are duplicated rather than imported because that module runs its whole sweep at import time.
# `turn_grid` takes the symbols as an argument instead of fitting its own symboliser, so detection
# and the proxies are guaranteed to be measured on one and the same symbolisation.
def turn_grid(events_sorted, symbols):
    """Turn-ordered integer series over the agents that speak enough to be testable."""
    cnt = collections.Counter(e[0] for e in events_sorted)
    agents = [a for a, n in cnt.most_common() if n >= 20][:15]
    keep = set(agents)
    stream = [(e[0], s) for e, s in zip(events_sorted, symbols) if e[0] in keep]
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


def detect(events_sorted, symbols, gt, n_sur, seed):
    """Pairwise and conditional edge sets at alpha 0.05, exactly as realism_sweep.py runs them."""
    coded, A, sil, chan = turn_grid(events_sorted, symbols)
    ctx = modal_ctx(chan, sil)
    pw = {k for k, v in fastte.directed_tests_permute(
        coded, A, sil, n_perm=n_sur, seed=seed).items() if v["p"] < 0.05}
    cd = {k for k, v in fastte.directed_tests_conditional(
        coded, ctx, A, A, sil, n_perm=n_sur, seed=seed).items() if v["p"] < 0.05}
    out = {"n_ticks": len(chan), "planted": len(gt)}
    for name, e in (("pairwise", pw), ("conditional", cd)):
        tp = len(e & gt)
        out[name] = {
            "edges": len(e), "true_positives": tp,
            "precision": round(tp / len(e), 4) if e else None,
            "recall": round(tp / len(gt), 4) if gt else None,
        }
    return out


def _vals(rows, key):
    return [r["proxies"].get(key) for r in rows]


def _summary(xs):
    x = np.asarray([v for v in xs if v is not None and np.isfinite(v)], dtype=float)
    if not x.size:
        return None
    return {"mean": round(float(x.mean()), 4), "min": round(float(x.min()), 4),
            "max": round(float(x.max()), 4),
            "sd": round(float(x.std(ddof=1)), 4) if x.size > 1 else 0.0, "n": int(x.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small instances and few resamples, for a wiring check only")
    ap.add_argument("--out", default="floor_proxy_sweep.json")
    args = ap.parse_args()

    n_msg = 1500 if args.quick else 8000
    n_boot = 3 if args.quick else 6
    seed_runs = 3 if args.quick else 5
    n_sur = 20 if args.quick else N_SUR
    overlaps = (0.0, 0.8, 0.9) if args.quick else OVERLAPS

    rows = []
    for ov in overlaps:
        for fam_i, (fam, spec) in enumerate(FAMILIES):
            for s in range(SEEDS):
                # the instance seed and the symboliser seed realism_sweep.py used at this cell
                inst_seed = 900 + s + 1000 * fam_i
                kw = dict(spec)
                kw["n_messages"] = n_msg
                inst = bench_gen.generate_instance(seed=inst_seed, vocab_overlap=ov, **kw)
                ev = sorted(inst["events"], key=lambda e: e[1])
                texts = [e[3] for e in ev]
                truth = [e[2] for e in ev]

                res = fx.all_proxies(texts, k=K, seed=s, n_boot=n_boot, seed_runs=seed_runs)
                sym = res["base_symbols"]
                true_ari = float(adjusted_rand_score(truth, fx.codes(sym)))
                det = detect(ev, sym, inst["ground_truth"], n_sur, s)

                row = {"vocab_overlap": ov, "family": fam, "instance_seed": inst_seed,
                       "symboliser_seed": s, "true_ari": round(true_ari, 4),
                       "proxies": {k: (None if v is None else round(float(v), 6))
                                   for k, v in res["proxies"].items()},
                       "diagnostics": res["diagnostics"], "detection": det}
                rows.append(row)
                log("ov %.2f %-18s seed %d: true ARI %.3f | boot %.3f char %.3f lsa %.3f "
                    "cos %.3f | cond edges %d rec %s"
                    % (ov, fam, s, true_ari,
                       res["proxies"]["boot_stability_full_mean"],
                       res["proxies"]["agree_char"], res["proxies"]["agree_lsa"],
                       res["proxies"]["centroid_cosine_mean"],
                       det["conditional"]["edges"], det["conditional"]["recall"]))

    out = {
        "question": ("whether a statistic computable without ground-truth labels tracks the true "
                     "adjusted Rand index well enough to tell an above-floor symboliser from a "
                     "below-floor one"),
        "k": K, "seeds_per_cell": SEEDS, "n_surrogates": n_sur, "n_messages": n_msg,
        "overlaps": list(overlaps), "stated_floor_ari": STATED_FLOOR,
        "proxy_direction": {k: ("higher_is_better" if v else "lower_is_better")
                            for k, v in fx.HIGHER_IS_BETTER.items()},
        "rows": rows,
    }

    # ---- per-overlap aggregate, including a reproduction check against realism_sweep.json
    per_overlap = []
    for ov in overlaps:
        sub = [r for r in rows if r["vocab_overlap"] == ov]
        rec = {"vocab_overlap": ov,
               "true_ari": _summary([r["true_ari"] for r in sub]),
               "proxies": {k: _summary(_vals(sub, k)) for k in fx.PROXY_NAMES}}
        for fam, _ in FAMILIES:
            fs = [r for r in sub if r["family"] == fam]
            rec.setdefault("detection", {})[fam] = {
                "pairwise_mean_edges": round(float(np.mean(
                    [r["detection"]["pairwise"]["edges"] for r in fs])), 2),
                "conditional_mean_edges": round(float(np.mean(
                    [r["detection"]["conditional"]["edges"] for r in fs])), 2),
                "conditional_mean_recall": (None if fs[0]["detection"]["planted"] == 0 else round(
                    float(np.mean([r["detection"]["conditional"]["recall"] for r in fs])), 4)),
                "conditional_pooled_precision": (
                    round(sum(r["detection"]["conditional"]["true_positives"] for r in fs)
                          / max(1, sum(r["detection"]["conditional"]["edges"] for r in fs)), 4)
                    if sum(r["detection"]["conditional"]["edges"] for r in fs) else None),
            }
        per_overlap.append(rec)
    out["per_overlap"] = per_overlap

    ref = HERE / "realism_sweep.json"
    if ref.exists():
        prev = {r["vocab_overlap"]: r["symboliser_ari_mean"]
                for r in json.loads(ref.read_text(encoding="utf-8"))["rows"]}
        check = []
        for rec in per_overlap:
            ov = rec["vocab_overlap"]
            if ov in prev:
                check.append({"vocab_overlap": ov, "realism_sweep_ari_mean": prev[ov],
                              "here_ari_mean": rec["true_ari"]["mean"],
                              "abs_diff": round(abs(prev[ov] - rec["true_ari"]["mean"]), 4)})
        out["reproduction_check"] = {
            "note": ("same instance seeds, same symboliser seeds, same k. A non-zero difference "
                     "would mean this harness is not measuring the configurations whose detection "
                     "behaviour is already known."),
            "max_abs_diff": max([c["abs_diff"] for c in check], default=None),
            "levels": check,
        }

    # ---- correlation and, more to the point, floor separation
    true = [r["true_ari"] for r in rows]
    above_ari = [r["true_ari"] >= STATED_FLOOR for r in rows]
    # the operational truth: on the structure family, did conditional detection keep every edge?
    struct = [r for r in rows if r["family"] == "structure"]
    held = [r["detection"]["conditional"]["recall"] == 1.0 for r in struct]

    ev = {}
    for name in fx.PROXY_NAMES:
        hib = fx.HIGHER_IS_BETTER[name]
        v = _vals(rows, name)
        rec = {"spearman_vs_true_ari": (lambda x: None if x is None else round(x, 4))(
            fx.spearman(v, true)), "direction": "higher_is_better" if hib else "lower_is_better"}
        rec["floor_by_stated_ari"] = fx.best_threshold(v, above_ari, hib)
        rec["floor_by_detection_held"] = fx.best_threshold(
            _vals(struct, name), held, hib)
        # leave-one-overlap-out: the cut is chosen on the other levels and judged on the held-out
        # one, so a threshold that only works because it was fitted on the same rows is exposed.
        agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "folds": []}
        for ov in overlaps:
            tr = [i for i, r in enumerate(rows) if r["vocab_overlap"] != ov]
            te = [i for i, r in enumerate(rows) if r["vocab_overlap"] == ov]
            cut = fx.best_threshold([v[i] for i in tr], [above_ari[i] for i in tr], hib)
            if cut is None:
                continue
            c = fx.confusion([v[i] for i in te], [above_ari[i] for i in te], cut["threshold"], hib)
            for key in ("tp", "fp", "fn", "tn"):
                agg[key] += c[key]
            agg["folds"].append({"held_out_overlap": ov, "threshold": cut["threshold"],
                                 "errors": c["errors"], "n": c["n"]})
        tot = agg["tp"] + agg["fp"] + agg["fn"] + agg["tn"]
        agg["errors"] = agg["fp"] + agg["fn"]
        agg["accuracy"] = round((agg["tp"] + agg["tn"]) / tot, 4) if tot else None
        sens = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else None
        spec = agg["tn"] / (agg["tn"] + agg["fp"]) if agg["tn"] + agg["fp"] else None
        agg["balanced_accuracy"] = (round((sens + spec) / 2.0, 4)
                                    if sens is not None and spec is not None else None)
        rec["floor_leave_one_overlap_out"] = agg
        ev[name] = rec
    out["evaluation"] = ev

    ranked = sorted((n for n in ev if ev[n]["floor_by_stated_ari"]),
                    key=lambda n: (-(ev[n]["floor_by_stated_ari"]["balanced_accuracy"] or 0),
                                   ev[n]["floor_by_stated_ari"]["errors"]))
    out["ranking_by_floor_separation"] = [
        {"proxy": n,
         "balanced_accuracy": ev[n]["floor_by_stated_ari"]["balanced_accuracy"],
         "errors": ev[n]["floor_by_stated_ari"]["errors"],
         "threshold": ev[n]["floor_by_stated_ari"]["threshold"],
         "direction": ev[n]["direction"],
         "spearman": ev[n]["spearman_vs_true_ari"],
         "loo_errors": ev[n]["floor_leave_one_overlap_out"]["errors"],
         "loo_balanced_accuracy": ev[n]["floor_leave_one_overlap_out"]["balanced_accuracy"]}
        for n in ranked]

    (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote %s" % args.out)
    for r in out["ranking_by_floor_separation"][:6]:
        log("  %-28s bal acc %-7s errors %-3s loo errors %-3s spearman %s"
            % (r["proxy"], r["balanced_accuracy"], r["errors"], r["loo_errors"], r["spearman"]))
    if "reproduction_check" in out:
        log("reproduction of realism_sweep ARI: max abs diff %s"
            % out["reproduction_check"]["max_abs_diff"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
