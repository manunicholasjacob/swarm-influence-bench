"""Turn-taking or non-stationary activity: which one breaks the circular-shift surrogate?

The audit measured a family-wise error rate of 0.90 for the circular-shift surrogate on the real
turn-ordered channel, and localised the cause in the speaker sequence: keep the real speaker order
and shuffle only the symbols and it fires 10 of 10; randomise the speaker order and it fires 0 of
10. That ablation could not go further, because the real speaker sequence carries two properties at
once and shuffling it destroys both. The same agent speaks twice in a row 36% of the time and
returns after exactly one other message 38% of the time, so speakers are not independent draws;
and each agent's share of messages rises and falls over time, median block-share coefficient of
variation 0.79. The report said plainly that it could not separate the two.

It could not because the generator drew speakers independently: `activity_drift` varied
non-stationarity alone and there was no knob for turn-taking at all, which is why the earlier sweep
(bench_stationarity_big.json) could only take the shift surrogate from 0.067 at drift 0.0 to 0.367
at drift 0.9, well short of 0.90. `bench_gen.generate_instance` now has `p_repeat` and `p_return`,
a speaker chain that is independent of `activity_drift` and reproduces the old generator bit for bit
when both are zero, so the two can be set one at a time.

This runs the factorial. Pure nulls only: no planted edges and no drifting common cause, so every
certified edge is a false positive by construction. Two arms:

  factorial   3 turn-taking levels (off, half, nominal real) x 3 drift levels (0.0, 0.45, 0.9),
              at the nominal knob values the question asks for.
  calibrated  a 2 x 2 at knob values tuned so the GENERATED channel reproduces the real channel's
              three measured statistics (repeat 0.36, return-after-one 0.38, median block-share CV
              0.79) rather than merely carrying those numbers as settings. No realised rate equals
              its knob: the activity draw lands on the previous speaker about a quarter of the time
              on its own, and drift raises the repeat rate further without any chain at all. The
              calibrated arm is therefore the one that answers "at real-data values".

Every cell reports both surrogates, the realised speaker statistics, and 95% Wilson intervals.
Replicate rows are kept in the output so the rates and intervals can be recomputed from them.

The (off, 0.0) and (off, 0.9) cells reuse the generator seeds and detector seeds of
stationarity_big.py, so their first 30 replicates are the same 30 measurements
bench_stationarity_big.json reports. The comparison is recomputed and written out, counts and
intervals both, rather than asserted. It is a check on the environment, not on the knob: that the
knob at zero leaves the generator byte-identical is established directly, against the committed
generator, over the whole grid and a spread of random configurations.

    python turntaking_factorial.py --workers 14

Writes turntaking_factorial.json. `--workers` only changes how the replicates are spread over
cores; every replicate is determined by its cell and its index, so the output does not depend on it.
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import sys
import time

# One BLAS/OpenMP thread per process. The symboliser's k-means is the only threaded step, and with
# several workers the default pools oversubscribe every core many times over. Checked first that
# pinning it to one thread leaves the symbol sequence byte-identical, so this is a scheduling
# decision and not a change to any measured number.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_eval
import bench_gen

# The real channel's own numbers, carried here so the table can be read against them.
REAL = {"repeat_rate": 0.36, "return_after_one_rate": 0.38, "median_block_share_cv": 0.79,
        "shift_fwer": 0.90, "permute_fwer": 0.063,
        "source": "AI Village audit, turn-ordered agent chat; permutation figure is 300 replicates"}

# Nominal arm: the knob values the question specifies. Half of the real-data pair is the moderate
# level on each axis, so the two knobs are swept off / half / full and the interaction is readable.
TURN_LEVELS = (("off", 0.0, 0.0), ("moderate", 0.18, 0.19), ("real_nominal", 0.36, 0.38))
DRIFT_LEVELS = (0.0, 0.45, 0.9)

# Calibrated arm: found by a coarse-to-fine grid search over (p_repeat, p_return, activity_drift)
# minimising relative error against the real channel's three statistics. At these settings the
# generated channel measures repeat 0.359, return-after-one 0.382, median block-share CV 0.807,
# against targets of 0.36, 0.38 and 0.79. The off-diagonal cells switch one of the two off, and the
# fourth corner of the 2 x 2 is the factorial's own (off, 0.0) cell.
CAL_REPEAT, CAL_RETURN, CAL_DRIFT = 0.22, 0.54, 0.27

# Held fixed at the earlier sweep's values so the two are comparable.
BASE_SEED = 7000
GEN = dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.0)

# bench_stationarity_big.json, the rows this run must reproduce over its first 30 replicates.
PUBLISHED = {("off", 0.0): {"shift": 2, "permute": 1}, ("off", 0.9): {"shift": 11, "permute": 3}}


def cells():
    """The cell list, in output order. Each cell is one (turn-taking, drift) configuration."""
    out = []
    for name, pr, pt in TURN_LEVELS:
        for d in DRIFT_LEVELS:
            out.append({"arm": "factorial", "turn_taking": name, "p_repeat": pr, "p_return": pt,
                        "activity_drift": d})
    out.append({"arm": "calibrated", "turn_taking": "off", "p_repeat": 0.0, "p_return": 0.0,
                "activity_drift": CAL_DRIFT})
    out.append({"arm": "calibrated", "turn_taking": "real_calibrated",
                "p_repeat": CAL_REPEAT, "p_return": CAL_RETURN, "activity_drift": 0.0})
    out.append({"arm": "calibrated", "turn_taking": "real_calibrated",
                "p_repeat": CAL_REPEAT, "p_return": CAL_RETURN, "activity_drift": CAL_DRIFT})
    return out


def speaker_stats(events, n_blocks=20):
    """The two turn-taking rates and the non-stationarity statistic, as the audit defined them.

    `return_after_one` requires the intervening message to come from a DIFFERENT agent, which is
    what "returns after exactly one other message" means and what makes it exclusive of a repeat.
    `return_any` drops that condition and is reported too, because the looser definition is the
    easy one to write by accident and the two differ a lot once repeats are common.
    """
    a = np.asarray([e[0] for e in events])
    rep = float((a[1:] == a[:-1]).mean()) if a.size > 1 else 0.0
    if a.size > 2:
        same2 = a[2:] == a[:-2]
        other = a[2:] != a[1:-1]
        ret = float((same2 & other).mean())
        ret_any = float(same2.mean())
    else:
        ret = ret_any = 0.0
    # Never more blocks than messages: np.array_split happily returns empty blocks, whose share is
    # a nan that then poisons the median.
    blocks = np.array_split(a, max(1, min(n_blocks, a.size)))
    cvs = []
    for who in np.unique(a):
        share = np.array([float((b == who).mean()) for b in blocks])
        if share.mean() > 0:
            cvs.append(float(share.std() / share.mean()))
    return {"repeat_rate": rep, "return_after_one_rate": ret, "return_any_rate": ret_any,
            "median_block_share_cv": float(np.median(cvs)) if cvs else 0.0,
            "n_speakers": int(np.unique(a).size)}


def run_replicate(job):
    """One replicate: generate a pure null, then test it with both surrogates.

    Pure function of (cell index, replicate index, surrogate count), which is what lets the
    replicates be spread over processes without the result depending on how they were spread.
    """
    ci, rep, n_sur = job
    cell = cells()[ci]
    inst = bench_gen.generate_instance(
        activity_drift=cell["activity_drift"], p_repeat=cell["p_repeat"],
        p_return=cell["p_return"], seed=BASE_SEED + rep, **GEN)
    assert not inst["ground_truth"], "a cell of this experiment is not a null"
    row = {"cell": ci, "rep": rep}
    row.update(speaker_stats(inst["events"]))
    for null in ("shift", "permute"):
        found = bench_eval.te_detector(inst["events"], null=null, n_surrogates=n_sur, seed=rep)
        row[null + "_edges"] = len(found)
        row[null + "_fired"] = bool(found)
    tested = bench_eval.te_detector.last_tested
    row["agents_tested"] = tested["tested"]
    row["family_size"] = tested["family_size"]
    return row


def summarise(rows):
    """Per-cell rates with Wilson intervals, from the replicate rows and nothing else.

    Takes only the rows, so a verifier can read the `rows` list back out of the JSON and rebuild
    every published rate and interval without rerunning a single detector.
    """
    out = []
    for ci, cell in enumerate(cells()):
        mine = sorted((r for r in rows if r["cell"] == ci), key=lambda r: r["rep"])
        rec = dict(cell)
        rec["cell"] = ci
        rec["n_replicates"] = len(mine)
        for null in ("shift", "permute"):
            fired = sum(r[null + "_fired"] for r in mine)
            lo, hi = bench_eval.wilson(fired, len(mine))
            rec[null] = {"fired": fired, "of": len(mine),
                         "fwer": round(fired / len(mine), 4) if mine else None,
                         "wilson95": [lo, hi],
                         "mean_edges": round(float(np.mean([r[null + "_edges"] for r in mine])), 3)
                         if mine else None}
        rec["realised"] = {k: round(float(np.mean([r[k] for r in mine])), 4)
                           for k in ("repeat_rate", "return_after_one_rate", "return_any_rate",
                                     "median_block_share_cv")} if mine else None
        if mine:
            rec["realised"]["mean_agents_tested"] = round(
                float(np.mean([r["agents_tested"] for r in mine])), 2)
            rec["realised"]["mean_family_size"] = round(
                float(np.mean([r["family_size"] for r in mine])), 1)
        out.append(rec)
    return out


def _find(cs, arm, turn, drift):
    for c in cs:
        if c["arm"] == arm and c["turn_taking"] == turn and c["activity_drift"] == drift:
            return c
    return None


def decompose(cs):
    """Main effects and the departure from additivity, for each surrogate and each arm.

    An additive prediction is the only comparison the design supports without a model: baseline
    plus the turn-taking-alone effect plus the drift-alone effect. Whether the observed both-on
    rate sits inside its own Wilson interval of that prediction is the whole interaction claim, so
    the prediction and the interval are both written out rather than a verdict word.
    """
    out = {}
    corners = {
        "factorial": (("factorial", "off", 0.0), ("factorial", "real_nominal", 0.0),
                      ("factorial", "off", 0.9), ("factorial", "real_nominal", 0.9)),
        "calibrated": (("factorial", "off", 0.0), ("calibrated", "real_calibrated", 0.0),
                       ("calibrated", "off", CAL_DRIFT),
                       ("calibrated", "real_calibrated", CAL_DRIFT)),
    }
    for arm, keys in corners.items():
        base, turn_only, drift_only, both = [_find(cs, *k) for k in keys]
        if not all((base, turn_only, drift_only, both)):
            continue
        per = {}
        for null in ("shift", "permute"):
            b = base[null]["fwer"]
            t = turn_only[null]["fwer"]
            d = drift_only[null]["fwer"]
            o = both[null]["fwer"]
            if None in (b, t, d, o):
                continue   # a partial dump mid-run; the corner has no replicates yet
            pred = b + (t - b) + (d - b)
            per[null] = {
                "baseline_both_off": b, "turn_taking_only": t, "drift_only": d, "both_on": o,
                "turn_taking_effect": round(t - b, 4), "drift_effect": round(d - b, 4),
                "additive_prediction": round(pred, 4),
                "both_on_minus_additive_prediction": round(o - pred, 4),
                "both_on_wilson95": both[null]["wilson95"],
                "additive_prediction_inside_both_on_wilson95": bool(
                    both[null]["wilson95"][0] <= pred <= both[null]["wilson95"][1]),
                "larger_single_effect": ("turn_taking" if (t - b) > (d - b)
                                         else "drift" if (d - b) > (t - b) else "tied"),
            }
        out[arm] = {"corners": {"baseline_both_off": base["cell"],
                                "turn_taking_only": turn_only["cell"],
                                "drift_only": drift_only["cell"], "both_on": both["cell"]},
                    "by_surrogate": per}
    return out


def check_published(cs, rows, first_n=30):
    """Do the turn-taking-off cells still reproduce the published drift sweep, replicate for replicate?

    The knob is only worth anything if switching it off leaves the earlier measurement untouched,
    and "bit for bit on the generator" is a weaker claim than "the same detector verdicts". These
    two cells use the generator seeds, detector seeds and surrogate count of stationarity_big.py,
    so their first 30 replicates are the same 30 measurements bench_stationarity_big.json reports.

    Read `recomputed_fired` against `published_fired` AND against `published_wilson95`. An exact
    match is the strong result. A one-replicate difference is not evidence that the generator
    changed: that the generator did not change is established separately and more directly, by
    comparing its output against the committed version over the whole grid and a spread of random
    configurations, where it is byte-identical. A difference here means the published counts came
    off a different estimator or library state, which is worth knowing and worth saying, and the
    200-replicate rate in the cell record is the number to use either way.
    """
    out = []
    for (turn, drift), want in PUBLISHED.items():
        cell = _find(cs, "factorial", turn, drift)
        if cell is None:
            continue
        mine = sorted((r for r in rows if r["cell"] == cell["cell"]), key=lambda r: r["rep"])[:first_n]
        got = {n: sum(r[n + "_fired"] for r in mine) for n in ("shift", "permute")}
        out.append({"turn_taking": turn, "activity_drift": drift, "first_n": len(mine),
                    "published_fired": want, "recomputed_fired": got,
                    "published_wilson95": {n: list(bench_eval.wilson(k, 30))
                                           for n, k in want.items()},
                    "this_run_fwer": {n: cell[n]["fwer"] for n in ("shift", "permute")},
                    "this_run_fwer_inside_published_wilson95": {
                        n: bool(cell[n]["fwer"] is not None
                                and bench_eval.wilson(want[n], 30)[0] <= cell[n]["fwer"]
                                <= bench_eval.wilson(want[n], 30)[1])
                        for n in ("shift", "permute")},
                    "matches": len(mine) == first_n and got == want})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=200, help="replicates per cell")
    ap.add_argument("--surrogates", type=int, default=100, help="surrogates per test")
    ap.add_argument("--workers", type=int, default=1, help="processes; does not change the output")
    ap.add_argument("--out", default="turntaking_factorial.json")
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute every rate, interval and effect from the replicate rows "
                         "already in --out, running no detector; this is the verifier's path")
    args = ap.parse_args()

    cs = cells()
    jobs = [(ci, rep, args.surrogates) for ci in range(len(cs)) for rep in range(args.reps)]
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    log(f"{len(cs)} cells x {args.reps} replicates x 2 surrogates = {2 * len(jobs)} tests, "
        f"{args.workers} worker(s)")
    out_path = HERE / args.out

    def dump(rows):
        done = sum(1 for _ in rows)
        cells_out = summarise(rows)
        json.dump({"reps": args.reps, "n_surrogates": args.surrogates, "nominal_alpha": 0.05,
                   "generator_seed_base": BASE_SEED, "generator": dict(GEN, edges=[]),
                   "turn_levels": [{"name": n, "p_repeat": a, "p_return": b}
                                   for n, a, b in TURN_LEVELS],
                   "drift_levels": list(DRIFT_LEVELS),
                   "calibrated_knobs": {"p_repeat": CAL_REPEAT, "p_return": CAL_RETURN,
                                        "activity_drift": CAL_DRIFT},
                   "real_channel": REAL,
                   "replicates_completed": done,
                   "complete": done == len(jobs),
                   "published_sweep_agreement": check_published(cells_out, rows),
                   "effects": decompose(cells_out),
                   "cells": cells_out,
                   "rows": sorted(rows, key=lambda r: (r["cell"], r["rep"]))},
                  open(out_path, "w", encoding="utf-8"), indent=2)

    def report(rows):
        for c in summarise(rows):
            if not c["n_replicates"]:
                log(f"{c['arm']:11s} turn={c['turn_taking']:15s} drift={c['activity_drift']:<5} "
                    f"no replicates")
                continue
            log(f"{c['arm']:11s} turn={c['turn_taking']:15s} drift={c['activity_drift']:<5} "
                f"shift {c['shift']['fwer']:.3f} {c['shift']['wilson95']} | "
                f"permute {c['permute']['fwer']:.3f} {c['permute']['wilson95']} | "
                f"realised repeat {c['realised']['repeat_rate']:.3f} "
                f"return {c['realised']['return_after_one_rate']:.3f} "
                f"cv {c['realised']['median_block_share_cv']:.3f}")
        for a in check_published(summarise(rows), rows):
            log(f"published-sweep agreement turn={a['turn_taking']} drift={a['activity_drift']}: "
                f"{a['recomputed_fired']} vs {a['published_fired']} (published 30-replicate "
                f"interval {a['published_wilson95']}) -> exact match {a['matches']}")

    if args.rebuild:
        # Everything published above the `rows` list is a pure function of that list, so the whole
        # table can be rebuilt and compared without a single detector run. That is the property
        # that makes the file checkable by someone who does not want to spend the CPU.
        rows = json.load(open(out_path, encoding="utf-8"))["rows"]
        dump(rows)
        report(rows)
        log(f"rebuilt {args.out} from {len(rows)} stored replicates, no detector run")
        return

    rows = []
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(run_replicate, jobs, chunksize=1):
                rows.append(row)
                if len(rows) % 25 == 0 or len(rows) == len(jobs):
                    dump(rows)
                    log(f"{len(rows)}/{len(jobs)} replicates")
    else:
        for job in jobs:
            rows.append(run_replicate(job))
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                dump(rows)
                log(f"{len(rows)}/{len(jobs)} replicates")
    dump(rows)
    report(rows)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
