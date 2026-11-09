"""The non-stationarity sweep, re-run with the environment recorded, in two arms.

`paper/README.md` lists one defect to fix before submission: `bench_stationarity_big.json`'s exact
counts do not reproduce, off by one or two replicates per cell. The generator, the estimator and
BLAS threading were ruled out, leaving a library or environment difference. The fix is not to chase
the old counts but to stop citing counts that nobody recorded an environment for.

  arm "reproduction"   The published configuration exactly: drifts 0.0/0.3/0.6/0.9, 30 replicates,
                       100 surrogates, the two published nulls, generator seed 7000+s and detector
                       seed s, through `bench_eval.te_detector` as the original did. Each cell is
                       compared against the committed JSON and the agreement is REPORTED, not
                       asserted, because a difference here is the finding.

  arm "precision"      All four surrogates, including the two variants added for the extended
                       paper, on a shared symbol grid at more replicates, so the drifted cells get
                       intervals narrow enough to call. The published 30-replicate sweep left the
                       permutation surrogate somewhere between 0.033 and 0.167, which is too wide
                       to say whether the repair holds up under drift.

The symboliser is the most likely source of the irreproducibility: it is TF-IDF plus k-means from
scikit-learn, and k-means output moves between scikit-learn versions. The environment block in the
output names the version actually used.

  python -X utf8 stationarity_pinned.py [--reps-precision N] [--workers N]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import pathlib
import platform
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "swarm-prep", HERE.parent / "swarm-sprint"):
    sys.path.insert(0, str(_p))

DRIFTS = [0.0, 0.3, 0.6, 0.9]
PUBLISHED_REPS, N_SUR, ALPHA = 30, 100, 0.05
VARIANTS = ("shift", "permute", "blockpermute", "localpermute")
BLOCK_SIZE = 500


def environment() -> dict:
    mods = {}
    for m in ("numpy", "scipy", "sklearn", "joblib", "threadpoolctl"):
        try:
            mods[m] = getattr(__import__(m), "__version__", "?")
        except Exception as exc:                       # noqa: BLE001
            mods[m] = f"absent ({type(exc).__name__})"
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": mods}
    try:
        from threadpoolctl import threadpool_info
        env["blas"] = [{k: v for k, v in d.items()
                        if k in ("user_api", "internal_api", "num_threads", "version")}
                       for d in threadpool_info()]
    except Exception:                                  # noqa: BLE001
        env["blas"] = "threadpoolctl unavailable"
    return env


def _pin_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def reproduction_cell(job) -> dict:
    """One (drift, replicate) of the published configuration, through the original detector."""
    drift, s = job
    _pin_threads()
    import bench_gen, bench_eval
    inst = bench_gen.generate_instance(n_agents=12, n_messages=8000, edges=(),
                                       shared_topic=0.0, activity_drift=drift, seed=7000 + s)
    out = {"activity_drift": drift, "rep": s}
    for null in ("shift", "permute"):
        found = bench_eval.te_detector(inst["events"], null=null, n_surrogates=N_SUR, seed=s)
        out[null] = {"fired": len(found) > 0, "n_edges": len(found)}
    return out


def precision_cell(job) -> dict:
    """One (drift, replicate) with all four surrogates on one shared symbol grid.

    Takes the draw count in the job rather than from the module constant: the pool spawns fresh
    interpreters, so a constant mutated in the parent would not reach the workers.
    """
    drift, s, n_sur = job
    _pin_threads()
    import bench_gen, swarm_te, fastte
    from swarm_load import TopicSymboliser
    inst = bench_gen.generate_instance(n_agents=12, n_messages=8000, edges=(),
                                       shared_topic=0.0, activity_drift=drift, seed=9000 + s)
    ev = sorted(inst["events"], key=lambda e: e[1])
    texts = [e[3] for e in ev]
    sym = TopicSymboliser(k=8, seed=s).fit(texts).transform(texts)
    counts = collections.Counter(e[0] for e in ev)
    agents = [a for a, n in counts.most_common() if n >= 20][:15]
    keep = set(agents)
    stream = [(e[0], x) for e, x in zip(ev, sym) if e[0] in keep]
    if len(agents) < 3 or len(stream) < 100:
        return {"activity_drift": drift, "rep": s, "skipped": True}
    n = len(stream)
    series = {a: [swarm_te.SILENT] * n for a in agents}
    for t, (sp, x) in enumerate(stream):
        series[sp][t] = x
    coded, A = fastte.code_series(series)
    alpha_syms = sorted({x for v in series.values() for x in v})
    silent = alpha_syms.index(swarm_te.SILENT)
    row = {"activity_drift": drift, "rep": s, "skipped": False, "n_agents_tested": len(agents)}
    for name in VARIANTS:
        if name == "shift":
            tests = fastte.directed_tests_fast(coded, A, n_shifts=n_sur, seed=s)
        elif name == "permute":
            tests = fastte.directed_tests_permute(coded, A, silent, n_perm=n_sur, seed=s)
        elif name == "blockpermute":
            tests = fastte.directed_tests_permute_block(coded, A, silent, n_perm=n_sur,
                                                        seed=s, block_size=BLOCK_SIZE)
        else:
            tests = fastte.directed_tests_permute_local(coded, A, silent, n_perm=n_sur, seed=s)
        edges = [k for k, v in tests.items() if v["p"] < ALPHA]
        row[name] = {"fired": bool(edges), "n_edges": len(edges)}
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-precision", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--precision-draws", type=int, default=N_SUR,
                    help="surrogate draws for the precision arm. The reproduction arm stays at "
                         "100 because that is what the published sweep used.")
    ap.add_argument("--skip-reproduction", action="store_true")
    ap.add_argument("--out", default=str(HERE / "stationarity_pinned.json"))
    args = ap.parse_args()
    from bench_eval import wilson
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    published = json.loads((HERE / "bench_stationarity_big.json").read_text(encoding="utf-8"))
    pub = {(r["activity_drift"], n): r[n]["fired"] for r in published["rows"]
           for n in ("shift", "permute")}

    result = {
        "what": ("The non-stationarity sweep re-run with the environment recorded. Pure nulls "
                 "only, so any certified edge is a false positive by construction."),
        "nominal_alpha": ALPHA, "n_surrogates_per_test": N_SUR, "drifts": DRIFTS,
        "block_size": BLOCK_SIZE, "environment": environment(),
    }

    # ---- arm 1: reproduction (skippable; it is fixed at 100 draws by definition) -------------------------------------------------------------
    jobs = [] if args.skip_reproduction else [(d, s) for d in DRIFTS for s in range(PUBLISHED_REPS)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(reproduction_cell, jobs):
            rows.append(r)
    repro = {"reps": PUBLISHED_REPS, "generator_seed_base": 7000, "cells": [], "rows": rows}
    agree = 0
    for d in DRIFTS:
        cell = {"activity_drift": d}
        for null in ("shift", "permute"):
            got = sum(1 for r in rows if r["activity_drift"] == d and r[null]["fired"])
            was = pub[(d, null)]
            lo, hi = wilson(got, PUBLISHED_REPS)
            same = got == was
            agree += same
            cell[null] = {"fired_now": got, "fired_published": was, "identical": same,
                          "delta": got - was, "fwer": round(got / PUBLISHED_REPS, 4),
                          "wilson95": [lo, hi],
                          "published_fwer_inside_interval": lo <= was / PUBLISHED_REPS <= hi}
        repro["cells"].append(cell)
    repro["skipped"] = args.skip_reproduction
    repro["cells_identical"] = agree
    repro["cells_total"] = len(DRIFTS) * 2
    repro["all_identical"] = agree == len(DRIFTS) * 2
    result["reproduction"] = repro
    print(f"[{time.time()-t0:5.0f}s] reproduction arm: {agree}/{len(DRIFTS)*2} cells identical "
          f"to the published counts", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- arm 2: precision ----------------------------------------------------------------
    jobs = [(d, s, args.precision_draws) for d in DRIFTS for s in range(args.reps_precision)]
    prows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(precision_cell, jobs), 1):
            prows.append(r)
            if i % 50 == 0:
                print(f"[{time.time()-t0:5.0f}s] precision arm {i}/{len(jobs)}", flush=True)
    cells = []
    for d in DRIFTS:
        done = [r for r in prows if r["activity_drift"] == d and not r.get("skipped")]
        cell = {"activity_drift": d, "replicates": len(done)}
        for name in VARIANTS:
            fired = sum(1 for r in done if r[name]["fired"])
            lo, hi = wilson(fired, len(done))
            cell[name] = {"fired": fired, "fwer": round(fired / len(done), 4) if done else None,
                          "wilson95": [lo, hi],
                          "excludes_nominal": not (lo <= ALPHA <= hi)}
        cells.append(cell)
    result["precision"] = {"reps": args.reps_precision, "generator_seed_base": 9000,
                           "n_surrogates_per_test": args.precision_draws,
                           "shared_symbol_grid": True, "cells": cells, "rows": prows}
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.0f}s -> {args.out}\n")
    print(f"{'drift':>6} " + " ".join(f"{n:>26}" for n in VARIANTS))
    for c in cells:
        print(f"{c['activity_drift']:>6} " + " ".join(
            f"{c[n]['fwer']:.3f} {str(c[n]['wilson95']):>18}" for n in VARIANTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
