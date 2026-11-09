"""Surrogate variants compared at the precision the headline already reached.

The sprint measured one repair, the global within-speaker permutation, at 300 replicates. The
extended paper needs the comparison BETWEEN variants at that same precision, which the sprint never
ran. Three permutation variants plus the conventional circular shift, on pure nulls:

  shift          the conventional circular time shift. No influence can be present, so every
                 certified edge is a false positive by construction.
  permute        global within-speaker permutation. The repair the sprint recommends.
  blockpermute   the same, confined to blocks of consecutive ticks, so drift slower than a block
                 survives into the null.
  localpermute   Runge (2018) local permutation in conditioning state space, as adapted to event
                 data by Shorten, Spinney and Lizier (2021). At history one the conditioning
                 variable is the target's own previous symbol, so the exact discrete analogue is
                 available and needs no neighbourhood size.

**All four surrogates see the identical symbol grid.** The text is symbolised once per replicate
and the same coded series is handed to each. Without that this would be a comparison between four
datasets rather than between four nulls, and the symboliser is stochastic enough to dominate the
difference being measured.

The environment is recorded in the output. `bench_stationarity_big.json`'s counts were found not to
reproduce across machines, and the cause was never a bug in the generator or estimator but a
library difference, so every sweep from now on states what it ran under.

  python -X utf8 surrogate_variants.py [--reps N] [--workers N] [--out PATH]
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

# Resolved from this file, never hardcoded: stationarity_big.py and bench_eval.py both pinned
# absolute C:\dev paths and so could only ever run on one machine.
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "swarm-prep", HERE.parent / "swarm-sprint"):
    sys.path.insert(0, str(p))

N_AGENTS, N_MESSAGES, N_SURROGATES, ALPHA = 12, 8000, 100, 0.05
BLOCK_SIZE = 500
VARIANTS = ("shift", "permute", "blockpermute", "localpermute")


def environment() -> dict:
    mods = {}
    for m in ("numpy", "scipy", "sklearn", "joblib", "threadpoolctl"):
        try:
            mods[m] = getattr(__import__(m), "__version__", "?")
        except Exception as exc:                      # noqa: BLE001
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


def one_replicate(job) -> dict:
    """One pure-null instance, symbolised once, then tested under each surrogate.

    Takes (rep, n_surrogates) as a tuple rather than reading a module constant: on Windows the
    pool spawns fresh interpreters that re-import this module, so a constant mutated in the parent
    would silently not reach the workers and the run would quietly use the default.
    """
    rep, n_sur = job
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import numpy as np                                  # noqa: F401
    import bench_gen, swarm_te, fastte
    from swarm_load import TopicSymboliser

    inst = bench_gen.generate_instance(n_agents=N_AGENTS, n_messages=N_MESSAGES,
                                       edges=(), shared_topic=0.0, seed=rep)
    ev = sorted(inst["events"], key=lambda e: e[1])
    sym = TopicSymboliser(k=8, seed=rep).fit([e[3] for e in ev]).transform([e[3] for e in ev])
    counts = collections.Counter(e[0] for e in ev)
    agents = [a for a, n in counts.most_common() if n >= 20][:15]
    keep = set(agents)
    stream = [(e[0], s) for e, s in zip(ev, sym) if e[0] in keep]
    if len(agents) < 3 or len(stream) < 100:
        return {"rep": rep, "skipped": True}
    n = len(stream)
    series = {a: [swarm_te.SILENT] * n for a in agents}
    for t, (sp, s) in enumerate(stream):
        series[sp][t] = s
    coded, A = fastte.code_series(series)
    alpha_syms = sorted({x for v in series.values() for x in v})
    silent = alpha_syms.index(swarm_te.SILENT)

    row = {"rep": rep, "n_agents_tested": len(agents), "n_ticks": n, "alphabet": A,
           "family_size": len(agents) * (len(agents) - 1), "skipped": False}
    for name in VARIANTS:
        t0 = time.time()
        if name == "shift":
            tests = fastte.directed_tests_fast(coded, A, n_shifts=n_sur, seed=rep)
        elif name == "permute":
            tests = fastte.directed_tests_permute(coded, A, silent, n_perm=n_sur, seed=rep)
        elif name == "blockpermute":
            tests = fastte.directed_tests_permute_block(coded, A, silent, n_perm=n_sur,
                                                        seed=rep, block_size=BLOCK_SIZE)
        else:
            tests = fastte.directed_tests_permute_local(coded, A, silent,
                                                        n_perm=n_sur, seed=rep)
        edges = [k for k, v in tests.items() if v["p"] < ALPHA]
        zs = [v["z"] for v in tests.values()]
        row[name] = {"fired": bool(edges), "n_edges": len(edges),
                     "max_z": float(max(zs)) if zs else None,
                     "seconds": round(time.time() - t0, 2)}
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n-surrogates", type=int, default=N_SURROGATES,
                    help="draws per test. The report's headline used 200; this defaults to 100.")
    ap.add_argument("--out", default=str(HERE / "surrogate_variants.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from bench_eval import wilson

    t0 = time.time()
    rows: list[dict] = []
    out_path = pathlib.Path(args.out)

    def flush():
        done = [r for r in rows if not r.get("skipped")]
        summary = {}
        for name in VARIANTS:
            fired = sum(1 for r in done if r[name]["fired"])
            k = len(done)
            summary[name] = {
                "replicates": k,
                "fwer": round(fired / k, 4) if k else None,
                "fired": fired,
                "wilson95": [round(x, 4) for x in wilson(fired, k)] if k else None,
                "mean_edges_when_fired": round(
                    sum(r[name]["n_edges"] for r in done if r[name]["fired"])
                    / max(1, fired), 3),
                "median_seconds": round(sorted(r[name]["seconds"] for r in done)[k // 2], 2)
                if k else None,
            }
        out_path.write_text(json.dumps({
            "what": ("Four surrogates on identical pure-null symbol grids. No influence and no "
                     "common cause is planted, so any certified edge is a false positive."),
            "nominal_alpha": ALPHA,
            "n_surrogates_per_test": args.n_surrogates,
            "block_size": BLOCK_SIZE,
            "config": {"n_agents": N_AGENTS, "n_messages": N_MESSAGES, "symboliser_k": 8,
                       "min_msgs": 20, "top_k": 15},
            "shared_symbol_grid": True,
            "requested_reps": args.reps,
            "replicates_completed": len(done),
            "complete": len(done) == args.reps,
            "environment": environment(),
            "summary": summary,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        return summary

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_replicate, (r, args.n_surrogates)): r for r in range(args.reps)}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 10 == 0 or i == args.reps:
                s = flush()
                el = time.time() - t0
                print(f"[{el:6.0f}s] {i}/{args.reps} "
                      + " | ".join(f"{n} {s[n]['fwer']:.3f}" for n in VARIANTS), flush=True)

    rows.sort(key=lambda r: r["rep"])
    summary = flush()
    print(f"\nDone in {time.time() - t0:.0f}s -> {out_path}")
    for name in VARIANTS:
        s = summary[name]
        print(f"  {name:14} FWER {s['fwer']:.4f}  95% CI {s['wilson95']}  "
              f"({s['fired']}/{s['replicates']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
