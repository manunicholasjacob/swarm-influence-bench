"""Validate the benchmark by running both surrogates on it.

If SwarmInfluence-Bench is a useful instrument, it should independently reproduce the AI Village
audit's central finding on data where the truth is known by construction: the circular-shift
surrogate should fail the null instances badly, and the permutation repair should not.
"""
import sys, json, time, functools
sys.path.insert(0, r"C:\dev\swarm-bench")
import bench_gen, bench_eval

SEEDS = 2
N_SUR = 100     # surrogates per test; 100 keeps the full grid tractable
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

specs = bench_gen.grid(seeds=SEEDS)
log(f"{len(specs)} instances, {SEEDS} seeds per configuration, {N_SUR} surrogates per test")

results = {}
for null in ("shift", "permute"):
    det = functools.partial(bench_eval.te_detector, null=null, n_surrogates=N_SUR)
    log(f"=== detector: {null} surrogate ===")
    summary = bench_eval.evaluate(
        det, instances=[dict(s) for s in specs],
        out_path=f"bench_results_{null}.json",
        log=lambda m: log(f"  {m.split('] ',1)[-1]}"))
    results[null] = summary
    log(f"{null}: FWER(null)={summary['familywise_error_pure_null']} "
        f"pure-null FWER + precision={summary['mean_precision']} recall={summary['mean_recall']}")
    log(f"{null}: power by coupling {summary['power_by_coupling']}")
    log(f"{null}: power by volume {summary['power_by_channel_volume']}")
    json.dump(results, open("bench_summary.json", "w", encoding="utf-8"), indent=2)

log("wrote bench_summary.json")
