"""Does the permutation repair itself hold up on non-stationary generated nulls?

The 8-replicate sweep in bench_stationarity.json suggested the permutation surrogate fires in 2 to 3
of 8 replicates once generated activity is strongly non-stationary, which is above nominal and is the
one place our own repair looks shaky. Eight replicates cannot distinguish 0.05 from 0.30, so this
re-runs the sweep with enough replicates to decide, and reports Wilson intervals so the answer is
honest either way.

Pure nulls only: no influence and no common cause, so any certified edge is an error by construction.
"""
import sys, json, time, math
sys.path.insert(0, r"C:\dev\swarm-bench")
import numpy as np
import bench_gen, bench_eval

DRIFTS = [0.0, 0.3, 0.6, 0.9]
REPS, N_SUR = 30, 100
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, c - m), 3), round(min(1.0, c + m), 3))


out = {"drifts": DRIFTS, "reps": REPS, "n_surrogates": N_SUR, "nominal": 0.05, "rows": []}
for d in DRIFTS:
    rec = {"activity_drift": d}
    for null in ("shift", "permute"):
        fired, edges = 0, []
        for s in range(REPS):
            inst = bench_gen.generate_instance(
                n_agents=12, n_messages=8000, edges=(), shared_topic=0.0,
                activity_drift=d, seed=7000 + s)
            found = bench_eval.te_detector(inst["events"], null=null, n_surrogates=N_SUR, seed=s)
            fired += len(found) > 0
            edges.append(len(found))
        lo, hi = wilson(fired, REPS)
        rec[null] = {"fired": fired, "of": REPS, "fwer": round(fired / REPS, 3),
                     "wilson95": [lo, hi], "mean_edges": round(float(np.mean(edges)), 2)}
        log(f"drift={d} {null:8s}: {fired}/{REPS} FWER {fired/REPS:.3f} "
            f"[{lo}, {hi}] mean edges {np.mean(edges):.2f}")
    out["rows"].append(rec)
    json.dump(out, open("bench_stationarity_big.json", "w", encoding="utf-8"), indent=2)
log("wrote bench_stationarity_big.json")
