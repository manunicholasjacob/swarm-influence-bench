"""Why does the shift surrogate fail catastrophically on real logs but mildly on synthetic ones?

On real AI Village data the circular-shift surrogate certified something in 9 of 10 structure-free
replicates. On the first version of this benchmark it managed only about 0.08, even though both have
exactly one speaker per tick. So mutual exclusivity alone does not explain the real failure.

The candidate: real agents are NON-STATIONARY. They have bursts and quiet spells, join and leave,
and shift in and out of a channel's active hours. Circularly shifting a series slides that activity
envelope against everyone else's, so the surrogate compares the observed data against a world with a
different joint activity pattern. With stationary speaking weights there is no envelope to misalign.

This sweeps activity_drift from stationary to strongly non-stationary on PURE nulls (no influence,
no common cause) and measures each surrogate's family-wise error rate.
"""
import sys, json, time, functools
sys.path.insert(0, r"C:\dev\swarm-bench")
import numpy as np
import bench_gen, bench_eval

DRIFTS = [0.0, 0.3, 0.6, 0.9]
REPS, N_SUR = 8, 100
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

out = {"drifts": DRIFTS, "reps": REPS, "n_surrogates": N_SUR, "rows": []}
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
        rec[null] = {"fwer": round(fired / REPS, 3), "mean_edges": round(float(np.mean(edges)), 2)}
        log(f"activity_drift={d} {null:8s}: FWER {fired}/{REPS} mean edges {np.mean(edges):.2f}")
    out["rows"].append(rec)
    json.dump(out, open("bench_stationarity.json", "w", encoding="utf-8"), indent=2)
log("wrote bench_stationarity.json")
