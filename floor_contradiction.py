"""A contradiction between two of this project's own measurements, and which one is wrong.

`floor_proxy_sweep.py` calibrated label-free stand-ins for the symbolisation floor and
`floor_proxy_real.py` applied them to the two real AI Village channels the audit analysed. They
came back saying main-village #general is narrowly ABOVE the floor and open-chat #general is well
BELOW it. The audit's own certified-edge counts, under the same symbol-permutation surrogate, run
the other way round: main-village certifies 2 pairwise edges and open-chat certifies 18.

The reading that makes those two statements contradictory is that a below-floor symboliser should
certify near-zero edges, because below the floor there is no topic signal left to detect. That
reading is what this script tests, on instances where the answer is known.

Four experiments, each a separate section of floor_contradiction.json:

  E1  the edge count across the whole overlap sweep, on the STRUCTURE family, which is the family
      with planted edges. The published "false edges fall to 0.0" claim is about the common-cause
      null, which has no planted edges at all, so it says nothing about what a below-floor instance
      WITH structure certifies. E1 also adds the circular-shift count beside the permutation count,
      because the two real channels disagree about which surrogate fires (main-village 11 shift
      against 2 permute, open-chat 3 shift against 18 permute) and the sweep never recorded shift.

  E2  the common-cause knob at FIXED symbolisation difficulty. `shared_topic` decides how much
      genuine topical dependence runs through the channel and has nothing to do with how hard the
      text is to cluster. If the edge count moves with it while the proxies and the true ARI do
      not, then an edge count is not a symboliser-quality statistic and the contradiction is a
      category error rather than a disagreement.

  E3  the published cuts tested OUT OF MECHANISM. The sweep degraded symbolisation one way only,
      by shared vocabulary, with the number of topics known and correct. Real channels differ in
      turn-taking, burstiness, swarm size, volume skew, message length and message count. E3 holds
      the vocabulary knob at zero, where the true ARI is pinned near 1.0 by construction, and moves
      each of those instead. Anything that drags a proxy below its published cut while the true ARI
      stays high is a false below-floor verdict waiting to happen on a real channel.

  E4  k. Both channels were symbolised at k=8 and real chat has no true k. E4 generates instances
      whose true topic count is 2 to 24 and symbolises every one of them at k=8, so the proxies can
      be read against a known k mismatch, and asks the question that decides whether the open-chat
      verdict means anything: when the proxies fall because k is wrong, do the symbols stop
      working?

  E5  whichever mechanisms E3 and E4 turned up, checked on the real channels themselves. Both are
      label free, so both are checkable there: message length is a property of the text, and the
      proxies can be recomputed at a k other than the audit's 8. This is the only section that
      opens the corpus, it is offline and cache-only by default, and what leaves it is counts,
      token-length quantiles, a vocabulary size and clustering indices. No message text, no token,
      no participant or room identifier.

E1 to E4 never touch the corpus. The real-channel counts quoted beside them are read from the
audit's committed facts files and the file records which fact came from where.

    python floor_contradiction.py [--only E1,E2,E3,E4,E5] [--quick] [--resume]
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
STATED_FLOOR = 0.44

# The sweep's own configurations, reproduced exactly so E1's permutation counts can be checked
# against the committed ones instead of being asserted to match.
SWEEP_OVERLAPS = (0.0, 0.3, 0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95)
SWEEP_FAMILIES = (
    ("common_cause_null", dict(n_agents=12, n_messages=8000, edges=(), shared_topic=0.55)),
    ("structure", dict(n_agents=12, n_messages=8000, shared_topic=0.55,
                       edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4)))),
)
STRUCT_EDGES = ((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4))
# four planted edges again, inside 0..4, for the five-agent cell in E3
SMALL_EDGES = ((0, 2, 0.4), (0, 3, 0.3), (2, 4, 0.5), (1, 4, 0.4))

# the measured AI Village turn-taking rates, from facts_mechanism.json
AIV_P_REPEAT = 0.36
AIV_P_RETURN = 0.38

# The two proxies floor_proxy_real.py actually voted with: ARI-based, zero in-sample calibration
# errors, at most one leave-one-overlap-out error. Their cuts are read from the sweep, not retyped.
VOTING = ("seed_stability_min", "agree_lsa")

t0 = time.time()


def log(m):
    print("[%5.0fs] %s" % (time.time() - t0, m), flush=True)


# ----------------------------------------------------------------------------- detection
# turn_grid and modal_ctx are the same code as floor_proxy_sweep.turn_grid and
# floor_proxy_sweep.modal_ctx, duplicated for the same reason: that module runs its whole sweep at
# import time. Both take the symbols as an argument so detection and the proxies are measured on
# one and the same symbolisation.

def turn_grid(events_sorted, symbols):
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
    return coded, A, sil, [alpha.index(s) for _, s in stream], agents


def modal_ctx(chan, sil, window=12):
    out = np.zeros(len(chan), dtype=np.int64)
    for t in range(len(chan)):
        lo = max(0, t - window)
        c = [x for x in chan[lo:max(lo, t - 1)] if x != sil]
        out[t] = collections.Counter(c).most_common(1)[0][0] if c else sil
    return out


def _edge_stats(tests, gt, alpha=0.05):
    """Certified edges plus the shape of the z distribution behind them.

    The z summary is here because the audit reports exactly these numbers for the real channels
    (facts_cond_diag.json) and the benchmark never did, so there was nothing to compare them with.
    A surrogate that sits ABOVE the observed statistic for most pairs is not a detector that found
    a lot, it is a detector whose null is wider than its data.
    """
    e = {k for k, v in tests.items() if v["p"] < alpha}
    z = np.array([v["z"] for v in tests.values()], dtype=float)
    tp = len(e & gt)
    return {
        "edges": len(e), "true_positives": tp,
        "precision": round(tp / len(e), 4) if e else None,
        "recall": round(tp / len(gt), 4) if gt else None,
        "n_pairs": len(tests),
        "mean_z": round(float(z.mean()), 4),
        "max_z": round(float(z.max()), 4),
        "frac_z_negative": round(float((z < 0).mean()), 4),
        "frac_z_exactly_zero": round(float((z == 0).mean()), 4),
    }


def detect(events_sorted, symbols, gt, n_sur, seed, with_shift=True, conditional=True):
    """Permutation and, optionally, circular-shift and conditional edge sets at alpha 0.05."""
    coded, A, sil, chan, agents = turn_grid(events_sorted, symbols)
    out = {"n_ticks": len(chan), "planted": len(gt), "n_agents_tested": len(agents),
           "planted_inside_tested_set": len(gt & {(a, b) for a in agents for b in agents if a != b})}
    pw = fastte.directed_tests_permute(coded, A, sil, n_perm=n_sur, seed=seed)
    out["pairwise"] = _edge_stats(pw, gt)
    if with_shift:
        sh = fastte.directed_tests_fast(coded, A, n_shifts=n_sur, seed=seed)
        out["shift"] = _edge_stats(sh, gt)
    if conditional:
        ctx = modal_ctx(chan, sil)
        cd = fastte.directed_tests_conditional(coded, ctx, A, A, sil, n_perm=n_sur, seed=seed)
        out["conditional"] = _edge_stats(cd, gt)
    return out


# ----------------------------------------------------------------------------- one cell

def run_cell(label, inst_seed, sym_seed, n_sur, proxies=True, n_boot=6, seed_runs=5, k=K, **kw):
    """Generate one instance, symbolise it once, and measure everything on that symbolisation."""
    inst = bench_gen.generate_instance(seed=inst_seed, **kw)
    ev = sorted(inst["events"], key=lambda e: e[1])
    texts = [e[3] for e in ev]
    truth = [e[2] for e in ev]
    row = {"cell": label, "instance_seed": inst_seed, "symboliser_seed": sym_seed,
           "symboliser_k": k, "config": {kk: inst["config"][kk] for kk in
                                         ("n_agents", "n_messages", "k_topics", "shared_topic",
                                          "vocab_overlap", "volume_skew", "p_repeat", "p_return",
                                          "activity_drift", "burstiness")
                                         if kk in inst["config"]}}
    row["config"]["words_per_msg"] = kw.get("words_per_msg", 12)
    row["config"]["volume_skew"] = kw.get("volume_skew", 1.4)
    if proxies:
        res = fx.all_proxies(texts, k=k, seed=sym_seed, n_boot=n_boot, seed_runs=seed_runs)
        sym = res["base_symbols"]
        row["proxies"] = {kk: (None if v is None else round(float(v), 6))
                          for kk, v in res["proxies"].items()}
        row["diagnostics"] = res["diagnostics"]
    else:
        from swarm_load import TopicSymboliser
        sym = TopicSymboliser(k=k, seed=sym_seed).fit(texts).transform(texts)
    row["true_ari"] = round(float(adjusted_rand_score(truth, fx.codes(sym))), 4)
    row["detection"] = detect(ev, sym, inst["ground_truth"], n_sur, sym_seed,
                              with_shift=True, conditional=proxies)
    return row


# ----------------------------------------------------------------------------- E1

def experiment_1(n_sur, n_msg, overlaps, seeds=SEEDS):
    """The permutation edge count against the true ARI, on both families, with shift beside it.

    The symboliser is refitted with the sweep's own instance seed and symboliser seed, so the
    permutation count here has to equal the committed one exactly. If it does not, this harness is
    not measuring the configurations the sweep measured and nothing below it means anything.
    """
    rows = []
    for ov in overlaps:
        for fam_i, (fam, spec) in enumerate(SWEEP_FAMILIES):
            for s in range(seeds):
                kw = dict(spec)
                kw["n_messages"] = n_msg
                kw["vocab_overlap"] = ov
                row = run_cell("%s_ov%.2f_s%d" % (fam, ov, s), 900 + s + 1000 * fam_i, s,
                               n_sur, proxies=False, **kw)
                row["family"] = fam
                row["vocab_overlap"] = ov
                rows.append(row)
                log("E1 ov %.2f %-18s s%d: ARI %.3f | permute %2d (tp %d) shift %2d (tp %d)"
                    % (ov, fam, s, row["true_ari"], row["detection"]["pairwise"]["edges"],
                       row["detection"]["pairwise"]["true_positives"],
                       row["detection"]["shift"]["edges"],
                       row["detection"]["shift"]["true_positives"]))
    return rows


def check_against_sweep(rows, sweep):
    """Did the refit reproduce the committed permutation counts and ARIs cell for cell?"""
    if not sweep:
        return {"available": False}
    want = {(r["family"], r["vocab_overlap"], r["symboliser_seed"]): r for r in sweep["rows"]}
    diffs = []
    for r in rows:
        key = (r["family"], r["vocab_overlap"], r["symboliser_seed"])
        if key not in want:
            continue
        w = want[key]
        diffs.append({"cell": r["cell"],
                      "edge_diff": r["detection"]["pairwise"]["edges"]
                      - w["detection"]["pairwise"]["edges"],
                      "ari_diff": round(r["true_ari"] - w["true_ari"], 4)})
    return {"available": True, "n_cells_compared": len(diffs),
            "max_abs_edge_diff": max((abs(d["edge_diff"]) for d in diffs), default=None),
            "max_abs_ari_diff": max((abs(d["ari_diff"]) for d in diffs), default=None),
            "mismatched_cells": [d for d in diffs if d["edge_diff"] or d["ari_diff"]]}


# ----------------------------------------------------------------------------- E2, E3, E4

def experiment_2(n_sur, n_msg, n_boot, seed_runs, seeds=SEEDS):
    """Common-cause strength at fixed symbolisation difficulty."""
    rows, cell = [], [0]
    for ov in (0.0, 0.8):
        for st in (0.0, 0.3, 0.55, 0.8, 0.95):
            cell[0] += 1
            for s in range(seeds):
                row = run_cell("shared_topic%.2f_ov%.2f_s%d" % (st, ov, s),
                               3000 + s + 100 * cell[0], s, n_sur,
                               n_boot=n_boot, seed_runs=seed_runs,
                               n_agents=12, n_messages=n_msg, edges=STRUCT_EDGES,
                               shared_topic=st, vocab_overlap=ov)
                row["axis"] = "shared_topic"
                row["vocab_overlap"] = ov
                rows.append(row)
                log("E2 ov %.2f shared %.2f s%d: ARI %.3f ss_min %.3f lsa %.3f | "
                    "permute %2d (tp %d) cond %2d (tp %d)"
                    % (ov, st, s, row["true_ari"], row["proxies"]["seed_stability_min"],
                       row["proxies"]["agree_lsa"], row["detection"]["pairwise"]["edges"],
                       row["detection"]["pairwise"]["true_positives"],
                       row["detection"]["conditional"]["edges"],
                       row["detection"]["conditional"]["true_positives"]))
    return rows


def experiment_3(n_sur, n_msg, n_boot, seed_runs, seeds=SEEDS):
    """The published cuts out of mechanism: vocabulary held at zero, everything else moved."""
    base = dict(n_agents=12, n_messages=n_msg, edges=STRUCT_EDGES, shared_topic=0.55,
                vocab_overlap=0.0)
    variants = [
        ("baseline", {}),
        ("turntaking_aiv", dict(p_repeat=AIV_P_REPEAT, p_return=AIV_P_RETURN)),
        ("activity_drift", dict(activity_drift=0.6)),
        ("turntaking_and_drift", dict(p_repeat=AIV_P_REPEAT, p_return=AIV_P_RETURN,
                                      activity_drift=0.6)),
        ("agents5", dict(n_agents=5, edges=SMALL_EDGES)),
        ("agents25", dict(n_agents=25)),
        ("volume_skew2.5", dict(volume_skew=2.5)),
        ("short_messages", dict(words_per_msg=4)),
        ("long_messages", dict(words_per_msg=30)),
        ("messages2000", dict(n_messages=max(2000, n_msg // 4))),
    ]
    rows, cell = [], [0]
    for name, over in variants:
        cell[0] += 1
        kw = dict(base)
        kw.update(over)
        for s in range(seeds):
            row = run_cell("%s_s%d" % (name, s), 4000 + s + 100 * cell[0], s, n_sur,
                           n_boot=n_boot, seed_runs=seed_runs, **kw)
            row["axis"] = "nuisance"
            row["variant"] = name
            row["vocab_overlap"] = kw["vocab_overlap"]
            rows.append(row)
            log("E3 %-22s s%d: ARI %.3f ss_min %.3f lsa %.3f | permute %2d shift %2d cond %2d "
                "rec %s"
                % (name, s, row["true_ari"], row["proxies"]["seed_stability_min"],
                   row["proxies"]["agree_lsa"], row["detection"]["pairwise"]["edges"],
                   row["detection"]["shift"]["edges"],
                   row["detection"]["conditional"]["edges"],
                   row["detection"]["conditional"]["recall"]))
    return rows


def experiment_4(n_sur, n_msg, n_boot, seed_runs, seeds=SEEDS):
    """A known k mismatch: true topic count 2 to 24, every instance symbolised at k=8.

    The second block holds the true topic count at 8 and moves the SYMBOLISER's k instead, which is
    the sweep the audit ran on open-chat (k=4 gave 8 edges, k=8 gave 18, k=16 gave 21).
    """
    rows, cell = [], [0]
    for kt in (2, 4, 6, 8, 12, 16, 24):
        cell[0] += 1
        for s in range(seeds):
            row = run_cell("ktopics%d_k8_s%d" % (kt, s), 5000 + s + 100 * cell[0], s, n_sur,
                           n_boot=n_boot, seed_runs=seed_runs, k=K,
                           n_agents=12, n_messages=n_msg, edges=STRUCT_EDGES,
                           shared_topic=0.55, vocab_overlap=0.0, k_topics=kt)
            row["axis"] = "true_k"
            row["k_topics"] = kt
            row["vocab_overlap"] = 0.0
            rows.append(row)
            log("E4 k_topics %2d k=8 s%d: ARI %.3f ss_min %.3f lsa %.3f | permute %2d cond %2d "
                "rec %s prec %s"
                % (kt, s, row["true_ari"], row["proxies"]["seed_stability_min"],
                   row["proxies"]["agree_lsa"], row["detection"]["pairwise"]["edges"],
                   row["detection"]["conditional"]["edges"],
                   row["detection"]["conditional"]["recall"],
                   row["detection"]["conditional"]["precision"]))
    for sk in (4, 16):
        cell[0] += 1
        for s in range(seeds):
            row = run_cell("ktopics8_k%d_s%d" % (sk, s), 5900 + s + 100 * cell[0], s, n_sur,
                           n_boot=n_boot, seed_runs=seed_runs, k=sk,
                           n_agents=12, n_messages=n_msg, edges=STRUCT_EDGES,
                           shared_topic=0.55, vocab_overlap=0.0, k_topics=8)
            row["axis"] = "symboliser_k"
            row["k_topics"] = 8
            row["vocab_overlap"] = 0.0
            rows.append(row)
            log("E4 k_topics  8 k=%-2d s%d: ARI %.3f ss_min %.3f lsa %.3f | permute %2d cond %2d "
                "rec %s"
                % (sk, s, row["true_ari"], row["proxies"]["seed_stability_min"],
                   row["proxies"]["agree_lsa"], row["detection"]["pairwise"]["edges"],
                   row["detection"]["conditional"]["edges"],
                   row["detection"]["conditional"]["recall"]))
    return rows


# ----------------------------------------------------------------------------- analysis

def _f(x):
    return None if x is None else round(float(x), 4)


def edge_count_as_a_proxy(rows):
    """Treat the certified edge count itself as a candidate floor statistic and score it.

    This is the question H1 asks in the form the rest of the repo asks it: is there a cut on the
    edge count that sorts above-floor from below-floor instances? Scored exactly as the sweep scores
    every other candidate, with fx.best_threshold, so the numbers are comparable.
    """
    out = {}
    for fam in ("structure", "common_cause_null", "both"):
        sub = [r for r in rows if fam == "both" or r["family"] == fam]
        if not sub:
            continue
        above = [r["true_ari"] >= STATED_FLOOR for r in sub]
        rec = {"n": len(sub)}
        for null in ("pairwise", "shift"):
            v = [r["detection"][null]["edges"] for r in sub]
            rec[null] = {
                "spearman_vs_true_ari": _f(fx.spearman(v, [r["true_ari"] for r in sub])),
                "floor_by_stated_ari": fx.best_threshold(v, above, True),
                "edges_above_floor": _f(np.mean([x for x, a in zip(v, above) if a])),
                "edges_below_floor": _f(np.mean([x for x, a in zip(v, above) if not a])),
                "max_edges_below_floor": int(max([x for x, a in zip(v, above) if not a],
                                                 default=0)),
                "min_edges_above_floor": int(min([x for x, a in zip(v, above) if a],
                                                 default=0)),
            }
        out[fam] = rec
    return out


def cuts_out_of_mechanism(rows, cuts, label):
    """Do the published cuts still separate above-floor from below-floor on these rows?

    The cut is NOT refitted. It is the number floor_proxy_real.py applied to the real channels, and
    the only question is how it behaves on instances degraded by a mechanism it never saw.
    """
    out = {"label": label, "n": len(rows)}
    above = [r["true_ari"] >= STATED_FLOOR for r in rows]
    for name in VOTING:
        cut = cuts.get(name)
        if cut is None:
            continue
        v = [r["proxies"][name] for r in rows]
        c = fx.confusion(v, above, cut["threshold"], cut["higher_is_better"])
        wrong = [{"cell": r["cell"], "true_ari": r["true_ari"], "value": r["proxies"][name],
                  "conditional_recall": r["detection"]["conditional"]["recall"],
                  "says": "above_floor" if (r["proxies"][name] >= cut["threshold"]) else
                          "below_floor"}
                 for r, a in zip(rows, above)
                 if (r["proxies"][name] >= cut["threshold"]) != a]
        out[name] = {"applied_threshold": cut["threshold"], "confusion": c,
                     "misclassified": wrong}
        # the operational question behind the cut: among rows the cut calls below-floor, how often
        # did the conditional detector in fact keep every planted edge?
        called_below = [r for r in rows if r["proxies"][name] < cut["threshold"]
                        and r["detection"].get("conditional")
                        and r["detection"]["conditional"]["recall"] is not None]
        out[name]["called_below_floor_but_detection_held"] = {
            "n_called_below": len(called_below),
            "n_full_recall": sum(r["detection"]["conditional"]["recall"] == 1.0
                                 for r in called_below),
            "cells": [r["cell"] for r in called_below
                      if r["detection"]["conditional"]["recall"] == 1.0],
        }
    return out


def pinned_ari_spread(rows, lo=0.9):
    """How far the proxies move among rows whose true ARI is pinned above `lo`.

    At vocab_overlap 0 the vocabularies are disjoint and the true ARI is near 1.0 whatever else the
    generator is doing, so any spread here is the proxy responding to something that is not
    symbolisation quality. The published cut is printed beside the range so a reader can see at once
    whether the spread crosses it.
    """
    sub = [r for r in rows if r["true_ari"] >= lo]
    out = {"n": len(sub), "true_ari_min": _f(min((r["true_ari"] for r in sub), default=None)),
           "true_ari_max": _f(max((r["true_ari"] for r in sub), default=None))}
    for name in VOTING:
        v = [r["proxies"][name] for r in sub]
        if not v:
            continue
        worst = min(sub, key=lambda r: r["proxies"][name])
        out[name] = {"min": _f(min(v)), "max": _f(max(v)), "mean": _f(np.mean(v)),
                     "range": _f(max(v) - min(v)),
                     "worst_cell": worst["cell"], "worst_true_ari": worst["true_ari"]}
    return out


def by_group(rows, key):
    """Per-group means of the things every experiment reports, for the tables in the write-up."""
    out = []
    for g in sorted({r[key] for r in rows}, key=lambda x: (isinstance(x, str), x)):
        sub = [r for r in rows if r[key] == g]
        rec = {key: g, "n": len(sub), "true_ari": _f(np.mean([r["true_ari"] for r in sub]))}
        if "proxies" in sub[0]:
            for name in VOTING:
                rec[name] = _f(np.mean([r["proxies"][name] for r in sub]))
        for null in ("pairwise", "shift", "conditional"):
            if null not in sub[0]["detection"]:
                continue
            rec[null + "_edges"] = _f(np.mean([r["detection"][null]["edges"] for r in sub]))
            rec[null + "_tp"] = _f(np.mean([r["detection"][null]["true_positives"] for r in sub]))
            recs = [r["detection"][null]["recall"] for r in sub
                    if r["detection"][null]["recall"] is not None]
            rec[null + "_recall"] = _f(np.mean(recs)) if recs else None
            rec[null + "_mean_z"] = _f(np.mean([r["detection"][null]["mean_z"] for r in sub]))
            rec[null + "_max_z"] = _f(np.mean([r["detection"][null]["max_z"] for r in sub]))
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- real-channel facts
# Counts copied from the audit's committed facts files, with the file named beside each one. They
# are here so the benchmark numbers above can be read against them in a single document. Nothing in
# this block is recomputed and no corpus content is reproduced: edge counts, tick counts, agent
# counts and z summaries only.

REAL = {
    "source_files": {
        "edges_by_null": "swarm-alignment-2026/deliverables/facts_symboliser.json",
        "conditional": "swarm-alignment-2026/deliverables/facts_conditional.json",
        "conditional_resolution": "swarm-alignment-2026/deliverables/facts_cond_diag.json",
        "replication_and_k": "swarm-alignment-2026/deliverables/facts_openchat.json",
        "humans": "swarm-alignment-2026/deliverables/facts_recompute.json",
        "power": ("swarm-alignment-2026/deliverables/facts_final.json and "
                  "facts_power_openchat.json"),
        "proxies": "floor_proxy_real.json",
    },
    "main-village": {
        "window_msgs": 8800, "n_ticks": 7958, "agents_tested": 15,
        "tfidf_permute_edges": 2, "tfidf_shift_edges": 11,
        "lsa_permute_edges": 1, "lsa_shift_edges": 12,
        "hash_permute_edges": 0, "hash_shift_edges": 1,
        "conditional_edges": 2, "conditional_survivors": 2,
        "conditional_mean_z": 0.343, "conditional_max_z": 13.5,
        "conditional_frac_z_negative": 0.51,
        "permute_power_at_coupling": {"0.1": 0.0, "0.2": 0.0, "0.4": 0.0, "0.8": 0.25},
        "permute_replication_jaccard": 0.0,
        "human_share_of_window": None,
        "seed_stability_min": 0.675886, "agree_lsa": 0.695958,
    },
    "open-chat-village": {
        "window_msgs": 8226, "n_ticks": 7735, "agents_tested": 15,
        "tfidf_permute_edges": 18, "tfidf_shift_edges": 3,
        "lsa_permute_edges": 22, "lsa_shift_edges": 2,
        "hash_permute_edges": 0, "hash_shift_edges": 0,
        "conditional_edges": 0, "conditional_survivors": 0,
        "conditional_mean_z": -0.338, "conditional_max_z": 4.43,
        "conditional_frac_z_negative": 0.652,
        "permute_power_at_coupling": {"0.1": 0.75, "0.2": 1.0, "0.4": 1.0, "0.8": 1.0},
        "permute_replication_jaccard": 0.2727,
        "permute_split_half_jaccard": 0.125,
        "permute_edges_by_k": {"4": 8, "8": 18, "16": 21},
        "permute_edge_jaccard_vs_k8": {"4": 0.4444, "16": 0.56},
        "human_share_of_window": 0.3219,
        "distinct_human_visitors": 124,
        "permute_edges_agents_only_humans_excluded": 17,
        "permute_edges_with_humans_as_one_node": 19,
        "permute_human_drives_agent_edges_when_humans_included": 4,
        "seed_stability_min": 0.287375, "agree_lsa": 0.455923,
    },
}


def experiment_5(days, seed, k_grid, allow_download, seed_runs=5):
    """The two mechanisms E3 and E4 found, measured on the real channels themselves.

    E3 found that shortening messages drags both voting proxies below their published cuts while
    the true ARI stays well above the floor, and E4 found that a wrong k does the same. Neither
    mechanism was ever checked against the real channels, because the proxies were only ever run
    at k=8 and the text shape was never reported. Both are checkable without labels.

    What leaves this function: counts, token-length quantiles, a vocabulary size, and clustering
    indices at each k. No message text, no token, no agent or participant identifier, no room
    identifier. The loader excludes human messages, as in the audit. Research use only.
    """
    import floor_proxy_real as fpr               # the audit's own channel loader, reused as is

    out = []
    for village in fpr.VILLAGES:
        try:
            ch = fpr.load_channel(village, days, allow_download)
        except Exception as exc:                  # the corpus is gated: say so, do not guess
            out.append({"village": village, "reachable": False,
                        "reason": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
            log("E5 %s: NOT REACHABLE (%s)" % (village, type(exc).__name__))
            continue
        texts = ch.pop("texts")
        toks = [fx.normalise(t).split() for t in texts]
        lens = np.array([len(t) for t in toks], dtype=float)
        vocab = set()
        for t in toks:
            vocab.update(t)
        q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        rec = dict(ch)
        rec["reachable"] = True
        rec["text_shape"] = {
            "n_messages": int(lens.size),
            "tokens_mean": round(float(lens.mean()), 2),
            "tokens_sd": round(float(lens.std(ddof=1)), 2),
            "tokens_quantiles": {str(p): round(float(np.percentile(lens, p)), 1) for p in q},
            "share_under_5_tokens": round(float((lens < 5).mean()), 4),
            "share_under_12_tokens": round(float((lens < 12).mean()), 4),
            "vocabulary_size": len(vocab),
            "type_token_ratio": round(len(vocab) / max(1.0, float(lens.sum())), 5),
            "distinct_tokens_per_message_mean": round(
                float(np.mean([len(set(t)) for t in toks])), 2),
        }
        # the k sweep: do the voting proxies read differently at a k other than the audit's 8?
        rec["by_k"] = []
        X, kept = fx.tfidf(texts)
        kept_idx = np.asarray(kept, dtype=np.int64)
        for k in k_grid:
            # one headline fit per k, handed to both the stability run and the second opinion, so
            # the three statistics describe one and the same clustering
            headline = fx.TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
            base = fx.codes(headline)
            # the same stability protocol floor_proxy.all_proxies uses, so a reading
            # here is comparable with the one floor_proxy_real.json published at k=8
            seeds = tuple([seed] + [seed + 101 * i for i in range(1, seed_runs)])
            ss = fx.seed_stability(texts, k=k, seeds=seeds, base=headline)
            lsa = fx.ari(base, fx.codes(
                fx.LsaSymboliser(k=k, seed=seed).fit(texts).transform(texts)))
            clustered = base[kept_idx]
            rec["by_k"].append({
                "k": k, "seed_runs": len(seeds),
                "seed_stability_min": round(ss["min"], 6),
                "seed_stability_mean": round(ss["mean"], 6),
                "agree_lsa": round(lsa, 6),
                "silhouette_cosine": round(fx.silhouette_cosine(X, clustered, seed=seed), 6),
                "size_entropy": round(fx.size_entropy(clustered), 6),
                "n_symbols_used": int(len(np.unique(base))),
            })
            log("E5 %-18s k %2d: ss_min %.3f ss_mean %.3f lsa %.3f sil %.4f"
                % (village, k, ss["min"], ss["mean"], lsa,
                   rec["by_k"][-1]["silhouette_cosine"]))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small instances and few levels, for a wiring check only")
    ap.add_argument("--only", default="E1,E2,E3,E4,E5")
    ap.add_argument("--seeds", type=int, default=None,
                    help="instances per cell; defaults to %d, or 1 with --quick" % SEEDS)
    ap.add_argument("--sweep", default="floor_proxy_sweep.json")
    ap.add_argument("--out", default="floor_contradiction.json")
    ap.add_argument("--resume", action="store_true",
                    help="start from the sections already in --out instead of an empty file, so "
                         "one experiment can be added to a finished run without discarding it")
    ap.add_argument("--days", type=int, default=14, help="E5 only: the audit's window length")
    ap.add_argument("--allow-download", action="store_true",
                    help="E5 only: permit a fetch of the gated corpus; off by default, cache only")
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()}

    n_msg = 700 if args.quick else 8000
    n_boot = 2 if args.quick else 6
    seed_runs = 2 if args.quick else 5
    n_sur = 10 if args.quick else N_SUR
    overlaps = (0.0, 0.9) if args.quick else SWEEP_OVERLAPS
    seeds = args.seeds if args.seeds else (1 if args.quick else SEEDS)

    sweep_path = HERE / args.sweep
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.exists() else None
    cuts = ({n: sweep["evaluation"][n]["floor_by_stated_ari"] for n in sweep["evaluation"]}
            if sweep else {})

    out = {
        "question": ("whether the certified-edge count and the label-free floor proxies really "
                     "contradict each other on the two real channels, and if so which of them is "
                     "measuring something other than symbolisation quality"),
        "k": K, "seeds_per_cell": seeds, "n_surrogates": n_sur, "n_messages": n_msg,
        "stated_floor_ari": STATED_FLOOR,
        "calibrated_cuts_applied": {n: cuts[n] for n in VOTING if n in cuts},
        "privacy": ("no corpus content. The real-channel block holds counts and z summaries copied "
                    "from the audit's committed facts files, with the source named for each."),
        "real_channels": REAL,
    }
    if args.resume:
        prior = HERE / args.out
        if not prior.exists():
            print("FAIL  --resume needs %s to exist" % args.out)
            return 1
        kept = json.loads(prior.read_text(encoding="utf-8"))
        for key in [k for k in kept if k.startswith("e") and k[1:2].isdigit()]:
            out[key] = kept[key]
        log("resumed with %d earlier section(s) from %s"
            % (sum(1 for k in out if k.startswith("e") and k[1:2].isdigit()), args.out))

    if "E1" in want:
        rows = experiment_1(n_sur, n_msg, overlaps, seeds)
        out["e1_edges_across_the_floor"] = {
            "hypothesis": ("H1: that a below-floor instance WITH planted structure might still "
                           "certify many permutation-surrogate edges, which would dissolve the "
                           "contradiction"),
            "reproduction_of_committed_sweep": check_against_sweep(rows, sweep),
            "per_overlap": by_group(rows, "vocab_overlap"),
            "edge_count_as_a_floor_statistic": edge_count_as_a_proxy(rows),
            "rows": rows,
        }
        (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        log("E1 done, %d rows" % len(rows))

    if "E2" in want:
        rows = experiment_2(n_sur, n_msg, n_boot, seed_runs, seeds)
        rec = {"hypothesis": ("whether the edge count tracks the amount of real dependence in the "
                              "channel rather than the quality of the symboliser")}
        # `by_group` keys on a top-level field, so the knob is lifted out of config and put back
        for r in rows:
            r["shared_topic"] = r["config"]["shared_topic"]
        for ov in sorted({r["vocab_overlap"] for r in rows}):
            rec["by_shared_topic_at_overlap_%.2f" % ov] = by_group(
                [r for r in rows if r["vocab_overlap"] == ov], "shared_topic")
        rec["rows"] = rows
        out["e2_common_cause_at_fixed_difficulty"] = rec
        (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        log("E2 done, %d rows" % len(rows))

    if "E3" in want:
        rows = experiment_3(n_sur, n_msg, n_boot, seed_runs, seeds)
        out["e3_cuts_out_of_mechanism"] = {
            "hypothesis": ("H2: that the proxies respond to properties of a channel other than "
                           "topic separability, so a cut calibrated on the vocabulary mechanism "
                           "misfires on a channel that differs in some other way"),
            "by_variant": by_group(rows, "variant"),
            "proxy_spread_at_pinned_true_ari": pinned_ari_spread(rows),
            "published_cuts_on_these_rows": cuts_out_of_mechanism(rows, cuts, "e3_nuisance"),
            "rows": rows,
        }
        (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        log("E3 done, %d rows" % len(rows))

    if "E4" in want:
        rows = experiment_4(n_sur, n_msg, n_boot, seed_runs, seeds)
        true_k = [r for r in rows if r["axis"] == "true_k"]
        sym_k = [r for r in rows if r["axis"] == "symboliser_k"]
        out["e4_k_mismatch"] = {
            "hypothesis": ("H4: that k=8 is wrong for open-chat, so a stable clustering at the "
                           "wrong k reads as low seed stability while the symbols still work"),
            "by_true_k": by_group(true_k, "k_topics"),
            "by_symboliser_k": by_group(sym_k, "symboliser_k"),
            "published_cuts_on_these_rows": cuts_out_of_mechanism(true_k, cuts, "e4_true_k"),
            "rows": rows,
        }
        (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        log("E4 done, %d rows" % len(rows))

    if "E5" in want:
        k_grid = (4, 8) if args.quick else (2, 4, 6, 8, 12, 16)
        chans = experiment_5(args.days, 0, k_grid, args.allow_download,
                             seed_runs=2 if args.quick else 5)
        rec = {"hypothesis": ("the two mechanisms E3 and E4 found, short messages and a wrong k, "
                              "checked on the real channels themselves, label free"),
               "privacy": ("counts, token-length quantiles, a vocabulary size and clustering "
                           "indices. No message text, no token, no participant or room "
                           "identifier. Human messages excluded by the loader."),
               "audit_k": K, "channels": chans}
        for ch in chans:
            if not ch.get("reachable"):
                continue
            best = max(ch["by_k"], key=lambda r: r["seed_stability_min"])
            at8 = next((r for r in ch["by_k"] if r["k"] == K), None)
            rec.setdefault("k_that_reads_best", {})[ch["village"]] = {
                "k": best["k"], "seed_stability_min": best["seed_stability_min"],
                "agree_lsa": best["agree_lsa"],
                "seed_stability_min_at_audit_k": None if at8 is None
                else at8["seed_stability_min"],
                "crosses_published_cut": (
                    None if "seed_stability_min" not in cuts else
                    best["seed_stability_min"] >= cuts["seed_stability_min"]["threshold"]),
            }
        out["e5_real_channel_text_shape_and_k"] = rec
        (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        log("E5 done, %d channels" % len(chans))

    (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
