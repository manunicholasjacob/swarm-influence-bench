"""The payoff: where do the two real AI Village channels sit relative to the symbolisation floor?

The audit report's largest stated limitation is that its floor control cannot be run on the real
channels, because the control needs an adjusted Rand index and ARI needs ground-truth topic labels
that real chat does not have. `floor_proxy_sweep.py` calibrates label-free stand-ins for that ARI
against configurations whose true ARI is known. This script applies them to the two channels the
audit actually analysed, main-village #general and open-chat-village #general, and reports which
side of the floor each one lands on.

What leaves this script: counts, and the proxy statistics. No message text, no agent or user
identifier, no room identifier, nothing that could be traced to a participant. Human messages are
excluded by the loader, as in the audit. Research use only.

Two honest limits on the transfer, both recorded in the output:

  1. A threshold calibrated on the benchmark is calibrated on the benchmark, and only one family of
     proxies survives the move. An adjusted Rand index between two clusterings of the same messages
     is dimensionless and bounded in [0, 1] whatever the corpus, so a cut on one of those means the
     same thing here as there. Every other statistic in the battery is a property of a particular
     embedding: a silhouette, a Calinski-Harabasz, a Davies-Bouldin or a centroid cosine depends on
     how many features the vectoriser produced and how dense they are, and a synthetic instance has
     a vocabulary of 54 to 320 words against the real channels' 20000. Those thresholds are carried
     into the output for completeness and grouped as `geometric`, and the summary verdict is taken
     from the `ari_based` family alone. The grouping is by construction, not by which answer it
     gives, and the result section reports what each family said so the reader can check that.
  2. The benchmark has exactly 8 topics by construction and the symboliser is given k=8. Real chat
     has no true k. A proxy saying "above the floor" on real text therefore means "this clustering
     is as stable and as reproducible as a benchmark clustering whose ARI was above the floor", not
     "its ARI against some true labelling is above the floor". The distinction is the residual
     limitation and it does not go away.

    python floor_proxy_real.py [--days 14] [--k 8]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

import floor_proxy as fx          # also puts the sibling swarm checkouts on sys.path

HERE = pathlib.Path(__file__).resolve().parent

VILLAGES = ("main-village", "open-chat-village")
ROOM_NAME = "general"

# The split is by what the statistic IS, not by what it says here. An adjusted Rand index between
# two clusterings of the same messages is dimensionless and bounded, so a benchmark cut on one
# carries over to a corpus with a different vocabulary. The rest describe a particular embedding
# and do not carry over. See the module docstring.
ARI_BASED = ("seed_stability_mean", "seed_stability_min", "boot_stability_full_mean",
             "boot_stability_full_min", "boot_stability_shared_mean",
             "agree_lsa", "agree_counts", "agree_char")
GEOMETRIC = ("silhouette_cosine", "calinski_harabasz", "davies_bouldin",
             "centroid_cosine_mean", "centroid_cosine_max", "size_entropy")

# The meaning-free symboliser is carried through the whole study as a null: cross-symboliser
# agreement is only evidence of anything if a symboliser that ignores meaning scores zero. It is
# therefore never allowed a vote on where a channel sits, only a sanity check that it still reads
# near zero on real text.
NULL_PROXIES = ("agree_hash_null",)

# a proxy only gets a vote if it actually separated the floor on the benchmark
MIN_CALIBRATION_BALANCED_ACCURACY = 0.8

t0 = time.time()


def log(m):
    print("[%5.0fs] %s" % (time.time() - t0, m), flush=True)


def load_channel(village, days, allow_download):
    """The audit's window for one village's #general, or a reason it could not be opened.

    The corpus is gated. Offline by default, so a run can only ever use what this machine was
    already granted and nothing new is fetched.
    """
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import collections
    import swarmlib as S
    msgs, _, rooms = S.load_village(village)
    # The room is picked by NAME, not by a hard-coded identifier, and the busiest room of that name
    # wins so the choice does not depend on dictionary order. In both villages the audit's channel
    # is the one this finds: main-village has a single #general, and in open-chat-village #general
    # is also the busiest room, which is how the audit selected it there.
    volume = collections.Counter(m["room_id"] for m in msgs)
    named = sorted((rid for rid, name in rooms.items() if str(name).lower() == ROOM_NAME),
                   key=lambda rid: -volume.get(rid, 0))
    room = named[0] if named else S.busiest_room(msgs)
    span, lo, hi = S.cut_window(msgs, room, days=days)
    return {
        "village": village,
        "room_name": ROOM_NAME if named else "busiest",
        "rooms_with_that_name": len(named),
        "window_days": days,
        "window_messages": len(span),
        "window_span_hours": round((hi - lo) / 3600.0, 1),
        "village_messages_total": len(msgs),
        "texts": [m["content"] for m in span],
    }


def verdict(name, value, cut):
    """Which side of a calibrated cut this value falls on."""
    if cut is None or value is None or not np.isfinite(value):
        return None
    above = value >= cut["threshold"] if cut["higher_is_better"] else value <= cut["threshold"]
    return {"proxy": name, "value": round(float(value), 6), "threshold": cut["threshold"],
            "direction": "higher_is_better" if cut["higher_is_better"] else "lower_is_better",
            "says": "above_floor" if above else "below_floor",
            "calibration_balanced_accuracy": cut["balanced_accuracy"],
            "calibration_errors": cut["errors"],
            "margin_to_threshold": round(
                float(value - cut["threshold"]) * (1.0 if cut["higher_is_better"] else -1.0), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="the audit's window length")
    ap.add_argument("--k", type=int, default=8, help="the audit's alphabet size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=6)
    ap.add_argument("--allow-download", action="store_true",
                    help="permit a fetch of the gated corpus; off by default, cache only")
    ap.add_argument("--sweep", default="floor_proxy_sweep.json")
    ap.add_argument("--out", default="floor_proxy_real.json")
    args = ap.parse_args()

    sweep_path = HERE / args.sweep
    if not sweep_path.exists():
        print("FAIL  %s is missing; run floor_proxy_sweep.py first, because the thresholds this "
              "script applies are calibrated there" % args.sweep)
        return 1
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    ranking = sweep.get("ranking_by_floor_separation", [])
    cuts = {n: sweep["evaluation"][n]["floor_by_stated_ari"] for n in sweep["evaluation"]}

    out = {
        "question": ("where the two real channels the audit analysed sit relative to the "
                     "symbolisation floor, judged by label-free proxies calibrated in "
                     + args.sweep),
        "k": args.k, "window_days": args.days, "symboliser_seed": args.seed,
        "calibrated_from": args.sweep,
        "stated_floor_ari": sweep.get("stated_floor_ari"),
        "privacy": ("aggregate statistics only: counts and clustering indices. No message text, "
                    "no participant identifier, no room identifier. Human messages excluded by "
                    "the loader. Research use only."),
        "proxy_families": {"ari_based": list(ARI_BASED), "geometric": list(GEOMETRIC)},
        "verdict_taken_from": "ari_based",
        "honesty_caveat": (
            "The family split is by construction, but the strict_reading rule was fixed after the "
            "benchmark calibration had been inspected, which is what calibration is for, and after "
            "a first pass over these two channels had already been run. It is therefore NOT a "
            "pre-registered test. Every proxy's raw value and cut is in the output so a reader can "
            "apply any other rule."),
        "excluded_as_null": list(NULL_PROXIES),
        "min_calibration_balanced_accuracy_to_vote": MIN_CALIBRATION_BALANCED_ACCURACY,
        "channels": [],
    }

    for village in VILLAGES:
        try:
            ch = load_channel(village, args.days, args.allow_download)
        except Exception as exc:                      # the corpus is gated: say so, do not guess
            out["channels"].append({"village": village, "reachable": False,
                                    "reason": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
            log("%s: NOT REACHABLE (%s)" % (village, type(exc).__name__))
            continue
        texts = ch.pop("texts")
        log("%s: %d messages in the %d-day window" % (village, len(texts), args.days))
        res = fx.all_proxies(texts, k=args.k, seed=args.seed, n_boot=args.n_boot)
        ch["reachable"] = True
        ch["proxies"] = {k: (None if v is None else round(float(v), 6))
                         for k, v in res["proxies"].items()}
        ch["diagnostics"] = res["diagnostics"]
        ch["verdicts"] = [v for v in (verdict(n, res["proxies"].get(n), cuts.get(n))
                                      for n in fx.PROXY_NAMES) if v is not None]

        def tally(names):
            vs = [v for v in ch["verdicts"]
                  if v["proxy"] in names and v["proxy"] not in NULL_PROXIES
                  and (v["calibration_balanced_accuracy"] or 0.0)
                  >= MIN_CALIBRATION_BALANCED_ACCURACY]
            says = {v["says"] for v in vs}
            return {"n_voting": len(vs),
                    "n_above": sum(v["says"] == "above_floor" for v in vs),
                    "n_below": sum(v["says"] == "below_floor" for v in vs),
                    "consensus": (says.pop() if len(says) == 1 else "split") if vs else None,
                    "proxies": [v["proxy"] for v in vs]}

        ch["by_family"] = {"ari_based": tally(ARI_BASED), "geometric": tally(GEOMETRIC)}
        # The strict reading: only the ARI-based cuts that made NO error separating the floor on
        # the benchmark. A cut with errors behind it is a cut with no evidence behind it, and a
        # majority vote over fifteen statistics of wildly different calibration quality is not a
        # measurement. Stated as its own field, with its caveat, rather than as the headline.
        perfect = [n for n in ARI_BASED
                   if (cuts.get(n) or {}).get("errors") == 0
                   and (sweep["evaluation"][n]["floor_leave_one_overlap_out"]["errors"] or 0) <= 1]
        ch["strict_reading"] = dict(tally(perfect), rule=(
            "ARI-based proxies with zero in-sample calibration errors and at most one "
            "leave-one-overlap-out error"))
        # the summary verdict comes from the ARI-based family, for the reason in the docstring
        fam = ch["by_family"]["ari_based"]
        ch["n_voting_proxies"] = fam["n_voting"]
        ch["n_above"] = fam["n_above"]
        ch["n_below"] = fam["n_below"]
        ch["verdict"] = fam["consensus"]
        # and from the three ARI-based proxies that separated the floor best on the benchmark
        top = [r["proxy"] for r in ranking if r["proxy"] in ARI_BASED][:3]
        ch["top_proxy_verdicts"] = [v for v in ch["verdicts"] if v["proxy"] in top]
        says = {v["says"] for v in ch["top_proxy_verdicts"]}
        ch["consensus_of_top_proxies"] = says.pop() if len(says) == 1 else "split"
        ch["null_proxy_sanity_check"] = {n: ch["proxies"].get(n) for n in NULL_PROXIES}
        out["channels"].append(ch)
        st = ch["strict_reading"]
        log("%s: STRICT %s (%d above / %d below of %d) | ari-based %d/%d | geometric %d/%d"
            % (village, st["consensus"], st["n_above"], st["n_below"], st["n_voting"],
               fam["n_above"], fam["n_below"],
               ch["by_family"]["geometric"]["n_above"],
               ch["by_family"]["geometric"]["n_below"]))

    reachable = [c for c in out["channels"] if c.get("reachable")]
    out["summary"] = {
        "channels_reachable": len(reachable),
        "channels_requested": len(VILLAGES),
        "per_channel": [{"village": c["village"], "window_messages": c["window_messages"],
                         "verdict": c["strict_reading"]["consensus"],
                         "strict_reading": c["strict_reading"],
                         "ari_family_consensus": c["verdict"],
                         "consensus_of_top_proxies": c["consensus_of_top_proxies"],
                         "n_above": c["n_above"], "n_below": c["n_below"],
                         "n_voting_proxies": c["n_voting_proxies"],
                         "geometric_family_consensus":
                             c["by_family"]["geometric"]["consensus"]}
                        for c in reachable],
    }
    (HERE / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
