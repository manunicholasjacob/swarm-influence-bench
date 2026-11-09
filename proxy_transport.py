"""Which floor proxies keep their calibrated cut when the channel changes for reasons other than
topic separability?

`floor_proxy_sweep.json` fits a cut per proxy on one degradation mechanism, vocabulary overlap, and
reports errors against the stated floor of ARI 0.44. **Three** proxies reach zero errors there:
`seed_stability_min`, `agree_lsa` and `calinski_harabasz`. Section 8 of the paper recommends the
first two and says nothing about the third, which is not defensible on error count, because on the
fitting mechanism all three are perfect. Something else has to do the choosing.

That something is transportability. A floor proxy is a claim about the symbolisation. If a cut moves
when the corpus changes size or message length while the symbolisation stays perfect, the cut is a
property of the corpus and the proxy cannot carry a published threshold.

`floor_contradiction.json`'s e3 arm is the test already sitting in the repository: 30 cells varying
turn-taking, drift, agent count, volume skew, message length and corpus size, 27 of them with true
ARI pinned at 1.000. Apply each proxy's fitted cut unchanged and count the errors, separating out
the errors made at true ARI 1.000, where there is nothing wrong with the symbolisation at all.

**Read the output with one caution.** A proxy that classifies badly on the fitting mechanism can
score zero nuisance errors by calling almost everything above floor. The comparison is only
meaningful among proxies that earned zero errors where they were fitted, which is why those three
are also printed in full.

    python -X utf8 proxy_transport.py [--out proxy_transport.json]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
FLOOR_ARI = 0.44
PERFECT = 0.999        # "the symboliser recovered the topics", allowing for float noise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "proxy_transport.json"))
    args = ap.parse_args()

    sweep = json.loads((HERE / "floor_proxy_sweep.json").read_text(encoding="utf-8"))
    contra = json.loads((HERE / "floor_contradiction.json").read_text(encoding="utf-8"))
    ev = sweep["evaluation"]
    e3 = contra["e3_cuts_out_of_mechanism"]
    rows = e3["rows"]
    n_perfect = sum(1 for r in rows if r["true_ari"] >= PERFECT)

    out = {
        "question": ("whether a floor proxy's cut, fitted on the vocabulary-overlap mechanism, "
                     "still separates above-floor from below-floor when the channel differs in "
                     "corpus size, message length, agent count, volume skew, turn-taking or "
                     "drift instead"),
        "stated_floor_ari": FLOOR_ARI,
        "fitting_mechanism": "vocabulary overlap, 9 levels x 6 cells, floor_proxy_sweep.json",
        "nuisance_set": {
            "source": "floor_contradiction.json :: e3_cuts_out_of_mechanism",
            "n_cells": len(rows),
            "n_cells_at_true_ari_1": n_perfect,
            "variants": sorted({r["cell"].rsplit("_s", 1)[0] for r in rows}),
        },
        "caution": ("a proxy that classifies badly where it was fitted can score zero nuisance "
                    "errors by calling nearly everything above floor; only the proxies with zero "
                    "fitting errors are comparable here"),
        "proxies": {},
    }

    for name, blk in ev.items():
        fit = blk.get("floor_by_stated_ari")
        if not fit:
            continue
        cut, hib = fit["threshold"], fit["higher_is_better"]
        errors = []
        for r in rows:
            v = r["proxies"].get(name)
            if v is None:
                continue
            says_above = (v >= cut) if hib else (v <= cut)
            truly_above = r["true_ari"] >= FLOOR_ARI
            if says_above != truly_above:
                errors.append({
                    "cell": r["cell"],
                    "true_ari": round(r["true_ari"], 4),
                    "value": v,
                    "n_messages": r.get("config", {}).get("n_messages"),
                    "says": "below_floor" if truly_above else "above_floor",
                    "symbolisation_was_perfect": r["true_ari"] >= PERFECT,
                })
        at_perfect = [e for e in errors if e["symbolisation_was_perfect"]]
        out["proxies"][name] = {
            "cut": cut,
            "higher_is_better": hib,
            "errors_on_fitting_mechanism": fit["errors"],
            "errors_on_nuisance_set": len(errors),
            "errors_at_true_ari_1": len(at_perfect),
            "cut_transports": len(errors) == 0,
            "cut_is_corpus_bound": len(at_perfect) > 0,
            "error_cells": errors,
        }

    clean = {k: v for k, v in out["proxies"].items() if v["errors_on_fitting_mechanism"] == 0}
    out["verdict"] = {
        "zero_error_on_fitting_mechanism": sorted(clean),
        "of_those_corpus_bound": sorted(k for k, v in clean.items() if v["cut_is_corpus_bound"]),
        "of_those_usable": sorted(k for k, v in clean.items() if not v["cut_is_corpus_bound"]),
        "statement": None,
    }
    bad = out["verdict"]["of_those_corpus_bound"]
    good = out["verdict"]["of_those_usable"]
    out["verdict"]["statement"] = (
        f"{len(clean)} proxies reach zero errors where they were fitted: {', '.join(sorted(clean))}. "
        f"Error count therefore cannot choose between them. Applying each cut unchanged to the "
        f"nuisance set separates them: {', '.join(good)} "
        f"{'makes' if len(good) == 1 else 'make'} errors only where the symbolisation genuinely "
        f"degraded, while {', '.join(bad)} {'also condemns' if len(bad) == 1 else 'also condemn'} "
        f"cells whose true ARI is exactly 1.000, on corpus size alone. A cut that moves with the "
        f"corpus is not a floor cut."
    )

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"stated floor ARI = {FLOOR_ARI}; nuisance cells = {len(rows)}, "
          f"{n_perfect} of them at true ARI 1.000\n")
    print(f"{'proxy':<28} {'cut':>11} {'fit errs':>9} {'nuis errs':>10} {'at ARI=1':>9}")
    for name, v in sorted(out["proxies"].items(),
                          key=lambda kv: (kv[1]["errors_on_nuisance_set"],
                                          kv[1]["errors_on_fitting_mechanism"])):
        print(f"{name:<28} {v['cut']:>11.4f} {v['errors_on_fitting_mechanism']:>9} "
              f"{v['errors_on_nuisance_set']:>10} {v['errors_at_true_ari_1']:>9}")

    print("\nThe three that were perfect where they were fitted:")
    for name, v in sorted(clean.items()):
        print(f"\n  {name} (cut {v['cut']:.4f}): {v['errors_on_nuisance_set']} nuisance errors, "
              f"{v['errors_at_true_ari_1']} at true ARI 1.000")
        for e in v["error_cells"]:
            tag = "  <-- symbolisation was PERFECT" if e["symbolisation_was_perfect"] else ""
            print(f"      {e['cell']:<24} true ARI {e['true_ari']:<7} "
                  f"value {e['value']:<11.4f} n_msg {e['n_messages']}{tag}")

    print("\n" + out["verdict"]["statement"])
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
