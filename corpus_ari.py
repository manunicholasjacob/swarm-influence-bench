"""Where does real LLM-written agent chat sit on the symbolisation-difficulty curve?

`realism_sweep.py` degrades symbolisation synthetically, by sharing vocabulary between topics, and
finds a floor: conditioning's advantage survives to an adjusted Rand index of about 0.44 and
collapses below it. That curve is only useful if we know where real text lands on it.

This measures that, using a corpus of LLM-written channel messages as the generator's text layer.
The generator still decides who speaks at which tick and on which topic, so the labels stay exact
and the ARI is measured against them.

Two honest limits on the number it produces, both reported in the output rather than buried:

  1. A finite pool REPEATS. With a few dozen distinct messages per topic spread over thousands of
     ticks, the symboliser is clustering a few dozen distinct strings, not thousands of unique
     ones. Real chat does not repeat like that, so the ARI here is an UPPER BOUND on what the same
     symboliser would achieve on genuinely unique text.
  2. The commissioned corpus arrived badly unbalanced, so only the best-covered topics are usable
     and the measurement runs at a smaller alphabet than the report's k=8.

Usage:  python -I corpus_ari.py <corpus.jsonl> [--topics N] [--per-topic N]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, r"C:\dev\swarm-prep")
sys.path.insert(0, r"C:\dev\swarm-sprint")
sys.path.insert(0, str(HERE))

from sklearn.metrics import adjusted_rand_score
from swarm_load import TopicSymboliser
import bench_gen
import text_pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--topics", type=int, default=3,
                    help="how many of the best-covered topics to use")
    ap.add_argument("--per-topic", type=int, default=0,
                    help="cap messages per topic; 0 means the smallest usable topic's count")
    ap.add_argument("--messages", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    pool, report = text_pool.load_pool(args.corpus, strict=False)
    counts = {i: len(v) for i, v in pool.items() if v}
    ranked = sorted(counts, key=lambda i: -counts[i])[:args.topics]
    per = args.per_topic or min(counts[i] for i in ranked)
    if per < 5:
        print(f"FAIL  best {args.topics} topics have only {per} messages each; not enough to cluster")
        return 1

    # Rebalance: equal pool size per topic, so an imbalanced corpus cannot flatter the symboliser
    # by letting one topic dominate the stream.
    balanced = {new: pool[old][:per] for new, old in enumerate(ranked)}
    names = [text_pool.TOPICS[i] for i in ranked]

    out = {
        "corpus": pathlib.Path(args.corpus).name,
        "corpus_rows": report["rows_seen"],
        "corpus_per_topic": report["per_topic"],
        "topics_used": names,
        "messages_per_topic": per,
        "distinct_messages_total": per * len(ranked),
        "seeds": args.seeds,
        "runs": [],
    }

    aris = []
    for s in range(args.seeds):
        inst = bench_gen.generate_instance(
            n_agents=12, n_messages=args.messages, edges=(), shared_topic=0.55,
            k_topics=len(ranked), seed=900 + s, text_pool=balanced)
        ev = sorted(inst["events"], key=lambda e: e[1])
        texts = [e[3] for e in ev]
        truth = [e[2] for e in ev]
        sym = TopicSymboliser(k=len(ranked), seed=s).fit(texts).transform(texts)
        ari = float(adjusted_rand_score(truth, list(sym)))
        aris.append(ari)
        out["runs"].append({"seed": 900 + s, "ari": round(ari, 4),
                            "distinct_texts": len(set(texts)),
                            "n_messages": len(texts)})
        print(f"seed {900 + s}: ARI {ari:.3f} over {len(texts)} messages "
              f"drawn from {len(set(texts))} distinct strings")

    out["ari_mean"] = round(float(np.mean(aris)), 4)
    out["ari_min"] = round(float(np.min(aris)), 4)
    out["upper_bound_caveat"] = (
        f"A pool of {out['distinct_messages_total']} distinct strings is reused across "
        f"{args.messages} ticks, so the symboliser clusters repeated text. Real chat does not "
        f"repeat this way, so this ARI is an upper bound.")

    # place it against the synthetic curve, if that sweep has been run
    sweep_path = HERE / "realism_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        pts = [(r["vocab_overlap"], r["symboliser_ari_mean"]) for r in sweep["rows"]]
        out["synthetic_curve"] = pts
        floor = min((a for _, a in pts if a >= 0.4), default=None)
        out["above_collapse_floor"] = bool(floor is not None and out["ari_mean"] >= 0.4)
        print(f"\nsynthetic curve (overlap, ARI): {pts}")
        print(f"real corpus ARI mean {out['ari_mean']:.3f} -> "
              f"{'ABOVE' if out['ari_mean'] >= 0.4 else 'BELOW'} the ~0.44 collapse floor")

    (HERE / "corpus_ari.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{out['upper_bound_caveat']}")
    print("wrote corpus_ari.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
