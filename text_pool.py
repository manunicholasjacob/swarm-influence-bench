"""Load an LLM-rendered message pool and hand it to the generator as its text layer.

The division of labour is the point of this file. The generator decides, in code, who speaks at
which tick, on which topic, and who is reacting to whom; that is the ground truth and no renderer
touches it. A pool only answers the separate question "what does a message about topic t look like",
so swapping the pool in changes realism and leaves `ground_truth` and every structural label exactly
as they were. `tests/test_text_pool.py` asserts that, by generating with and without a pool at the
same seed and checking the (agent, time, topic) stream is identical.

The expected input is an instruction dataset whose instructions name the topic verbatim:

    {"instruction": "Write a short channel message about: outreach email drafting",
     "response": "Second pass on the partner list is done, 40 names left to verify."}

JSONL, JSON array and CSV are all accepted, because exports differ. Field names are sniffed from a
few common spellings rather than hardcoded.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import Counter

# The eight topics the corpus was generated against, in the order the generator indexes them.
TOPICS = [
    "web server deployment and debugging",
    "marketing copy writing and review",
    "shared budget and expense tracking",
    "logo design and visual branding",
    "meeting scheduling and availability",
    "survey spreadsheet analysis",
    "outreach email drafting",
    "file permissions and storage troubleshooting",
]

_INSTRUCTION_KEYS = ("instruction", "prompt", "input", "question")
_RESPONSE_KEYS = ("response", "completion", "output", "answer", "text")
_ABOUT = re.compile(r"about\s*:\s*(.+?)\s*$", re.I)


def _rows(path: pathlib.Path):
    """Yield dict rows from JSONL, a JSON array, or CSV."""
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        yield from csv.DictReader(raw.splitlines())
        return
    stripped = raw.lstrip()
    if stripped.startswith("["):
        data = json.loads(raw)
        if isinstance(data, dict):           # {"rows": [...]} and friends
            for key in ("rows", "data", "records", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        yield from (r for r in data if isinstance(r, dict))
        return
    for line in raw.splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _pick(row: dict, keys) -> str:
    """Find a field by one of several spellings, exact match first.

    Exports decorate these names: Adaption's JSONL ships `enhanced_prompt` and
    `enhanced_completion` rather than `prompt` and `completion`, so a suffix match is tried after
    the exact ones. Exact wins, so a row carrying both `prompt` and `enhanced_prompt` is never
    resolved by the decorated name.
    """
    lowered = {k.lower(): k for k in row}
    for k in keys:
        if k in lowered and row[lowered[k]]:
            return str(row[lowered[k]]).strip()
    for k in keys:
        for low, original in lowered.items():
            if low.endswith("_" + k) and row[original]:
                return str(row[original]).strip()
    return ""


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def load_pool(path, topics=None, min_per_topic=10, strict=True):
    """Return {topic_index: [message, ...]} plus a report on what was parsed.

    Rows whose topic cannot be matched to one of `topics` are counted and dropped rather than
    guessed at, because a misfiled message is a mislabelled one and the labels are the whole point.
    """
    topics = list(topics or TOPICS)
    index = {normalise(t): i for i, t in enumerate(topics)}
    pool = {i: [] for i in range(len(topics))}
    seen = unmatched = empty = 0
    unmatched_examples = Counter()

    for row in _rows(pathlib.Path(path)):
        seen += 1
        instruction = _pick(row, _INSTRUCTION_KEYS)
        message = _pick(row, _RESPONSE_KEYS)
        if not message:
            empty += 1
            continue
        m = _ABOUT.search(instruction)
        label = normalise(m.group(1) if m else instruction)
        if label in index:
            pool[index[label]].append(message)
            continue
        # fall back to an unambiguous containment match before giving up
        hits = [i for key, i in index.items() if key and key in label]
        if len(hits) == 1:
            pool[hits[0]].append(message)
        else:
            unmatched += 1
            unmatched_examples[label[:60]] += 1

    report = {
        "rows_seen": seen,
        "rows_used": sum(len(v) for v in pool.values()),
        "rows_unmatched": unmatched,
        "rows_without_text": empty,
        "per_topic": {topics[i]: len(v) for i, v in pool.items()},
        "unmatched_examples": unmatched_examples.most_common(5),
    }
    thin = [topics[i] for i, v in pool.items() if len(v) < min_per_topic]
    if thin:
        msg = (f"{len(thin)} topic(s) have fewer than {min_per_topic} messages: {thin}. "
               f"Parsed {report['rows_used']} of {seen} rows.")
        if strict:
            raise ValueError(msg)
        report["warning"] = msg
    return pool, report


if __name__ == "__main__":
    import sys
    pool, report = load_pool(sys.argv[1], strict=False)
    print(json.dumps(report, indent=2))
    for i, msgs in sorted(pool.items()):
        if msgs:
            print(f"  [{i}] {TOPICS[i]}: {len(msgs)} e.g. {msgs[0][:70]}")
