"""Tests for the pluggable text layer and the loader for an LLM-rendered corpus.

The property that matters: swapping the renderer must not move a single structural label.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bench_gen
import text_pool


def _pool(n=20):
    return {i: [f"{text_pool.TOPICS[i]} note {k}" for k in range(n)]
            for i in range(len(text_pool.TOPICS))}


def test_renderer_swap_leaves_every_structural_label_identical():
    kw = dict(edges=((0, 5, 0.4), (2, 9, 0.5)), seed=7, n_messages=1200)
    a = bench_gen.generate_instance(**kw)
    b = bench_gen.generate_instance(text_pool=_pool(), **kw)
    assert a["ground_truth"] == b["ground_truth"]
    assert a["realised_coupling"] == b["realised_coupling"]
    # (agent, time, topic) must match exactly; only the text differs
    assert [e[:3] for e in a["events"]] == [e[:3] for e in b["events"]]
    assert [e[3] for e in a["events"]] != [e[3] for e in b["events"]]


def test_rendered_text_comes_from_the_right_topic():
    inst = bench_gen.generate_instance(edges=(), seed=8, n_messages=600, text_pool=_pool())
    for _, _, topic, text in inst["events"]:
        assert text.startswith(text_pool.TOPICS[topic])


def test_config_records_which_renderer_was_used():
    assert bench_gen.generate_instance(edges=(), seed=9, n_messages=200,
                                       text_pool=_pool())["config"]["text_renderer"] == "pool"
    assert bench_gen.generate_instance(edges=(), seed=9,
                                       n_messages=200)["config"]["text_renderer"] == "vocab"


def test_a_pool_missing_a_topic_is_refused_not_silently_filled():
    partial = {0: ["only topic zero"]}
    with pytest.raises(ValueError, match="no messages for topic"):
        bench_gen.generate_instance(edges=(), seed=10, n_messages=400, text_pool=partial)


def _write(tmp_path, name, rows):
    p = tmp_path / name
    if name.endswith(".jsonl"):
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    else:
        p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def _rows(n_each=12):
    return [{"instruction": f"Write a short channel message about: {t}",
             "response": f"message {k} on {t}"}
            for t in text_pool.TOPICS for k in range(n_each)]


def test_loader_reads_jsonl_and_maps_every_topic(tmp_path):
    pool, report = text_pool.load_pool(_write(tmp_path, "c.jsonl", _rows()))
    assert report["rows_unmatched"] == 0
    assert report["rows_used"] == report["rows_seen"]
    assert sorted(pool) == list(range(8))
    assert all(len(v) == 12 for v in pool.values())


def test_loader_reads_a_json_array(tmp_path):
    pool, report = text_pool.load_pool(_write(tmp_path, "c.json", _rows()))
    assert report["rows_unmatched"] == 0
    assert len(pool[0]) == 12


def test_loader_reads_csv(tmp_path):
    p = tmp_path / "c.csv"
    lines = ["instruction,response"]
    lines += [f'"Write a short channel message about: {t}","note {k} about {t}"'
              for t in text_pool.TOPICS for k in range(12)]
    p.write_text("\n".join(lines), encoding="utf-8")
    pool, report = text_pool.load_pool(p)
    assert report["rows_unmatched"] == 0
    assert len(pool[3]) == 12


def test_loader_accepts_alternative_field_names(tmp_path):
    rows = [{"prompt": f"Write a short channel message about: {t}", "completion": f"x {k} {t}"}
            for t in text_pool.TOPICS for k in range(12)]
    pool, report = text_pool.load_pool(_write(tmp_path, "c.jsonl", rows))
    assert report["rows_unmatched"] == 0


def test_loader_reads_the_prefixed_field_names_a_real_export_uses(tmp_path):
    """Adaption's JSONL ships enhanced_prompt / enhanced_completion, not prompt / completion."""
    rows = [{"enhanced_prompt": f"Write a short channel message about: {t}",
             "enhanced_completion": f"status {k} on {t}"}
            for t in text_pool.TOPICS for k in range(12)]
    pool, report = text_pool.load_pool(_write(tmp_path, "c.jsonl", rows))
    assert report["rows_unmatched"] == 0
    assert report["rows_used"] == 96


def test_an_exact_field_name_wins_over_a_prefixed_one(tmp_path):
    rows = [{"prompt": f"Write a short channel message about: {t}",
             "enhanced_prompt": "Write a short channel message about: logo design and visual branding",
             "completion": f"right {k}", "enhanced_completion": f"wrong {k}"}
            for t in text_pool.TOPICS for k in range(12)]
    pool, _ = text_pool.load_pool(_write(tmp_path, "c.jsonl", rows))
    flat = [m for v in pool.values() for m in v]
    assert all(m.startswith("right") for m in flat)
    assert len(pool[0]) == 12, "topic should come from `prompt`, not `enhanced_prompt`"


def test_unmatched_rows_are_counted_and_dropped_not_guessed(tmp_path):
    rows = _rows() + [{"instruction": "Write about: quantum basket weaving", "response": "nope"}]
    pool, report = text_pool.load_pool(_write(tmp_path, "c.jsonl", rows), strict=False)
    assert report["rows_unmatched"] == 1
    assert all("nope" not in m for v in pool.values() for m in v)
    assert report["unmatched_examples"]


def test_a_thin_topic_is_refused_in_strict_mode(tmp_path):
    rows = [r for r in _rows() if text_pool.TOPICS[5] not in r["instruction"]]
    with pytest.raises(ValueError, match="fewer than"):
        text_pool.load_pool(_write(tmp_path, "c.jsonl", rows))
    pool, report = text_pool.load_pool(_write(tmp_path, "c.jsonl", rows), strict=False)
    assert "warning" in report and len(pool[5]) == 0


def test_loaded_pool_drives_the_generator_end_to_end(tmp_path):
    pool, _ = text_pool.load_pool(_write(tmp_path, "c.jsonl", _rows(30)))
    inst = bench_gen.generate_instance(edges=((0, 5, 0.4),), seed=11, n_messages=900,
                                       text_pool=pool)
    assert inst["ground_truth"] == {("agent00", "agent05")}
    assert all(isinstance(e[3], str) and e[3] for e in inst["events"])
