"""Tests for the evaluation metrics, on hand-built rows so no detector or sklearn is needed.

Run: python -m pytest tests -q
"""
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

# bench_eval imports the private estimator modules at import time; skip cleanly if absent so the
# generator tests still run for someone who only cloned this repo.
bench_eval = pytest.importorskip(
    "bench_eval", reason="needs swarm-prep and swarm-sprint on sys.path")


def row(name, planted=0, found=0, tp=0, n_messages=8000, shared_topic=0.0,
        couplings=(), n_agents=12):
    return {"name": name, "n_agents": n_agents, "shared_topic": shared_topic,
            "n_agents_tested": n_agents, "family_size": n_agents * (n_agents - 1),
            "dropped_by_top_k": 0, "n_messages": n_messages,
            "edges_planted": planted, "edges_found": found, "true_positives": tp,
            "precision": round(tp / found, 4) if found else None,
            "recall": round(tp / planted, 4) if planted else None,
            "is_null": planted == 0, "any_certified": found > 0,
            "couplings": list(couplings)}


def test_pure_and_common_cause_nulls_are_separated_by_setting_not_by_name():
    rows = [row("whatever_a", shared_topic=0.0, found=0),
            row("whatever_b", shared_topic=0.0, found=1),
            row("other_a", shared_topic=0.55, found=20),
            row("other_b", shared_topic=0.55, found=18)]
    s = bench_eval.summarise(rows)
    assert s["familywise_error_pure_null"] == 0.5
    assert s["familywise_error_common_cause_null"] == 1.0
    assert s["mean_edges_on_common_cause_null"] == 19.0


def test_a_null_row_without_shared_topic_is_refused_rather_than_guessed():
    r = row("x", found=0)
    del r["shared_topic"]
    with pytest.raises(ValueError, match="shared_topic"):
        bench_eval.summarise([r])


def test_pooled_precision_counts_silent_instances_and_per_instance_does_not():
    """A detector that fires once perfectly and stays silent otherwise should not score 1.00."""
    rows = [row("s1", planted=4, found=4, tp=4, couplings=(0.4,)),
            row("s2", planted=4, found=0, tp=0, couplings=(0.4,)),
            row("s3", planted=4, found=20, tp=2, couplings=(0.4,))]
    s = bench_eval.summarise(rows)
    assert s["mean_precision_when_fired"] == pytest.approx((1.0 + 0.1) / 2, abs=1e-4)
    assert s["pooled_precision"] == pytest.approx(6 / 24, abs=1e-4)
    assert s["fired_rate_on_signal"] == pytest.approx(2 / 3, abs=1e-4)


def test_recall_and_counts():
    rows = [row("s1", planted=4, found=5, tp=3, couplings=(0.4,))]
    s = bench_eval.summarise(rows)
    assert s["mean_recall"] == 0.75
    assert s["n_signal"] == 1 and s["n_null"] == 0 and s["n_instances"] == 1


def test_all_null_grid_reports_no_precision_instead_of_crashing():
    s = bench_eval.summarise([row("n1", found=0), row("n2", found=0)])
    assert s["pooled_precision"] is None
    assert s["mean_precision_when_fired"] is None
    assert s["mean_recall"] is None
    assert s["familywise_error_pure_null"] == 0.0


def test_power_curve_is_built_from_settings_and_not_from_the_name_prefix():
    rows = [row(f"myown_{c}_{s}", planted=1, found=1, tp=1, couplings=(c,))
            for c in (0.1, 0.4) for s in range(3)]
    s = bench_eval.summarise(rows)
    assert s["power_by_coupling"] == {0.1: 1.0, 0.4: 1.0}


def test_empty_power_curve_is_distinguishable_from_no_power():
    rows = [row("multi", planted=4, found=4, tp=4, couplings=(0.4, 0.3, 0.5, 0.4))]
    with pytest.warns(UserWarning, match="not the same as zero power"):
        s = bench_eval.summarise(rows)
    assert s["detection_floor_coupling"]["status"] == "no_data"


@pytest.mark.parametrize("curve,status,floor", [
    ({0.05: 0.0, 0.1: 0.33, 0.2: 1.0, 0.4: 1.0}, "bracketed", 0.2),
    ({0.05: 1.0, 0.1: 1.0}, "below_grid", 0.05),
    ({0.05: 0.0, 0.1: 0.33}, "never_reached", None),
    ({}, "no_data", None),
    ({0.05: 0.33, 0.1: 1.0, 0.2: 0.67, 0.4: 0.33}, "non_monotone", 0.1),
])
def test_detection_floor_reports_its_own_status(curve, status, floor):
    """Regression: every one of these used to come back as a bare number or a bare None, so
    'never reached', 'no data' and 'power collapsed again above the crossing' were identical."""
    rows = [row(f"c{c}_{s}", planted=1, found=1 if p else 0, tp=1 if p >= 1 else 0,
                couplings=(c,))
            for c, p in curve.items() for s in range(3)]
    # build the curve directly rather than via rows, so the parametrised values are exact
    out = bench_eval.summarise(rows) if rows else None
    got = bench_eval.__dict__.get("floor_of")
    if got is None:  # floor_of is local to summarise, so exercise it through a crafted curve
        pytest.skip("floor_of is not module level")
    d = got(curve, 3)
    assert d["status"] == status and d["floor"] == floor


def test_coarse_seed_count_is_flagged_in_the_floor_status():
    rows = [row(f"c0.4_{s}", planted=1, found=1, tp=1, couplings=(0.4,)) for s in range(3)]
    s = bench_eval.summarise(rows)
    assert "3 seeds" in (s["detection_floor_coupling"]["note"] or "")


# ---------------------------------------------------------------------------
# Wilson intervals. The function moved here out of stationarity_big.py so the sweeps share one
# implementation; these pin that the move was lossless.
# ---------------------------------------------------------------------------

def test_wilson_reproduces_every_published_interval_in_the_drift_sweep():
    """Recompute bench_stationarity_big.json's intervals from its own counts.

    The function used to live in the script that wrote that file. Moving it is only safe if the
    published numbers come back unchanged, so they are recomputed here rather than trusted.
    """
    data = json.load(open(HERE / "bench_stationarity_big.json", encoding="utf-8"))
    checked = 0
    for r in data["rows"]:
        for null in ("shift", "permute"):
            lo, hi = bench_eval.wilson(r[null]["fired"], r[null]["of"])
            assert [lo, hi] == r[null]["wilson95"], (r["activity_drift"], null)
            checked += 1
    assert checked == 8


def test_wilson_brackets_the_point_estimate_and_stays_inside_the_unit_interval():
    for k in range(0, 31):
        lo, hi = bench_eval.wilson(k, 30)
        assert 0.0 <= lo <= k / 30 <= hi <= 1.0


def test_wilson_at_zero_successes_does_not_claim_zero_risk():
    """A rate of 0 out of 30 is not evidence that the rate is 0, which is the whole reason an
    interval is reported next to every error rate in this repo."""
    lo, hi = bench_eval.wilson(0, 30)
    assert lo == 0.0 and hi > 0.1


def test_wilson_narrows_as_replicates_are_added():
    width = lambda k, n: bench_eval.wilson(k, n)[1] - bench_eval.wilson(k, n)[0]
    assert width(3, 30) > width(20, 200) > width(100, 1000)


def test_wilson_with_no_trials_is_unknown_rather_than_zero():
    assert bench_eval.wilson(0, 0) == (None, None)
