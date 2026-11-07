"""Tests for the turn-taking factorial's own measurement and analysis code.

The detector runs are far too slow for a test suite, so what is tested here is everything between
the generator and the table: the realised speaker statistics, the cell list, and the effect
decomposition that the paper's additivity claim rests on. Hand-built sequences and hand-built cell
records, so the expected values are exact rather than approximate.

Run: python -m pytest tests -q
"""
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

# The module imports bench_eval, which imports the private estimator at import time; skip cleanly
# if it is absent so the generator tests still run from a bare clone.
tf = pytest.importorskip("turntaking_factorial",
                         reason="needs swarm-prep and swarm-sprint on sys.path")


def ev(speakers):
    """Events in the generator's shape, carrying only the speaker column the statistics read."""
    return [(s, float(i), 0, "x") for i, s in enumerate(speakers)]


def test_repeat_rate_counts_consecutive_messages_from_the_same_agent():
    # a a b b b c -> repeats at positions 1, 3, 4 of 5 adjacent pairs
    assert tf.speaker_stats(ev("aabbbc"))["repeat_rate"] == pytest.approx(3 / 5)


def test_return_after_one_requires_a_different_agent_in_between():
    """aba is a return; aaa is a repeat and must not be counted twice.

    Getting this wrong is the easy mistake, and it matters here because the knob is calibrated
    against two rates the audit measured as mutually exclusive.
    """
    s = tf.speaker_stats(ev("abab"))
    assert s["return_after_one_rate"] == pytest.approx(1.0)
    assert s["repeat_rate"] == 0.0
    s = tf.speaker_stats(ev("aaaa"))
    assert s["return_after_one_rate"] == 0.0
    assert s["return_any_rate"] == pytest.approx(1.0)
    assert s["repeat_rate"] == pytest.approx(1.0)


def test_the_loose_return_definition_is_reported_alongside_the_strict_one():
    """They differ exactly where a repeat sits in between, which is the whole point of keeping both."""
    s = tf.speaker_stats(ev("aab"))
    assert s["return_any_rate"] == 0.0
    s = tf.speaker_stats(ev("aaa"))
    assert s["return_any_rate"] == 1.0 and s["return_after_one_rate"] == 0.0


def test_block_share_cv_is_zero_for_a_perfectly_stationary_channel():
    s = tf.speaker_stats(ev("ab" * 100), n_blocks=10)
    assert s["median_block_share_cv"] == pytest.approx(0.0, abs=1e-9)
    assert s["n_speakers"] == 2


def test_block_share_cv_is_large_when_agents_speak_only_in_one_block():
    """Two of three agents confined to a single block of ten is the extreme the statistic has to
    notice. Each confined agent has shares [1, 0 x 9], so its CV is 3.0 and the median is too."""
    s = tf.speaker_stats(ev("a" * 10 + "b" * 10 + "c" * 80), n_blocks=10)
    assert s["median_block_share_cv"] > 2.0
    assert s["n_speakers"] == 3


def test_short_sequences_do_not_raise_or_return_a_nan():
    """Regression: np.array_split into more blocks than messages returns empty blocks, whose
    share is a nan, and one nan poisons the median for the whole cell."""
    for seq in ("", "a", "ab"):
        got = tf.speaker_stats(ev(seq))
        assert 0.0 <= got["repeat_rate"] <= 1.0
        assert got["median_block_share_cv"] == got["median_block_share_cv"], seq


def test_the_cell_list_covers_the_full_factorial_plus_the_calibrated_arm():
    cs = tf.cells()
    fac = [c for c in cs if c["arm"] == "factorial"]
    cal = [c for c in cs if c["arm"] == "calibrated"]
    assert len(fac) == len(tf.TURN_LEVELS) * len(tf.DRIFT_LEVELS)
    assert len(cal) == 3, "the fourth corner of the calibrated 2 x 2 is the factorial's own origin"
    assert {(c["p_repeat"], c["p_return"], c["activity_drift"]) for c in cs}.__len__() == len(cs)


def test_the_factorial_contains_the_real_data_knob_values_and_a_both_off_origin():
    cs = tf.cells()
    assert any(c["p_repeat"] == 0.36 and c["p_return"] == 0.38 and c["activity_drift"] == 0.0
               for c in cs), "turn-taking alone at the real-data values"
    assert any(c["p_repeat"] == 0.36 and c["p_return"] == 0.38 and c["activity_drift"] == 0.9
               for c in cs), "both knobs at their top values"
    assert any(c["p_repeat"] == 0.0 and c["p_return"] == 0.0 and c["activity_drift"] == 0.0
               for c in cs), "the origin, which must match the earlier drift sweep"


def test_every_cell_is_a_pure_null():
    assert tf.GEN["edges"] == () and tf.GEN["shared_topic"] == 0.0


def _cell(ci, arm, turn, drift, shift_fwer, permute_fwer, n=200):
    """A summarised cell record of the shape summarise() produces, with chosen rates."""
    rec = {"cell": ci, "arm": arm, "turn_taking": turn, "p_repeat": 0.0, "p_return": 0.0,
           "activity_drift": drift, "n_replicates": n}
    for null, f in (("shift", shift_fwer), ("permute", permute_fwer)):
        k = round(f * n)
        rec[null] = {"fired": k, "of": n, "fwer": round(k / n, 4),
                     "wilson95": list(tf.bench_eval.wilson(k, n)), "mean_edges": 0.0}
    return rec


def _factorial_corners(base, turn_only, drift_only, both):
    return [_cell(0, "factorial", "off", 0.0, base, base),
            _cell(1, "factorial", "real_nominal", 0.0, turn_only, turn_only),
            _cell(2, "factorial", "off", 0.9, drift_only, drift_only),
            _cell(3, "factorial", "real_nominal", 0.9, both, both)]


def test_the_decomposition_reports_both_main_effects_and_the_additive_prediction():
    d = tf.decompose(_factorial_corners(0.05, 0.25, 0.35, 0.55))["factorial"]["by_surrogate"]["shift"]
    assert d["turn_taking_effect"] == pytest.approx(0.20, abs=1e-4)
    assert d["drift_effect"] == pytest.approx(0.30, abs=1e-4)
    assert d["additive_prediction"] == pytest.approx(0.55, abs=1e-4)
    assert d["both_on_minus_additive_prediction"] == pytest.approx(0.0, abs=1e-4)
    assert d["additive_prediction_inside_both_on_wilson95"] is True


def test_the_decomposition_flags_a_both_on_rate_the_additive_prediction_misses():
    """A superadditive corner is the interaction claim, and it is only a claim if the additive
    prediction falls outside the corner's own interval."""
    d = tf.decompose(_factorial_corners(0.05, 0.10, 0.15, 0.90))["factorial"]["by_surrogate"]["shift"]
    assert d["both_on_minus_additive_prediction"] > 0.6
    assert d["additive_prediction_inside_both_on_wilson95"] is False


def test_the_decomposition_names_which_single_knob_did_more():
    f = tf.decompose(_factorial_corners(0.05, 0.40, 0.10, 0.50))
    assert f["factorial"]["by_surrogate"]["shift"]["larger_single_effect"] == "turn_taking"
    f = tf.decompose(_factorial_corners(0.05, 0.10, 0.40, 0.50))
    assert f["factorial"]["by_surrogate"]["shift"]["larger_single_effect"] == "drift"
    f = tf.decompose(_factorial_corners(0.05, 0.20, 0.20, 0.50))
    assert f["factorial"]["by_surrogate"]["shift"]["larger_single_effect"] == "tied"


def test_a_partially_filled_run_decomposes_without_crashing():
    """Regression: the incremental dump used to run before every corner had a replicate, and the
    missing rate came through as None straight into the arithmetic."""
    cells = _factorial_corners(0.05, 0.25, 0.35, 0.55)
    cells[3]["shift"]["fwer"] = None
    out = tf.decompose(cells)["factorial"]["by_surrogate"]
    assert "shift" not in out and "permute" in out


def test_published_sweep_agreement_is_false_until_thirty_replicates_exist():
    """The check must not report success from a short run that happens to agree so far."""
    cells = [_cell(0, "factorial", "off", 0.0, 0.0, 0.0), _cell(1, "factorial", "off", 0.9, 0.0, 0.0)]
    rows = [{"cell": 0, "rep": r, "shift_fired": r < 2, "permute_fired": r < 1} for r in range(10)]
    got = {(a["turn_taking"], a["activity_drift"]): a
           for a in tf.check_published(cells, rows, first_n=30)}
    assert got[("off", 0.0)]["matches"] is False
    assert got[("off", 0.0)]["first_n"] == 10


def test_published_sweep_agreement_is_true_only_on_the_exact_published_counts():
    cells = [_cell(0, "factorial", "off", 0.0, 0.0, 0.0)]
    want = tf.PUBLISHED[("off", 0.0)]
    rows = [{"cell": 0, "rep": r, "shift_fired": r < want["shift"],
             "permute_fired": r < want["permute"]} for r in range(30)]
    assert tf.check_published(cells, rows, first_n=30)[0]["matches"] is True
    rows[29]["shift_fired"] = True
    assert tf.check_published(cells, rows, first_n=30)[0]["matches"] is False


def test_published_sweep_agreement_also_reports_the_published_interval():
    """A one-replicate difference in 30 is not a contradiction, and the record has to let a reader
    see that rather than only a boolean. The published count's own Wilson interval is written out
    next to this run's 200-replicate rate."""
    cells = [_cell(0, "factorial", "off", 0.0, 0.055, 0.105)]
    rows = [{"cell": 0, "rep": r, "shift_fired": False, "permute_fired": False} for r in range(30)]
    got = tf.check_published(cells, rows, first_n=30)[0]
    assert got["published_wilson95"]["shift"] == list(tf.bench_eval.wilson(2, 30))
    assert got["this_run_fwer"]["shift"] == 0.055
    assert got["this_run_fwer_inside_published_wilson95"]["shift"] is True
