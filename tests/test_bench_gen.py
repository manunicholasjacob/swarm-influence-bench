"""Tests for the generator's labels. Every one of these is a regression test for a real defect.

Run: python -m pytest tests -q    (needs only numpy, not scikit-learn)
"""
import collections
import hashlib
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bench_gen


def _lag_ratio(events, leader, follower, lag=1):
    """P(follower speaks at t+lag on the leader's topic) / P(follower speaks at all).

    A labelled edge has to show up as a lag-specific excess, not merely as a busy follower.
    """
    agents = [a for a, _, _, _ in events]
    topics = [tp for _, _, tp, _ in events]
    base = agents.count(follower) / len(agents)
    hits = opps = 0
    for i in range(lag, len(agents)):
        if agents[i - lag] == leader:
            opps += 1
            hits += agents[i] == follower and topics[i] == topics[i - lag]
    return (hits / opps) / base if opps and base else 0.0


def test_planted_edge_is_present_at_the_labelled_lag():
    inst = bench_gen.generate_instance(edges=((0, 5, 0.4),), n_messages=6000, seed=1)
    assert ("agent00", "agent05") in inst["ground_truth"]
    assert _lag_ratio(inst["events"], "agent00", "agent05") > 2.0


def test_ground_truth_matches_the_requested_edges():
    edges = ((0, 5, 0.4), (2, 9, 0.5))
    inst = bench_gen.generate_instance(edges=edges, seed=2)
    assert inst["ground_truth"] == {("agent00", "agent05"), ("agent02", "agent09")}


def test_zero_coupling_edges_are_not_labelled():
    inst = bench_gen.generate_instance(edges=((0, 5, 0.0),), seed=3)
    assert inst["ground_truth"] == set()


def test_realised_coupling_is_reported_and_below_the_label_for_a_shared_leader():
    """Only one agent speaks per tick, so a leader's second follower cannot also land at the
    stated lag. The label would overstate it, which is why the realised value is reported."""
    inst = bench_gen.generate_instance(edges=((0, 5, 0.4), (0, 7, 0.4)), n_messages=8000, seed=4)
    r = inst["realised_coupling"]
    assert set(r) == {"0->5", "0->7"}
    assert r["0->5"] > r["0->7"], "the first follower should realise more of its label"
    assert r["0->7"] < 0.4


def test_unshared_edge_realises_close_to_its_label():
    inst = bench_gen.generate_instance(edges=((2, 9, 0.5),), n_messages=8000, seed=5)
    assert abs(inst["realised_coupling"]["2->9"] - 0.5) < 0.05


def test_vocabulary_does_not_encode_the_topic():
    """Regression: words were built as syllables + str(topic), so one regex recovered the
    generator's own label from every message and the symbolisation step was free."""
    inst = bench_gen.generate_instance(edges=(), k_topics=8, n_messages=500, seed=6)
    words = {w for _, _, _, text in inst["events"] for w in text.split()}
    assert words, "no text generated"
    assert not any(re.search(r"\d", w) for w in words)


def test_vocabularies_are_disjoint_across_topics():
    v = bench_gen._vocab(8, seed=7)
    flat = [w for ws in v.values() for w in ws]
    assert len(flat) == len(set(flat))
    assert len(v) == 8


def test_generation_is_reproducible_for_a_fixed_seed():
    a = bench_gen.generate_instance(edges=((0, 5, 0.4),), n_messages=800, seed=8)
    b = bench_gen.generate_instance(edges=((0, 5, 0.4),), n_messages=800, seed=8)
    assert a["events"] == b["events"]
    assert a["realised_coupling"] == b["realised_coupling"]


def test_different_seeds_give_different_streams():
    a = bench_gen.generate_instance(edges=(), n_messages=800, seed=9)
    b = bench_gen.generate_instance(edges=(), n_messages=800, seed=10)
    assert a["events"] != b["events"]


def test_grid_cells_are_not_prefixes_of_one_another():
    """Regression: cells shared a base seed, so a 2000-message instance was an exact prefix of an
    8000-message one at the same seed, which understated the variance of anything averaged."""
    specs = {s["name"]: s for s in bench_gen.grid(seeds=1)}
    short = dict(specs["volume2000_s0"]); short.pop("name")
    long_ = dict(specs["volume8000_s0"]); long_.pop("name")
    a = bench_gen.generate_instance(**short)["events"]
    b = bench_gen.generate_instance(**long_)["events"]
    assert a != b[:len(a)]


def test_grid_seeds_are_all_distinct():
    seeds = [s["seed"] for s in bench_gen.grid(seeds=3)]
    assert len(seeds) == len(set(seeds))


def test_pure_null_has_no_planted_edges_and_no_drifting_common_cause():
    specs = [s for s in bench_gen.grid(seeds=1) if s["name"].startswith("nullpure")]
    assert specs
    for s in specs:
        assert s["edges"] == ()
        assert s["shared_topic"] == 0.0


def test_common_cause_null_has_no_edges_but_a_shared_topic():
    specs = [s for s in bench_gen.grid(seeds=1) if s["name"].startswith("nullcommon")]
    assert specs
    for s in specs:
        assert s["edges"] == ()
        assert s["shared_topic"] > 0.0


def test_shared_topic_creates_temporal_coherence_not_marginal_skew():
    """What `shared_topic` buys is that consecutive messages agree, which is the common cause a
    pairwise test cannot tell from influence.

    It does NOT concentrate the marginal topic distribution, and the direction is the reverse of
    what one might assume: the drifting current topic wanders over the whole alphabet, so a high
    shared_topic gives a nearly uniform marginal, while at shared_topic=0 agents fall back on home
    topics that the busiest speakers dominate. The two null families therefore differ in marginal
    skew as well as in coherence, which is a property of this design worth knowing about.
    """
    coherence = lambda i: float(np.mean([a == b for a, b in zip(
        [tp for _, _, tp, _ in i["events"]][:-1], [tp for _, _, tp, _ in i["events"]][1:])]))
    lo = bench_gen.generate_instance(edges=(), shared_topic=0.0, n_messages=4000, seed=11)
    hi = bench_gen.generate_instance(edges=(), shared_topic=0.9, n_messages=4000, seed=11)
    assert coherence(hi) > 2 * coherence(lo)
    top = lambda i: collections.Counter(tp for _, _, tp, _ in i["events"]).most_common(1)[0][1]
    assert top(lo) > top(hi), "documented: the pure null is the more marginally skewed of the two"


@pytest.mark.parametrize("bad,msg", [
    (((0, 0, 0.5),), "self-edge"),
    (((0, 99, 0.5),), "outside"),
    (((0, 5, 1.7),), "probability"),
    (((0, 5, -0.1),), "probability"),
])
def test_invalid_edges_are_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        bench_gen.generate_instance(edges=bad, seed=12)


def test_one_speaker_per_tick():
    inst = bench_gen.generate_instance(edges=((0, 5, 0.4), (0, 7, 0.4)), n_messages=1000, seed=13)
    times = [t for _, t, _, _ in inst["events"]]
    assert times == sorted(times)
    assert len(times) == len(inst["events"])


def test_activity_drift_makes_volumes_non_stationary():
    def cv(d):
        inst = bench_gen.generate_instance(edges=(), activity_drift=d, n_messages=6000,
                                           n_blocks=10, seed=14)
        ag = [a for a, _, _, _ in inst["events"]]
        blocks = np.array_split(np.array(ag), 10)
        shares = [collections.Counter(b).get("agent00", 0) / len(b) for b in blocks]
        return float(np.std(shares) / (np.mean(shares) + 1e-9))
    assert cv(0.9) > cv(0.0)


# ---------------------------------------------------------------------------
# Turn-taking (p_repeat, p_return)
# ---------------------------------------------------------------------------

def _speaker_rates(**kw):
    """Realised repeat rate, return-after-exactly-one-other rate, and median block-share CV."""
    inst = bench_gen.generate_instance(n_agents=12, n_messages=4000, edges=(),
                                       shared_topic=0.0, **kw)
    a = np.asarray([x for x, _, _, _ in inst["events"]])
    repeat = float((a[1:] == a[:-1]).mean())
    ret = float(((a[2:] == a[:-2]) & (a[2:] != a[1:-1])).mean())
    cvs = []
    for who in np.unique(a):
        share = np.array([float((b == who).mean()) for b in np.array_split(a, 10)])
        if share.mean() > 0:
            cvs.append(float(share.std() / share.mean()))
    return repeat, ret, float(np.median(cvs))


# Digests of the generator's output taken from the commit BEFORE p_repeat and p_return existed,
# over three configurations that between them exercise planted edges, shared leaders, drift,
# burstiness, a non-default lag, a non-default topic count and vocabulary overlap. The whole point
# of the new knobs is that the factorial compares one generator against itself at different
# settings rather than two different generators, and that claim is only worth as much as this test.
_PRE_TURN_TAKING_DIGESTS = [
    (dict(edges=((0, 5, 0.4),), n_messages=800, seed=8),
     "a959e62b3365121b4f47f8720e5625bbfd3b86a52a13c83a4bfcfb90121c84de"),
    (dict(edges=(), n_messages=1200, shared_topic=0.0, activity_drift=0.9, n_blocks=10, seed=14),
     "4a74d21e144153c52e975ca243208762c91087d098d3bbdbd409967c879005e1"),
    (dict(edges=((0, 2, 0.4), (0, 5, 0.3), (1, 6, 0.5)), n_messages=600, k_topics=5, n_agents=7,
          burstiness=0.4, reaction_lag=3, vocab_overlap=0.5, seed=21),
     "fc5e77d788186ce5abdfc09dc0d9d3bc2d0a404ce03e07f09c5e3cf35dadfa85"),
]


def _digest(inst):
    return hashlib.sha256((repr(inst["events"]) + repr(sorted(inst["ground_truth"]))
                           + repr(sorted(inst["realised_coupling"].items()))).encode()).hexdigest()


@pytest.mark.parametrize("cfg,digest", _PRE_TURN_TAKING_DIGESTS)
@pytest.mark.parametrize("knobs", [{}, dict(p_repeat=0.0, p_return=0.0)])
def test_turn_taking_at_zero_reproduces_the_earlier_generator_bit_for_bit(cfg, digest, knobs):
    """Zero means OFF, not a chain with zero probabilities.

    If the branch drew a random number and then threw it away, the stream would still be a valid
    null but every number measured before the knob existed would have to be remeasured, and the
    drift sweep in bench_stationarity_big.json could not be compared with the new cells. Both the
    default and an explicit zero have to land on the same bytes as the generator that preceded
    them, which is what these digests pin.
    """
    assert _digest(bench_gen.generate_instance(**cfg, **knobs)) == digest


def test_p_repeat_raises_the_immediate_repeat_rate():
    base_rep, _, _ = _speaker_rates(seed=30)
    rep, _, _ = _speaker_rates(p_repeat=0.5, seed=30)
    assert rep > 1.5 * base_rep and rep > 0.5


def test_p_return_raises_the_return_after_one_rate():
    _, base_ret, _ = _speaker_rates(seed=30)
    _, ret, _ = _speaker_rates(p_return=0.5, seed=30)
    assert ret > 2 * base_ret and ret > 0.4


def test_the_two_turn_taking_knobs_move_different_statistics():
    """p_repeat must not be a slow way of writing p_return. Each one raises its own rate and
    pushes the other DOWN, because a tick spent repeating is a tick not spent returning."""
    rep_only, ret_at_rep, _ = _speaker_rates(p_repeat=0.5, seed=30)
    rep_at_ret, ret_only, _ = _speaker_rates(p_return=0.5, seed=30)
    assert rep_only > rep_at_ret
    assert ret_only > ret_at_rep


def test_the_return_branch_requires_a_different_intervening_speaker():
    """Regression: the obvious version of this chain runs away.

    If `p_return` can pick the speaker from two back when that is also the last speaker, then one
    repeat makes both branches repeat, and at p_repeat 0.36 with p_return 0.38 the realised repeat
    rate came out at 0.678 instead of anywhere near the 0.36 the knob is calibrated against. The
    branch is conditional for that reason, and the bound here is below the runaway value, so a
    regression to the unconditional form fails this test rather than quietly inflating the cell
    the experiment calls real.
    """
    rep, _, _ = _speaker_rates(p_repeat=0.36, p_return=0.38, seed=30)
    assert rep < 0.58


def test_turn_taking_and_activity_drift_are_separate_knobs():
    """The whole reason the knob exists: the real speaker sequence carries turn-taking and
    non-stationary activity at once, and shuffling it destroys both. Turn-taking alone must raise
    the repeat rate without making activity non-stationary, and drift alone must do the reverse."""
    base_rep, _, base_cv = _speaker_rates(seed=30)
    turn_rep, _, turn_cv = _speaker_rates(p_repeat=0.36, p_return=0.38, seed=30)
    drift_rep, _, drift_cv = _speaker_rates(activity_drift=0.9, seed=30)
    assert turn_rep > 1.5 * base_rep, "turn-taking should raise the repeat rate"
    assert drift_cv > 2 * turn_cv, "drift should dominate the block-share CV"
    assert turn_cv < drift_cv


def test_activity_drift_also_raises_the_repeat_rate_so_the_statistics_are_not_separable():
    """Documented, because it is easy to read the knob as owning the statistic.

    The knobs are independent; the statistics they move are not. Concentrating activity into fewer
    agents per block makes repeats more likely all by itself, so a measured repeat rate cannot be
    read back as a turn-taking setting and any cell has to report what it realised.
    """
    base_rep, _, _ = _speaker_rates(seed=30)
    drift_rep, _, _ = _speaker_rates(activity_drift=0.9, seed=30)
    assert drift_rep > base_rep


def test_turn_taking_does_not_disturb_the_planted_edge_label():
    """Reactions are scheduled by the leader speaking and bypass the chain, so an unshared edge
    should still realise close to its stated coupling with turn-taking switched on."""
    inst = bench_gen.generate_instance(edges=((2, 9, 0.5),), n_messages=8000,
                                       p_repeat=0.36, p_return=0.38, seed=40)
    assert inst["ground_truth"] == {("agent02", "agent09")}
    assert abs(inst["realised_coupling"]["2->9"] - 0.5) < 0.06


def test_turn_taking_knobs_are_recorded_in_the_config():
    cfg = bench_gen.generate_instance(edges=(), n_messages=200, p_repeat=0.1, p_return=0.2,
                                      seed=41)["config"]
    assert cfg["p_repeat"] == 0.1 and cfg["p_return"] == 0.2
    assert bench_gen.generate_instance(edges=(), n_messages=200,
                                       seed=41)["config"]["p_repeat"] == 0.0


@pytest.mark.parametrize("bad,msg", [
    (dict(p_repeat=1.2), "p_repeat"),
    (dict(p_repeat=-0.1), "p_repeat"),
    (dict(p_return=1.2), "p_return"),
    (dict(p_return=-0.1), "p_return"),
    (dict(p_repeat=0.6, p_return=0.6), "exceeds 1"),
])
def test_invalid_turn_taking_settings_are_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        bench_gen.generate_instance(edges=(), n_messages=200, seed=42, **bad)


def test_turn_taking_still_puts_one_speaker_in_each_tick():
    inst = bench_gen.generate_instance(edges=((0, 5, 0.4),), n_messages=1000,
                                       p_repeat=0.36, p_return=0.38, seed=43)
    times = [t for _, t, _, _ in inst["events"]]
    assert times == sorted(times) and len(times) == 1000
