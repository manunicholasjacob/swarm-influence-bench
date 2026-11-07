"""Tests for the generator's labels. Every one of these is a regression test for a real defect.

Run: python -m pytest tests -q    (needs only numpy, not scikit-learn)
"""
import collections
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
