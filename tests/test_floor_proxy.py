"""Tests for the label-free symbolisation-quality proxies.

Two things need defending here. The first is arithmetic: the Calinski-Harabasz and Davies-Bouldin
implementations have to agree with sklearn on the sparse path they exist for, and the rank
correlation and the threshold search have to be right. The second is the trap the whole module is
built around: an adjusted Rand index between clusterings of DIFFERENT point sets is not defined,
and the shortcuts that avoid restricting to the shared points are wrong in a measurable way.
Padding the undrawn points with one shared label collapses the index towards zero even where the
true stability is 1.0. Measuring agreement against the full-data fit instead of against another
resample gives a different number again, by enough to change a verdict, though the sweep showed its
direction is corpus-size dependent so no direction is asserted. Both are pinned here as numbers.

Run: python -m pytest tests -q
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bench_gen
import floor_proxy as fx

from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------------- the trap

def test_ari_refuses_two_labellings_that_do_not_cover_the_same_points():
    with pytest.raises(ValueError, match="same points"):
        fx.ari([0, 1, 0, 1], [0, 1, 0])


def test_ari_on_shared_uses_exactly_the_points_both_clusterings_saw():
    a = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0}
    b = {2: 5, 3: 5, 4: 9, 5: 9, 6: 9}
    got, n_shared = fx.ari_on_shared(a, b)
    assert n_shared == 3
    want = adjusted_rand_score([a[i] for i in (2, 3, 4)], [b[i] for i in (2, 3, 4)])
    assert got == pytest.approx(want)


def test_ari_on_shared_refuses_a_pair_with_nothing_in_common():
    with pytest.raises(ValueError, match="share"):
        fx.ari_on_shared({0: 0, 1: 1}, {7: 0, 8: 1})


def test_padding_the_points_a_resample_missed_wrecks_the_stability_estimate():
    """The measured size of the trap, in the direction it actually goes.

    Two clusterings see half the corpus each and agree substantially where they overlap. Restricted
    to the points both saw, that agreement is recovered. Padding the rest with one shared "not
    sampled" label does not merely bias the number: the two undrawn sets only partly coincide, so
    the pad group disagrees with itself across the two labellings and the index collapses towards
    zero. Either way it is not a stability estimate, which is the whole point.
    """
    rng = np.random.default_rng(0)
    n = 1200
    truth = rng.integers(0, 4, size=n)
    a_idx = rng.choice(n, size=n // 2, replace=False)
    b_idx = rng.choice(n, size=n // 2, replace=False)
    noisy = np.where(rng.random(n) < 0.5, truth, rng.integers(0, 4, size=n))

    honest, n_shared = fx.ari_on_shared({int(i): int(truth[i]) for i in a_idx},
                                        {int(i): int(noisy[i]) for i in b_idx})
    pad_a, pad_b = np.full(n, -1), np.full(n, -1)
    pad_a[a_idx] = truth[a_idx]
    pad_b[b_idx] = noisy[b_idx]
    padded = fx.ari(pad_a, pad_b)

    assert 0 < n_shared < n
    assert honest > 0.15, "the shared-point restriction should recover the real agreement"
    assert abs(padded - honest) > 0.1, (
        "padding must not be allowed to pass for the restricted index; honest %.4f padded %.4f"
        % (honest, padded))


def test_agreement_with_the_full_fit_is_a_different_statistic_and_is_reported_as_one():
    """Stability measured against the clustering fitted on ALL the data is not stability between
    resamples, and the two can differ by enough to change a verdict.

    No DIRECTION is asserted, deliberately. It looks like it must read high, since every resample
    shares most of its messages with the reference and the reference saw everything, and on this
    1500-message corpus it does. The 8000-message sweep found per-level gaps from -0.040 to +0.012,
    so the effect is corpus-size dependent and claiming a one-way bias would be overreaching. What
    is defensible, and what this pins, is that the two readings are not interchangeable and that
    the gap is reported rather than buried.
    """
    texts, _ = _corpus(0.8, n_messages=1500)
    out = fx.bootstrap_stability(texts, k=8, seed=7919, n_boot=4,
                                 base=fx.codes(fx.TopicSymboliser(k=8, seed=0)
                                               .fit(texts).transform(texts)))
    assert out["vs_base_inflation"] == pytest.approx(
        out["vs_base_mean"] - out["full_mean"], abs=1e-9)
    assert abs(out["vs_base_inflation"]) > 0.01


def test_the_shared_point_restriction_agrees_with_the_unrestricted_full_data_reading():
    """Where the danger is NOT. A drawn message gets the same label whether the refitted model is
    asked about it alone or about the whole corpus, so restricting to the shared points is simply
    the same statistic on a subsample. That is why `full` is the headline and the restriction is
    cheap rather than contentious."""
    texts, _ = _corpus(0.5, n_messages=1500)
    out = fx.bootstrap_stability(texts, k=8, seed=7919, n_boot=4)
    assert 0 < out["shared_points_mean"] < len(texts)
    assert out["shared_mean"] == pytest.approx(out["full_mean"], abs=0.05)


def test_the_mis_readings_are_diagnostics_and_never_thresholdable_proxies():
    for name in ("boot_naive_pad_mean", "boot_vs_base_mean", "naive_pad_mean", "vs_base_mean"):
        assert name not in fx.PROXY_NAMES
        assert name not in fx.HIGHER_IS_BETTER


def test_every_reported_proxy_declares_which_direction_is_better():
    """The sweep iterates PROXY_NAMES and thresholds by direction, so a proxy added to the battery
    without a declared direction would be silently dropped from the evaluation."""
    texts, _ = _corpus(0.5, n_messages=600)
    got = fx.all_proxies(texts, k=4, seed=0, n_boot=2, seed_runs=2, sample_size=300)
    assert set(got["proxies"]) == set(fx.PROXY_NAMES) == set(fx.HIGHER_IS_BETTER)


# ----------------------------------------------------------------------------- arithmetic

def _toy_sparse(seed=0, n=240, k=3, vocab=12):
    """A sparse matrix with real cluster structure, built through sklearn so no scipy import is
    needed here: scipy is not declared in requirements.txt even though sklearn pulls it in."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, k, size=n)
    docs = []
    for c in lab:
        words = rng.choice([f"w{c}_{j}" for j in range(vocab)], size=8)
        words = list(words) + [f"shared{int(rng.integers(3))}"]
        docs.append(" ".join(words))
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(min_df=1).fit_transform(docs), lab


def test_calinski_harabasz_matches_sklearn_on_the_sparse_path():
    x, lab = _toy_sparse()
    assert fx.calinski_harabasz(x, lab) == pytest.approx(
        calinski_harabasz_score(x.toarray(), lab), rel=1e-8)


def test_davies_bouldin_matches_sklearn_on_the_sparse_path():
    x, lab = _toy_sparse(seed=1)
    assert fx.davies_bouldin(x, lab) == pytest.approx(
        davies_bouldin_score(x.toarray(), lab), rel=1e-8)


def test_the_internal_indices_agree_between_the_sparse_and_dense_paths():
    x, lab = _toy_sparse(seed=2)
    dense = x.toarray().astype(float)
    assert fx.calinski_harabasz(dense, lab) == pytest.approx(
        fx.calinski_harabasz(x, lab), rel=1e-8)
    assert fx.davies_bouldin(dense, lab) == pytest.approx(
        fx.davies_bouldin(x, lab), rel=1e-8)


def test_internal_indices_return_none_when_there_is_only_one_cluster():
    x, _ = _toy_sparse()
    lab = np.zeros(x.shape[0], dtype=int)
    assert fx.calinski_harabasz(x, lab) is None
    assert fx.davies_bouldin(x, lab) is None
    assert fx.silhouette_cosine(x, lab) is None
    assert fx.centroid_cosine(x, lab) == (None, None)


def test_centroid_cosine_is_zero_when_clusters_share_no_vocabulary():
    texts = ["aa bb cc"] * 50 + ["dd ee ff"] * 50
    x = TfidfVectorizer(min_df=1).fit_transform(texts)
    mean, mx = fx.centroid_cosine(x, [0] * 50 + [1] * 50)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert mx == pytest.approx(0.0, abs=1e-9)


def test_centroid_cosine_rises_as_clusters_share_vocabulary():
    shared = TfidfVectorizer(min_df=1).fit_transform(
        ["aa bb cc"] * 50 + ["aa bb dd"] * 50)
    mean_shared, _ = fx.centroid_cosine(shared, [0] * 50 + [1] * 50)
    assert mean_shared > 0.4


def test_size_entropy_is_one_for_a_balanced_alphabet_and_lower_for_a_skewed_one():
    assert fx.size_entropy([0] * 50 + [1] * 50 + [2] * 50) == pytest.approx(1.0)
    assert fx.size_entropy([0] * 148 + [1] + [2]) < 0.2


def test_codes_keeps_an_unplaced_message_as_its_own_symbol():
    c = fx.codes(["t1", fx.SILENT, "t1", "t2"])
    assert len(set(c.tolist())) == 3
    assert c[0] == c[2] and c[1] != c[0]


# ----------------------------------------------------------------------------- evaluation helpers

def test_spearman_is_one_under_any_increasing_transform():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
    assert fx.spearman(x, [v ** 3 for v in x]) == pytest.approx(1.0)
    assert fx.spearman(x, [-v for v in x]) == pytest.approx(-1.0)


def test_spearman_averages_tied_ranks():
    # ranks of [1,2,2,3] are [1, 2.5, 2.5, 4]; against [1,2,3,4] this is a known value
    got = fx.spearman([1, 2, 2, 3], [1, 2, 3, 4])
    assert got == pytest.approx(0.9486832980505138, rel=1e-9)


def test_spearman_ignores_missing_values_and_gives_up_when_too_few_remain():
    assert fx.spearman([1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 99.0]) == pytest.approx(1.0)
    assert fx.spearman([1.0, np.nan, np.nan], [1.0, 2.0, 3.0]) is None


def test_best_threshold_separates_a_clean_case_without_error():
    v = [0.9, 0.8, 0.85, 0.2, 0.1, 0.15]
    above = [True, True, True, False, False, False]
    cut = fx.best_threshold(v, above, higher_is_better=True)
    assert cut["errors"] == 0 and cut["balanced_accuracy"] == 1.0
    assert 0.2 < cut["threshold"] < 0.8


def test_best_threshold_honours_a_lower_is_better_proxy():
    v = [1.0, 1.2, 1.1, 5.0, 6.0, 5.5]        # small value means a good symboliser
    above = [True, True, True, False, False, False]
    cut = fx.best_threshold(v, above, higher_is_better=False)
    assert cut["errors"] == 0
    assert 1.2 < cut["threshold"] < 5.0


def test_best_threshold_gives_up_when_one_class_is_absent():
    assert fx.best_threshold([0.1, 0.2, 0.3], [True, True, True], True) is None


def test_the_positive_class_is_above_the_floor_so_a_wrong_pass_is_a_false_positive():
    c = fx.confusion([0.9], [False], threshold=0.5, higher_is_better=True)
    assert c["fp"] == 1 and c["tp"] == 0
    c = fx.confusion([0.1], [True], threshold=0.5, higher_is_better=True)
    assert c["fn"] == 1


# ----------------------------------------------------------------------------- end to end

def _corpus(overlap, n_messages=1200, seed=900):
    inst = bench_gen.generate_instance(n_agents=12, n_messages=n_messages, edges=(),
                                       shared_topic=0.55, vocab_overlap=overlap, seed=seed)
    ev = sorted(inst["events"], key=lambda e: e[1])
    return [e[3] for e in ev], [e[2] for e in ev]


def test_all_proxies_is_deterministic_for_a_fixed_seed():
    texts, _ = _corpus(0.5, n_messages=600)
    a = fx.all_proxies(texts, k=4, seed=0, n_boot=2, seed_runs=2, sample_size=300)["proxies"]
    b = fx.all_proxies(texts, k=4, seed=0, n_boot=2, seed_runs=2, sample_size=300)["proxies"]
    assert a == b


def test_all_proxies_never_sees_a_label():
    """The integrity property: the battery takes messages, and nothing else.

    If a label ever reached it the whole result would be circular, so the signature is the
    guarantee and this test pins it.
    """
    import inspect
    params = set(inspect.signature(fx.all_proxies).parameters)
    assert params == {"texts", "k", "seed", "n_boot", "seed_runs", "sample_size"}
    assert not any("label" in p or "truth" in p or "topic" in p for p in params)


def test_the_headline_proxies_rank_a_hard_corpus_below_an_easy_one():
    """The claim the sweep is built to quantify, at a size a test can afford."""
    easy_t, easy_truth = _corpus(0.0)
    hard_t, hard_truth = _corpus(0.9)
    easy = fx.all_proxies(easy_t, k=8, seed=0, n_boot=3, seed_runs=3, sample_size=600)
    hard = fx.all_proxies(hard_t, k=8, seed=0, n_boot=3, seed_runs=3, sample_size=600)

    easy_ari = adjusted_rand_score(easy_truth, fx.codes(easy["base_symbols"]))
    hard_ari = adjusted_rand_score(hard_truth, fx.codes(hard["base_symbols"]))
    assert easy_ari > 0.8 > hard_ari, "the generator's difficulty knob must still bite"

    for name in ("boot_stability_full_mean", "agree_char", "silhouette_cosine"):
        assert easy["proxies"][name] > hard["proxies"][name], name
    # lower is better for these two
    for name in ("centroid_cosine_mean", "davies_bouldin"):
        assert easy["proxies"][name] < hard["proxies"][name], name


def test_the_hash_symboliser_stays_a_null_at_every_difficulty():
    """Cross-symboliser agreement is only evidence if a meaning-free symboliser scores zero."""
    for overlap in (0.0, 0.9):
        texts, _ = _corpus(overlap, n_messages=800)
        res = fx.all_proxies(texts, k=8, seed=0, n_boot=2, seed_runs=2, sample_size=400)
        assert abs(res["proxies"]["agree_hash_null"]) < 0.05


# ----------------------------------------------------------------------------- hygiene

@pytest.mark.parametrize("name", ["floor_proxy.py", "floor_proxy_sweep.py", "floor_proxy_real.py",
                                  "tests/test_floor_proxy.py"])
def test_sources_are_pure_ascii_with_no_dashes_of_the_long_kind(name):
    raw = (pathlib.Path(__file__).resolve().parents[1] / name).read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 127]
    assert not bad, "non-ASCII byte at offset %d in %s" % (bad[0][0], name)
    text = raw.decode("ascii")
    for code in (0x2014, 0x2013):      # em dash, en dash: written as code points to stay ASCII
        assert chr(code) not in text
