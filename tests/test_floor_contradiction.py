"""Tests for the experiment that resolves the edge-count against floor-proxy contradiction.

Three things need defending here.

The first is that this script measures the same thing the calibration sweep measured. It rebuilds
the turn grid and refits the symboliser, so if either differs by a line the permutation counts it
compares against the committed sweep are counts of something else. `turn_grid` is therefore checked
against the sweep's own copy on the same instance, element for element, rather than by inspection.

The second is the arithmetic of the summaries: the edge count at a given alpha, the z summary that
goes beside it, and the rule that the published cut is APPLIED and never refitted. A function that
quietly refit the threshold on the new rows would turn an out-of-mechanism test into an in-sample
one and would always pass.

The third is the finding itself, pinned as a number against the committed sweep so that a later
change to the sweep cannot leave the claim standing on nothing: across the whole overlap sweep, no
below-floor instance on the structure family certifies anything close to the count the audit
recorded for open-chat.

Run: python -m pytest tests -q
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bench_gen
import floor_contradiction as fc
import floor_proxy as fx

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _instance(n_messages=600, **kw):
    kw.setdefault("n_agents", 6)
    kw.setdefault("edges", ((0, 2, 0.5), (1, 4, 0.4)))
    kw.setdefault("shared_topic", 0.55)
    inst = bench_gen.generate_instance(seed=11, n_messages=n_messages, **kw)
    return sorted(inst["events"], key=lambda e: e[1]), inst["ground_truth"]


def _symbols(events):
    """A cheap deterministic symbolisation: the generator's own topic, as a symbol string.

    The tests below are about the grid, the counting and the summaries, not about clustering, so
    paying for a k-means fit in every one of them would buy nothing.
    """
    return ["T%d" % e[2] for e in events]


# ----------------------------------------------------------------------------- same measurement

def test_turn_grid_is_identical_to_the_one_the_calibration_sweep_uses():
    """The reproduction check in the output is only meaningful if the grids agree exactly."""
    import floor_proxy_sweep as sweep
    events, _ = _instance()
    syms = _symbols(events)
    coded, A, sil, chan, agents = fc.turn_grid(events, syms)
    w_coded, w_A, w_sil, w_chan = sweep.turn_grid(events, syms)
    assert A == w_A and sil == w_sil
    assert chan == w_chan
    assert sorted(coded) == sorted(w_coded)
    for a in coded:
        assert np.array_equal(coded[a], w_coded[a]), a
    assert set(agents) == set(w_coded), "the agent list must be the grid's own key set"


def test_modal_context_is_identical_to_the_one_the_calibration_sweep_uses():
    import floor_proxy_sweep as sweep
    events, _ = _instance()
    _, _, sil, chan, _ = fc.turn_grid(events, _symbols(events))
    assert np.array_equal(fc.modal_ctx(chan, sil), sweep.modal_ctx(chan, sil))


def test_detect_is_deterministic_at_a_fixed_seed():
    events, gt = _instance()
    syms = _symbols(events)
    a = fc.detect(events, syms, gt, n_sur=10, seed=3)
    b = fc.detect(events, syms, gt, n_sur=10, seed=3)
    assert a == b


def test_detect_measures_both_surrogates_on_the_one_grid():
    """Permutation and shift must see the same symbolisation, or the comparison is of two runs."""
    events, gt = _instance()
    out = fc.detect(events, _symbols(events), gt, n_sur=10, seed=0)
    assert out["pairwise"]["n_pairs"] == out["shift"]["n_pairs"] == out["conditional"]["n_pairs"]
    assert out["n_ticks"] > 0
    assert out["planted"] == len(gt)


# ----------------------------------------------------------------------------- the summaries

def test_edge_stats_counts_exactly_the_pairs_below_alpha():
    tests = {("a", "b"): {"te": 0.1, "z": 4.0, "p": 0.01},
             ("b", "a"): {"te": 0.0, "z": -1.0, "p": 0.9},
             ("a", "c"): {"te": 0.2, "z": 0.0, "p": 0.05},     # not strictly below alpha
             ("c", "a"): {"te": 0.3, "z": 2.0, "p": 0.049}}
    got = fc._edge_stats(tests, {("a", "b"), ("x", "y")})
    assert got["edges"] == 2
    assert got["true_positives"] == 1
    assert got["precision"] == 0.5
    assert got["recall"] == 0.5
    assert got["n_pairs"] == 4
    assert got["mean_z"] == pytest.approx(1.25)
    assert got["max_z"] == 4.0
    assert got["frac_z_negative"] == 0.25
    assert got["frac_z_exactly_zero"] == 0.25


def test_edge_stats_reports_no_precision_when_nothing_was_certified():
    tests = {("a", "b"): {"te": 0.0, "z": 0.1, "p": 0.5}}
    got = fc._edge_stats(tests, {("a", "b")})
    assert got["edges"] == 0 and got["precision"] is None and got["recall"] == 0.0


def test_edge_stats_reports_no_recall_on_a_null_instance():
    tests = {("a", "b"): {"te": 0.1, "z": 4.0, "p": 0.001}}
    got = fc._edge_stats(tests, set())
    assert got["edges"] == 1 and got["recall"] is None and got["precision"] == 0.0


# ----------------------------------------------------------------------------- do not refit

def _row(cell, ari, ss, lsa, recall=1.0):
    return {"cell": cell, "true_ari": ari,
            "proxies": {"seed_stability_min": ss, "agree_lsa": lsa},
            "detection": {"conditional": {"recall": recall}}}


def test_cuts_out_of_mechanism_applies_the_given_threshold_and_does_not_refit():
    """The whole point of the test is that the cut is the published one.

    These rows are perfectly separable at 0.5, so a function that refit would score zero errors.
    Applied at the published 0.6228 it must instead report the two above-floor rows it misses.
    """
    cuts = {"seed_stability_min": {"threshold": 0.6227555, "higher_is_better": True},
            "agree_lsa": {"threshold": 0.6461735, "higher_is_better": True}}
    rows = [_row("a", 0.95, 0.55, 0.55), _row("b", 0.90, 0.58, 0.58),
            _row("c", 0.20, 0.30, 0.30), _row("d", 0.10, 0.20, 0.20)]
    got = fc.cuts_out_of_mechanism(rows, cuts, "unit")
    rec = got["seed_stability_min"]
    assert rec["applied_threshold"] == 0.6227555
    assert rec["confusion"]["fn"] == 2 and rec["confusion"]["fp"] == 0
    assert [m["cell"] for m in rec["misclassified"]] == ["a", "b"]
    # and the operational counterweight: rows the cut condemns where detection in fact held
    held = rec["called_below_floor_but_detection_held"]
    assert held["n_called_below"] == 4 and held["n_full_recall"] == 4

    best = fx.best_threshold([r["proxies"]["seed_stability_min"] for r in rows],
                             [r["true_ari"] >= fc.STATED_FLOOR for r in rows], True)
    assert best["errors"] == 0, "these rows ARE separable, which is what makes the test bite"
    assert best["threshold"] != rec["applied_threshold"]


def test_pinned_ari_spread_names_the_worst_cell_it_found():
    rows = [_row("good", 0.99, 0.90, 0.90), _row("bad", 0.97, 0.31, 0.40),
            _row("ignored", 0.10, 0.05, 0.05)]
    got = fc.pinned_ari_spread(rows, lo=0.9)
    assert got["n"] == 2
    assert got["seed_stability_min"]["min"] == 0.31
    assert got["seed_stability_min"]["max"] == 0.90
    assert got["seed_stability_min"]["range"] == pytest.approx(0.59)
    assert got["seed_stability_min"]["worst_cell"] == "bad"


def test_by_group_averages_within_a_group_and_keeps_the_key():
    rows = [{"v": 1, "true_ari": 1.0, "detection": {"pairwise": {
                "edges": 10, "true_positives": 4, "recall": 1.0, "mean_z": 0.1, "max_z": 9.0}}},
            {"v": 1, "true_ari": 0.5, "detection": {"pairwise": {
                "edges": 20, "true_positives": 2, "recall": 0.5, "mean_z": 0.3, "max_z": 7.0}}}]
    got = fc.by_group(rows, "v")
    assert len(got) == 1
    assert got[0]["v"] == 1 and got[0]["n"] == 2
    assert got[0]["true_ari"] == 0.75
    assert got[0]["pairwise_edges"] == 15.0
    assert got[0]["pairwise_recall"] == 0.75


# ----------------------------------------------------------------------------- generator contract

def test_the_five_agent_cell_plants_four_edges_that_the_generator_accepts():
    """`SMALL_EDGES` exists because the baseline edge list names agents 5, 7, 9 and 11."""
    assert len(fc.SMALL_EDGES) == len(fc.STRUCT_EDGES)
    with pytest.raises(ValueError, match="outside"):
        bench_gen.generate_instance(n_agents=5, n_messages=200, edges=fc.STRUCT_EDGES)
    inst = bench_gen.generate_instance(n_agents=5, n_messages=400, edges=fc.SMALL_EDGES, seed=1)
    assert len(inst["ground_truth"]) == 4


def test_run_cell_measures_the_proxies_and_the_detection_on_one_symbolisation():
    row = fc.run_cell("unit", inst_seed=7, sym_seed=0, n_sur=8, n_boot=2, seed_runs=2,
                      n_agents=6, n_messages=500, edges=((0, 2, 0.5),), shared_topic=0.55,
                      vocab_overlap=0.0)
    assert 0.0 <= row["true_ari"] <= 1.0
    assert row["proxies"]["seed_stability_min"] is not None
    assert "pairwise" in row["detection"] and "shift" in row["detection"]
    assert "conditional" in row["detection"]
    assert row["diagnostics"]["n_messages"] == 500
    assert row["config"]["vocab_overlap"] == 0.0


def test_run_cell_without_proxies_still_reports_the_true_ari_and_both_surrogates():
    row = fc.run_cell("unit", inst_seed=7, sym_seed=0, n_sur=8, proxies=False,
                      n_agents=6, n_messages=500, edges=((0, 2, 0.5),), shared_topic=0.55,
                      vocab_overlap=0.0)
    assert "proxies" not in row
    assert "conditional" not in row["detection"], "skipped when the proxies are skipped"
    assert row["detection"]["pairwise"]["n_pairs"] == row["detection"]["shift"]["n_pairs"]


# ----------------------------------------------------------------------------- the finding

def _sweep():
    p = ROOT / "floor_proxy_sweep.json"
    if not p.exists():
        pytest.skip("floor_proxy_sweep.json is not in the tree")
    return json.loads(p.read_text(encoding="utf-8"))


def test_no_below_floor_benchmark_cell_certifies_as_many_edges_as_open_chat_did():
    """H1's verdict, pinned.

    H1 proposed that a below-floor instance WITH planted structure might still certify many
    permutation edges, which would have made the audit's 18 edges on open-chat compatible with a
    below-floor symboliser. On the committed sweep it does not: the permutation count falls with
    the true ARI on the family that HAS planted edges. The two classes very nearly separate, and
    they overlap on exactly one boundary cell, at vocabulary overlap 0.80 where the mean ARI sits
    on the floor itself. The most any below-floor cell certifies is 12, against the 18 recorded for
    open-chat.
    """
    sweep = _sweep()
    struct = [r for r in sweep["rows"] if r["family"] == "structure"]
    below = [r for r in struct if r["true_ari"] < fc.STATED_FLOOR]
    above = [r for r in struct if r["true_ari"] >= fc.STATED_FLOOR]
    assert len(below) == 10 and len(above) == 17
    max_below = max(r["detection"]["pairwise"]["edges"] for r in below)
    min_above = min(r["detection"]["pairwise"]["edges"] for r in above)
    assert max_below == 12 and min_above == 11
    # the single overlapping cell, and it is the boundary level
    straddle = [r for r in below if r["detection"]["pairwise"]["edges"] >= min_above]
    assert [r["vocab_overlap"] for r in straddle] == [0.8]
    # the finding: open-chat certifies more than EVERY below-floor cell, boundary one included
    assert fc.REAL["open-chat-village"]["tfidf_permute_edges"] > max_below
    # and the edge count is a strong floor statistic on the benchmark, which is what makes the
    # real-channel disagreement a problem rather than a curiosity
    cut = fx.best_threshold([r["detection"]["pairwise"]["edges"] for r in struct],
                            [r["true_ari"] >= fc.STATED_FLOOR for r in struct], True)
    assert cut["errors"] == 1 and cut["threshold"] == 12.5


def test_the_two_real_channels_sit_on_opposite_sides_of_the_benchmark_relation():
    """The contradiction, stated as the two numbers that make it.

    Open-chat certifies more edges than any below-floor benchmark cell while the proxies place it
    below the floor; main-village certifies fewer than any above-floor benchmark cell while the
    proxies place it above. One channel cannot be explained by a drifting threshold, and two
    channels failing in OPPOSITE directions cannot be explained by one at all.
    """
    sweep = _sweep()
    struct = [r for r in sweep["rows"] if r["family"] == "structure"]
    above = [r for r in struct if r["true_ari"] >= fc.STATED_FLOOR]
    below = [r for r in struct if r["true_ari"] < fc.STATED_FLOOR]
    cut = sweep["evaluation"]["seed_stability_min"]["floor_by_stated_ari"]["threshold"]

    oc, mv = fc.REAL["open-chat-village"], fc.REAL["main-village"]
    assert oc["seed_stability_min"] < cut < mv["seed_stability_min"]
    assert oc["tfidf_permute_edges"] > max(r["detection"]["pairwise"]["edges"] for r in below)
    assert mv["tfidf_permute_edges"] < min(r["detection"]["pairwise"]["edges"] for r in above)


def test_the_surrogate_that_fires_is_the_other_one_on_each_real_channel():
    """Why an edge count cannot be read as a symboliser score on these two channels.

    The two nulls disagree in opposite directions on the two channels, which is a property of their
    speaker sequences rather than of their text.
    """
    oc, mv = fc.REAL["open-chat-village"], fc.REAL["main-village"]
    assert mv["tfidf_shift_edges"] > mv["tfidf_permute_edges"]
    assert oc["tfidf_permute_edges"] > oc["tfidf_shift_edges"]
    # and the meaning-free null stays a null under both, on both channels
    for ch in (oc, mv):
        assert ch["hash_permute_edges"] == 0
        assert ch["hash_shift_edges"] <= 1


def test_open_chats_certified_edges_do_not_survive_conditioning_or_replication():
    """H3's evidence, as the audit recorded it.

    Open-chat's 18 pairwise edges fall to 0 under conditioning, its largest z is 4.43 against
    main-village's 13.5, its surrogate sits above the observed statistic for 65 percent of pairs,
    and the edge set does not replicate across windows or across halves of the same window. Two
    edges that survive conditioning are more evidence than eighteen that do not.
    """
    oc, mv = fc.REAL["open-chat-village"], fc.REAL["main-village"]
    assert oc["conditional_edges"] == 0 and mv["conditional_edges"] == 2
    assert oc["conditional_max_z"] < mv["conditional_max_z"]
    assert oc["conditional_mean_z"] < 0 < mv["conditional_mean_z"]
    assert oc["conditional_frac_z_negative"] > 0.6
    assert oc["permute_replication_jaccard"] < 0.3
    assert oc["permute_split_half_jaccard"] < 0.2
    # and the count is strongly k dependent, which a stable structure claim would not be
    assert oc["permute_edges_by_k"]["4"] < oc["permute_edges_by_k"]["8"] \
        < oc["permute_edges_by_k"]["16"]


def test_the_benchmark_null_family_has_open_chats_exact_signature():
    """Many pairwise edges and zero conditional edges is the common-cause null's signature.

    It is what the generator produces when there is NO influence at all but every agent tracks the
    same drifting topic, and it is what open-chat shows. The structure family, which has planted
    edges, shows pairwise many AND conditional non-zero, which is what main-village shows.
    """
    sweep = _sweep()
    nulls = [r for r in sweep["rows"]
             if r["family"] == "common_cause_null" and r["true_ari"] >= 0.8]
    assert nulls
    assert all(r["detection"]["conditional"]["edges"] == 0 for r in nulls)
    assert min(r["detection"]["pairwise"]["edges"] for r in nulls) >= 10
    struct = [r for r in sweep["rows"]
              if r["family"] == "structure" and r["true_ari"] >= 0.8]
    assert all(r["detection"]["conditional"]["edges"] > 0 for r in struct)


def test_the_real_channel_block_carries_counts_only():
    """The corpus is gated: nothing but numbers may be copied out of it."""
    for village, rec in fc.REAL.items():
        if village == "source_files":
            continue
        for key, value in rec.items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    assert isinstance(k2, str) and len(k2) <= 4, (village, key, k2)
                    assert v2 is None or isinstance(v2, (int, float)), (village, key, k2)
            else:
                assert value is None or isinstance(value, (int, float)), (village, key)


# ----------------------------------------------------------------------------- the result

def _result():
    p = ROOT / "floor_contradiction.json"
    if not p.exists():
        pytest.skip("floor_contradiction.json is not in the tree; run floor_contradiction.py")
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_refit_reproduced_the_committed_sweep_cell_for_cell():
    """If this ever fails, every comparison in the file is a comparison of two different runs."""
    rec = _result()["e1_edges_across_the_floor"]["reproduction_of_committed_sweep"]
    assert rec["available"] and rec["n_cells_compared"] == 54
    assert rec["max_abs_edge_diff"] == 0 and rec["max_abs_ari_diff"] == 0.0
    assert rec["mismatched_cells"] == []


def test_the_edge_count_moves_with_common_cause_at_perfect_symbolisation():
    """E2's finding, which is what makes the edge count unusable as a symboliser score.

    At zero vocabulary overlap the true ARI is 1.0 and both voting proxies read 1.0, so the
    symbolisation is not merely good but identical in quality across these rows. The permutation
    count still runs from about 25 to about 73 as the common-cause knob is raised, so the count is
    reporting how much dependence the channel contains, not how well its text was symbolised.
    """
    rows = _result()["e2_common_cause_at_fixed_difficulty"]["by_shared_topic_at_overlap_0.00"]
    perfect = [r for r in rows if r["true_ari"] == 1.0 and r["seed_stability_min"] == 1.0]
    assert len(perfect) >= 3, "the pinned rows must really be at perfect symbolisation"
    edges = [r["pairwise_edges"] for r in perfect]
    assert min(edges) < 30 < max(edges)
    assert max(edges) / min(edges) > 2.5


def test_the_published_cut_flips_with_common_cause_at_nearly_fixed_ari():
    """The same confound seen from the proxy side, at the hard end of the vocabulary knob."""
    rows = _result()["e2_common_cause_at_fixed_difficulty"]["by_shared_topic_at_overlap_0.80"]
    cut = 0.6227555
    aris = [r["true_ari"] for r in rows]
    assert max(aris) - min(aris) < 0.20, "the true ARI must be roughly pinned for this to bite"
    says_below = [r["shared_topic"] for r in rows if r["seed_stability_min"] < cut]
    says_above = [r["shared_topic"] for r in rows if r["seed_stability_min"] >= cut]
    assert says_below and says_above, "the verdict must flip on the common-cause knob alone"
    assert max(says_below) < min(says_above), "and flip monotonically, not at random"


def test_a_wrong_k_breaks_the_published_cuts_in_both_directions():
    """E4's finding. The cuts scored a balanced accuracy of 1.00 in calibration."""
    pc = _result()["e4_k_mismatch"]["published_cuts_on_these_rows"]
    ss, lsa = pc["seed_stability_min"], pc["agree_lsa"]
    assert ss["confusion"]["errors"] == 7 and ss["confusion"]["balanced_accuracy"] == 0.75
    assert lsa["confusion"]["errors"] == 9 and lsa["confusion"]["balanced_accuracy"] == 0.5714
    # seed stability fails only one way: it condemns symbolisations that in fact worked
    assert ss["confusion"]["fp"] == 0 and ss["confusion"]["fn"] == 7
    assert all(m["says"] == "below_floor" for m in ss["misclassified"])
    assert all(m["conditional_recall"] == 1.0 for m in ss["misclassified"])
    # cross-symboliser agreement fails both ways, including calling ARI 0.25 above the floor
    assert lsa["confusion"]["fp"] == 3
    assert min(m["true_ari"] for m in lsa["misclassified"]
               if m["says"] == "above_floor") < 0.30


def test_a_below_floor_instance_can_certify_a_healthy_number_of_edges():
    """H1, which the vocabulary mechanism refutes and the k mechanism confirms.

    At a true topic count of 24 symbolised at k=8 the true ARI is about 0.27, far below the floor,
    and yet the permutation surrogate certifies about 22 edges, which is what a perfectly
    symbolised channel certifies, and the conditional detector keeps every planted edge. So a
    below-floor ARI does NOT imply a near-zero edge count. It only implied one on the single
    mechanism the calibration sweep used.
    """
    by_k = {r["k_topics"]: r for r in _result()["e4_k_mismatch"]["by_true_k"]}
    bad, good = by_k[24], by_k[8]
    assert bad["true_ari"] < 0.30 < fc.STATED_FLOOR
    assert bad["pairwise_edges"] > 20
    assert abs(bad["pairwise_edges"] - good["pairwise_edges"]) < 3
    assert bad["conditional_recall"] == 1.0 and bad["pairwise_tp"] == 4.0


def test_the_k_sweep_reproduces_the_published_reading_at_the_published_k():
    """The k sweep is only evidence about the published verdict if it reproduces it at k=8.

    Same loader, same window, same stability protocol of five seeds, so the k=8 row has to equal
    the number floor_proxy_real.json published to the digit. It does, for both channels and for
    both voting proxies, which is what licenses reading the other rows as the same statistic at a
    different k.
    """
    real = ROOT / "floor_proxy_real.json"
    if not real.exists():
        pytest.skip("floor_proxy_real.json is not in the tree")
    pub = {c["village"]: c["proxies"] for c in
           json.loads(real.read_text(encoding="utf-8"))["channels"] if c.get("reachable")}
    for ch in _result()["e5_real_channel_text_shape_and_k"]["channels"]:
        if not ch.get("reachable"):
            continue
        at8 = next(r for r in ch["by_k"] if r["k"] == fc.K)
        assert at8["seed_runs"] == 5, "the published control uses five stability seeds"
        for name in ("seed_stability_min", "agree_lsa"):
            assert at8[name] == pytest.approx(pub[ch["village"]][name], abs=5e-6), \
                (ch["village"], name)


def test_both_real_channels_are_more_stably_symbolised_at_k4_than_at_the_audits_k8():
    """E5. The verdict the floor control published on open-chat is an artifact of k.

    At the audit's k=8 open-chat reads 0.345 against a published cut of 0.6228 and is condemned.
    At k=4 the same channel reads 0.992, which is the highest reading either channel produces at
    any k on the grid, so on this evidence open-chat is the BETTER symbolised of the two.
    """
    best = _result()["e5_real_channel_text_shape_and_k"]["k_that_reads_best"]
    for village in ("main-village", "open-chat-village"):
        rec = best[village]
        assert rec["k"] == 4
        assert rec["crosses_published_cut"] is True
        assert rec["seed_stability_min"] > rec["seed_stability_min_at_audit_k"]
    oc, mv = best["open-chat-village"], best["main-village"]
    assert oc["seed_stability_min_at_audit_k"] < 0.6227555 < oc["seed_stability_min"]
    assert oc["seed_stability_min"] > mv["seed_stability_min"]


def test_the_short_message_mechanism_does_not_explain_open_chat():
    """E5's other half, which rules out the rival explanation rather than supporting one.

    E3 showed that short messages drag both voting proxies below their cuts. If that were the
    reason open-chat was condemned, open-chat would be the channel with the short messages. It is
    the other way round: 27 percent of main-village's messages are under five tokens against 2
    percent of open-chat's, and main-village is the channel the control cleared.
    """
    chans = {c["village"]: c for c in
             _result()["e5_real_channel_text_shape_and_k"]["channels"] if c.get("reachable")}
    mv = chans["main-village"]["text_shape"]
    oc = chans["open-chat-village"]["text_shape"]
    assert mv["share_under_5_tokens"] > 0.25
    assert oc["share_under_5_tokens"] < 0.05
    assert oc["tokens_mean"] > mv["tokens_mean"]


def test_the_real_channel_section_leaks_no_text_and_no_identifier():
    """The corpus is gated, so the shape of this section is a privacy contract, not a convenience."""
    rec = _result()["e5_real_channel_text_shape_and_k"]
    allowed = {"village", "room_name", "rooms_with_that_name", "window_days", "window_messages",
               "window_span_hours", "village_messages_total", "reachable", "text_shape", "by_k",
               "reason"}
    for ch in rec["channels"]:
        assert set(ch) <= allowed, set(ch) - allowed
        for value in ch.get("text_shape", {}).values():
            if isinstance(value, dict):
                assert all(isinstance(v, (int, float)) for v in value.values())
            else:
                assert isinstance(value, (int, float))
        for row in ch.get("by_k", []):
            assert all(isinstance(v, (int, float)) for v in row.values())


# ----------------------------------------------------------------------------- hygiene

@pytest.mark.parametrize("name", ["floor_contradiction.py",
                                  "tests/test_floor_contradiction.py"])
def test_sources_are_pure_ascii_with_no_dashes_of_the_long_kind(name):
    raw = (ROOT / name).read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 127]
    assert not bad, "non-ASCII byte at offset %d in %s" % (bad[0][0], name)
    text = raw.decode("ascii")
    for code in (0x2014, 0x2013):      # em dash, en dash: written as code points to stay ASCII
        assert chr(code) not in text


@pytest.mark.parametrize("name", ["floor_contradiction.py",
                                  "tests/test_floor_contradiction.py"])
def test_sources_hard_code_nobodys_drive_layout(name):
    text = (ROOT / name).read_text(encoding="ascii")
    # built from code points so that this test file does not trip over its own needles, the same
    # trick the dash check above uses
    drive = chr(67) + chr(58)                      # C:
    slash, back = chr(47), chr(92)
    for bad in (drive + back, drive.lower() + back, drive + slash, drive.lower() + slash,
                slash + "Users" + slash, slash + "home" + slash):
        assert bad not in text, "%s contains an absolute path %r" % (name, bad)
