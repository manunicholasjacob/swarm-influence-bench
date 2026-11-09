# SwarmInfluence-Bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24064140.svg)](https://doi.org/10.5281/zenodo.24064140)

A benchmark for **coordination and influence detectors** on multi-agent chat, where the true
influence graph is known by construction.

Built out of an audit which found that the standard information-theoretic recipe, applied to real AI
Village logs, certified 11 influence edges on one channel of which **none** survived a valid
surrogate. The reason nobody had noticed is that there was nowhere to get calibrated ground truth.

Companion audit: **[swarm-influence-audit](https://github.com/manunicholasjacob/swarm-influence-audit)**.

Author: Manu Nicholas Jacob, Independent Researcher (manunicholasjacob@gmail.com).

## What makes the labels trustworthy, and where they fall short

**The generator decides, in code, who speaks at which tick, on which topic, and who is reacting to
whom.** That is the ground truth and it is exact for the edges it names. Surface text is a separate,
replaceable layer drawn from disjoint per-topic vocabularies, so the generator runs offline with no
model and no API. Swapping in an LLM renderer changes realism without touching a single label.

A detector sees only `(agent, time, text)` and must symbolise the text itself. That claim used to be
false: words were built as syllables followed by the topic index, so one regular expression recovered
the generator's own label from every message. The vocabularies are now built as a single pool and
partitioned across topics, so they stay disjoint while nothing inside a word says which slice it
landed in.

Two honest limits on the labels, both measured rather than argued:

**Requested coupling is not realised coupling.** Only one agent speaks per tick, so when a leader has
two followers whose reaction coins both land, the second is pushed to the next tick and no longer
arrives at `reaction_lag`. `generate_instance` therefore returns `realised_coupling` beside the
request. A labelled 0.3 realises about 0.18 when it shares its leader with another follower, while an
unshared edge realises its label to within 0.03. Use the realised value when you index anything by
coupling.

**A planted edge does not only couple its own pair, and the label set does not say so.** Because a
queued reaction consumes the tick, every other agent becomes less likely to speak immediately after
the leader. On the standard four-edge instance the probability that a non-follower speaks in the tick
after the leader falls to between **0.19 and 0.83** of its own base rate, against **0.94 to 1.05** on
an otherwise identical instance with no planted edges. That displacement is a real lag-1 dependency
and `ground_truth` does not contain it, so **precision on this benchmark is a lower bound for any
detector**. What the benchmark supports is the comparison between two detectors, not the absolute
precision of either. `tests/test_bench_gen.py` measures the effect so you can see its size.

## Swapping in a realistic text layer

The text layer is the last thing the generator decides and the only replaceable one, so a renderer
swap changes realism without moving a single structural label. `tests/test_text_pool.py` asserts
exactly that: generate twice at the same seed, once with the vocabulary sampler and once with a
message pool, and the `(agent, time, topic)` stream is identical while only the text differs.

```python
import bench_gen, text_pool

pool, report = text_pool.load_pool("agent_team_channel_messages.jsonl")
print(report["per_topic"])          # how many messages landed in each topic
inst = bench_gen.generate_instance(n_agents=12, n_messages=8000,
                                   edges=((0, 5, 0.4),), text_pool=pool, seed=1)
inst["config"]["text_renderer"]     # "pool" rather than "vocab"
```

`load_pool` expects an instruction dataset whose instructions name the topic verbatim, as in
`"Write a short channel message about: outreach email drafting"`, and accepts JSONL, a JSON array or
CSV with the common field spellings. Rows whose topic cannot be matched are **counted and dropped,
never guessed at**, because a misfiled message is a mislabelled one and the labels are the point.
A topic left thinner than `min_per_topic` raises rather than quietly producing a lopsided corpus.

If you have no rendered corpus, you do not need one to ask the question it answers: see the
vocabulary-overlap sweep below, which degrades symbolisation in a controlled way instead.

## Two kinds of null, and this is the point

- **Pure nulls** (`shared_topic=0`): no influence and no drifting common cause. These measure
  **calibration**.
- **Common-cause nulls** (`shared_topic>0`): no influence between agents, but everyone tracks a
  shared drifting channel topic, so every pair is genuinely dependent *through the channel*. These
  measure **confounding**, which pairwise transfer entropy cannot distinguish from influence.

Separating these was not in the original design. It came out of smoke-testing and noticing that both
surrogates flagged edges on what had been called a null, because common cause was still switched on.

One subtlety worth knowing: `shared_topic` buys **temporal coherence**, not marginal concentration.
The drifting current topic wanders over the whole alphabet, so a high `shared_topic` gives a nearly
uniform marginal, while at `shared_topic=0` agents fall back on home topics that the busiest speakers
dominate. The two null families therefore differ in marginal skew as well as in coherence. Home
topics are also drawn with replacement, so a pure null still contains a *static* shared-topic
structure between some pairs; lag-1 transfer entropy is blind to it, but a lag-0 or marginal
co-topic detector will fire and be scored as miscalibrated.

## What it measured

**It refuted the hypothesis it was built to confirm.** The audit expected the benchmark to reproduce
the circular-shift surrogate's failure. It does not: on stationary pure nulls the shift surrogate is
roughly calibrated, nowhere near the 0.90 measured on real logs, even though generated channels also
have exactly one speaker per tick. The generator draws speakers independently, so it has none of the
real sequence's turn-taking. That negative result is what sent the audit to isolate the speaker
sequence as the real cause.

**Non-stationary activity is one sufficient route to the failure, and it damages both surrogates.**
Sweeping `activity_drift` on pure nulls, 30 replicates per cell, 95% Wilson intervals
(`bench_stationarity_big.json`):

| `activity_drift` | shift FWER | permutation FWER |
|---|---|---|
| 0.0 | 0.067 [0.018, 0.213] | 0.033 [0.006, 0.167] |
| 0.3 | 0.133 [0.053, 0.297] | 0.100 [0.035, 0.256] |
| 0.6 | 0.200 [0.095, 0.373] | 0.167 [0.073, 0.336] |
| 0.9 | **0.367** [0.219, 0.545] | **0.100** [0.035, 0.256] |

The shift surrogate inflates about fivefold across the sweep. An earlier 8-replicate version
reported 0.375 for the permutation surrogate at drift 0.6, which looked alarming and was noise; this
is why the sweep was re-run at 30.

**Superseded in part by `turntaking_factorial.json`, which runs 200 replicates per cell.** At that
precision the permutation surrogate is no longer "not shown to fail": pooled over 2,400 replicates
it sits at **0.0942, 95% Wilson [0.083, 0.107]**, which excludes the nominal 0.05, and it is flat
across every cell of a turn-taking by drift factorial with no trend in either knob. So the honest
statement is **stable at about 0.09, which is not 0.05**, rather than nominal. It does not degrade
under turn-taking, which is the thing it was built to survive. The audit's headline 0.063 [0.041,
0.097] comes from 300 replicates on the real channel's own speaker sequence rather than from this
generator; the two intervals overlap, so they are consistent, and the generator's larger sample is
the more precise of the two.

**The conclusions survive bad symbolisation, down to a sharp floor.** The fair objection to a
synthetic text layer is that disjoint per-topic vocabularies make symbolisation trivial, so every
number here assumes a symboliser that does not exist. `vocab_overlap` is the answer: it is the
fraction of each topic's vocabulary shared with every other topic, and raising it degrades the
symboliser the way real text does, measurably. Adjusted Rand index is computed against the true
topic labels, so the degradation is quantified rather than asserted (`realism_sweep.json`):

| `vocab_overlap` | symboliser ARI | common-cause null, pairwise | structure, pairwise precision | structure, conditional precision | conditional recall |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 22.7 | 0.18 | **0.80** | 1.00 |
| 0.5 | 0.92 | 20.0 | 0.18 | **0.80** | 1.00 |
| 0.8 | 0.44 | 10.7 | 0.33 | **0.80** | 1.00 |
| 0.9 | 0.08 | 0.0 | 0.56 | 1.00 | **0.08** |

The conditional detector's advantage is not an artifact of easy text: at ARI 0.44, where the
symboliser recovers less than half the topic structure, it still holds precision 0.80 at full recall
while pairwise sits at 0.33. The zero row also reproduces the headline table from an independent
seed stream.

**But between ARI 0.44 and 0.08 everything collapses.** Conditional recall falls from 1.00 to 0.08,
and the common-cause null stops producing false edges only because there is no longer any topic
signal left to confound with. A detector that certifies nothing on a badly symbolised channel is not
reporting an absence of influence, it is reporting that its own front end failed, and nothing
downstream of the symboliser can tell those apart. If you publish an influence graph over open text,
report your symboliser's agreement with something.

**But the floor is not a property of the symboliser alone, and the label-free proxies for it do not
transfer.** `floor_contradiction.py` was written to settle a disagreement between two of this
repository's own measurements. The label-free cuts calibrated in `floor_proxy_sweep.json` place the
audit's open-chat channel below the floor; the audit's certified-edge counts place it above anything
a below-floor instance of this generator produces. Both measurements turn out to be confounded, and
the overlap sweep hid it by holding one knob fixed (`floor_contradiction.json`):

| what was held fixed | what was moved | what happened |
|---|---|---|
| true ARI 1.00, both voting proxies 1.00 | `shared_topic` 0.30 to 0.95 | permutation count 6.0 to 72.7 |
| `vocab_overlap` 0.80, true ARI 0.31 to 0.48 | `shared_topic` 0.00 to 0.95 | `seed_stability_min` 0.39 to 0.73, crossing its published cut |
| `vocab_overlap` 0.00, true ARI 1.00 | `n_messages` 8000 to 2000 | permutation count 23.7 to 7.3 |
| `vocab_overlap` 0.00 | true topic count 8 to 24, symboliser still at k=8 | true ARI 1.00 to 0.27, permutation count 23.0 to 22.3, conditional recall 1.00 throughout |

Three consequences, each measured rather than argued:

- **The certified-edge count is not a symboliser score.** Within the overlap sweep it tracked the
  true ARI at Spearman 0.93, which is better than any proxy in `floor_proxy.py`, but only because
  `shared_topic` was pinned at 0.55 at every level. Unpin it and the count varies twelvefold at
  perfect symbolisation.
- **The ARI floor moves with the common cause.** At `shared_topic` 0 the conditional detector keeps
  all four planted edges at a true ARI of 0.30, below the 0.44 this section names. The floor is a
  property of the pair (symboliser, channel), not of the symboliser.
- **A below-floor ARI does not imply a near-zero edge count.** It does on the vocabulary mechanism
  and not on the k mechanism, where ARI 0.27 still certifies 22.3 edges at full recall.

**The published cuts fail out of mechanism.** Calibrated on shared vocabulary with the topic count
known and correct, they scored a balanced accuracy of 1.00. Re-applied, unchanged, to instances that
differ only in their true topic count, `seed_stability_min` scores 0.75 and `agree_lsa` 0.57. All
seven of seed stability's errors condemn a symbolisation under which the detector in fact kept every
planted edge; three of cross-symboliser agreement's nine call a true ARI of 0.25 to 0.29 safe.
Message length is the other mechanism that breaks them: at four words per message, true ARI 0.65 to
0.69, both cuts read below the floor at full recall.

**On the real channels the verdict is an artifact of k.** Recomputed over a k grid of 2 to 16 with
the same loader, the same window and the same five-seed stability protocol, which reproduces
`floor_proxy_real.json`'s k=8 readings to the digit:

| k | main-village `seed_stability_min` | `agree_lsa` | open-chat `seed_stability_min` | `agree_lsa` |
|---|---|---|---|---|
| 2 | 0.634 | -0.040 | 0.077 | 1.000 |
| **4** | **0.874** | **0.878** | **0.950** | **0.978** |
| 6 | 0.628 | 0.753 | 0.408 | 0.759 |
| 8 | 0.676 | 0.696 | **0.287** | **0.456** |
| 12 | 0.761 | 0.694 | 0.355 | 0.336 |
| 16 | 0.818 | 0.780 | 0.277 | 0.478 |

Open-chat is the channel the control condemned, and at k=4 it produces the highest stability reading
either channel reaches anywhere on the grid. The published cuts of 0.6228 and 0.6462 clear it there
by a wide margin. The rival explanation, that short messages depressed its proxies, is ruled out by
the text itself: 27.1 percent of main-village's messages are under five tokens against 2.3 percent
of open-chat's, so the short-message channel is the one the control cleared.

**So `floor_proxy_real.json`'s two verdicts are statements about k=8 and nothing more.** The floor
control should not be run at a single k. The defensible version reports the proxy as a curve over k
and declines to certify a channel whose verdict changes along it, which both of these do.

**Precision, not recall, is the weak point of pairwise transfer entropy.** That is what motivated the
conditional detector.

**The conditional detector fixes most of it.** Three seeds per family, each with its own seed stream,
100 surrogates per test, the symbol-permutation null throughout, so the two rows differ *only* in
whether the channel's recent state is conditioned on (`bench_validation.json`, from `revalidate.py`):

| detector | pure null | common-cause null (0 planted) | structure (4 planted) | pooled precision | recall |
|---|---|---|---|---|---|
| pairwise | fires 1/3 | 22.33 false edges | 23.33 edges, 4 TP | 0.17 | 1.0 |
| **conditional** | fires 1/3 | **0.33** | **5.0 edges, 4 TP** | **0.80** | **1.0** |

and it costs no power: the planted edge is still found 3/3 at every coupling from 0.05 to 0.8, while
certifying only 1.0 to 1.7 edges in total.

Read `pooled precision` as total true positives over total certified edges across all instances. The
per-instance mean (`mean_precision_when_fired`) only averages instances that fired, which flatters a
detector that stays silent, so both are reported and the pooled one is the headline. Remember the
label-set caveat above: 0.80 is a floor, not a ceiling.

**Caveat, stated plainly:** this benchmark's common cause *is* a shared modal topic and the
conditioning variable *is* the channel's modal topic, so these numbers are close to a best case. The
real-channel result in the audit is the harder test.

## The detector contract

A detector is a callable:

```python
Callable[[list[tuple[str, float, int, str]]], set[tuple[str, str]]]
```

It receives the event list and must return a **set** of `(source, target)` tuples using the
generator's agent names, which are `"agent00"`, `"agent01"` and so on. It is given the `topic` field
in each event but must not use it; that is the label. Returning a list rather than a set raises
`TypeError` in the harness, because scoring does `found & ground_truth`.

## Usage

```python
import bench_gen, bench_eval, functools

# one instance with a known graph. shared_topic: 0 is a pure null, >0 adds common cause.
inst = bench_gen.generate_instance(n_agents=12, n_messages=8000,
                                   edges=((0, 5, 0.4),), shared_topic=0.55, seed=1)
inst["events"]             # [(agent, time, topic, text)] - a detector uses agent, time and text
inst["ground_truth"]       # {("agent00", "agent05")}
inst["realised_coupling"]  # {"0->5": 0.41} - what the stream actually contains

detector = functools.partial(bench_eval.te_detector, null="permute")
summary = bench_eval.evaluate(detector)
```

`evaluate` returns the numbers every influence claim should carry:

| key | meaning |
|---|---|
| `familywise_error_pure_null` | share of pure-null instances certifying at least one edge; should be near 0.05 |
| `familywise_error_common_cause_null` | the same on common-cause nulls; measures confounding, not calibration |
| `pooled_precision` | total TP over total certified, across every signal instance |
| `mean_precision_when_fired` | per-instance precision averaged over instances that fired only |
| `fired_rate_on_signal` | share of signal instances where anything was certified |
| `power_by_coupling`, `power_by_channel_volume` | mean recall at each swept value |
| `detection_floor_coupling`, `detection_floor_volume` | `{"floor", "status", "note"}`; status is one of `bracketed`, `below_grid`, `never_reached`, `non_monotone`, `no_data` |
| `n_agents_tested`, `family_size`, `dropped_by_top_k` | per row: how many agents the detector actually tested, and how many it dropped |

The detection floor returns a status rather than a bare number because "never reached", "no data" and
"power collapsed again above the crossing" used to be indistinguishable. It also warns when fewer
than ten seeds are used, since with *n* seeds recall can only take multiples of 1/*n* and an 80%
threshold is not estimable.

`n_agents_tested` matters: `te_detector` caps at `top_k=15`, so on a 25-agent channel all 25 can
clear `min_msgs` while only 15 are tested, and a planted edge may not be inside the tested set.

## Requirements and layout

Python 3.10+, `numpy`, `scikit-learn` (see `requirements.txt`). scikit-learn is imported lazily
inside the symboliser's `fit()`, so a missing install used to fail minutes into a run rather than at
startup.

**`bench_gen.py` and `tests/test_bench_gen.py` are standalone and are the reusable part.** The
*evaluator* is not, and this is a real limitation rather than a path-editing problem: `bench_eval.py`
and every script that uses it import the estimator and symbolisers from `swarm-prep` and
`swarm-sprint`, and **`swarm-prep` is a separate private dependency that is not redistributed here**.
Editing the `sys.path.insert` lines will not conjure it. Without it you can generate instances and
run the generator tests, but not reproduce the evaluator tables above. Vendoring the estimator is on
the roadmap.

```
swarm-prep     preexisting estimator and symbolisers (private, NOT redistributed)
swarm-sprint   the audit, including the vectorised estimator
swarm-bench    this repository
```

The MIT licence here covers this repository's own code.

## Scripts

| file | what | runtime |
|---|---|---|
| `bench_gen.py` | the generator and the standard configuration grid | seconds |
| `bench_eval.py` | detector harness and metrics | - |
| `revalidate.py` | **the authoritative validation**: both estimators on all three families plus the power sweep, writing `bench_validation.json` | ~3 min |
| `bench_run.py` | scores both surrogates across the whole grid | ~5 min at `SEEDS=2, N_SUR=100` |
| `realism_sweep.py` | sweeps `vocab_overlap` and measures symboliser ARI against the truth | ~3 min |
| `floor_proxy.py` | the label-free symbolisation-quality statistics, as a library | - |
| `floor_proxy_sweep.py` | calibrates those statistics against known ARI over 9 overlap levels, writing `floor_proxy_sweep.json` | ~42 min |
| `floor_proxy_real.py` | applies the calibrated cuts to the two real channels, writing `floor_proxy_real.json`; needs the gated corpus in the local cache | ~3 min |
| `floor_contradiction.py` | tests the floor proxies against the certified-edge count, which disagree with them on the real channels, writing `floor_contradiction.json`; reads the audit's committed counts and never the corpus | ~60 min |
| `text_pool.py` | loads an LLM-rendered message corpus as the text layer | - |
| `turntaking_factorial.py` | separates turn-taking from non-stationary activity, which the real speaker sequence carries at once, writing `turntaking_factorial.json` | ~2 h |
| `surrogate_variants.py` | all four surrogates on **identical** pure-null symbol grids, writing `surrogate_variants.json`. Records the environment | ~12 min at 300 replicates, 12 workers |
| `stationarity_pinned.py` | the drift sweep re-run with the environment recorded: one arm reproducing the published configuration exactly and reporting agreement, one arm running all four surrogates at higher precision. Writes `stationarity_pinned.json` | ~20 min |
| `stationarity_big.py` | 30-replicate activity-drift sweep with Wilson intervals. **Its counts are not reproducible across environments**; use `stationarity_pinned.py` | ~15 min |
| `bench_stationarity.py` | the original 8-replicate drift sweep, **superseded** by the above | ~4 min |
| `cond_strong.py`, `cond_power.py`, `bench_conditional.py` | **superseded pilots**, kept for the record | - |

**Do not cite the superseded files.** `cond_strong.json`, `cond_power.json`,
`bench_conditional.json`, `bench_summary.json`, `bench_results_*.json` and `bench_stationarity.json`
were all produced before three defects were fixed: the generator dropped queued reactions, grid cells
shared a base seed so a short instance was an exact prefix of a longer one, and the vocabulary
encoded the topic index. `bench_validation.json` and `bench_stationarity_big.json` are the current
files.

## Tests

```bash
python -m pytest tests -q
```

150 tests across the generator, the detector harness, the text layer, the symbolisation-floor
analysis and the turn-taking factorial. Many began as regression tests for a defect that was
actually found. `tests/test_bench_gen.py` needs only numpy; `tests/test_bench_eval.py` skips
cleanly if the private estimator is not on the path.

## Roadmap

1. **Vendor the estimator** so the evaluator is reproducible from a clean clone.
2. **Label the displacement.** Either add the leader-to-everyone suppression edges to `ground_truth`,
   or let a reaction co-occur with the organic draw instead of consuming the tick, so precision is
   not capped by construction.
3. **Realistic text layer.** Replace the vocabulary sampler with generated agent dialogue, keeping
   the generator in charge of structure, and publish the corpus.
4. ~~**Turn-taking in the generator.**~~ **Done, and the stated expectation was wrong.** The
   generator now has `p_repeat` and `p_return` knobs (a Markov speaker chain, bit-identical to the
   old generator at zero). The guess above was that adding turn-taking would reproduce the real
   surrogate failure. It does the opposite: turn-taking **suppresses** the shift surrogate, taking
   it from 0.055 at no turn-taking and no drift to **0.000** at real-channel values, while drift is
   the only inflating factor and reaches 0.400 with turn-taking off. Neither knob nor both together
   reaches the real channel's 0.90, so the generator still does not explain that number. See
   `turntaking_factorial.py`.
5. **More detectors.** Net-influence scores, multivariate transfer entropy, and continuous-time
   estimators.

## Licence

MIT, see [LICENSE](LICENSE). No AI Village data is used or redistributed here; the corpus is gated
and its terms forbid training on it. This benchmark's *data* is synthetic and independent.
