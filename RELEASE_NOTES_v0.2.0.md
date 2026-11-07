# SwarmInfluence-Bench v0.2.0

A benchmark for coordination and influence detectors on multi-agent chat, where the true influence
graph is known by construction.

**Do not cut this release until the repository is enabled in Zenodo.** The Zenodo webhook is
installed only when you enable the repository, and the documented flow is enable, then release. If
you release first, Zenodo will not archive it and you will need a fresh release.

## Why this release exists

An external audit of v0.1 found three defects serious enough that every published number had to be
regenerated. All three are fixed here, and a fourth was found, measured, and documented rather than
quietly patched.

## Fixed

**The generator leaked its own answer.** Words were built as syllables followed by the topic index,
so a one-line regular expression recovered the generator's label from every message. The
symbolisation step this benchmark claims to require was solvable exactly, for free. Vocabularies are
now built as one pool and partitioned, so they stay disjoint while nothing inside a word says which
slice it landed in.

**Queued reactions were dropped** when a leader had two or more followers whose coins both landed,
which understated coupling for every follower after the first. They are now re-queued, and
`generate_instance` reports `realised_coupling` beside the request, because only one agent speaks
per tick and the second follower still cannot land at the stated lag.

**Grid cells shared a base seed**, so a short instance was an exact prefix of a longer one at the
same seed, which understated the variance of anything averaged over the grid. Each cell now gets its
own stream.

**Evaluator metrics.** `floor_of` returned a bare number that conflated no-data, never-reached,
below-grid and non-monotone curves; it now returns a status and flags when there are too few seeds
to estimate an 80 percent power threshold at all. Power curves are selected on the generator's
settings rather than a name prefix, with a warning when a curve comes back empty. `te_detector`
records how many agents it actually tested, since `top_k` silently drops agents past the cap.
scikit-learn is now declared in requirements, where its lazy import used to let a run fail minutes
in rather than at install time.

## Found and documented, not fixed

**A planted edge does not only couple its own pair.** Because a queued reaction consumes the tick,
every other agent becomes less likely to speak immediately after the leader: on the standard
four-edge instance a non-follower's probability falls to between 0.19 and 0.83 of its base rate,
against 0.94 to 1.05 with no planted edges. That is a real lag-1 dependency absent from
`ground_truth`, so **precision on this benchmark is a lower bound for any detector**. Patching the
labels until our own method scored well would not have been a benchmark. The roadmap says how to fix
it properly.

## New

**A symbolisation-difficulty knob and the floor it reveals.** `vocab_overlap` is the fraction of
each topic's vocabulary shared with every other topic, which degrades symbolisation the way real
text does. Sweeping it with adjusted Rand index measured against the true labels:

| overlap | ARI | conditional precision | conditional recall |
|---|---|---|---|
| 0.0 | 1.00 | 0.80 | 1.00 |
| 0.5 | 0.92 | 0.80 | 1.00 |
| 0.8 | 0.44 | 0.80 | 1.00 |
| 0.9 | 0.08 | 1.00 | 0.08 |

The conditional detector's advantage is not an artifact of easy text: at ARI 0.44 it holds precision
0.80 at full recall while pairwise sits at 0.33. But between 0.44 and 0.08 everything collapses, and
the common-cause nulls stop producing false edges only because there is no topic signal left to
confound with. A detector that certifies nothing on a badly symbolised channel is reporting that its
own front end failed, not an absence of influence.

**A pluggable text layer.** `generate_instance` takes `text_pool`, so an LLM-rendered corpus can
replace the vocabulary sampler. A test asserts the property that makes this safe: generate twice at
one seed, with and without a pool, and the `(agent, time, topic)` stream is bit-identical while only
the text differs. `text_pool.load_pool` reads JSONL, JSON or CSV, counts and drops rows whose topic
it cannot match rather than guessing, and refuses a corpus that leaves a topic thin.

**Measured against real text:** a corpus of LLM-written channel messages gives a mean ARI of 0.937,
above the collapse floor, so the synthetic text layer was never the weak point.

## Tests

46, each a regression test for a defect that was actually found. `tests/test_bench_gen.py` needs
only numpy; `tests/test_bench_eval.py` skips cleanly without the private estimator.

```
python -m pytest tests -q
```

## Superseded result files

`cond_strong.json`, `cond_power.json`, `bench_conditional.json`, `bench_summary.json`,
`bench_results_*.json` and `bench_stationarity.json` all predate the fixes above. **Do not cite
them.** `bench_validation.json`, `bench_stationarity_big.json`, `realism_sweep.json` and
`corpus_ari.json` are current.

## Known limitations

- The evaluator depends on a private estimator package and is not reproducible from a clean clone.
  The generator and its tests are. Vendoring is on the roadmap.
- The generator draws speakers independently, so it does not reproduce the turn-taking that breaks
  the shift surrogate on real logs. That is a limitation found rather than designed, and it is why
  the companion audit could not separate turn-taking from non-stationary activity.
- Home topics are drawn with replacement, so a "pure null" still contains a static shared-topic
  structure between some pairs. Lag-1 transfer entropy is blind to it; a lag-0 detector would not be.

MIT. No AI Village data is used or redistributed; this benchmark's data is synthetic and
independent.
