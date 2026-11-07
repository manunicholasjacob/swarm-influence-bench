# SwarmInfluence-Bench

A benchmark for **coordination and influence detectors** on multi-agent chat, where the true
influence graph is known by construction.

Built out of an audit which found that the standard information-theoretic recipe, applied to real AI
Village logs, certified 11 influence edges on one channel of which **none** survived a valid
surrogate. The reason nobody had noticed is that there was nowhere to get calibrated ground truth.

Companion audit: **[swarm-influence-audit](https://github.com/manunicholasjacob/swarm-influence-audit)**.

Author: Manu Nicholas Jacob, Independent Researcher (manunicholasjacob@gmail.com).

## What makes the labels trustworthy

**The generator decides, in code, who speaks at which tick, on which topic, and who is reacting to
whom.** That is the ground truth and it is exact. Surface text is a separate, replaceable layer drawn
from disjoint per-topic vocabularies, so the generator runs offline with no model and no API.
Swapping in an LLM renderer changes realism without touching a single label.

A detector sees only `(agent, time, text)` and must symbolise the text itself, so it cannot read the
generator's topic labels.

## Two kinds of null, and this is the point

- **Pure nulls** (`shared_topic=0`): no influence and no common cause. These measure **calibration**.
- **Common-cause nulls** (`shared_topic>0`): no influence between agents, but everyone tracks a
  shared drifting channel topic, so every pair is genuinely dependent *through the channel*. These
  measure **confounding**, which pairwise transfer entropy cannot distinguish from influence.

Separating these was not in the original design. It came out of smoke-testing and noticing that both
surrogates flagged edges on what had been called a null, because common cause was still switched on.

## What it measured

**It refuted the hypothesis it was built to confirm.** The audit expected the benchmark to reproduce
the circular-shift surrogate's failure. It does not: on pure nulls the shift surrogate is roughly
calibrated at **0.083**, not the 0.90 measured on real logs, even though generated channels also have
exactly one speaker per tick. The generator draws speakers independently, so it has none of the real
sequence's turn-taking. That negative result is what sent the audit to isolate the speaker sequence
as the real cause. Making generated activity strongly non-stationary raises the shift surrogate to
0.625 (`bench_stationarity.json`), so non-stationarity is one sufficient route but not the whole story.

**It found that precision, not recall, is the weak point of pairwise TE.** With the permutation
surrogate: recall 1.0, but mean precision **0.087** and a common-cause-null error rate of **1.0**
(18.8 mean false edges). That is what motivated the conditional detector.

**The conditional detector then fixes it** (`cond_strong.json`, `cond_power.json`):

| detector | common-cause null (0 planted) | structure (4 planted) | precision |
|---|---|---|---|
| pairwise | 22.7 false edges | 24 edges, 4 TP | 0.17 |
| **conditional** | **0.0** | **4.0 edges, 4 TP** | **1.00** |

and it costs no power: the planted edge is still found 3/3 at every coupling from 0.05 to 0.8.

**Caveat, stated plainly:** this benchmark's common cause *is* a shared modal topic and the
conditioning variable *is* the channel's modal topic, so these numbers are close to a best case. The
real-channel result in the audit is the harder test.

## Usage

```python
import bench_gen, bench_eval, functools

# one instance with a known graph. Note shared_topic: 0 is a pure null, >0 adds common cause.
inst = bench_gen.generate_instance(n_agents=12, n_messages=8000,
                                   edges=((0, 5, 0.4),), shared_topic=0.55, seed=1)
inst["events"]        # [(agent, time, topic, text)] - a detector sees agent, time and text only
inst["ground_truth"]  # {("agent00", "agent05")}

detector = functools.partial(bench_eval.te_detector, null="permute")
summary = bench_eval.evaluate(detector)
```

`evaluate` returns the numbers every influence claim should carry: `familywise_error_pure_null`,
`familywise_error_common_cause_null`, `power_by_coupling`, `power_by_channel_volume`, the detection
floors, and precision/recall against the planted edges.

## Requirements and layout

Python 3.12+, `numpy`.

**`bench_gen.py` is standalone and is the reusable part.** The *evaluator* is not: `bench_eval.py`,
`bench_run.py`, `bench_conditional.py`, `cond_strong.py`, `cond_power.py` and
`bench_stationarity.py` import the preexisting estimator and symbolisers via absolute paths, so they
expect:

```
C:\dev\swarm-prep     preexisting estimator and symbolisers (private, not redistributed)
C:\dev\swarm-sprint   the audit
C:\dev\swarm-bench    this repository
```

Edit the `sys.path.insert` lines at the top of those files for any other layout. The MIT licence here
covers this repository's own code; `swarm-prep` is a separate private dependency.

## Scripts

| file | what | runtime |
|---|---|---|
| `bench_gen.py` | the generator and the standard configuration grid | seconds |
| `bench_eval.py` | detector harness and metrics | - |
| `bench_run.py` | scores both surrogates across the grid | ~5 min at `SEEDS=2, N_SUR=100`; longer at defaults |
| `cond_strong.py` | the conditional detector, modal-topic conditioning | ~1 min |
| `cond_power.py` | conditional power sweep across couplings | ~3 min |
| `bench_stationarity.py` | does non-stationary activity reproduce the failure | ~4 min |
| `bench_conditional.py` | **superseded pilot**, coarse 3-level conditioning | ~2 min |

**`bench_conditional.json` is kept for the record and contradicts the headline**: it reports 15.3
residual false edges because it uses a coarse 3-level context. `cond_strong.json` uses the modal-topic
conditioning the audit actually specifies, and reports 0.0. Use `cond_strong.py`.

## Roadmap

1. **Realistic text layer.** Replace the vocabulary sampler with generated agent dialogue, keeping
   the generator in charge of structure, and publish the corpus.
2. **Turn-taking in the generator.** It currently draws speakers independently, which is why it does
   not reproduce the real surrogate failure. Adding realistic turn-taking would make it a sharper
   instrument and let it test surrogates properly.
3. **More detectors.** Net-influence scores, multivariate transfer entropy, and continuous-time
   estimators.

## Licence

MIT, see [LICENSE](LICENSE). No AI Village data is used or redistributed here; the corpus is gated
and its terms forbid training on it. This benchmark's *data* is synthetic and independent.
