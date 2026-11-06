# SwarmInfluence-Bench

A benchmark for **coordination and influence detectors** on multi-agent chat, where the true
influence graph is known by construction.

Built out of a finding: applying the standard information-theoretic recipe to real agent logs
(the AI Village corpus) produced 11 certified influence edges on one channel, **none of which
survived a valid surrogate**, and 3 on another where a valid surrogate finds 18. The reason nobody
had noticed is that there was nowhere to get calibrated ground truth. This is that.

See the companion audit: `C:\dev\swarm-sprint` (Apart Swarm Alignment Hackathon 2026).

## What makes the labels trustworthy

**The generator decides, in code, who speaks at which tick, on which topic, and who is reacting to
whom.** That is the ground truth, and it is exact. Surface text is a separate, replaceable layer.
Today it is drawn from disjoint per-topic vocabularies, so the whole benchmark runs offline with no
model and no API. Swapping in an LLM renderer changes realism without touching a single label.

This is the opposite of the usual approach of running real agents and then trying to infer what
happened, and it is the only way to get labels that are not themselves an inference problem.

## The parameters are the ones that turned out to matter

From the audit, not from a wishlist: coupling strength, **follower volume** (power is driven by how
much an agent says, not only by how strongly it copies), channel volume, swarm size, burstiness,
reply latency, and topic drift.

## Two kinds of null, and this is the point

Most benchmarks put signal in every instance, which cannot measure the failure mode the audit found.
This one has two distinct null families:

- **Pure nulls** (`shared_topic=0`): no influence and no common cause. This is the calibration check.
  A correctly calibrated detector should certify something in about 5% of these and no more.
- **Common-cause nulls** (`shared_topic>0`): no influence between agents, but everyone tracks a
  shared drifting channel topic. Every pair is now genuinely dependent *through the channel*.
  Pairwise transfer entropy cannot distinguish that from direct influence, and this family measures
  how badly it conflates them.

Separating these two was not in the original design. It came out of smoke-testing the benchmark and
noticing that both surrogates flagged edges on what I had called a null, because I had left common
cause switched on. Reporting them apart is more honest and more useful than reporting one number.

## Usage

```python
import bench_gen, bench_eval, functools

# one instance with a known graph
inst = bench_gen.generate_instance(n_agents=12, n_messages=8000, edges=((0, 5, 0.4),), seed=1)
inst["events"]        # [(agent, time, topic, text)] - a detector sees agent, time and text only
inst["ground_truth"]  # {("agent00", "agent05")}

# score any detector: events -> set of certified (src, dst)
detector = functools.partial(bench_eval.te_detector, null="permute")
summary = bench_eval.evaluate(detector)
```

`evaluate` returns the three numbers every influence claim should carry:

| | |
|---|---|
| `familywise_error_pure_null` | how often anything is certified when there is nothing to find |
| `familywise_error_common_cause_null` | how often shared context is mistaken for influence |
| `power_by_coupling` / `power_by_channel_volume`, and the detection floors | what the detector can see |

plus precision and recall against the planted edges.

## Reproduce the validation

```
python bench_gen.py     # sanity check the generator
python bench_run.py     # score both surrogates across the grid -> bench_summary.json
```

`bench_run.py` runs the detector twice, once with the conventional circular-shift surrogate and once
with the symbol-permutation repair, and should independently reproduce the audit's central finding on
data where the truth is known.

## Roadmap

1. **Realistic text layer.** Replace the vocabulary sampler with generated agent dialogue, keeping
   the generator in charge of structure. The Adaption platform's `invent` and `extend_trajectories`
   endpoints fit this exactly, and its HuggingFace connector publishes the result.
2. **A learned baseline.** Train a detector on the benchmark and compare it against calibrated
   transfer entropy. If a learned detector beats a corrected information-theoretic one, that is worth
   knowing; if it does not, that is worth knowing too.
3. **More detectors.** Net-influence scores, conditional and multivariate transfer entropy (which
   should handle the common-cause nulls far better than pairwise does), and continuous-time
   estimators.

## Licence

MIT. No AI Village data is used or redistributed here; the corpus is gated and its terms forbid
training on it. This benchmark is synthetic and independent, which is also what makes it freely
licensable.
