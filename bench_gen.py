"""SwarmInfluence-Bench: generate multi-agent conversations with a KNOWN influence graph.

The design decision that makes the labels trustworthy: the generator decides, in code, who speaks at
which tick, on which topic, and who is reacting to whom. That is the ground truth and it is exact.
Surface text is a separate, replaceable layer. Today it is drawn from per-topic vocabularies, so the
whole benchmark runs offline; swapping in an LLM renderer changes realism without touching labels.

Parameters are the ones the AI Village audit showed actually decide the answer: coupling, follower
volume, channel volume, swarm size, burstiness, reply latency and topic drift. Null instances, with
no planted edges at all, are first class, because a benchmark where every instance contains signal
cannot measure the failure mode that audit found.
"""
from __future__ import annotations
import json, hashlib
import numpy as np

SILENT = "<silent>"


def _vocab(k_topics, words_per_topic=40, seed=0):
    """Disjoint vocabularies, so a topic model has something real to recover."""
    rng = np.random.default_rng(seed)
    syll = ["ka", "re", "mo", "ti", "lu", "sa", "ne", "vo", "di", "pa", "zu", "fe"]
    vocab = {}
    for t in range(k_topics):
        words = set()
        while len(words) < words_per_topic:
            w = "".join(rng.choice(syll, size=3)) + str(t)
            words.add(w)
        vocab[t] = sorted(words)
    return vocab


def generate_instance(n_agents=12, n_messages=6000, edges=(), k_topics=8, seed=0,
                      burstiness=1.0, drift=0.02, reaction_lag=1, mean_gap_s=25.0,
                      volume_skew=1.4, words_per_msg=12, shared_topic=0.55):
    """One channel with a known influence graph.

    edges: iterable of (leader, follower, coupling). `coupling` is the probability that, when the
           leader speaks, the follower reacts `reaction_lag` ticks later copying the leader's topic.

    shared_topic: probability that a message speaks to the channel's drifting current topic. This is
           the common-cause knob and it matters more than it looks. Above zero, every pair of agents
           is genuinely topically dependent through the channel even with no influence between them,
           and pairwise transfer entropy cannot tell that apart from direct influence. Set it to 0
           for a PURE null (no influence and no common cause), which is what a calibration check
           needs; leave it high for a COMMON-CAUSE null, which tests a different and equally real
           failure. Reporting both is the point.

    Returns events [(agent, time, topic, text)] and the ground-truth edge set.
    """
    rng = np.random.default_rng(seed)
    vocab = _vocab(k_topics, seed=seed)
    edges = [(int(a), int(b), float(c)) for a, b, c in edges]

    # per-agent speaking weights: a realistic heavy head and long tail
    w = 1.0 / np.power(np.arange(1, n_agents + 1), volume_skew)
    w = w / w.sum()

    # the channel has a drifting current topic; agents mostly speak to it
    cur_topic = int(rng.integers(k_topics))
    home = rng.integers(0, k_topics, size=n_agents)

    followers = {}
    for a, b, c in edges:
        followers.setdefault(a, []).append((b, c))

    agents, times, topics = [], [], []
    t = 0.0
    pending = []          # scheduled reactions: (ticks_remaining, follower, topic)
    for _ in range(n_messages):
        # burstiness: gamma inter-arrival, shape<1 is bursty, 1 is Poisson
        gap = rng.gamma(shape=max(1e-3, burstiness), scale=mean_gap_s / max(1e-3, burstiness))
        t += float(gap)
        if rng.random() < drift:
            cur_topic = int(rng.integers(k_topics))

        due = [p for p in pending if p[0] <= 0]
        pending = [(n - 1, f, tp) for n, f, tp in pending if n > 0]
        if due:
            _, speaker, topic = due[0]
            pending = [p for p in pending if p is not due[0]] + pending[:0]
        else:
            speaker = int(rng.choice(n_agents, p=w))
            r = rng.random()
            rest = shared_topic + (1.0 - shared_topic) * 0.55
            topic = (cur_topic if r < shared_topic
                     else (int(home[speaker]) if r < rest else int(rng.integers(k_topics))))
        agents.append(int(speaker)); times.append(t); topics.append(int(topic))

        for fol, c in followers.get(speaker, ()):
            if rng.random() < c:
                pending.append((reaction_lag - 1, int(fol), int(topic)))

    texts = [" ".join(rng.choice(vocab[tp], size=words_per_msg)) for tp in topics]
    events = [(f"agent{a:02d}", float(tt), int(tp), tx)
              for a, tt, tp, tx in zip(agents, times, topics, texts)]
    gt = {(f"agent{a:02d}", f"agent{b:02d}") for a, b, c in edges if c > 0}
    return {"events": events, "ground_truth": gt, "n_agents": n_agents,
            "config": {"n_agents": n_agents, "n_messages": n_messages, "k_topics": k_topics,
                       "seed": seed, "burstiness": burstiness, "drift": drift,
                       "reaction_lag": reaction_lag, "mean_gap_s": mean_gap_s,
                       "edges": [[a, b, c] for a, b, c in edges]}}


def grid(seeds=3):
    """The benchmark's standard configuration grid, including null instances."""
    out = []
    # 1a. PURE nulls: no influence and no common cause. This is the calibration check.
    for n_ag in (5, 12, 25):
        for n_msg in (2000, 8000):
            for s in range(seeds):
                out.append(dict(name=f"nullpure_a{n_ag}_m{n_msg}_s{s}", n_agents=n_ag,
                                n_messages=n_msg, edges=(), shared_topic=0.0, seed=100 + s))
    # 1b. COMMON-CAUSE nulls: no influence, but everyone tracks a shared drifting topic.
    for n_ag in (5, 12, 25):
        for s in range(seeds):
            out.append(dict(name=f"nullcommon_a{n_ag}_s{s}", n_agents=n_ag, n_messages=8000,
                            edges=(), shared_topic=0.55, seed=150 + s))
    # 2. coupling sweep, one planted edge, fixed volume
    for c in (0.05, 0.1, 0.2, 0.4, 0.8):
        for s in range(seeds):
            out.append(dict(name=f"coupling{c}_s{s}", n_agents=12, n_messages=8000,
                            edges=((0, 5, c),), seed=200 + s))
    # 3. channel-volume sweep at fixed strong coupling
    for n_msg in (1000, 2000, 4000, 8000, 16000):
        for s in range(seeds):
            out.append(dict(name=f"volume{n_msg}_s{s}", n_agents=12, n_messages=n_msg,
                            edges=((0, 5, 0.4),), seed=300 + s))
    # 4. multi-edge structure, which is what a real claim looks like
    for s in range(seeds):
        out.append(dict(name=f"structure_s{s}", n_agents=12, n_messages=8000,
                        edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4)),
                        seed=400 + s))
    # 5. burstiness and delayed reaction, the realism challenges
    for b in (0.4, 1.0):
        for lag in (1, 3):
            for s in range(seeds):
                out.append(dict(name=f"burst{b}_lag{lag}_s{s}", n_agents=12, n_messages=8000,
                                edges=((0, 5, 0.4),), burstiness=b, reaction_lag=lag,
                                seed=500 + s))
    return out


if __name__ == "__main__":
    inst = generate_instance(edges=((0, 5, 0.4),), seed=1)
    ev = inst["events"]
    print(f"{len(ev)} messages, {inst['n_agents']} agents, ground truth {inst['ground_truth']}")
    import collections
    print("speaker volumes:", collections.Counter(a for a, _, _, _ in ev).most_common(5))
    print("example text:", ev[0][3][:70])
    print(f"grid: {len(grid())} instances")
