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
import numpy as np


def _vocab(k_topics, words_per_topic=40, seed=0, overlap=0.0):
    """Per-topic vocabularies, optionally overlapping, so a topic model has something to recover.

    The words must NOT encode which topic they belong to. An earlier version built each word as
    three syllables followed by the topic index, which meant a one-line regex recovered the
    generator's own label from every message: the symbolisation step the benchmark claims to
    require was solvable exactly, for free. Here one pool of words is built and then PARTITIONED
    across topics, so nothing inside a word says which slice it landed in.

    `overlap` is the fraction of each topic's vocabulary drawn from a pool SHARED by every topic,
    and it is the knob that decides how hard symbolisation is. At 0 the vocabularies are disjoint
    and clustering the text recovers the topic almost perfectly, which is not what real text looks
    like: in real messages most words are common to every subject and only a few are distinctive.
    Raising it makes the symboliser lossy in the way a real one is, without any model in the loop,
    and lets a user ask the question that matters, which is whether a detector's conclusions
    survive imperfect symbolisation rather than whether they hold given a perfect one.
    """
    rng = np.random.default_rng(seed)
    syll = ["ka", "re", "mo", "ti", "lu", "sa", "ne", "vo", "di", "pa", "zu", "fe"]
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    n_shared = int(round(overlap * words_per_topic))
    n_private = words_per_topic - n_shared
    need = n_shared + k_topics * n_private
    pool = set()
    while len(pool) < need:
        pool.add("".join(rng.choice(syll, size=4)))
    words = sorted(pool)
    rng.shuffle(words)
    shared, rest = words[:n_shared], words[n_shared:]
    return {t: sorted(shared + rest[t * n_private:(t + 1) * n_private])
            for t in range(k_topics)}


def generate_instance(n_agents=12, n_messages=6000, edges=(), k_topics=8, seed=0,
                      burstiness=1.0, drift=0.02, reaction_lag=1, mean_gap_s=25.0,
                      volume_skew=1.4, words_per_msg=12, shared_topic=0.55,
                      activity_drift=0.0, n_blocks=20, vocab_overlap=0.0, text_pool=None):
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
    vocab = _vocab(k_topics, seed=seed, overlap=vocab_overlap)
    edges = [(int(a), int(b), float(c)) for a, b, c in edges]
    # Validate, because every one of these used to pass silently and corrupt the labels. A
    # self-edge lands in ground_truth but no estimator tests i->i, so recall was capped below 1
    # with no diagnostic; an out-of-range follower injected an extra speaker into the channel.
    for a, b, c in edges:
        if not (0 <= a < n_agents and 0 <= b < n_agents):
            raise ValueError(f"edge ({a},{b}) outside 0..{n_agents - 1}")
        if a == b:
            raise ValueError(f"self-edge ({a},{b}): no pairwise estimator tests i->i")
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"coupling {c} for ({a},{b}) is not a probability")

    # per-agent speaking weights: a realistic heavy head and long tail
    w0 = 1.0 / np.power(np.arange(1, n_agents + 1), volume_skew)
    w0 = w0 / w0.sum()

    # activity_drift makes those weights NON-STATIONARY: the timeline is cut into blocks and each
    # agent's share is resampled per block, so agents have bursts and quiet spells the way real ones
    # do. This turned out to matter more than anything else for whether a circular-shift surrogate
    # behaves, because shifting an agent's series slides its activity envelope against everyone
    # else's. With stationary weights a shift is nearly harmless; with realistic bursts it is not.
    block_w = []
    for b in range(max(1, n_blocks)):
        if activity_drift <= 0:
            block_w.append(w0)
            continue
        # each agent may go quiet for a block, and those that speak get a heavy-tailed multiplier.
        # This reproduces how real agents behave: bursts, idle spells, joining and leaving.
        live = rng.random(n_agents) > (activity_drift * 0.6)
        if not live.any():
            live[int(rng.integers(n_agents))] = True
        mult = np.exp(rng.normal(0.0, 1.5 * activity_drift, size=n_agents))
        wb = w0 * live * mult
        block_w.append(wb / wb.sum())
    w = block_w[0]

    # the channel has a drifting current topic; agents mostly speak to it
    cur_topic = int(rng.integers(k_topics))
    home = rng.integers(0, k_topics, size=n_agents)

    followers = {}
    for a, b, c in edges:
        followers.setdefault(a, []).append((b, c))

    agents, times, topics = [], [], []
    t = 0.0
    pending = []          # scheduled reactions: (ticks_remaining, follower, topic)
    for msg_i in range(n_messages):
        w = block_w[min(len(block_w) - 1, int(msg_i * len(block_w) / max(1, n_messages)))]
        # burstiness: gamma inter-arrival, shape<1 is bursty, 1 is Poisson
        gap = rng.gamma(shape=max(1e-3, burstiness), scale=mean_gap_s / max(1e-3, burstiness))
        t += float(gap)
        if rng.random() < drift:
            cur_topic = int(rng.integers(k_topics))

        due = [p for p in pending if p[0] <= 0]
        pending = [(n - 1, f, tp) for n, f, tp in pending if n > 0]
        if due:
            _, speaker, topic = due[0]
            # only one agent speaks per tick, so re-queue the rest for the next tick instead of
            # dropping them. Dropping them silently biased every follower after the first, which
            # made the labelled coupling wrong for any leader with two or more followers.
            pending = [(0, f, tp) for _, f, tp in due[1:]] + pending
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

    # The text layer is deliberately the LAST thing decided and the only replaceable one. Structure
    # is already fixed above, so swapping the renderer changes realism without touching a label.
    if text_pool is None:
        texts = [" ".join(rng.choice(vocab[tp], size=words_per_msg)) for tp in topics]
    else:
        missing = sorted({tp for tp in topics if not text_pool.get(tp)})
        if missing:
            raise ValueError(f"text_pool has no messages for topic(s) {missing}; "
                             f"it must cover all {k_topics} topics")
        texts = [str(rng.choice(text_pool[tp])) for tp in topics]
    events = [(f"agent{a:02d}", float(tt), int(tp), tx)
              for a, tt, tp, tx in zip(agents, times, topics, texts)]
    gt = {(f"agent{a:02d}", f"agent{b:02d}") for a, b, c in edges if c > 0}

    # REALISED coupling, not just the requested one. Only one agent speaks per tick, so when a
    # leader has two or more followers whose coins both land, they cannot all reply at the same
    # lag; the later ones are pushed back and no longer land at `reaction_lag`. The label would
    # otherwise overstate the coupling for every follower after the first, so we measure what
    # actually happened and report it alongside the request.
    realised = {}
    for a, b, c in edges:
        la, fa = a, b
        hits = opps = 0
        for i in range(reaction_lag, len(agents)):
            if agents[i - reaction_lag] == la:
                opps += 1
                if agents[i] == fa and topics[i] == topics[i - reaction_lag]:
                    hits += 1
        realised[f"{a}->{b}"] = round(hits / opps, 4) if opps else 0.0

    return {"events": events, "ground_truth": gt, "n_agents": n_agents,
            "realised_coupling": realised,
            "config": {"n_agents": n_agents, "n_messages": n_messages, "k_topics": k_topics,
                       "seed": seed, "burstiness": burstiness, "drift": drift,
                       "reaction_lag": reaction_lag, "mean_gap_s": mean_gap_s,
                       "shared_topic": shared_topic, "activity_drift": activity_drift,
                       "vocab_overlap": vocab_overlap,
                       "text_renderer": "pool" if text_pool is not None else "vocab",
                       "edges": [[a, b, c] for a, b, c in edges]}}


def _next(cell):
    cell[0] += 1
    return cell[0]


def grid(seeds=3):
    """The benchmark's standard configuration grid, including both null families.

    Every configuration cell gets its own RNG stream. An earlier version reused a base seed across
    cells, which made a short instance an exact prefix of a longer one at the same seed and so
    understated the variance of anything averaged over the grid."""
    cell = [0]
    out = []
    # 1a. PURE nulls: no influence and no common cause. This is the calibration check.
    for n_ag in (5, 12, 25):
        for n_msg in (2000, 8000):
            for s in range(seeds):
                out.append(dict(name=f"nullpure_a{n_ag}_m{n_msg}_s{s}", n_agents=n_ag,
                                n_messages=n_msg, edges=(), shared_topic=0.0, seed=100 + s + 1000 * _next(cell)))
    # 1b. COMMON-CAUSE nulls: no influence, but everyone tracks a shared drifting topic.
    for n_ag in (5, 12, 25):
        for s in range(seeds):
            out.append(dict(name=f"nullcommon_a{n_ag}_s{s}", n_agents=n_ag, n_messages=8000,
                            edges=(), shared_topic=0.55, seed=150 + s + 1000 * _next(cell)))
    # 2. coupling sweep, one planted edge, fixed volume
    for c in (0.05, 0.1, 0.2, 0.4, 0.8):
        for s in range(seeds):
            out.append(dict(name=f"coupling{c}_s{s}", n_agents=12, n_messages=8000,
                            edges=((0, 5, c),), seed=200 + s + 1000 * _next(cell)))
    # 3. channel-volume sweep at fixed strong coupling
    for n_msg in (1000, 2000, 4000, 8000, 16000):
        for s in range(seeds):
            out.append(dict(name=f"volume{n_msg}_s{s}", n_agents=12, n_messages=n_msg,
                            edges=((0, 5, 0.4),), seed=300 + s + 1000 * _next(cell)))
    # 4. multi-edge structure, which is what a real claim looks like
    for s in range(seeds):
        out.append(dict(name=f"structure_s{s}", n_agents=12, n_messages=8000,
                        edges=((0, 5, 0.4), (0, 7, 0.3), (2, 9, 0.5), (9, 11, 0.4)),
                        seed=400 + s + 1000 * _next(cell)))
    # 5. burstiness and delayed reaction, the realism challenges
    for b in (0.4, 1.0):
        for lag in (1, 3):
            for s in range(seeds):
                out.append(dict(name=f"burst{b}_lag{lag}_s{s}", n_agents=12, n_messages=8000,
                                edges=((0, 5, 0.4),), burstiness=b, reaction_lag=lag,
                                seed=500 + s + 1000 * _next(cell)))
    return out


if __name__ == "__main__":
    inst = generate_instance(edges=((0, 5, 0.4),), seed=1)
    ev = inst["events"]
    print(f"{len(ev)} messages, {inst['n_agents']} agents, ground truth {inst['ground_truth']}")
    import collections
    print("speaker volumes:", collections.Counter(a for a, _, _, _ in ev).most_common(5))
    print("example text:", ev[0][3][:70])
    print(f"grid: {len(grid())} instances")
