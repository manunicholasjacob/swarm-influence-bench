"""Can symbolisation quality be judged WITHOUT ground-truth labels?

`realism_sweep.py` establishes a symbolisation floor: the conditional detector keeps its precision
and recall down to an adjusted Rand index of about 0.44 between symboliser output and the true
topics, and collapses below it. The report's own largest stated limitation is that this control
cannot be run on the real channels, because ARI needs labels and real chat has none.

This file asks whether some LABEL-FREE statistic tracks ARI closely enough to place a symboliser
on the right side of that floor. If one does, the floor control becomes runnable on real logs.

Every statistic here is computable from the messages and the symboliser alone:

  seed stability      mean pairwise ARI across k-means runs that differ only in initialisation
  bootstrap stability mean pairwise ARI across refits on resampled messages
  silhouette          cosine silhouette on the TF-IDF vectors the clusterer actually saw
  Calinski-Harabasz   between-cluster over within-cluster dispersion
  Davies-Bouldin      worst-case ratio of cluster spread to cluster separation
  cross-symboliser    ARI against a structurally different but still meaning-bearing symboliser
  centroid cosine     how similar the cluster centroids are to each other
  size entropy        how evenly the messages spread over the alphabet

THE TRAP, handled explicitly in `ari_on_shared`: an adjusted Rand index is defined between two
labellings of THE SAME points. Bootstrap resamples see different subsets, so their label vectors
cannot be lined up position by position, because position 7 of one resample and position 7 of
another are different messages. Labels from a resample are therefore carried here as a dict keyed
by the ORIGINAL message index, so the only way to compare two of them is on the indices they share.
`bootstrap_stability` reports the two mis-readings alongside the honest one, with their measured
sizes, rather than leaving the warning as prose. Padding the undrawn points with one shared label
collapses the statistic, by 0.60 on average over the sweep. Comparing each resample with the
full-data fit rather than with another resample turned out NOT to be the systematic inflation it
looks like: it inflated by 0.09 on a 2000-message probe and by between -0.04 and +0.01 per level
on the 8000-message sweep, so it is corpus-size dependent and reported rather than corrected.

Nothing in this file touches ground truth. `true_ari` is computed by the callers, which have labels
only because the benchmark generator wrote them.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np


def add_sibling_repos():
    """Put the sibling swarm checkouts on sys.path without hard-coding anybody's drive layout.

    The audit's symboliser lives in swarm-prep and its estimator in swarm-sprint, both checked out
    beside this repo. SWARM_PREP and SWARM_SPRINT override either location.
    """
    here = pathlib.Path(__file__).resolve().parent
    out = []
    for env, name in (("SWARM_PREP", "swarm-prep"), ("SWARM_SPRINT", "swarm-sprint")):
        raw = os.environ.get(env)
        cand = pathlib.Path(raw).expanduser() if raw else here.parent / name
        if cand.is_dir():
            out.append(str(cand))
    for p in [str(here)] + out:
        if p not in sys.path:
            sys.path.insert(0, p)
    return out


add_sibling_repos()

from sklearn.cluster import KMeans                                  # noqa: E402
from sklearn.decomposition import TruncatedSVD                      # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score   # noqa: E402

from swarm_load import SILENT, HashSymboliser, TopicSymboliser, normalise  # noqa: E402

K_DEFAULT = 8
N_INIT = 10          # matches TopicSymboliser, so a refit differs only in what it was fitted on
MAX_FEATURES = 20000
MIN_DF = 1


# ----------------------------------------------------------------------------- label bookkeeping

def codes(symbols):
    """Symbol strings to dense integer codes.

    SILENT keeps a code of its own. A message the symboliser could not place is its own outcome and
    merging it into a topic would flatter every agreement statistic below.
    """
    order = {s: i for i, s in enumerate(sorted(set(symbols)))}
    return np.array([order[s] for s in symbols], dtype=np.int64)


def ari(a, b):
    """Adjusted Rand index between two labellings of the same points, in the same order."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("labellings cover %d and %d points; ARI needs the same points"
                         % (a.shape[0], b.shape[0]))
    return float(adjusted_rand_score(a, b))


def ari_on_shared(a, b):
    """ARI between two clusterings given as {original message index: label}.

    This signature exists to make the trap impossible. A resample sees its own subset of messages,
    so position 7 of one resample and position 7 of another are different messages, and an ARI
    taken over those vectors is not a stability estimate of anything. Keying by the original index
    means the only legal comparison is on the indices both resamples drew, which is what happens
    here.

    Returns (ari, n_shared).
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        raise ValueError("the two clusterings share %d points; ARI needs at least 2" % len(shared))
    return float(adjusted_rand_score([a[i] for i in shared], [b[i] for i in shared])), len(shared)


# ----------------------------------------------------------------------------- feature space

def tfidf(texts, min_df=MIN_DF, max_features=MAX_FEATURES):
    """The feature space TopicSymboliser actually clusters in, plus the rows it kept.

    TopicSymboliser drops messages whose normalised text is empty, so an internal-validity score
    taken over all rows would be scoring points the clusterer never saw.

    Returns (X, kept_row_indices).
    """
    corpus = [normalise(t) for t in texts]
    kept = [i for i, c in enumerate(corpus) if c]
    if not kept:
        raise ValueError("every message normalises to empty text")
    vec = TfidfVectorizer(min_df=min_df, max_features=max_features)
    return vec.fit_transform([corpus[i] for i in kept]), kept


def _rowsq(X):
    """Squared L2 norm of each row, for a sparse matrix or a dense array."""
    if hasattr(X, "multiply"):
        return np.asarray(X.multiply(X).sum(axis=1)).ravel()
    return np.asarray(np.square(np.asarray(X)).sum(axis=1)).ravel()


def _dense_labels(labels):
    """Relabel to 0..m-1 so cluster arithmetic can index by label."""
    uniq, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64), len(uniq)


def _centroids(X, labels, m):
    cent = np.zeros((m, X.shape[1]), dtype=float)
    sizes = np.zeros(m, dtype=float)
    for c in range(m):
        mask = labels == c
        sizes[c] = float(mask.sum())
        if sizes[c]:
            cent[c] = np.asarray(X[mask].mean(axis=0)).ravel()
    return cent, sizes


def _sq_to_point(X, rowsq, c):
    return np.maximum(0.0, rowsq - 2.0 * np.asarray(X @ c).ravel() + float(c @ c))


# ----------------------------------------------------------------------------- internal validity

def calinski_harabasz(X, labels):
    """Between-cluster over within-cluster dispersion. Takes X sparse or dense.

    sklearn's version wants a dense array. The real channels' TF-IDF matrix is about 8800 by 20000,
    which densifies to well over a gigabyte, so the standard formula is evaluated here directly on
    the sparse matrix. The tests check it against sklearn's value on the same data.
    """
    lab, m = _dense_labels(labels)
    n = X.shape[0]
    if m < 2 or n <= m:
        return None
    rowsq = _rowsq(X)
    grand = np.asarray(X.mean(axis=0)).ravel()
    cent, sizes = _centroids(X, lab, m)
    within = 0.0
    for c in range(m):
        mask = lab == c
        if not mask.any():
            continue
        within += float(_sq_to_point(X[mask], rowsq[mask], cent[c]).sum())
    between = float((sizes * ((cent - grand) ** 2).sum(axis=1)).sum())
    if within <= 0.0:
        return 1.0
    return float(between * (n - m) / (within * (m - 1)))


def davies_bouldin(X, labels):
    """Worst-case ratio of cluster spread to cluster separation. Lower is better.

    Takes X sparse or dense, for the same reason as `calinski_harabasz`.
    """
    lab, m = _dense_labels(labels)
    if m < 2:
        return None
    rowsq = _rowsq(X)
    cent, _ = _centroids(X, lab, m)
    spread = np.zeros(m, dtype=float)
    for c in range(m):
        mask = lab == c
        if not mask.any():
            continue
        spread[c] = float(np.sqrt(_sq_to_point(X[mask], rowsq[mask], cent[c])).mean())
    sq = (cent ** 2).sum(axis=1)
    gap = np.sqrt(np.maximum(0.0, sq[:, None] + sq[None, :] - 2.0 * (cent @ cent.T)))
    worst = []
    for i in range(m):
        ratios = [(spread[i] + spread[j]) / gap[i, j]
                  for j in range(m) if j != i and gap[i, j] > 0]
        worst.append(max(ratios) if ratios else 0.0)
    return float(np.mean(worst))


def centroid_cosine(X, labels):
    """Mean and max cosine similarity between cluster centroids. Lower means better separated.

    Scale free, so it stays comparable across the sweep even though the benchmark's vocabulary
    shrinks as the topics share more of it.
    """
    lab, m = _dense_labels(labels)
    if m < 2:
        return None, None
    cent, _ = _centroids(X, lab, m)
    norms = np.linalg.norm(cent, axis=1)
    keep = norms > 0
    cent, norms = cent[keep], norms[keep]
    if cent.shape[0] < 2:
        return None, None
    sim = (cent @ cent.T) / np.outer(norms, norms)
    iu = np.triu_indices(cent.shape[0], k=1)
    return float(sim[iu].mean()), float(sim[iu].max())


def size_entropy(labels):
    """Shannon entropy of cluster occupancy, normalised by log of the number of clusters used."""
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    if len(counts) < 2:
        return 0.0
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return h / math.log(len(counts))


def silhouette_cosine(X, labels, sample_size=2000, seed=0):
    """Cosine silhouette on the clustered rows, on a fixed random subsample.

    The full index is quadratic in the number of messages, which at 8000 messages is 64 million
    pairs for no extra information. `seed` fixes the subsample, so the number is reproducible.
    """
    lab, m = _dense_labels(labels)
    if m < 2:
        return None
    n = X.shape[0]
    size = None if sample_size is None or sample_size >= n else int(sample_size)
    return float(silhouette_score(X, lab, metric="cosine", sample_size=size, random_state=seed))


# ----------------------------------------------------------------------------- second opinions

class LsaSymboliser:
    """TF-IDF then truncated SVD then k-means: a different route to the same kind of symbol.

    Structurally different from the headline symboliser (it clusters in a dense low-rank space
    where co-occurring words have been folded together) but still meaning-bearing, which is what a
    second opinion has to be. The audit's HashSymboliser is meaning-free by construction, so it is
    a null, not a second opinion; it is reported alongside for exactly that reason.
    """

    def __init__(self, k=K_DEFAULT, seed=0, n_components=50):
        self.k, self.seed, self.n_components = k, seed, n_components
        self._vec = self._svd = self._km = None

    def fit(self, texts):
        corpus = [normalise(t) for t in texts]
        keep = [c for c in corpus if c]
        self._vec = TfidfVectorizer(min_df=MIN_DF, max_features=MAX_FEATURES)
        x = self._vec.fit_transform(keep)
        comp = min(self.n_components, max(2, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=comp, random_state=self.seed)
        z = self._svd.fit_transform(x)
        self._km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=N_INIT).fit(z)
        return self

    def transform(self, texts):
        corpus = [normalise(t) for t in texts]
        idx = [i for i, c in enumerate(corpus) if c]
        out = [SILENT] * len(corpus)
        if idx:
            z = self._svd.transform(self._vec.transform([corpus[i] for i in idx]))
            for i, lab in zip(idx, self._km.predict(z)):
                out[i] = "L%d" % lab
        return out


class CountSymboliser:
    """Raw term counts then k-means. No IDF and no length normalisation.

    Meaning-bearing, and a genuinely different weighting. Its weakness is worth stating: without
    normalisation, k-means on counts partly clusters by message LENGTH. The benchmark's messages
    are all the same length so that cannot bite there, but real chat lengths vary by an order of
    magnitude, so on real logs this one is the least trustworthy of the three.
    """

    def __init__(self, k=K_DEFAULT, seed=0):
        self.k, self.seed = k, seed
        self._vec = self._km = None

    def fit(self, texts):
        corpus = [normalise(t) for t in texts]
        keep = [c for c in corpus if c]
        self._vec = CountVectorizer(min_df=MIN_DF, max_features=MAX_FEATURES)
        x = self._vec.fit_transform(keep)
        self._km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=N_INIT).fit(x)
        return self

    def transform(self, texts):
        corpus = [normalise(t) for t in texts]
        idx = [i for i, c in enumerate(corpus) if c]
        out = [SILENT] * len(corpus)
        if idx:
            for i, lab in zip(idx, self._km.predict(self._vec.transform([corpus[i] for i in idx]))):
                out[i] = "c%d" % lab
        return out


class CharSymboliser:
    """Character n-gram TF-IDF then k-means. Never sees a word boundary.

    This is the most independent of the three second opinions: it agrees with the headline
    symboliser only when the topics really do differ in their surface strings, and it is immune to
    the tokeniser the headline symboliser depends on.
    """

    def __init__(self, k=K_DEFAULT, seed=0, ngram_range=(3, 5)):
        self.k, self.seed, self.ngram_range = k, seed, ngram_range
        self._vec = self._km = None

    def fit(self, texts):
        corpus = [normalise(t) for t in texts]
        keep = [c for c in corpus if c]
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                                    min_df=2, max_features=MAX_FEATURES)
        x = self._vec.fit_transform(keep)
        self._km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=N_INIT).fit(x)
        return self

    def transform(self, texts):
        corpus = [normalise(t) for t in texts]
        idx = [i for i, c in enumerate(corpus) if c]
        out = [SILENT] * len(corpus)
        if idx:
            for i, lab in zip(idx, self._km.predict(self._vec.transform([corpus[i] for i in idx]))):
                out[i] = "g%d" % lab
        return out


# ----------------------------------------------------------------------------- stability

def seed_stability(texts, k=K_DEFAULT, seeds=(0, 101, 202, 303, 404), base=None):
    """Mean pairwise ARI across k-means runs that differ only in initialisation.

    `base` is an already-computed labelling for seeds[0], so a caller that has fitted the headline
    symboliser already does not pay for it twice.
    """
    labs = []
    for i, s in enumerate(seeds):
        if i == 0 and base is not None:
            labs.append(codes(base))
        else:
            labs.append(codes(TopicSymboliser(k=k, seed=s).fit(texts).transform(texts)))
    vals = [ari(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_pairs": len(vals), "n_runs": len(labs)}


def bootstrap_stability(texts, k=K_DEFAULT, seed=0, n_boot=6, frac=1.0, replace=True, base=None):
    """How much the clustering moves when the messages it was fitted on are resampled.

    Four readings, because which points the index is taken over, and which pairs of clusterings are
    compared, are both real choices and both change the answer. Only the first is a proxy; the
    other three are reported so the size and the DIRECTION of each mistake is on the record instead
    of being asserted in a comment.

      full      each refitted model is applied back to EVERY message (predict, not fit), so the two
                label vectors cover the same points by construction and no restriction is needed.
                This is the one to use.
      shared    the same predictions restricted to the messages BOTH resamples actually drew. For a
                fit-then-predict symboliser a drawn message gets the same label either way, so this
                is `full` measured on the roughly 40 percent of the corpus the two resamples have
                in common, and it agrees with `full` to within sampling noise. Reported because it
                shows where the danger is NOT: restricting to the shared points is correct and
                costs almost nothing, so there is no excuse for skipping it.
      naive_pad the version that skips the restriction by padding, and the one that really does
                damage. A resample's in-sample labels exist only for the messages it drew, so the
                tempting fix is to give the rest one extra "not sampled" label and compare
                full-length vectors. The two undrawn sets only partly coincide, so the pad group
                disagrees with itself and the index does not merely shift, it collapses: over the
                sweep it sits 0.60 BELOW the restricted reading on average and up to 0.88 below,
                reading about 0.12 even where the honest stability is 1.0. It would condemn a
                perfect symboliser.
      vs_base   a different statistic that is easy to mistake for this one: instead of comparing
                resamples with each other, each resample is compared with the clustering fitted on
                ALL the data. The worry is that it should read high, since every resample shares
                most of its messages with that reference and the reference saw everything. On a
                2000-message probe it did, by about 0.09 at the boundary. Over the 8000-message
                sweep it did NOT: the per-level gap runs from -0.040 to +0.012, which is inside
                the row-to-row noise, although single rows differ by as much as +0.12. So the
                honest statement is that it is corpus-size dependent and not a bias anyone should
                correct for in one direction. It is reported, and not used, because a reader
                cannot tell the two apart from a number alone.

    `base` is the full-data labelling, supplied by the caller so it is not fitted twice. Without it
    the vs_base reading is skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(texts)
    size = max(k + 1, int(round(frac * n)))
    full_labels, in_sample, padded = [], [], []
    for _ in range(n_boot):
        idx = (rng.integers(0, n, size=size) if replace
               else rng.permutation(n)[:size])
        sym = TopicSymboliser(k=k, seed=seed).fit([texts[i] for i in idx])
        lab_all = codes(sym.transform(texts))
        full_labels.append(lab_all)
        drawn = np.unique(idx)
        # keyed by the ORIGINAL message index, which is what makes a legal comparison possible
        in_sample.append({int(i): int(lab_all[i]) for i in drawn})
        pad = np.full(n, -1, dtype=np.int64)
        pad[drawn] = lab_all[drawn]
        padded.append(pad)
    pairs = [(i, j) for i in range(n_boot) for j in range(i + 1, n_boot)]
    full = [ari(full_labels[i], full_labels[j]) for i, j in pairs]
    naive = [ari(padded[i], padded[j]) for i, j in pairs]
    sh, counts = [], []
    for i, j in pairs:
        v, c = ari_on_shared(in_sample[i], in_sample[j])
        sh.append(v)
        counts.append(c)
    out = {
        "full_mean": float(np.mean(full)), "full_min": float(np.min(full)),
        "full_sd": float(np.std(full, ddof=1)) if len(full) > 1 else 0.0,
        "shared_mean": float(np.mean(sh)), "shared_min": float(np.min(sh)),
        "shared_points_mean": float(np.mean(counts)),
        "naive_pad_mean": float(np.mean(naive)),
        "naive_pad_error": float(np.mean(naive) - np.mean(sh)),
        "n_boot": n_boot, "n_pairs": len(pairs), "resample_size": size, "with_replacement": replace,
    }
    if base is not None:
        b = np.asarray(base)
        vs = [ari(b, lab) for lab in full_labels]
        out["vs_base_mean"] = float(np.mean(vs))
        out["vs_base_inflation"] = float(np.mean(vs) - np.mean(full))
    return out


# ----------------------------------------------------------------------------- the whole battery

def all_proxies(texts, k=K_DEFAULT, seed=0, n_boot=6, seed_runs=5, sample_size=2000):
    """Every label-free statistic in this file, plus the headline symbols so a caller with labels
    can compute the true ARI on exactly the same run.

    `seed` is the headline symboliser's random_state. The stability seeds and the bootstrap stream
    are derived from it, and are disjoint from it, so a lucky initialisation cannot make a run look
    both good and stable for the same reason.
    """
    base = TopicSymboliser(k=k, seed=seed).fit(texts).transform(texts)
    base_codes = codes(base)
    X, kept = tfidf(texts)
    clustered = base_codes[np.asarray(kept, dtype=np.int64)]

    sil = silhouette_cosine(X, clustered, sample_size=sample_size, seed=seed)
    ch = calinski_harabasz(X, clustered)
    db = davies_bouldin(X, clustered)
    cc_mean, cc_max = centroid_cosine(X, clustered)

    ss = seed_stability(texts, k=k, seeds=tuple([seed] + [seed + 101 * i for i in range(1, seed_runs)]),
                        base=base)
    bs = bootstrap_stability(texts, k=k, seed=seed + 7919, n_boot=n_boot, base=base_codes)

    second = {}
    for name, sym in (("lsa", LsaSymboliser(k=k, seed=seed)),
                      ("counts", CountSymboliser(k=k, seed=seed)),
                      ("char", CharSymboliser(k=k, seed=seed)),
                      ("hash_null", HashSymboliser(k=k))):
        second[name] = ari(base_codes, codes(sym.fit(texts).transform(texts)))

    return {
        "base_symbols": list(base),
        "proxies": {
            "seed_stability_mean": ss["mean"],
            "seed_stability_min": ss["min"],
            "boot_stability_full_mean": bs["full_mean"],
            "boot_stability_full_min": bs["full_min"],
            "boot_stability_shared_mean": bs["shared_mean"],
            "silhouette_cosine": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "centroid_cosine_mean": cc_mean,
            "centroid_cosine_max": cc_max,
            "size_entropy": size_entropy(clustered),
            "agree_lsa": second["lsa"],
            "agree_counts": second["counts"],
            "agree_char": second["char"],
            "agree_hash_null": second["hash_null"],
        },
        "diagnostics": {
            "n_messages": len(texts),
            "n_clustered": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_distinct_texts": len(set(texts)),
            "n_symbols_used": int(len(np.unique(base_codes))),
            "silent_share": round(float(np.mean([s == SILENT for s in base])), 4),
            "seed_stability_sd": ss["sd"],
            "boot_stability_full_sd": bs["full_sd"],
            "boot_shared_points_mean": bs["shared_points_mean"],
            "boot_resample_size": bs["resample_size"],
            # the two mis-readings of bootstrap stability, kept as diagnostics and never
            # thresholded on. See the docstring of bootstrap_stability for what each one is.
            "boot_naive_pad_mean": bs["naive_pad_mean"],
            "boot_naive_pad_error": bs["naive_pad_error"],
            "boot_vs_base_mean": bs.get("vs_base_mean"),
            "boot_vs_base_inflation": bs.get("vs_base_inflation"),
        },
    }


# a larger value of these means a BETTER symboliser; the rest are inverted
HIGHER_IS_BETTER = {
    "seed_stability_mean": True, "seed_stability_min": True,
    "boot_stability_full_mean": True, "boot_stability_full_min": True,
    "boot_stability_shared_mean": True,
    "silhouette_cosine": True, "calinski_harabasz": True,
    "davies_bouldin": False,
    "centroid_cosine_mean": False, "centroid_cosine_max": False,
    "size_entropy": True,
    "agree_lsa": True, "agree_counts": True, "agree_char": True,
    "agree_hash_null": True,
}

PROXY_NAMES = tuple(HIGHER_IS_BETTER)


# ----------------------------------------------------------------------------- evaluation helpers

def _ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    """Spearman rank correlation, average ranks for ties.

    Written out rather than imported: scipy is not declared in requirements.txt, and this is a
    Pearson correlation of ranks.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra, rb = _ranks(a[ok]), _ranks(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else None


def confusion(values, above, threshold, higher_is_better):
    """Confusion of 'proxy says above floor' against 'truly above floor'.

    Positive class is ABOVE the floor, which is the claim a user of the control wants to make and
    therefore the one whose false positives matter: a false positive here is a channel reported as
    safely symbolised when it is not.
    """
    pred = (np.asarray(values, dtype=float) >= threshold if higher_is_better
            else np.asarray(values, dtype=float) <= threshold)
    truth = np.asarray(above, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    bal = (sens + spec) / 2.0 if sens is not None and spec is not None else None
    return {"threshold": float(threshold), "higher_is_better": bool(higher_is_better),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": None if sens is None else round(sens, 4),
            "specificity": None if spec is None else round(spec, 4),
            "balanced_accuracy": None if bal is None else round(bal, 4),
            "errors": fp + fn, "n": int(len(truth))}


def best_threshold(values, above, higher_is_better):
    """The cut on a proxy that best separates above-floor from below-floor rows.

    Candidate cuts are midpoints between consecutive distinct observed values, so the chosen cut
    never sits exactly on an observation. Scored by balanced accuracy: the two classes are not the
    same size here and plain accuracy would reward a proxy that simply called everything
    above-floor. Ties are broken towards the cut furthest from the nearest observation, which is
    the one most likely to survive on new data.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2 or len(set(np.asarray(above, dtype=bool)[ok].tolist())) < 2:
        return None
    v, truth = v[ok], np.asarray(above, dtype=bool)[ok]
    uniq = np.unique(v)
    cuts = [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(len(uniq) - 1)]
    if not cuts:
        return None
    margins = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    best, best_key = None, None
    for cut, margin in zip(cuts, margins):
        c = confusion(v, truth, cut, higher_is_better)
        key = (c["balanced_accuracy"] or 0.0, -c["errors"], float(margin))
        if best_key is None or key > best_key:
            best, best_key = c, key
    best["margin"] = float(best_key[2])
    return best
