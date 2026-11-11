"""Paired comparison of original and edited weights on the same items and seeds (tool T14).

Error bars resample items, not samples, since samples of one item are correlated (Miller 2024, "Adding Error Bars to
Evals"). McNemar's exact test counts the samples that flipped between wrong and right.
"""
from math import comb

import numpy as np


def paired(original, edited, resamples=2000, seed=0):
    """Accuracies, the difference with a 95% bootstrap interval, flips, and McNemar's p-value."""
    before = np.array([r["correct"] for r in original], dtype=float)
    after = np.array([r["correct"] for r in edited], dtype=float)
    items = np.array([r["item"] for r in original])
    per_item = np.array([(after[items == i] - before[items == i]).mean() for i in np.unique(items)])
    boots = np.random.default_rng(seed).choice(per_item, size=(resamples, len(per_item))).mean(1)
    wrong_to_right = int(((before == 0) & (after == 1)).sum())
    right_to_wrong = int(((before == 1) & (after == 0)).sum())
    flips = wrong_to_right + right_to_wrong
    p = min(1.0, 2 * sum(comb(flips, k) for k in range(min(wrong_to_right, right_to_wrong) + 1)) / 2 ** flips) \
        if flips else 1.0
    summary = {"original": 100 * before.mean(), "edited": 100 * after.mean(), "diff": 100 * per_item.mean(),
               "ci": [100 * float(q) for q in np.percentile(boots, [2.5, 97.5])],
               "wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong, "mcnemar_p": p,
               "items": len(per_item)}
    if "hit_limit" in original[0]:
        summary["hit_limit"] = [100 * np.mean([r["hit_limit"] for r in records]) for records in (original, edited)]
    return summary
