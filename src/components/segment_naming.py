"""Auto-assign plain-language business segment names to numeric clusters.

Cluster labels are meaningless to non-technical users. This module ranks
clusters by a "customer value" score derived from RFM centroids and maps
them to a fixed set of business archetypes (Champions, Loyal, ...).

Works for any clustering model (K-Means, DBSCAN, GMM). DBSCAN noise
(label == -1) is always mapped to "Outliers".
"""
from __future__ import annotations
import numpy as np

# Ordered high -> low customer value. Index 0 = most valuable cluster.
# Each archetype carries plain-language content for non-technical users.
ARCHETYPES = [
    {
        "name": "Champions",
        "who": "Your best customers. They buy often, recently, and spend the most.",
        "churn_risk": "Low",
        "customer_value": "Critical",
        "actions": [
            "Give them early access to new products before everyone else.",
            "Invite them to a VIP or loyalty program.",
            "Don't over-discount - they'll happily buy at full price.",
            "Ask them for reviews and referrals - they already love you.",
        ],
        "tactic": "Keep them happy. Don't discount, reward instead.",
    },
    {
        "name": "Loyal Customers",
        "who": "Solid repeat buyers who spend well and come back often.",
        "churn_risk": "Low-Medium",
        "customer_value": "High",
        "actions": [
            "Send a personal thank-you with a small surprise.",
            "Recommend products they haven't tried yet (related to past buys).",
            "Enrol them in a rewards program if you have one.",
            "Ask for quick feedback to keep them feeling heard.",
        ],
        "tactic": "Grow the relationship. Sell them more of what they love.",
    },
    {
        "name": "Potential Loyalists",
        "who": "Recent buyers who haven't bought many times yet, but show promise.",
        "churn_risk": "Medium",
        "customer_value": "Medium",
        "actions": [
            "Send a welcome series teaching them about your products.",
            "Offer a small discount on their next purchase to build the habit.",
            "Show them what other happy customers buy most.",
            "Stay in touch with a friendly monthly newsletter.",
        ],
        "tactic": "Turn a one-time buyer into a regular. Build the habit.",
    },
    {
        "name": "At Risk",
        "who": "Used to buy often and spend well, but haven't come back in a while.",
        "churn_risk": "High",
        "customer_value": "High (slipping)",
        "actions": [
            "Send a 'we miss you' email with a strong, time-limited offer.",
            "Remind them of items they bought before.",
            "Show what's new since their last visit.",
            "For big spenders, reach out personally (call or handwritten note).",
        ],
        "tactic": "Win them back NOW, before they leave for a competitor.",
    },
    {
        "name": "Hibernating",
        "who": "Haven't bought in a long time, rarely order, and spend little.",
        "churn_risk": "Very High",
        "customer_value": "Low",
        "actions": [
            "Try one cheap win-back campaign with a time-limited deal.",
            "Don't overspend - use your lowest-cost channel only.",
            "Ask why they stopped (a 1-question survey).",
            "If no response after 2 tries, stop marketing to them.",
        ],
        "tactic": "Cheap win-back test. If it fails, let them go.",
    },
    {
        "name": "Lost",
        "who": "Gone for a very long time with almost no purchases.",
        "churn_risk": "Critical",
        "customer_value": "Minimal",
        "actions": [
            "Stop regular marketing to them to save budget.",
            "Only re-target with a deep discount if they were ever a big spender.",
            "Use them as a lesson - understand what went wrong.",
        ],
        "tactic": "Drop from active marketing. Reclaim budget.",
    },
]

OUTLIER_ARCHETYPE = {
    "name": "Outliers",
    "who": "This customer doesn't fit any clear pattern. Buying behaviour is unusual.",
    "churn_risk": "Unknown",
    "customer_value": "Unknown",
    "actions": [
        "Review this customer by hand.",
        "Check for a data error or possible fraud.",
        "Decide what to do case-by-case.",
    ],
    "tactic": "Needs a human to look at it.",
}


def _value_score(centroid_rfm: np.ndarray) -> float:
    """Higher = more valuable. Recency is bad (lower better), F & M good."""
    r, f, m = centroid_rfm
    # Normalize contributions by log to tame scale differences between R and M.
    return (np.log1p(f) + np.log1p(m)) - np.log1p(r)


def assign_segments(labels: np.ndarray, centroids_rfm: dict) -> dict:
    """Map cluster labels -> business segment metadata.

    Logic (designed for RFM data, works for any clustering model):
      1. Rank real clusters by a customer-value score (high F+M, low R = best).
      2. The top cluster -> Champions, the next -> Loyal Customers.
         (Guard: if the top-value cluster is also the oldest, it is "At Risk"
         instead - a high-value but slipping group - and the next recent
         high-value cluster becomes Champion.)
      3. Remaining clusters are ordered by recency (most recent first) and
         mapped to [Potential Loyalists, At Risk, Hibernating, Lost, ...].
         This makes "At Risk" = used-to-spend-well-but-slipping, and
         "Potential" = recent-but-low-engagement - the correct business meaning.
      4. DBSCAN noise (label == -1) is always "Outliers".

    Args:
        labels: array of integer cluster labels (may include -1 for DBSCAN noise).
        centroids_rfm: {label: [recency, frequency, monetary]} in original scale.

    Returns:
        {str(label): segment metadata dict} for every label present.
    """
    unique = sorted(int(l) for l in np.unique(labels))
    noise = -1 in unique
    real = [l for l in unique if l != -1]

    arr = np.array([[centroids_rfm[l][0], centroids_rfm[l][1], centroids_rfm[l][2]] for l in real],
                   dtype=float)
    R, F, M = arr[:, 0], arr[:, 1], arr[:, 2]
    value = (np.log1p(F) + np.log1p(M)) - np.log1p(R)  # higher = better

    # Recency rank: 0 = most recent (smallest R).
    recency_order = np.argsort(R)  # indices into `real`, ascending R

    # Value order: 0 = highest value.
    value_order = np.argsort(-value)

    assignment = {}  # label -> archetype index
    used = set()

    # Step 2: Champion + Loyal from top value, with the "old top-value" guard.
    top_val_idx = value_order[0]
    top_is_oldest = (recency_order[-1] == top_val_idx) and len(real) > 2
    if top_is_oldest:
        # High-value but slipping -> At Risk. Champion = best recent cluster.
        assignment[real[top_val_idx]] = 3  # At Risk archetype index
        used.add(top_val_idx)
        champion_idx = next(i for i in value_order if i not in used)
        assignment[real[champion_idx]] = 0  # Champions
        used.add(champion_idx)
        # Loyal = next best value not used.
        loyal_idx = next(i for i in value_order if i not in used)
        assignment[real[loyal_idx]] = 1  # Loyal Customers
        used.add(loyal_idx)
    else:
        assignment[real[top_val_idx]] = 0  # Champions
        used.add(top_val_idx)
        if len(value_order) > 1:
            loyal_idx = value_order[1]
            assignment[real[loyal_idx]] = 1  # Loyal Customers
            used.add(loyal_idx)

    # Step 3: remaining clusters ordered by recency (most recent first) ->
    # [Potential Loyalists, At Risk, Hibernating, Lost, ...] = archetypes 2,3,4,5
    remaining_archetypes = [2, 3, 4, 5]  # Potential, At Risk, Hibernating, Lost
    remaining = [i for i in recency_order if i not in used]  # already most-recent-first
    for slot, idx in enumerate(remaining):
        arch_idx = remaining_archetypes[slot] if slot < len(remaining_archetypes) else 5
        assignment[real[idx]] = arch_idx

    out: dict = {}
    for idx, label in enumerate(real):
        arch = ARCHETYPES[assignment[label]]
        out[str(label)] = {
            **arch,
            "size": int((labels == label).sum()),
            "centroid_rfm": [round(float(v), 2) for v in centroids_rfm[label]],
        }
    if noise:
        out["-1"] = {
            **OUTLIER_ARCHETYPE,
            "size": int((labels == -1).sum()),
            "centroid_rfm": [0.0, 0.0, 0.0],
        }
    return out
