"""Evaluation helpers for future claim-to-evidence retrieval experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def reciprocal_rank(retrieved_ids: Iterable[str], relevant_ids: set[str]) -> float:
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> dict[str, float | int]:
    if not relevant_ids:
        raise ValueError("At least one relevant evidence ID is required")
    return {
        "recall_at_1": int(bool(set(retrieved_ids[:1]) & relevant_ids)),
        "recall_at_3": int(bool(set(retrieved_ids[:3]) & relevant_ids)),
        "recall_at_5": int(bool(set(retrieved_ids[:5]) & relevant_ids)),
        "mean_reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        "evidence_hit_rate": int(bool(set(retrieved_ids) & relevant_ids)),
    }


def grouped_results(
    examples: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate already-computed example metrics by language/category.

    This helper does not run on the synthetic fixture in the project workflow.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        groups[f"language:{example['language']}"].append(example)
        groups[f"source_category:{example['source_category']}"].append(example)
    return {
        group: {
            metric: sum(float(item[metric]) for item in items) / len(items)
            for metric in (
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "mean_reciprocal_rank",
                "evidence_hit_rate",
            )
        }
        for group, items in groups.items()
    }
