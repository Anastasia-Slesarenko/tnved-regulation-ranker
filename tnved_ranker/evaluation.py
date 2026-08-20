import pandas as pd


def calculate_ranking_metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, int | float]:
    """Calculate Hit@K and MRR for one expected regulation per declaration."""

    if labels.empty:
        raise ValueError("Labels must not be empty")

    if labels["declaration_id"].duplicated().any():
        raise ValueError("declaration_id must be unique in labels")

    ks = tuple(sorted(set(ks)))

    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    missing_declarations = (
        set(labels["declaration_id"])
        - set(predictions["declaration_id"])
    )

    if missing_declarations:
        raise ValueError(
            "Predictions are missing for declarations: "
            f"{sorted(missing_declarations)}"
        )

    comparison = predictions.merge(
        labels[
            [
                "declaration_id",
                "expected_regulation_id",
            ]
        ],
        on="declaration_id",
        how="inner",
    )

    relevant_predictions = comparison[
        comparison["regulation_id"].eq(
            comparison["expected_regulation_id"]
        )
    ]

    best_ranks = relevant_predictions.groupby("declaration_id")["rank"].min()

    relevant_ranks = labels["declaration_id"].map(best_ranks)

    metrics = {"n_queries": len(labels)}

    for k in ks:
        metrics[f"hit_at_{k}"] = float((relevant_ranks <= k).mean())

    max_k = max(ks)

    reciprocal_ranks = (
        1 / relevant_ranks.where(relevant_ranks.le(max_k))
    ).fillna(0)

    metrics[f"mrr_at_{max_k}"] = float(reciprocal_ranks.mean())

    return metrics
