import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


PREDICTION_COLUMNS = (
    "declaration_id",
    "rank",
    "regulation_id",
    "score",
)


def validate_predictions(
    predictions: pd.DataFrame,
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
    top_k: int = 10,
) -> None:
    """Validate predictions."""

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            "predictions must have exactly these columns in this order: "
            f"{list(PREDICTION_COLUMNS)}"
        )

    if predictions.isna().to_numpy().any():
        raise ValueError("predictions must not contain missing values")

    if not is_integer_dtype(predictions["rank"]):
        raise ValueError("rank must have an integer dtype")

    if not is_numeric_dtype(predictions["score"]):
        raise ValueError("score must have a numeric dtype")

    if not np.isfinite(predictions["score"]).all():
        raise ValueError("score must contain only finite values")

    expected_declaration_ids = set(declarations["declaration_id"])
    predicted_declaration_ids = set(predictions["declaration_id"])

    if predicted_declaration_ids != expected_declaration_ids:
        missing = sorted(expected_declaration_ids - predicted_declaration_ids)
        unknown = sorted(predicted_declaration_ids - expected_declaration_ids)
        raise ValueError(
            "declaration_id mismatch: "
            f"missing={missing[:5]}, unknown={unknown[:5]}"
        )

    valid_regulation_ids = set(regulations["regulation_id"])
    unknown_regulation_ids = sorted(
        set(predictions["regulation_id"]) - valid_regulation_ids
    )

    if unknown_regulation_ids:
        raise ValueError(
            "predictions contain unknown regulations: "
            f"{unknown_regulation_ids[:5]}"
        )

    if predictions.duplicated(
        subset=["declaration_id", "regulation_id"]
    ).any():
        raise ValueError(
            "regulation_id values must be unique within each declaration"
        )

    expected_ranks = list(range(1, top_k + 1))
    invalid_sizes = []
    invalid_ranks = []
    invalid_score_order = []

    for declaration_id, group in predictions.groupby(
        "declaration_id", sort=False
    ):
        ordered = group.sort_values("rank")

        if len(ordered) != top_k:
            invalid_sizes.append(declaration_id)

        if ordered["rank"].tolist() != expected_ranks:
            invalid_ranks.append(declaration_id)

        if not ordered["score"].is_monotonic_decreasing:
            invalid_score_order.append(declaration_id)

    if invalid_sizes:
        raise ValueError(
            f"each declaration must have exactly {top_k} predictions: "
            f"{invalid_sizes[:5]}"
        )

    if invalid_ranks:
        raise ValueError(
            f"ranks must be exactly 1..{top_k} for each declaration: "
            f"{invalid_ranks[:5]}"
        )

    if invalid_score_order:
        raise ValueError(
            "scores must not increase as rank increases: "
            f"{invalid_score_order[:5]}"
        )
