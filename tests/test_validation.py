import numpy as np
import pandas as pd
import pytest

from tnved_ranker.validation import PREDICTION_COLUMNS, validate_predictions


TOP_K = 3


@pytest.fixture
def declarations() -> pd.DataFrame:
    return pd.DataFrame({"declaration_id": ["D0001", "D0002"]})


@pytest.fixture
def regulations() -> pd.DataFrame:
    return pd.DataFrame(
        {"regulation_id": ["R0001", "R0002", "R0003", "R0004"]}
    )


@pytest.fixture
def valid_predictions() -> pd.DataFrame:
    rows = [
        ("D0001", 1, "R0001", 0.90),
        ("D0001", 2, "R0002", 0.70),
        ("D0001", 3, "R0003", 0.50),
        ("D0002", 1, "R0002", 0.85),
        ("D0002", 2, "R0003", 0.65),
        ("D0002", 3, "R0004", 0.45),
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def run_validation(
    predictions: pd.DataFrame,
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
) -> None:
    validate_predictions(
        predictions,
        declarations,
        regulations,
        top_k=TOP_K,
    )


def test_valid_predictions_pass(
    valid_predictions,
    declarations,
    regulations,
):
    run_validation(valid_predictions, declarations, regulations)


def test_wrong_columns_fail(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions[
        ["rank", "declaration_id", "regulation_id", "score"]
    ]

    with pytest.raises(ValueError, match="columns"):
        run_validation(invalid, declarations, regulations)


def test_missing_declaration_fails(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions[
        valid_predictions["declaration_id"] != "D0002"
    ]

    with pytest.raises(ValueError, match="declaration_id mismatch"):
        run_validation(invalid, declarations, regulations)


def test_unknown_regulation_fails(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions.copy()
    invalid.loc[0, "regulation_id"] = "R9999"

    with pytest.raises(ValueError, match="unknown regulations"):
        run_validation(invalid, declarations, regulations)


def test_duplicate_regulation_fails(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions.copy()
    invalid.loc[1, "regulation_id"] = invalid.loc[0, "regulation_id"]

    with pytest.raises(ValueError, match="must be unique"):
        run_validation(invalid, declarations, regulations)


def test_wrong_group_size_fails(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions.drop(index=0)

    with pytest.raises(ValueError, match="exactly 3 predictions"):
        run_validation(invalid, declarations, regulations)


def test_invalid_ranks_fail(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions.copy()
    invalid.loc[2, "rank"] = 4

    with pytest.raises(ValueError, match="ranks must be exactly"):
        run_validation(invalid, declarations, regulations)


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
def test_non_finite_score_fails(
    invalid_score,
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions.copy()
    invalid.loc[0, "score"] = invalid_score

    with pytest.raises(ValueError, match="missing values|finite values"):
        run_validation(invalid, declarations, regulations)


def test_wrong_score_order_fails(
    valid_predictions,
    declarations,
    regulations,
):
    invalid = valid_predictions.copy()
    invalid.loc[1, "score"] = 1.0

    with pytest.raises(ValueError, match="rank increases"):
        run_validation(invalid, declarations, regulations)
