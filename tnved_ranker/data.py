from pathlib import Path

import pandas as pd


DECLARATION_COLUMNS = {
    "declaration_id",
    "G31_1",
    "desc_extention",
}

REGULATION_COLUMNS = {
    "regulation_id",
    "code",
    "description",
}


def _validate_table(
    table: pd.DataFrame,
    required_columns: set[str],
    id_column: str,
) -> None:
    missing_columns = required_columns - set(table.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if table[id_column].isna().any() or table[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be non-empty and unique")


def load_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load declarations and regulations."""

    data_dir = Path(data_dir)

    declarations = pd.read_json(
        data_dir / "declarations.jsonl",
        lines=True,
        dtype={"declaration_id": "string"},
    )
    regulations = pd.read_json(
        data_dir / "regulations.jsonl",
        lines=True,
        dtype={
            "regulation_id": "string",
            "code": "string",
        },
    )

    _validate_table(
        declarations,
        DECLARATION_COLUMNS,
        "declaration_id",
    )
    _validate_table(
        regulations,
        REGULATION_COLUMNS,
        "regulation_id",
    )

    return declarations, regulations