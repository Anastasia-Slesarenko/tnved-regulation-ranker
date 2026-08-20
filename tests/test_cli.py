import sys

import pandas as pd

import run
from tnved_ranker.validation import PREDICTION_COLUMNS, validate_predictions


def test_main_writes_valid_predictions(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()

    declarations = pd.DataFrame({"declaration_id": ["D0001", "D0002"]})

    regulation_ids = [f"R{index:04d}" for index in range(1, 13)]
    regulations = pd.DataFrame({"regulation_id": regulation_ids})

    rows = [
        {
            "declaration_id": declaration_id,
            "rank": rank,
            "regulation_id": regulation_id,
            "score": 1.0 / rank,
        }
        for declaration_id in declarations["declaration_id"]
        for rank, regulation_id in enumerate(regulation_ids[:10], start=1)
    ]

    expected_predictions = pd.DataFrame(
        rows,
        columns=PREDICTION_COLUMNS,
    )

    def fake_load_data(received_data_dir):
        assert received_data_dir == data_dir
        return declarations, regulations

    def fake_rank_regulations_hybrid(
        received_declarations,
        received_regulations,
        **kwargs,
    ):
        assert received_declarations is declarations
        assert received_regulations is regulations

        assert kwargs == {
            "top_k": 10,
            "rrf_k": 60,
            "batch_size": 32,
            "device": "cpu",
            "local_files_only": True,
        }

        return expected_predictions.copy()

    monkeypatch.setattr(run, "load_data", fake_load_data)
    monkeypatch.setattr(run, "rank_regulations_hybrid", fake_rank_regulations_hybrid)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--data",
            str(data_dir),
            "--out",
            str(out_dir),
        ],
    )

    run.main()

    output_path = out_dir / "predictions.csv"
    assert output_path.is_file()

    saved_predictions = pd.read_csv(output_path)

    validate_predictions(
        saved_predictions,
        declarations,
        regulations,
        top_k=10,
    )

    pd.testing.assert_frame_equal(
        saved_predictions,
        expected_predictions,
        check_exact=False,
    )
