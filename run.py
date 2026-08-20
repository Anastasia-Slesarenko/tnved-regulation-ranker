import argparse
from pathlib import Path
from time import perf_counter

from tnved_ranker.data import load_data
from tnved_ranker.retrieval import rank_regulations_hybrid
from tnved_ranker.validation import validate_predictions


TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank regulations for customs declarations",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory with declarations.jsonl and regulations.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for predictions.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = perf_counter()

    declarations, regulations = load_data(args.data)

    predictions = rank_regulations_hybrid(
        declarations,
        regulations,
        top_k=TOP_K,
        rrf_k=60,
        batch_size=32,
        device="cpu",
        local_files_only=True,
    )

    validate_predictions(
        predictions,
        declarations,
        regulations,
        top_k=TOP_K,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "predictions.csv"
    predictions.to_csv(output_path, index=False)

    elapsed = perf_counter() - started_at
    print(f"Saved {len(predictions)} predictions to {output_path}")
    print(f"Elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
