import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tnved_ranker.validation import PREDICTION_COLUMNS


E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_MODEL_REVISION = "d1d99a1efae6779390caba937d92c54b5bc70e51"


def _build_query_texts(
    declarations: pd.DataFrame,
) -> pd.Series:
    return (
        declarations[["G31_1", "desc_extention"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )


def _build_document_texts(
    regulations: pd.DataFrame,
    document_columns: tuple[str, ...],
) -> pd.Series:
    missing_columns = set(document_columns) - set(regulations.columns)

    if missing_columns:
        raise ValueError(f"Missing document columns: {sorted(missing_columns)}")

    return (
        regulations[list(document_columns)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )


def _predictions_from_scores(
    scores: np.ndarray,
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    scores = np.asarray(scores)

    expected_shape = (len(declarations), len(regulations))

    if scores.shape != expected_shape:
        raise ValueError(f"scores must have shape {expected_shape}")

    if not 1 <= top_k <= len(regulations):
        raise ValueError("top_k must be between 1 and the number of regulations")

    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    top_indices = np.argsort(
        -scores,
        axis=1,
        kind="stable",
    )[:, :top_k]

    declaration_ids = declarations["declaration_id"].to_numpy()
    regulation_ids = regulations["regulation_id"].to_numpy()

    rows = []

    for query_index, regulation_indices in enumerate(top_indices):
        for rank, regulation_index in enumerate(
            regulation_indices,
            start=1,
        ):
            rows.append(
                {
                    "declaration_id": declaration_ids[query_index],
                    "rank": rank,
                    "regulation_id": regulation_ids[regulation_index],
                    "score": float(scores[query_index, regulation_index]),
                }
            )

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def rank_regulations(
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
    top_k: int = 10,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 1),
    document_columns: tuple[str, ...] = ("description",),
) -> pd.DataFrame:
    """Rank regulations using TF-IDF similarity."""

    queries = _build_query_texts(declarations)

    documents = _build_document_texts(regulations, document_columns)

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
    )

    document_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(queries)

    scores = cosine_similarity(query_matrix, document_matrix)

    return _predictions_from_scores(
        scores,
        declarations,
        regulations,
        top_k,
    )


def rank_regulations_e5(
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
    top_k: int = 10,
    model_name: str = E5_MODEL_NAME,
    model_revision: str = E5_MODEL_REVISION,
    batch_size: int = 32,
    device: str = "cpu",
    local_files_only: bool = True,
) -> pd.DataFrame:
    """Rank regulations using multilingual E5 embeddings."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    model = SentenceTransformer(
        model_name,
        revision=model_revision,
        device=device,
        local_files_only=local_files_only,
    )

    queries = ("query: " + _build_query_texts(declarations)).tolist()

    documents = (
        "passage: " + _build_document_texts(regulations, ("description",))
    ).tolist()

    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    document_embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    scores = query_embeddings @ document_embeddings.T

    return _predictions_from_scores(
        scores,
        declarations,
        regulations,
        top_k,
    )


def reciprocal_rank_fusion(
    lexical_predictions: pd.DataFrame,
    semantic_predictions: pd.DataFrame,
    top_k: int = 10,
    rrf_k: int = 60,
) -> pd.DataFrame:
    """Combine two complete rankings using RRF."""

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    lexical_ranking = lexical_predictions[
        ["declaration_id", "regulation_id", "rank"]
    ].rename(columns={"rank": "lexical_rank"})

    semantic_ranking = semantic_predictions[
        ["declaration_id", "regulation_id", "rank"]
    ].rename(columns={"rank": "semantic_rank"})

    fused = lexical_ranking.merge(
        semantic_ranking,
        on=[
            "declaration_id",
            "regulation_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(fused) != len(lexical_ranking) or len(fused) != len(semantic_ranking):
        raise ValueError(
            "RRF inputs must contain the same "
            "declaration-regulation pairs"
        )

    fused["score"] = (
        1 / (rrf_k + fused["lexical_rank"])
        + 1 / (rrf_k + fused["semantic_rank"])
    )

    fused = fused.sort_values(
        [
            "declaration_id",
            "score",
            "lexical_rank",
            "semantic_rank",
            "regulation_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
            True,
        ],
        kind="stable",
    )

    fused["rank"] = fused.groupby("declaration_id", sort=False).cumcount().add(1)

    return (
        fused.loc[
            fused["rank"].le(top_k),
            list(PREDICTION_COLUMNS),
        ]
        .reset_index(drop=True)
    )


def rank_regulations_hybrid(
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
    top_k: int = 10,
    rrf_k: int = 60,
    batch_size: int = 32,
    device: str = "cpu",
    local_files_only: bool = True,
) -> pd.DataFrame:
    """Rank regulations using the final E6 configuration."""

    candidate_count = len(regulations)

    if not 1 <= top_k <= candidate_count:
        raise ValueError("top_k must be between 1 and the number of regulations")

    lexical_predictions = rank_regulations(
        declarations,
        regulations,
        top_k=candidate_count,
        analyzer="char_wb",
        ngram_range=(3, 5),
    )

    semantic_predictions = rank_regulations_e5(
        declarations,
        regulations,
        top_k=candidate_count,
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
    )

    return reciprocal_rank_fusion(
        lexical_predictions,
        semantic_predictions,
        top_k=top_k,
        rrf_k=rrf_k,
    )
