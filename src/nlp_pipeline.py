from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Additional domain-specific noise words to keep matched keywords crisp and informative
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "book",
        "books",
        "story",
        "stories",
        "read",
        "reading",
        "reader",
        "readers",
        "novel",
        "novels",
        "set",
        "world",
        "tale",
        "tales",
        "character",
        "characters",
        "feel",
        "feels",
        "felt",
        "like",
        "just",
        "very",
        "new",
        "old",
        "one",
        "two",
        "three",
    }
)


def clean_text(text: Optional[str]) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize cleaned text and optionally filter out stop words and single-character tokens."""
    cleaned = clean_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", cleaned)
    if not remove_stopwords:
        return [tok for tok in raw_tokens if len(tok) > 1]
    return [
        tok for tok in raw_tokens if len(tok) > 1 and tok not in CUSTOM_STOP_WORDS
    ]


def preprocess_for_vectorizer(text: Optional[str]) -> str:
    """Return a whitespace-joined token string ready for TfidfVectorizer."""
    return " ".join(tokenize(text, remove_stopwords=True))


class BookNLPPipeline:
    """Text preprocessing, TF-IDF vectorization, and weighted reader-profile builder."""

    def __init__(
        self,
        max_features: int = 1500,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.items_df: Optional[pd.DataFrame] = None
        self.item_tfidf_matrix = None
        self.feature_names: np.ndarray = np.array([])
        self.item_id_to_idx: Dict[str, int] = {}
        self.idx_to_item_id: Dict[int, str] = {}

    def build_item_corpus(self, items_df: pd.DataFrame) -> pd.Series:
        """
        Combine title, category, keywords, and blurb into a single weighted document per book.
        Keywords and categories are repeated to emphasize thematic tags during TF-IDF matching.
        """
        corpus: List[str] = []
        for _, row in items_df.iterrows():
            title_tokens = tokenize(row.get("title", ""))
            category_tokens = tokenize(row.get("category", ""))
            keyword_tokens = tokenize(row.get("keywords", ""))
            blurb_tokens = tokenize(row.get("blurb", ""))

            combined_tokens = (
                blurb_tokens
                + keyword_tokens * 2
                + category_tokens * 2
                + title_tokens
            )
            corpus.append(" ".join(combined_tokens))
        return pd.Series(corpus, index=items_df.index)

    def fit(self, items_df: pd.DataFrame) -> "BookNLPPipeline":
        """Fit the TF-IDF vectorizer on the catalog of books."""
        required_cols = {"item_id", "title", "category", "blurb", "keywords"}
        missing = required_cols - set(items_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in items_df: {sorted(missing)}")

        self.items_df = items_df.reset_index(drop=True).copy()
        self.item_id_to_idx = {
            str(item_id): idx for idx, item_id in enumerate(self.items_df["item_id"])
        }
        self.idx_to_item_id = {
            idx: item_id for item_id, idx in self.item_id_to_idx.items()
        }

        corpus = self.build_item_corpus(self.items_df)
        self.item_tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def transform_query(self, query_text: str):
        """Transform a free-text query or cold-start preference description into TF-IDF space."""
        if self.item_tfidf_matrix is None:
            raise RuntimeError("Call fit(items_df) before calling transform_query().")
        processed = preprocess_for_vectorizer(query_text)
        return self.vectorizer.transform([processed])

    def build_user_profiles(
        self,
        ratings_df: pd.DataFrame,
        rating_threshold_shift: float = 3.0,
        fallback_to_positive: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Build a weighted TF-IDF taste profile for each reader from their rated books.
        Ratings above `rating_threshold_shift` contribute positively, while lower ratings
        down-weight disliked themes.
        """
        if self.item_tfidf_matrix is None:
            raise RuntimeError("Call fit(items_df) before calling build_user_profiles().")

        profiles: Dict[str, np.ndarray] = {}
        dense_items = self.item_tfidf_matrix.toarray()
        num_features = dense_items.shape[1]

        for user_id, group in ratings_df.groupby("user_id"):
            weighted_sum = np.zeros(num_features, dtype=np.float64)
            total_abs_weight = 0.0
            positive_vectors: List[np.ndarray] = []
            all_user_vectors: List[np.ndarray] = []

            for _, row in group.iterrows():
                item_id = str(row["item_id"])
                if item_id not in self.item_id_to_idx:
                    continue
                idx = self.item_id_to_idx[item_id]
                vec = dense_items[idx]
                rating = float(row["rating"])
                weight = rating - rating_threshold_shift

                all_user_vectors.append(vec)
                if rating >= 4.0:
                    positive_vectors.append(vec * max(weight, 0.5))

                weighted_sum += vec * weight
                total_abs_weight += abs(weight)

            if total_abs_weight > 0 and np.any(weighted_sum > 0):
                profile_vec = np.maximum(weighted_sum, 0.0) / total_abs_weight
            elif fallback_to_positive and positive_vectors:
                profile_vec = np.mean(positive_vectors, axis=0)
            elif all_user_vectors:
                profile_vec = np.mean(all_user_vectors, axis=0)
            else:
                profile_vec = np.zeros(num_features, dtype=np.float64)

            norm = np.linalg.norm(profile_vec)
            if norm > 0:
                profile_vec = profile_vec / norm

            profiles[str(user_id)] = profile_vec

        return profiles

    def get_top_matching_keywords(
        self,
        query_vec,
        item_idx: int,
        top_k: int = 4,
    ) -> List[str]:
        """Extract the highest-contributing overlapping TF-IDF terms between a query and a book."""
        if self.item_tfidf_matrix is None or len(self.feature_names) == 0:
            return []

        q_dense = (
            query_vec.toarray().ravel()
            if hasattr(query_vec, "toarray")
            else np.asarray(query_vec).ravel()
        )
        d_dense = self.item_tfidf_matrix[item_idx].toarray().ravel()

        overlap_scores = q_dense * d_dense
        positive_indices = np.where(overlap_scores > 0)[0]

        if len(positive_indices) > 0:
            ranked_idx = positive_indices[np.argsort(-overlap_scores[positive_indices])[:top_k]]
            return [str(self.feature_names[i]) for i in ranked_idx]

        top_doc_idx = np.argsort(-d_dense)[:top_k]
        return [str(self.feature_names[i]) for i in top_doc_idx if d_dense[i] > 0]

    def rank_items_by_vector(
        self,
        query_vec,
        top_n: int = 5,
        exclude_item_ids: Optional[Sequence[str]] = None,
        top_keywords: int = 4,
    ) -> List[Dict[str, object]]:
        """Compute cosine similarity against all books and return Top-N matches with keyword explanations."""
        if self.item_tfidf_matrix is None or self.items_df is None:
            raise RuntimeError("Call fit(items_df) before calling rank_items_by_vector().")

        exclude = set(str(item_id) for item_id in (exclude_item_ids or []))

        # Normalize query_vec to 2D for cosine_similarity (handles both sparse matrices and 1D ndarray profiles)
        if hasattr(query_vec, "toarray"):
            # Sparse matrix from TfidfVectorizer.transform — already 2D (1, n_features)
            qv = query_vec
        else:
            qv = np.asarray(query_vec)
            if qv.ndim == 1:
                qv = qv.reshape(1, -1)

        similarities = cosine_similarity(qv, self.item_tfidf_matrix).ravel()
        ranked_indices = np.argsort(-similarities)

        results: List[Dict[str, object]] = []
        for idx in ranked_indices:
            item_id = str(self.idx_to_item_id[int(idx)])
            if item_id in exclude:
                continue

            score = float(similarities[idx])
            # Only collect items with positive similarity score
            if score <= 0:
                break

            matched_keywords = self.get_top_matching_keywords(query_vec, int(idx), top_k=top_keywords)
            row = self.items_df.iloc[int(idx)]
            results.append(
                {
                    "item_id": item_id,
                    "title": row.get("title", ""),
                    "author": row.get("author", ""),
                    "category": row.get("category", ""),
                    "blurb": row.get("blurb", ""),
                    "keywords": row.get("keywords", ""),
                    "score": score,
                    "matched_keywords": matched_keywords,
                }
            )

            if len(results) >= top_n:
                break

        return results