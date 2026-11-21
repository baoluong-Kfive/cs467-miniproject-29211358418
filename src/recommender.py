from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from nlp_pipeline import BookNLPPipeline


class PopularityRecommender:
    """Popularity baseline using Bayesian weighted average rating with a minimum count threshold."""

    def __init__(self, min_ratings: int = 2) -> None:
        self.min_ratings = min_ratings
        self.items_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.popular_df: Optional[pd.DataFrame] = None
        self.global_mean: float = 3.5

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "PopularityRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.global_mean = float(ratings_df["rating"].mean()) if len(ratings_df) > 0 else 3.5

        stats = (
            ratings_df.groupby("item_id")["rating"]
            .agg(rating_count="count", avg_rating="mean")
            .reset_index()
        )
        merged = items_df.merge(stats, on="item_id", how="left")
        merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
        merged["avg_rating"] = merged["avg_rating"].fillna(0.0)

        # Bayesian weighted score: (v / (v + m)) * R + (m / (v + m)) * C
        m = float(self.min_ratings)
        v = merged["rating_count"].astype(float)
        r = merged["avg_rating"].astype(float)
        merged["weighted_score"] = (v / (v + m)) * r + (m / (v + m)) * self.global_mean

        self.popular_df = merged.sort_values(
            by=["weighted_score", "rating_count", "avg_rating"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return self

    def recommend(
        self,
        user_id: Optional[str] = None,
        top_n: int = 5,
        exclude_item_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        if self.popular_df is None or self.ratings_df is None:
            raise RuntimeError("Call fit(items_df, ratings_df) before calling recommend().")

        excluded = {str(x) for x in (exclude_item_ids or [])}
        if user_id is not None:
            user_items = self.ratings_df.loc[
                self.ratings_df["user_id"] == str(user_id), "item_id"
            ].astype(str)
            excluded.update(user_items.tolist())

        eligible = self.popular_df[
            (self.popular_df["rating_count"] >= self.min_ratings)
            & (~self.popular_df["item_id"].astype(str).isin(excluded))
        ]
        if len(eligible) < top_n:
            fallback = self.popular_df[~self.popular_df["item_id"].astype(str).isin(excluded)]
            eligible = pd.concat([eligible, fallback]).drop_duplicates(subset=["item_id"])

        results: List[Dict[str, object]] = []
        for _, row in eligible.head(top_n).iterrows():
            kw_list = [k.strip() for k in str(row.get("keywords", "")).split(",") if k.strip()]
            if len(kw_list) <= 1:
                kw_list = str(row.get("keywords", "")).split()[:4]
            results.append(
                {
                    "item_id": str(row["item_id"]),
                    "title": str(row["title"]),
                    "author": str(row["author"]),
                    "category": str(row["category"]),
                    "score": round(float(row["weighted_score"]), 4),
                    "avg_rating": round(float(row["avg_rating"]), 2),
                    "rating_count": int(row["rating_count"]),
                    "matched_keywords": kw_list[:4],
                }
            )
        return results


class ContentBasedRecommender:
    """NLP Content-Based recommender supporting reader taste profiles and free-text queries."""

    def __init__(self, nlp_pipeline: Optional[BookNLPPipeline] = None) -> None:
        self.nlp = nlp_pipeline if nlp_pipeline is not None else BookNLPPipeline()
        self.items_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.user_profiles: Dict[str, np.ndarray] = {}

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "ContentBasedRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.nlp.fit(items_df)
        self.user_profiles = self.nlp.build_user_profiles(ratings_df)
        return self

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 5,
        exclude_item_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        if self.ratings_df is None:
            raise RuntimeError("Call fit(items_df, ratings_df) before recommending.")

        uid = str(user_id)
        if uid not in self.user_profiles:
            return []

        excluded = {str(x) for x in (exclude_item_ids or [])}
        seen = self.ratings_df.loc[self.ratings_df["user_id"] == uid, "item_id"].astype(str)
        excluded.update(seen.tolist())

        return self.nlp.rank_items_by_vector(
            query_vec=self.user_profiles[uid],
            top_n=top_n,
            exclude_item_ids=list(excluded),
        )

    def recommend_by_query(
        self,
        query_text: str,
        top_n: int = 5,
        exclude_item_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        query_vec = self.nlp.transform_query(query_text)
        return self.nlp.rank_items_by_vector(
            query_vec=query_vec,
            top_n=top_n,
            exclude_item_ids=exclude_item_ids,
        )

    def recommend_similar_items(
        self,
        item_id: str,
        top_n: int = 5,
    ) -> List[Dict[str, object]]:
        iid = str(item_id)
        if iid not in self.nlp.item_id_to_idx or self.nlp.item_tfidf_matrix is None:
            return []
        idx = self.nlp.item_id_to_idx[iid]
        item_vec = self.nlp.item_tfidf_matrix[idx]
        return self.nlp.rank_items_by_vector(
            query_vec=item_vec,
            top_n=top_n,
            exclude_item_ids=[iid],
        )


class UserBasedCFRecommender:
    """
    User-Based Collaborative Filtering with Cosine or Pearson similarity over co-rated items
    and mean-centered rating prediction.
    """

    def __init__(
        self,
        similarity_metric: Literal["cosine", "pearson"] = "pearson",
        k_neighbors: int = 5,
        min_co_rated: int = 1,
        shrinkage: float = 2.0,
    ) -> None:
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.min_co_rated = min_co_rated
        self.shrinkage = shrinkage

        self.items_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.rating_matrix: Optional[pd.DataFrame] = None
        self.user_means: Dict[str, float] = {}
        self.global_mean: float = 3.5
        self.similarity_matrix: Optional[pd.DataFrame] = None

    def _compute_pair_similarity(
        self,
        u_vec: np.ndarray,
        v_vec: np.ndarray,
        u_mean: float,
        v_mean: float,
    ) -> float:
        co_mask = (~np.isnan(u_vec)) & (~np.isnan(v_vec))
        n_common = int(np.sum(co_mask))
        if n_common < self.min_co_rated:
            return 0.0

        u_common = u_vec[co_mask]
        v_common = v_vec[co_mask]

        if self.similarity_metric == "pearson":
            u_centered = u_common - u_mean
            v_centered = v_common - v_mean
            denom = float(np.linalg.norm(u_centered) * np.linalg.norm(v_centered))
            if denom == 0.0:
                return 0.0
            raw_sim = float(np.dot(u_centered, v_centered) / denom)
        elif self.similarity_metric == "cosine":
            denom = float(np.linalg.norm(u_common) * np.linalg.norm(v_common))
            if denom == 0.0:
                return 0.0
            raw_sim = float(np.dot(u_common, v_common) / denom)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        # Apply significance weighting so single-item overlaps do not dominate
        weight_factor = n_common / (n_common + self.shrinkage)
        return raw_sim * weight_factor

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "UserBasedCFRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.global_mean = float(ratings_df["rating"].mean()) if len(ratings_df) > 0 else 3.5

        all_users = sorted(ratings_df["user_id"].astype(str).unique().tolist())
        all_items = items_df["item_id"].astype(str).tolist()

        pivot = ratings_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            aggfunc="mean",
        )
        self.rating_matrix = pivot.reindex(index=all_users, columns=all_items)

        self.user_means = {}
        for uid in all_users:
            row_vals = self.rating_matrix.loc[uid].dropna().values
            self.user_means[uid] = float(np.mean(row_vals)) if len(row_vals) > 0 else self.global_mean

        n_users = len(all_users)
        sim_mat = np.eye(n_users, dtype=np.float64)
        matrix_values = self.rating_matrix.values

        for i in range(n_users):
            for j in range(i + 1, n_users):
                sim = self._compute_pair_similarity(
                    matrix_values[i],
                    matrix_values[j],
                    self.user_means[all_users[i]],
                    self.user_means[all_users[j]],
                )
                sim_mat[i, j] = sim
                sim_mat[j, i] = sim

        self.similarity_matrix = pd.DataFrame(sim_mat, index=all_users, columns=all_users)
        return self

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating for (user_id, item_id) using mean-centered User-Based CF."""
        if self.rating_matrix is None or self.similarity_matrix is None:
            raise RuntimeError("Call fit(items_df, ratings_df) before predicting ratings.")

        uid = str(user_id)
        iid = str(item_id)
        base_mean = self.user_means.get(uid, self.global_mean)

        if uid not in self.rating_matrix.index or iid not in self.rating_matrix.columns:
            return base_mean

        item_ratings = self.rating_matrix[iid].dropna()
        if len(item_ratings) == 0:
            return base_mean

        neighbors = []
        for other_uid, other_rating in item_ratings.items():
            other_uid_str = str(other_uid)
            if other_uid_str == uid:
                continue
            sim = float(self.similarity_matrix.loc[uid, other_uid_str])
            if sim > 0:
                other_mean = self.user_means.get(other_uid_str, self.global_mean)
                neighbors.append((sim, float(other_rating) - other_mean))

        if not neighbors:
            # Fallback to item average blended with user mean
            item_avg = float(item_ratings.mean())
            return float(np.clip(0.5 * base_mean + 0.5 * item_avg, 1.0, 5.0))

        neighbors.sort(key=lambda x: x[0], reverse=True)
        top_neighbors = neighbors[: self.k_neighbors]

        numerator = sum(sim * dev for sim, dev in top_neighbors)
        denominator = sum(abs(sim) for sim, _ in top_neighbors)
        pred = base_mean + (numerator / denominator if denominator > 0 else 0.0)
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: str,
        top_n: int = 5,
        exclude_item_ids: Optional[Sequence[str]] = None,
        nlp_pipeline: Optional[BookNLPPipeline] = None,
        user_profiles: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, object]]:
        if self.items_df is None or self.ratings_df is None:
            raise RuntimeError("Call fit(items_df, ratings_df) before recommending.")

        uid = str(user_id)
        excluded = {str(x) for x in (exclude_item_ids or [])}
        seen = self.ratings_df.loc[self.ratings_df["user_id"] == uid, "item_id"].astype(str)
        excluded.update(seen.tolist())

        scored_items: List[Dict[str, object]] = []
        for _, row in self.items_df.iterrows():
            iid = str(row["item_id"])
            if iid in excluded:
                continue
            pred_score = self.predict_rating(uid, iid)

            matched_kw: List[str] = []
            if (
                nlp_pipeline is not None
                and user_profiles is not None
                and uid in user_profiles
                and iid in nlp_pipeline.item_id_to_idx
            ):
                idx = nlp_pipeline.item_id_to_idx[iid]
                matched_kw = nlp_pipeline.get_top_matching_keywords(
                    user_profiles[uid], idx, top_k=4
                )
            if not matched_kw:
                matched_kw = str(row.get("keywords", "")).split()[:4]

            scored_items.append(
                {
                    "item_id": iid,
                    "title": str(row["title"]),
                    "author": str(row["author"]),
                    "category": str(row["category"]),
                    "score": round(pred_score, 4),
                    "matched_keywords": matched_kw,
                }
            )

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_n]


class HybridBookRecommender:
    """Combines NLP Content-Based similarity with User-Based Collaborative Filtering."""

    def __init__(
        self,
        content_weight: float = 0.55,
        cf_weight: float = 0.45,
        cf_metric: Literal["cosine", "pearson"] = "pearson",
    ) -> None:
        self.content_weight = content_weight
        self.cf_weight = cf_weight
        self.cb_model = ContentBasedRecommender()
        self.cf_model = UserBasedCFRecommender(similarity_metric=cf_metric)
        self.items_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "HybridBookRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.cb_model.fit(items_df, ratings_df)
        self.cf_model.fit(items_df, ratings_df)
        return self

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 5,
        exclude_item_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        if self.items_df is None or self.ratings_df is None:
            raise RuntimeError("Call fit(items_df, ratings_df) before recommending.")

        uid = str(user_id)
        excluded = {str(x) for x in (exclude_item_ids or [])}
        seen = self.ratings_df.loc[self.ratings_df["user_id"] == uid, "item_id"].astype(str)
        excluded.update(seen.tolist())

        # Fetch full candidate scores from both models
        cb_all = self.cb_model.recommend_for_user(uid, top_n=len(self.items_df), exclude_item_ids=list(excluded))
        cf_all = self.cf_model.recommend(
            uid,
            top_n=len(self.items_df),
            exclude_item_ids=list(excluded),
            nlp_pipeline=self.cb_model.nlp,
            user_profiles=self.cb_model.user_profiles,
        )

        cb_map = {item["item_id"]: item for item in cb_all}
        cf_map = {item["item_id"]: item for item in cf_all}

        hybrid_results: List[Dict[str, object]] = []
        for iid, cf_item in cf_map.items():
            cb_sim = float(cb_map[iid]["score"]) if iid in cb_map else 0.0
            # Normalize CF predicted rating from [1, 5] into [0, 1] for clean blending
            cf_norm = (float(cf_item["score"]) - 1.0) / 4.0
            blended_score = self.content_weight * cb_sim + self.cf_weight * cf_norm

            matched_kw = (
                cb_map[iid]["matched_keywords"]
                if iid in cb_map and cb_map[iid]["matched_keywords"]
                else cf_item["matched_keywords"]
            )
            hybrid_results.append(
                {
                    "item_id": iid,
                    "title": cf_item["title"],
                    "author": cf_item["author"],
                    "category": cf_item["category"],
                    "score": round(blended_score, 4),
                    "cb_similarity": round(cb_sim, 4),
                    "cf_pred_rating": round(float(cf_item["score"]), 4),
                    "matched_keywords": matched_kw,
                }
            )

        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_n]