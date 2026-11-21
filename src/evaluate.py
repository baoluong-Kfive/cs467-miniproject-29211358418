from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import (
    ContentBasedRecommender,
    HybridBookRecommender,
    PopularityRecommender,
    UserBasedCFRecommender,
)


def train_test_split_per_user(
    ratings_df: pd.DataFrame,
    n_holdout: int = 2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out `n_holdout` interactions per user (prioritizing liked items with rating >= 4)
    to evaluate top-N recommendation precision and rating prediction error.
    """
    rng = np.random.default_rng(random_state)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in ratings_df.groupby("user_id"):
        if len(group) <= n_holdout + 2:
            train_parts.append(group)
            continue

        liked = group[group["rating"] >= 4]
        if len(liked) >= n_holdout:
            chosen_idx = rng.choice(liked.index.to_numpy(), size=n_holdout, replace=False)
        else:
            chosen_idx = rng.choice(group.index.to_numpy(), size=n_holdout, replace=False)

        test_mask = group.index.isin(chosen_idx)
        test_parts.append(group[test_mask])
        train_parts.append(group[~test_mask])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def evaluate_ranking_metrics(
    recommendations_by_user: Dict[str, List[Dict[str, object]]],
    test_df: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """Compute Precision@K, Recall@K, and HitRate@K across evaluated users."""
    precisions: List[float] = []
    recalls: List[float] = []
    hits: List[float] = []

    for uid, group in test_df.groupby("user_id"):
        uid_str = str(uid)
        relevant_items = set(
            group.loc[group["rating"] >= relevance_threshold, "item_id"].astype(str).tolist()
        )
        if not relevant_items or uid_str not in recommendations_by_user:
            continue

        recs = recommendations_by_user[uid_str][:k]
        rec_ids = [str(r["item_id"]) for r in recs]
        n_rel_in_top_k = len(set(rec_ids) & relevant_items)

        precisions.append(n_rel_in_top_k / float(k))
        recalls.append(n_rel_in_top_k / float(len(relevant_items)))
        hits.append(1.0 if n_rel_in_top_k > 0 else 0.0)

    return {
        f"Precision@{k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"Recall@{k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        f"HitRate@{k}": round(float(np.mean(hits)) if hits else 0.0, 4),
    }


def evaluate_rating_error(
    cf_model: UserBasedCFRecommender,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute RMSE and MAE for numeric rating predictions on the held-out test set."""
    errors: List[float] = []
    for _, row in test_df.iterrows():
        pred = cf_model.predict_rating(str(row["user_id"]), str(row["item_id"]))
        actual = float(row["rating"])
        errors.append(pred - actual)

    err_arr = np.asarray(errors, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(err_arr**2))) if len(err_arr) > 0 else 0.0
    mae = float(np.mean(np.abs(err_arr))) if len(err_arr) > 0 else 0.0
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


def plot_dataset_distributions(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a two-panel chart showing rating distribution and book category breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left panel: Rating distribution
    rating_counts = ratings_df["rating"].value_counts().sort_index()
    bars1 = axes[0].bar(
        rating_counts.index.astype(str),
        rating_counts.values,
        color="#3B6E8C",
        width=0.55,
        edgecolor="#1F3B4D",
    )
    axes[0].set_title("Reader Rating Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rating (1 to 5 Stars)", fontsize=10)
    axes[0].set_ylabel("Number of Ratings", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # Right panel: Book count by category
    cat_counts = items_df["category"].value_counts().sort_values(ascending=True)
    bars2 = axes[1].barh(
        cat_counts.index,
        cat_counts.values,
        color="#4A8B71",
        edgecolor="#244739",
    )
    axes[1].set_title("Catalog Books by Category", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Books", fontsize=10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars2:
        width = bar.get_width()
        axes[1].annotate(
            f"{int(width)}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_comparison(
    eval_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a grouped bar chart comparing Precision@5, Recall@5, and HitRate@5 across models."""
    metrics = ["Precision@5", "Recall@5", "HitRate@5"]
    models = eval_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ["#2E5B88", "#4E9A76", "#D9822B"]

    for idx, metric in enumerate(metrics):
        values = eval_df[metric].tolist()
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width=width, label=metric, color=colors[idx])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

    ax.set_title("Offline Evaluation Comparison Across Recommendation Methods", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_recommendation_list(recs: List[Dict[str, object]]) -> str:
    """Format top-N recommendations into readable lines for terminal and sample_outputs.txt."""
    lines: List[str] = []
    for rank, item in enumerate(recs, start=1):
        keywords = ", ".join(item.get("matched_keywords", []))
        lines.append(
            f"  {rank}. [{item['item_id']}] {item['title']} by {item['author']} "
            f"({item['category']}) | Score: {item['score']:.4f} | Keywords: {keywords}"
        )
    return "\n".join(lines)


def run_full_evaluation(
    items_path: str = "data/items.csv",
    ratings_path: str = "data/ratings.csv",
    results_dir: str = "results",
) -> Tuple[pd.DataFrame, str]:
    """Run offline split evaluation, save plots, and generate outputs for all 3 required test cases."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)

    # Compute interaction matrix sparsity
    n_users = ratings_df["user_id"].nunique()
    n_items = items_df["item_id"].nunique()
    n_interactions = len(ratings_df)
    sparsity = 1.0 - (n_interactions / float(n_users * n_items))

    # 1. Train/test split evaluation
    train_df, test_df = train_test_split_per_user(ratings_df, n_holdout=2, random_state=42)
    test_users = sorted(test_df["user_id"].astype(str).unique().tolist())

    pop_model = PopularityRecommender(min_ratings=2).fit(items_df, train_df)
    cb_model = ContentBasedRecommender().fit(items_df, train_df)
    cf_cosine = UserBasedCFRecommender(similarity_metric="cosine", k_neighbors=5).fit(items_df, train_df)
    cf_pearson = UserBasedCFRecommender(similarity_metric="pearson", k_neighbors=5).fit(items_df, train_df)
    hybrid_model = HybridBookRecommender(content_weight=0.55, cf_weight=0.45, cf_metric="pearson").fit(
        items_df, train_df
    )

    pop_recs = {uid: pop_model.recommend(user_id=uid, top_n=5) for uid in test_users}
    cb_recs = {uid: cb_model.recommend_for_user(user_id=uid, top_n=5) for uid in test_users}
    cf_cos_recs = {uid: cf_cosine.recommend(user_id=uid, top_n=5) for uid in test_users}
    cf_pear_recs = {uid: cf_pearson.recommend(user_id=uid, top_n=5) for uid in test_users}
    hybrid_recs = {uid: hybrid_model.recommend_for_user(user_id=uid, top_n=5) for uid in test_users}

    cos_err = evaluate_rating_error(cf_cosine, test_df)
    pear_err = evaluate_rating_error(cf_pearson, test_df)

    rows = [
        {"Model": "Popularity Baseline", **evaluate_ranking_metrics(pop_recs, test_df, k=5), "RMSE": None, "MAE": None},
        {"Model": "NLP Content-Based", **evaluate_ranking_metrics(cb_recs, test_df, k=5), "RMSE": None, "MAE": None},
        {"Model": "User-Based CF (Cosine)", **evaluate_ranking_metrics(cf_cos_recs, test_df, k=5), **cos_err},
        {"Model": "User-Based CF (Pearson)", **evaluate_ranking_metrics(cf_pear_recs, test_df, k=5), **pear_err},
        {"Model": "Hybrid (NLP + Pearson CF)", **evaluate_ranking_metrics(hybrid_recs, test_df, k=5), "RMSE": None, "MAE": None},
    ]
    eval_df = pd.DataFrame(rows)

    # 2. Save charts
    plot_dataset_distributions(items_df, ratings_df, out_dir / "dataset_distribution.png")
    plot_model_comparison(eval_df, out_dir / "model_comparison.png")

    # 3. Fit full models on all data for the 3 required test cases
    full_cb = ContentBasedRecommender().fit(items_df, ratings_df)
    full_cf = UserBasedCFRecommender(similarity_metric="pearson", k_neighbors=5).fit(items_df, ratings_df)
    full_hybrid = HybridBookRecommender(content_weight=0.55, cf_weight=0.45, cf_metric="pearson").fit(
        items_df, ratings_df
    )

    # Test Case 1: Existing reader with >= 6 rated books (U01 has 7 rated books)
    tc1_user = "U01"
    tc1_history = ratings_df[ratings_df["user_id"] == tc1_user].merge(
        items_df[["item_id", "title", "category"]], on="item_id"
    )
    tc1_cb = full_cb.recommend_for_user(tc1_user, top_n=5)
    tc1_cf = full_cf.recommend(
        tc1_user,
        top_n=5,
        nlp_pipeline=full_cb.nlp,
        user_profiles=full_cb.user_profiles,
    )
    tc1_hybrid = full_hybrid.recommend_for_user(tc1_user, top_n=5)

    # Test Case 2: Required free-text search query
    tc2_query = "historical mystery set in Asia with detective"
    tc2_recs = full_cb.recommend_by_query(tc2_query, top_n=5)

    # Test Case 3: Cold-start reader with a 1-sentence reading preference
    tc3_sentence = "I enjoy cozy stories about warm neighborhood cafes, nature, and gentle everyday life."
    tc3_recs = full_cb.recommend_by_query(tc3_sentence, top_n=5)

    report_sections = [
        "========================================================================",
        "CS467 MINI-PROJECT (MP-A02): BOOK SUMMARY AND READER-TASTE RECOMMENDER",
        "========================================================================",
        f"Dataset Summary: {n_items} books | {n_users} readers | {n_interactions} ratings | Matrix Sparsity: {sparsity:.2%}",
        "",
        "--- OFFLINE EVALUATION RESULTS (Holdout: 2 liked items per user) ---",
        eval_df.to_string(index=False),
        "",
        "========================================================================",
        "TEST CASE 1: EXISTING READER RECOMMENDATIONS (User ID: U01, 7 rated books)",
        "========================================================================",
        "Rated History for U01:",
        "\n".join(
            f"  - [{r['item_id']}] {r['title']} ({r['category']}) -> Rating: {r['rating']}"
            for _, r in tc1_history.iterrows()
        ),
        "\n[1A] NLP Content-Based Reader Profile Top-5:",
        format_recommendation_list(tc1_cb),
        "\n[1B] User-Based Collaborative Filtering (Pearson) Top-5:",
        format_recommendation_list(tc1_cf),
        "\n[1C] Hybrid (NLP + Collaborative Filtering) Top-5:",
        format_recommendation_list(tc1_hybrid),
        "",
        "========================================================================",
        "TEST CASE 2: FREE-TEXT SEARCH QUERY",
        f"Query: '{tc2_query}'",
        "========================================================================",
        format_recommendation_list(tc2_recs),
        "",
        "========================================================================",
        "TEST CASE 3: COLD-START NEW READER (1-Sentence Preference)",
        f"Input: '{tc3_sentence}'",
        "========================================================================",
        format_recommendation_list(tc3_recs),
        "",
    ]

    full_report = "\n".join(report_sections)
    (out_dir / "sample_outputs.txt").write_text(full_report, encoding="utf-8")
    return eval_df, full_report


if __name__ == "__main__":
    _, report = run_full_evaluation()
    print(report)