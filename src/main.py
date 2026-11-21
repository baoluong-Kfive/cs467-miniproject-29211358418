from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluate import format_recommendation_list, run_full_evaluation
from recommender import (
    ContentBasedRecommender,
    HybridBookRecommender,
    PopularityRecommender,
    UserBasedCFRecommender,
)


def load_dataset(
    items_path: str = "data/items.csv",
    ratings_path: str = "data/ratings.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load book items and reader ratings CSV files."""
    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)
    return items_df, ratings_df


def run_single_request(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    user_id: Optional[str] = None,
    query: Optional[str] = None,
    method: str = "hybrid",
    top_n: int = 5,
) -> None:
    """Run a single recommendation request by user_id or free-text query."""
    cb_model = ContentBasedRecommender().fit(items_df, ratings_df)
    cf_model = UserBasedCFRecommender(similarity_metric="pearson", k_neighbors=5).fit(
        items_df, ratings_df
    )
    hybrid_model = HybridBookRecommender(
        content_weight=0.55, cf_weight=0.45, cf_metric="pearson"
    ).fit(items_df, ratings_df)
    pop_model = PopularityRecommender(min_ratings=2).fit(items_df, ratings_df)

    if query:
        recs = cb_model.recommend_by_query(query, top_n=top_n)
        print(f"\n=== Free-Text / Cold-Start Query: '{query}' (Top-{top_n}) ===")
        print(format_recommendation_list(recs))
        return

    target_user = user_id if user_id else "U01"
    if target_user not in set(ratings_df["user_id"].astype(str)):
        print(f"\nUser '{target_user}' not found in ratings.csv. Using Popularity baseline:")
        print(format_recommendation_list(pop_model.recommend(top_n=top_n)))
        return

    print(f"\n=== Recommendations for Reader '{target_user}' | Method: {method.upper()} (Top-{top_n}) ===")
    if method == "content":
        recs = cb_model.recommend_for_user(target_user, top_n=top_n)
    elif method == "cf":
        recs = cf_model.recommend(
            target_user,
            top_n=top_n,
            nlp_pipeline=cb_model.nlp,
            user_profiles=cb_model.user_profiles,
        )
    elif method == "popularity":
        recs = pop_model.recommend(user_id=target_user, top_n=top_n)
    else:
        recs = hybrid_model.recommend_for_user(target_user, top_n=top_n)

    print(format_recommendation_list(recs))


def interactive_cli(items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    """Interactive command-line loop for testing user profiles and text queries."""
    available_users = sorted(ratings_df["user_id"].astype(str).unique().tolist())
    print("\n--- CS467 Book Recommender Interactive CLI ---")
    print(f"Available users: {', '.join(available_users)}")
    print("Commands: 'user <UID> [hybrid|content|cf|popularity]', 'query <text>', or 'exit'")

    while True:
        try:
            raw_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive CLI.")
            break

        if not raw_input:
            continue
        if raw_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if raw_input.lower().startswith("user "):
            parts = raw_input.split()
            uid = parts[1].upper() if len(parts) > 1 else "U01"
            method = parts[2].lower() if len(parts) > 2 else "hybrid"
            run_single_request(items_df, ratings_df, user_id=uid, method=method, top_n=5)
        elif raw_input.lower().startswith("query "):
            q_text = raw_input[6:].strip()
            run_single_request(items_df, ratings_df, query=q_text, top_n=5)
        else:
            run_single_request(items_df, ratings_df, query=raw_input, top_n=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Book Summary and Reader-Taste Recommender CLI (CS467 Mini-Project)"
    )
    parser.add_argument("--items", default="data/items.csv", help="Path to items.csv")
    parser.add_argument("--ratings", default="data/ratings.csv", help="Path to ratings.csv")
    parser.add_argument("--results-dir", default="results", help="Output directory for charts and reports")
    parser.add_argument("--user", type=str, default=None, help="Target reader ID (e.g., U01)")
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["hybrid", "content", "cf", "popularity"],
        help="Recommendation method when querying by user ID",
    )
    parser.add_argument("--query", type=str, default=None, help="Free-text search or cold-start preference")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations to display")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run offline evaluation, save charts, and execute all 3 required test cases",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive CLI session",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.items).exists() or not Path(args.ratings).exists():
        raise FileNotFoundError("Missing data/items.csv or data/ratings.csv. Complete Step 1 first.")

    items_df, ratings_df = load_dataset(args.items, args.ratings)

    if args.interactive:
        interactive_cli(items_df, ratings_df)
        return

    if args.run_tests or (args.user is None and args.query is None):
        _, report = run_full_evaluation(
            items_path=args.items,
            ratings_path=args.ratings,
            results_dir=args.results_dir,
        )
        print(report)
        return

    run_single_request(
        items_df=items_df,
        ratings_df=ratings_df,
        user_id=args.user,
        query=args.query,
        method=args.method,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()