from __future__ import annotations

import argparse
import json

from dataclean_workshop.train_model import (
    FIXED_MODEL_CONFIG,
    build_pipeline,
    evaluate_model,
    load_dataset,
    prepare_training_frame,
    split_training_frame,
)

SUBMISSION_FEATURES = [
    "origin_station",
    "destination_station",
    "district",
    "transport_type",
    "transport_detail",
    "mode",
    "service_level",
    "operator",
    "day_of_week",
    "is_holiday",
    "weather_condition",
    "country_code",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed workshop model on a cleaned candidate submission."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the cleaned submission CSV.",
    )
    parser.add_argument(
        "--labels",
        help=(
            "Optional labels CSV with record_id and target column. Use this for "
            "local tournament dry runs when the cleaned feature file does not "
            "include delay_risk."
        ),
    )
    parser.add_argument(
        "--target-col",
        default="delay_risk",
        help="Target column name. Default: delay_risk.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved for the test split. Default: 0.20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    return parser.parse_args()


def attach_labels_if_needed(
    df,
    labels_path: str | None,
    target_col: str,
):
    if target_col in df.columns:
        return df
    if labels_path is None:
        return df
    if "record_id" not in df.columns:
        raise ValueError("--labels requires the cleaned input to contain record_id")

    labels = load_dataset(labels_path)
    required = {"record_id", target_col}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels file is missing columns: {sorted(missing)}")
    if labels["record_id"].duplicated().any():
        raise ValueError("labels file contains duplicate record_id values")

    merged = df.merge(
        labels[["record_id", target_col]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if merged[target_col].isna().any():
        raise ValueError("labels file does not cover every cleaned record_id")
    return merged


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input)
    df = attach_labels_if_needed(df, args.labels, args.target_col)
    if "record_id" in df.columns:
        df = df.sort_values("record_id").reset_index(drop=True)

    X, y = prepare_training_frame(df, args.target_col, SUBMISSION_FEATURES)
    X_train, X_eval, y_train, y_eval, evaluation_mode = split_training_frame(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        ordered_split=True,
    )
    pipeline = build_pipeline(X_train, random_state=args.random_state)
    metrics = evaluate_model(pipeline, X_train, X_eval, y_train, y_eval)

    summary = {
        "input": args.input,
        "target_col": args.target_col,
        "evaluation_mode": evaluation_mode,
        "feature_cols": SUBMISSION_FEATURES,
        "model_config": FIXED_MODEL_CONFIG,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_eval)),
        "f1_score": round(metrics["f1_score"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
