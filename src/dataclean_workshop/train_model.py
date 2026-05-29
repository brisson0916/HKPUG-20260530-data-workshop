from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

DEFAULT_BENCHMARK_FEATURES = [
    "origin_station",
    "destination_station",
    "district",
    "encoded_transport",
    "day_of_week",
    "is_holiday",
    "weather_condition",
    "country_code",
]

FIXED_MODEL_CONFIG = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "n_iter_no_change": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gradient boosting model on messy or cleaned transport data."
    )
    parser.add_argument("--input", required=True, help="Path to the CSV dataset.")
    parser.add_argument(
        "--target-col",
        required=True,
        help="Target column name, for example delay_risk.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        help=(
            "Feature columns to use. Accepts repeated names or comma-separated lists. "
            "If omitted, the fixed workshop benchmark feature set is used."
        ),
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help=(
            "Use every column except the target. Intended for local experimentation, "
            "not workshop scoring."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of rows reserved for the test split. Default: 0.20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model. Default: 42.",
    )
    parser.add_argument(
        "--model-out",
        help="Optional path to save the trained sklearn pipeline as a pickle.",
    )
    return parser.parse_args()


def parse_feature_columns(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None

    feature_columns: list[str] = []
    for value in raw_values:
        parts = [part.strip() for part in value.split(",")]
        feature_columns.extend([part for part in parts if part])
    return feature_columns


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None,
    use_all_features: bool = False,
) -> list[str]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    if feature_cols is None and use_all_features:
        resolved_feature_cols = [
            column for column in df.columns if column != target_col
        ]
        if not resolved_feature_cols:
            raise ValueError(
                "No feature columns were found after excluding the target column."
            )
    elif feature_cols is None:
        resolved_feature_cols = DEFAULT_BENCHMARK_FEATURES
    else:
        resolved_feature_cols = feature_cols

    missing_features = [
        column for column in resolved_feature_cols if column not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Feature columns not found: {missing_features}")

    return resolved_feature_cols


def prepare_training_frame(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None,
    use_all_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    resolved_feature_cols = resolve_feature_columns(
        df, target_col, feature_cols, use_all_features=use_all_features
    )

    working = df[resolved_feature_cols + [target_col]].copy()
    working[target_col] = pd.to_numeric(working[target_col], errors="coerce")
    working = working.dropna(subset=[target_col])

    X = coerce_numeric_like_columns(working[resolved_feature_cols])
    y = working[target_col].astype(int)
    return X, y


def coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        if converted[column].dtype == object:
            numeric_version = pd.to_numeric(converted[column], errors="coerce")
            non_null_ratio = numeric_version.notna().mean()
            if non_null_ratio >= 0.8:
                converted[column] = numeric_version
    return converted


def build_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        # Keep numeric NaN values intact so missing-value handling remains part of
        # the participant's cleaning quality rather than being silently repaired here.
        transformers.append(("numeric", "passthrough", numeric_columns))
    if categorical_columns:
        # Encode missing categorical values explicitly instead of imputing them to
        # the most frequent class, preserving missingness as model-visible signal.
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=random_state,
                    **FIXED_MODEL_CONFIG,
                ),
            ),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_train: pd.Series,
    y_eval: pd.Series,
) -> dict[str, float]:
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_eval)
    y_proba = pipeline.predict_proba(X_eval)[:, 1]

    return {
        "f1_score": f1_score(y_eval, y_pred),
        "roc_auc": roc_auc_score(y_eval, y_proba),
    }


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(pipeline, handle)


def should_use_ordered_split(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    use_all_features: bool,
) -> bool:
    return feature_cols is None and not use_all_features and "record_id" in df.columns


def maybe_order_training_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    use_all_features: bool,
) -> pd.DataFrame:
    if should_use_ordered_split(df, feature_cols, use_all_features):
        return df.sort_values("record_id").reset_index(drop=True)
    return df


def split_training_frame(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
    ordered_split: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if ordered_split:
        split_idx = max(1, min(len(X) - 1, int(len(X) * (1 - test_size))))
        X_train = X.iloc[:split_idx].reset_index(drop=True)
        X_eval = X.iloc[split_idx:].reset_index(drop=True)
        y_train = y.iloc[:split_idx].reset_index(drop=True)
        y_eval = y.iloc[split_idx:].reset_index(drop=True)
        return X_train, X_eval, y_train, y_eval, "ordered_record_split"

    stratify = y if y.nunique() > 1 else None
    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_eval, y_train, y_eval, "random_split"


def main() -> None:
    args = parse_args()
    requested_feature_cols = parse_feature_columns(args.feature_cols)
    use_all_features = args.all_features and requested_feature_cols is None

    df = load_dataset(args.input)
    df = maybe_order_training_dataframe(df, requested_feature_cols, use_all_features)
    feature_cols = resolve_feature_columns(
        df,
        args.target_col,
        requested_feature_cols,
        use_all_features=use_all_features,
    )
    X, y = prepare_training_frame(
        df,
        args.target_col,
        feature_cols,
        use_all_features=use_all_features,
    )
    X_train, X_eval, y_train, y_eval, evaluation_mode = split_training_frame(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        ordered_split=should_use_ordered_split(
            df, requested_feature_cols, use_all_features
        ),
    )

    pipeline = build_pipeline(X_train, random_state=args.random_state)
    metrics = evaluate_model(pipeline, X_train, X_eval, y_train, y_eval)

    summary = {
        "input": str(args.input),
        "target_col": args.target_col,
        "evaluation_mode": evaluation_mode,
        "feature_policy": (
            "custom"
            if requested_feature_cols is not None
            else (
                "all_except_target" if use_all_features else "fixed_workshop_benchmark"
            )
        ),
        "feature_cols": feature_cols,
        "model_config": FIXED_MODEL_CONFIG,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_eval)),
        "f1_score": round(metrics["f1_score"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
    }
    print(json.dumps(summary, indent=2))

    if args.model_out:
        save_model(pipeline, args.model_out)


if __name__ == "__main__":
    main()
