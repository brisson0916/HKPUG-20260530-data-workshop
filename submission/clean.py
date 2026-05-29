from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dataclean_workshop.cleaning import CleaningConfig, clean_transport_data


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

OUTPUT_COLUMNS = ["record_id", *SUBMISSION_FEATURES]


def clean(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _clean_frame(train), _clean_frame(test)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    config = CleaningConfig(filter_irrelevant_rows=False, drop_duplicates=False)
    cleaned = clean_transport_data(frame, config=config)
    if "record_id" not in cleaned.columns:
        raise ValueError("input data must contain record_id")

    for column in OUTPUT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA

    cleaned["record_id"] = cleaned["record_id"].astype(str)
    cleaned = cleaned.drop_duplicates(subset=["record_id"], keep="first")
    return cleaned[OUTPUT_COLUMNS].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--test-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_input)
    test = pd.read_csv(args.test_input)
    cleaned_train, cleaned_test = clean(train, test)

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_output).parent.mkdir(parents=True, exist_ok=True)
    cleaned_train.to_csv(args.train_output, index=False)
    cleaned_test.to_csv(args.test_output, index=False)


if __name__ == "__main__":
    main()
