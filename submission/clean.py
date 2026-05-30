from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-input")
    parser.add_argument("--test-output")
    return parser.parse_args()


# Start input your code
#
#
#
# End of your code


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_input)
    # Start input your code
    #
    #
    #
    # End of your code
    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    cleaned_train.to_csv(args.train_output, index=False)

    if args.test_input or args.test_output:
        if not args.test_input or not args.test_output:
            raise ValueError("--test-input and --test-output must be provided together")
        test = pd.read_csv(args.test_input)
        cleaned_test = _clean_frame(test)
        Path(args.test_output).parent.mkdir(parents=True, exist_ok=True)
        cleaned_test.to_csv(args.test_output, index=False)


if __name__ == "__main__":
    main()
