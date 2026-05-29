from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import unquote

import pandas as pd


DISTRICT_MAP = {
    "Central Station": "Central and Western",
    "Wan Chai": "Wan Chai",
    "Causeway Bay": "Wan Chai",
    "Admiralty": "Central and Western",
    "Tsim Sha Tsui": "Yau Tsim Mong",
    "Mong Kok": "Yau Tsim Mong",
    "Sha Tin": "Sha Tin",
    "Tsuen Wan": "Tsuen Wan",
    "Kennedy Town": "Central and Western",
    "North Point": "Eastern",
}

DAY_MAP = {
    "mon": "Mon",
    "tue": "Tue",
    "wed": "Wed",
    "thu": "Thu",
    "fri": "Fri",
    "sat": "Sat",
    "sun": "Sun",
}

WEATHER_MAP = {
    "sunny": "Sunny",
    "cloudy": "Cloudy",
    "rain": "Rain",
    "heavy rain": "Heavy Rain",
    "heavy_rain": "Heavy Rain",
    "heavyrain": "Heavy Rain",
    "wx_s": "Sunny",
    "wx.sun.sig": "Sunny",
    "wx-s-v2": "Sunny",
    "wx_c": "Cloudy",
    "wx.cld.ops": "Cloudy",
    "cld-shift": "Cloudy",
    "wx_r": "Rain",
    "wx.rn.alert": "Rain",
    "rn-v2": "Rain",
    "wx_hr": "Heavy Rain",
    "wx.hr.critical": "Heavy Rain",
    "hrn-ops": "Heavy Rain",
}

MISSING_WEATHER_VALUES = {
    "unknown",
    "clr",
    "heavy-rn",
    "rain??",
    "tbd",
    "wx_unknown",
    "wx/legacy",
    "system",
    "audit",
    "test",
}

COUNTRY_CODE_MAP = {
    "hk": "HK",
    "hkg": "HK",
    "geo::hkg": "HK",
    "geo::852": "HK",
    "852": "HK",
    "territory-hk": "HK",
    "hk-zone": "HK",
    "legacy": "HK",
    "mo": "MO",
    "mac": "MO",
    "cn": "CN",
    "chn": "CN",
    "na": "NA",
    "na_region": "NA",
}

COUNTRY_LIKE_VALUES = set(COUNTRY_CODE_MAP)
WEATHER_LIKE_VALUES = set(WEATHER_MAP) | MISSING_WEATHER_VALUES

CANONICAL_STATION_ALIASES = {
    "Central Station": [
        "central station",
        "central stn",
        "cen station",
        "central",
        "c. station",
        "stn_central",
        "src::stn_central::l2",
        "src-stn_central",
    ],
    "Wan Chai": [
        "wan chai",
        "wanchai",
        "w chai",
        "wanchai_stop",
        "src::wanchai_stop::l2",
        "src-wanchai_stop",
        "dst::wanchai_stop::l2",
        "dst-wanchai_stop",
    ],
    "Causeway Bay": [
        "causeway bay",
        "causewaybay",
        "cwb",
        "causewaybay",
        "cwb_hub",
        "src::cwb_hub::l2",
        "dst::cwb_hub::l2",
        "src-cwb_hub",
        "dst-cwb_hub",
    ],
    "Admiralty": [
        "admiralty",
        "admiralty station",
        "admiraltyy",
        "adm",
        "adm_xfer",
        "src::adm_xfer::l2",
        "dst::adm_xfer::l2",
        "src-adm_xfer",
        "dst-adm_xfer",
    ],
    "Tsim Sha Tsui": [
        "tsim sha tsui",
        "tst",
        "tsim sha tsui east",
        "tsimshatsui",
        "tst_east",
        "src::tst_east::l2",
        "dst::tst_east::l2",
        "src-tst_east",
        "dst-tst_east",
    ],
    "Mong Kok": [
        "mong kok",
        "mk",
        "mongkok",
        "mongkok",
        "mk_hub",
        "src::mk_hub::l2",
        "dst::mk_hub::l2",
        "src-mk_hub",
        "dst-mk_hub",
    ],
    "Sha Tin": [
        "sha tin",
        "shatin",
        "shatin_term",
        "src::shatin_term::l2",
        "dst::shatin_term::l2",
        "src-shatin_term",
        "dst-shatin_term",
        "s tin",
    ],
    "Tsuen Wan": [
        "tsuen wan",
        "tsuenwan",
        "tswan",
        "twn",
        "tsw_line",
        "src::tsw_line::l2",
        "dst::tsw_line::l2",
        "src-tsw_line",
        "dst-tsw_line",
    ],
    "Kennedy Town": [
        "kennedy town",
        "kennedytown",
        "k town",
        "kennedy tn",
        "ktown_stop",
        "src::ktown_stop::l2",
        "dst::ktown_stop::l2",
        "src-ktown_stop",
        "dst-ktown_stop",
    ],
    "North Point": [
        "north point",
        "north pt",
        "northpoint",
        "np",
        "np_ferry",
        "src::np_ferry::l2",
        "dst::np_ferry::l2",
        "src-np_ferry",
        "dst-np_ferry",
    ],
}

STATION_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in CANONICAL_STATION_ALIASES.items()
    for alias in aliases
}

STATION_COMPACT_ALIAS_TO_CANONICAL = {
    re.sub(r"[\s._:-]+", "", alias): canonical
    for alias, canonical in STATION_ALIAS_TO_CANONICAL.items()
}

TRANSPORT_TYPE_MAP = {
    "bus": "bus",
    "tram": "tram",
    "ferry": "ferry",
}

TRANSPORT_DETAIL_MAP = {
    "general": "general",
    "airport": "airport",
    "night": "night",
    "crossharbour": "crossharbour",
    "cross-harbour": "crossharbour",
}

MODE_MAP = {
    "local": "local",
    "express": "express",
}

SERVICE_LEVEL_MAP = {
    "standard": "standard",
    "premium": "premium",
}

OPERATOR_MAP = {
    "kmb": "KMB",
    "k.m.b.": "KMB",
    "kowloon motor bus": "KMB",
    "ctb": "CTB",
    "citybus": "CTB",
    "hkkf": "HKKF",
    "hk ferry": "HKKF",
    "hong kong ferry": "HKKF",
}

TRANSPORT_TYPE_ALIASES = {
    "bus": ["bus"],
    "tram": ["tram"],
    "ferry": ["ferry"],
}

TRANSPORT_DETAIL_ALIASES = {
    "airport": ["airport"],
    "night": ["night"],
    "crossharbour": ["crossharbour", "cross-harbour", "crossharb", "xhbr"],
}

MODE_ALIASES = {
    "local": ["local", "loc"],
    "express": ["express", "exp"],
}

SERVICE_LEVEL_ALIASES = {
    "standard": ["standard", "std"],
    "premium": ["premium", "prem"],
}

OPERATOR_ALIASES = {
    "KMB": ["kmb", "k.m.b.", "kowloon motor bus", "kowloon"],
    "CTB": ["ctb", "citybus"],
    "HKKF": ["hkkf", "hk ferry", "hong kong ferry", "ferryhk"],
}

NUMERIC_COLUMNS = [
    "fare_hkd",
    "distance_km",
    "scheduled_duration_min",
    "hour_of_day",
    "is_holiday",
    "delay_risk",
]


@dataclass(slots=True)
class CleaningConfig:
    decode_station_names: bool = True
    normalize_station_aliases: bool = True
    filter_irrelevant_rows: bool = True
    repair_field_drift: bool = True
    remap_districts: bool = True
    normalize_day_of_week: bool = True
    normalize_weather: bool = True
    normalize_country_code: bool = True
    normalize_encoded_transport: bool = True
    extract_transport_fields: bool = True
    coerce_numeric_columns: bool = True
    drop_duplicates: bool = True
    drop_outliers: bool = False


def clean_transport_data(
    df: pd.DataFrame, config: CleaningConfig | None = None
) -> pd.DataFrame:
    """Starter cleaning pipeline for workshop use.

    This intentionally keeps the steps explicit so participants can swap out
    or refine the logic as they work through the challenge.
    """
    cfg = config or CleaningConfig()
    cleaned = df.copy()

    if cfg.decode_station_names:
        cleaned = decode_station_names(cleaned)

    if cfg.normalize_station_aliases:
        cleaned = normalize_station_names(cleaned)

    if cfg.filter_irrelevant_rows:
        cleaned = filter_irrelevant_rows(cleaned)

    if cfg.repair_field_drift:
        cleaned = repair_known_field_drift(cleaned)

    if cfg.remap_districts:
        cleaned = remap_district_from_origin(cleaned)

    if cfg.normalize_day_of_week:
        cleaned = normalize_day_of_week(cleaned)

    if cfg.normalize_weather:
        cleaned = normalize_weather_condition(cleaned)

    if cfg.normalize_country_code:
        cleaned = normalize_country_codes(cleaned)

    if cfg.normalize_encoded_transport:
        cleaned = normalize_encoded_transport_column(cleaned)

    if cfg.extract_transport_fields:
        cleaned = extract_transport_features(cleaned)

    if cfg.coerce_numeric_columns:
        cleaned = coerce_numeric_columns(cleaned, NUMERIC_COLUMNS)

    if cfg.drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if cfg.drop_outliers:
        cleaned = drop_simple_outliers(cleaned)

    return cleaned


def decode_station_names(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("origin_station", "destination_station"):
        if column in df.columns:
            df[column] = df[column].map(_decode_if_string)
    return df


def normalize_station_names(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("origin_station", "destination_station"):
        if column not in df.columns:
            continue
        df[column] = df[column].map(_normalize_station_name)
    return df


def remap_district_from_origin(df: pd.DataFrame) -> pd.DataFrame:
    if {"origin_station", "district"}.issubset(df.columns):
        df["district"] = df["origin_station"].map(DISTRICT_MAP).fillna(df["district"])
    return df


def normalize_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    if "day_of_week" not in df.columns:
        return df

    df["day_of_week"] = df["day_of_week"].map(
        lambda value: DAY_MAP.get(str(value).strip().lower(), value)
        if pd.notna(value)
        else value
    )
    return df


def normalize_weather_condition(df: pd.DataFrame) -> pd.DataFrame:
    if "weather_condition" not in df.columns:
        return df

    df["weather_condition"] = df["weather_condition"].map(_normalize_weather_value)
    return df


def normalize_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "country_code" not in df.columns:
        return df

    df["country_code"] = df["country_code"].map(_normalize_country_value)
    return df


def normalize_encoded_transport_column(df: pd.DataFrame) -> pd.DataFrame:
    if "encoded_transport" not in df.columns:
        return df

    df["encoded_transport"] = df["encoded_transport"].map(normalize_encoded_transport)
    return df


def normalize_encoded_transport(value: object) -> object:
    if pd.isna(value):
        return value

    parsed = parse_encoded_transport(value)
    transport_type = parsed["transport_type"]
    mode = parsed["mode"]
    service_level = parsed["service_level"]
    operator = parsed["operator"]

    if any(pd.isna(part) for part in [transport_type, mode, service_level, operator]):
        return _clean_transport_text(value)

    detail = parsed["transport_detail"]
    type_and_detail = (
        transport_type
        if pd.isna(detail) or detail == "general"
        else f"{transport_type}-{detail}"
    )
    return f"{type_and_detail}_{mode}_{service_level}_{operator}"


def extract_transport_features(df: pd.DataFrame) -> pd.DataFrame:
    if "encoded_transport" not in df.columns:
        return df

    extracted = df["encoded_transport"].map(parse_encoded_transport)
    extracted_df = pd.DataFrame(extracted.tolist(), index=df.index)

    for column in extracted_df.columns:
        df[column] = extracted_df[column]
    return df


def parse_encoded_transport(value: object) -> dict[str, object]:
    result = {
        "transport_type": pd.NA,
        "transport_detail": pd.NA,
        "mode": pd.NA,
        "service_level": pd.NA,
        "operator": pd.NA,
    }

    if pd.isna(value):
        return result

    cleaned_text = _clean_transport_text(value)
    lowered = cleaned_text.lower()

    transport_type = _find_alias_value(lowered, TRANSPORT_TYPE_ALIASES)
    detail = _find_alias_value(lowered, TRANSPORT_DETAIL_ALIASES)
    mode = _find_alias_value(lowered, MODE_ALIASES)
    service_level = _find_alias_value(lowered, SERVICE_LEVEL_ALIASES)
    operator = _find_alias_value(lowered, OPERATOR_ALIASES)

    result["transport_type"] = transport_type if transport_type is not None else pd.NA
    result["transport_detail"] = detail if detail is not None else "general"
    result["mode"] = mode if mode is not None else pd.NA
    result["service_level"] = service_level if service_level is not None else pd.NA
    result["operator"] = operator if operator is not None else pd.NA
    return result


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def filter_irrelevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    encoded_transport = (
        df["encoded_transport"].astype(str) if "encoded_transport" in df.columns else ""
    )
    origin_station = (
        df["origin_station"].astype(str) if "origin_station" in df.columns else ""
    )
    destination_station = (
        df["destination_station"].astype(str)
        if "destination_station" in df.columns
        else ""
    )

    mask = pd.Series(False, index=df.index)
    if "survey_code" in df.columns:
        mask |= df["survey_code"].astype(str).str.upper().eq("SYS")
    if "campaign_tag" in df.columns:
        mask |= df["campaign_tag"].isin(
            ["internal_audit", "staff_shuttle", "maintenance_ops"]
        )
    if "ops_comment_code" in df.columns:
        mask |= df["ops_comment_code"].astype(str).eq("OPS-SYS")
    if "service_note" in df.columns:
        mask |= (
            df["service_note"]
            .astype(str)
            .str.contains(
                r"test record|sandbox move|internal audit|staff shuttle|depot transfer",
                case=False,
                regex=True,
            )
        )
    if "encoded_transport" in df.columns:
        mask |= encoded_transport.str.contains(
            r"admin_move|test_run|maintenance_shift|staff_shuttle",
            case=False,
            regex=True,
        )
    mask |= origin_station.isin(["Depot", "Workshop", "Audit Hub"])
    mask |= destination_station.isin(["Depot", "Workshop", "Audit Hub"])

    return df.loc[~mask].reset_index(drop=True)


def repair_known_field_drift(df: pd.DataFrame) -> pd.DataFrame:
    if {"weather_condition", "country_code"}.issubset(df.columns):
        weather_norm = df["weather_condition"].map(_normalize_string_key)
        country_norm = df["country_code"].map(_normalize_string_key)
        swap_mask = weather_norm.isin(COUNTRY_LIKE_VALUES) & country_norm.isin(
            WEATHER_LIKE_VALUES
        )
        if swap_mask.any():
            weather_snapshot = df.loc[swap_mask, "weather_condition"].copy()
            df.loc[swap_mask, "weather_condition"] = df.loc[swap_mask, "country_code"]
            df.loc[swap_mask, "country_code"] = weather_snapshot

    return df


def drop_simple_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Very conservative starter rule set for obvious workshop outliers."""
    if "fare_hkd" in df.columns:
        df = df[df["fare_hkd"].isna() | (df["fare_hkd"] <= 200)]
    if "distance_km" in df.columns:
        df = df[df["distance_km"].isna() | (df["distance_km"] <= 100)]
    if "scheduled_duration_min" in df.columns:
        df = df[
            df["scheduled_duration_min"].isna() | (df["scheduled_duration_min"] <= 180)
        ]
    return df.reset_index(drop=True)


def _decode_if_string(value: object) -> object:
    return unquote(value) if isinstance(value, str) else value


def _normalize_lookup(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return value

    normalized_key = str(value).strip().lower()
    return mapping.get(normalized_key, value)


def _normalize_string_key(value: object) -> str:
    return str(value).strip().lower() if pd.notna(value) else ""


def _normalize_station_name(value: object) -> object:
    if pd.isna(value):
        return value

    raw_key = _normalize_string_key(value)
    normalized_key = raw_key.replace(".", " ").replace("-", " ").replace("_", " ")
    normalized_key = re.sub(r"\s+", " ", normalized_key).strip()
    compact_key = re.sub(r"[\s._:-]+", "", raw_key)

    if raw_key in STATION_ALIAS_TO_CANONICAL:
        return STATION_ALIAS_TO_CANONICAL[raw_key]
    if normalized_key in STATION_ALIAS_TO_CANONICAL:
        return STATION_ALIAS_TO_CANONICAL[normalized_key]
    if compact_key in STATION_COMPACT_ALIAS_TO_CANONICAL:
        return STATION_COMPACT_ALIAS_TO_CANONICAL[compact_key]

    return value


def _normalize_weather_value(value: object) -> object:
    if pd.isna(value):
        return value

    normalized_key = _normalize_string_key(value)
    if normalized_key in MISSING_WEATHER_VALUES:
        return pd.NA
    return WEATHER_MAP.get(normalized_key, value)


def _normalize_country_value(value: object) -> object:
    if pd.isna(value):
        return value

    normalized_key = _normalize_string_key(value)
    return COUNTRY_CODE_MAP.get(normalized_key, value)


def _clean_transport_text(value: object) -> str:
    text = unquote(str(value).strip())
    text = re.sub(r"^svc:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^legacy::", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^L2::", "", text, flags=re.IGNORECASE)

    meta_blob = re.search(r";value=(.*?);flag=", text, flags=re.IGNORECASE)
    if meta_blob:
        text = meta_blob.group(1)
    elif text.lower().startswith("meta") and ":" in text:
        text = text.split(":", 1)[1]

    legacy_blob = re.search(
        r"legacy\[src=.*?;kind=(.*?);tier=(.*?);run=(.*?);op=(.*?)\]",
        text,
        flags=re.IGNORECASE,
    )
    if legacy_blob:
        kind, tier, run, operator = legacy_blob.groups()
        text = f"{kind}_{run}_{tier}_{operator}"

    text = re.sub(r":v\d+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^op=", "", text, flags=re.IGNORECASE)
    text = re.sub(r";backup=.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(src=.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(batch=.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[run=", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"\[tier=", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"\[op=", "_", text, flags=re.IGNORECASE)
    text = text.replace("][", "_").replace("[", "").replace("]", "")
    text = text.replace("%20", "_").replace(" / ", "_").replace("|", "_")
    text = text.replace("::", "_").replace(";", "_")
    text = (
        text.replace("kind=", "")
        .replace("tier=", "")
        .replace("run=", "")
        .replace("op=", "")
    )
    text = text.replace("__", "_")
    text = (
        text.replace("apt", "airport")
        .replace("ngt", "night")
        .replace("xhbr", "crossharbour")
    )

    return text


def _find_alias_value(text: str, alias_map: dict[str, list[str]]) -> str | None:
    for canonical, aliases in alias_map.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in text:
                return canonical
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply starter cleaning or transport extraction to a transport CSV."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--extract-transport-only",
        action="store_true",
        help=(
            "Only normalize encoded_transport and append extracted transport fields. "
            "This is useful for quick parsing experiments before fuller cleaning."
        ),
    )
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.extract_transport_only:
        result = normalize_encoded_transport_column(df.copy())
        result = extract_transport_features(result)
    else:
        result = clean_transport_data(df)

    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    run_cli()
