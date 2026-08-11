#!/usr/bin/env python3
"""Refresh the dashboard's checked-in World Bank data snapshot."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.worldbank.org/v2"
SOURCE_ID = 2
START_YEAR = 2010
END_YEAR = 2024
COUNTRIES = {
    "BDI": "Burundi",
    "COD": "Democratic Republic of the Congo",
    "KEN": "Kenya",
    "RWA": "Rwanda",
    "SOM": "Somalia",
    "SSD": "South Sudan",
    "TZA": "Tanzania",
    "UGA": "Uganda",
}
INDICATORS = {
    "NY.GDP.PCAP.KD": "gdp_per_capita_constant_2015_usd",
    "NY.GDP.PCAP.KD.ZG": "gdp_per_capita_growth_annual_pct",
    "SP.DYN.LE00.IN": "life_expectancy_years",
    "SE.SEC.ENRR": "secondary_enrolment_gross_pct",
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def request_json(
    session: requests.Session,
    url: str,
    params: dict,
    max_attempts: int = 5,
) -> tuple[list, str]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) < 2:
                raise ValueError(f"Unexpected API response: {str(payload)[:300]}")
            return payload, response.url
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == max_attempts:
                break
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"World Bank request failed: {last_error}")


def fetch_indicator(
    session: requests.Session, country_codes: str, indicator_code: str
) -> tuple[list[dict], list[dict]]:
    url = f"{BASE_URL}/country/{country_codes}/indicator/{indicator_code}"
    page = 1
    raw_pages: list[dict] = []
    records: list[dict] = []
    while True:
        params = {
            "date": f"{START_YEAR}:{END_YEAR}",
            "format": "json",
            "source": SOURCE_ID,
            "per_page": 20000,
            "page": page,
        }
        payload, request_url = request_json(session, url, params)
        metadata = payload[0] or {}
        page_records = payload[1] or []
        raw_pages.append(
            {
                "request_url": request_url,
                "metadata": metadata,
                "records": page_records,
            }
        )
        records.extend(page_records)
        if page >= int(metadata.get("pages", 1)):
            break
        page += 1
    return raw_pages, records


def main() -> None:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "EAC-Quality-of-Life-Streamlit-Capstone/1.0"}
    )
    country_codes = ";".join(COUNTRIES)
    raw_bundle = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": "World Bank Indicators API v2",
        "source_id": SOURCE_ID,
        "study_period": {"start_year": START_YEAR, "end_year": END_YEAR},
        "countries": COUNTRIES,
        "indicators": INDICATORS,
        "responses": {},
    }
    tidy_rows: list[dict] = []

    for indicator_code, output_column in INDICATORS.items():
        print(f"Fetching {indicator_code}...")
        raw_pages, records = fetch_indicator(session, country_codes, indicator_code)
        raw_bundle["responses"][indicator_code] = {"pages": raw_pages}
        for record in records:
            tidy_rows.append(
                {
                    "country_code": record.get("countryiso3code"),
                    "year": record.get("date"),
                    "indicator_code": indicator_code,
                    "variable": output_column,
                    "value": record.get("value"),
                }
            )

    tidy = pd.DataFrame(tidy_rows)
    tidy["year"] = pd.to_numeric(tidy["year"], errors="coerce")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.loc[
        tidy["country_code"].isin(COUNTRIES)
        & tidy["year"].between(START_YEAR, END_YEAR)
    ].copy()
    tidy["year"] = tidy["year"].astype(int)
    tidy = tidy.drop_duplicates(["country_code", "year", "variable"], keep="first")

    wide_values = (
        tidy.pivot(index=["country_code", "year"], columns="variable", values="value")
        .reset_index()
    )
    wide_values.columns.name = None
    grid = pd.MultiIndex.from_product(
        [COUNTRIES.keys(), range(START_YEAR, END_YEAR + 1)],
        names=["country_code", "year"],
    ).to_frame(index=False)
    grid["country"] = grid["country_code"].map(COUNTRIES)
    wide = grid.merge(wide_values, on=["country_code", "year"], how="left")
    for output_column in INDICATORS.values():
        if output_column not in wide:
            wide[output_column] = pd.NA
    wide = wide[
        ["country_code", "country", "year", *INDICATORS.values()]
    ].sort_values(["country_code", "year"])

    csv_path = DATA_DIR / "world_bank_eac_2010_2024.csv"
    raw_path = DATA_DIR / "world_bank_eac_2010_2024_raw.json"
    metadata_path = DATA_DIR / "snapshot_metadata.json"
    wide.to_csv(csv_path, index=False)
    raw_path.write_text(
        json.dumps(raw_bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "retrieved_at_utc": raw_bundle["retrieved_at_utc"],
                "source": "World Bank World Development Indicators",
                "source_url": "https://api.worldbank.org/v2",
                "rows": len(wide),
                "countries": len(COUNTRIES),
                "start_year": START_YEAR,
                "end_year": END_YEAR,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(wide)} rows to {csv_path}")


if __name__ == "__main__":
    main()
