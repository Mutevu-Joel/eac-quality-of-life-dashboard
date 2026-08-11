"""Reusable data preparation and analytics for the EAC Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

GDP_LEVEL = "gdp_per_capita_constant_2015_usd"
GDP_GROWTH = "gdp_per_capita_growth_annual_pct"
LIFE = "life_expectancy_years"
EDUCATION = "secondary_enrolment_gross_pct"
CORE_METRICS = [GDP_LEVEL, GDP_GROWTH, LIFE, EDUCATION]

METRIC_META = {
    GDP_LEVEL: {
        "label": "Real GDP per capita",
        "short_label": "GDP per capita",
        "unit": "constant 2015 US$",
        "format": ",.0f",
    },
    GDP_GROWTH: {
        "label": "GDP per capita growth",
        "short_label": "GDP growth",
        "unit": "annual %",
        "format": ".1f",
    },
    LIFE: {
        "label": "Life expectancy at birth",
        "short_label": "Life expectancy",
        "unit": "years",
        "format": ".1f",
    },
    EDUCATION: {
        "label": "Secondary-school enrolment",
        "short_label": "Secondary enrolment",
        "unit": "% gross",
        "format": ".1f",
    },
}

REQUIRED_COLUMNS = ["country_code", "country", "year", *CORE_METRICS]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the checked-in World Bank snapshot."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dashboard dataset was not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["country_code"] = frame["country_code"].astype(str).str.upper().str.strip()
    frame = frame.loc[frame["country_code"].isin(COUNTRIES)].copy()
    frame["country"] = frame["country_code"].map(COUNTRIES)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.dropna(subset=["year"])
    frame["year"] = frame["year"].astype(int)
    frame = frame.loc[frame["year"].between(START_YEAR, END_YEAR)].copy()

    for column in CORE_METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.loc[frame[GDP_LEVEL].le(0), GDP_LEVEL] = np.nan
    frame.loc[~frame[LIFE].between(20, 100) & frame[LIFE].notna(), LIFE] = np.nan
    frame.loc[
        ~frame[EDUCATION].between(0, 200) & frame[EDUCATION].notna(), EDUCATION
    ] = np.nan

    return (
        frame.drop_duplicates(["country_code", "year"], keep="first")
        .sort_values(["country_code", "year"])
        .reset_index(drop=True)
    )


def filter_data(
    frame: pd.DataFrame,
    country_codes: Iterable[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Apply the dashboard's country and year slicers."""
    selected_codes = list(country_codes)
    return frame.loc[
        frame["country_code"].isin(selected_codes)
        & frame["year"].between(start_year, end_year)
    ].copy()


def endpoint_changes(
    frame: pd.DataFrame,
    value_column: str,
    start_year: int,
    end_year: int,
    exact_endpoints: bool = False,
) -> pd.DataFrame:
    """Calculate country changes using exact or first/last available endpoints."""
    output_columns = [
        "country_code",
        "country",
        "start_year",
        "end_year",
        "start_value",
        "end_value",
        "span_years",
        "absolute_change",
        "percent_change",
        "annualized_percent",
        "annualized_absolute_change",
        "exact_requested_endpoints",
    ]
    rows: list[dict] = []
    window = frame.loc[frame["year"].between(start_year, end_year)].copy()

    for country_code, group in window.groupby("country_code"):
        group = group.sort_values("year")
        observed = group[["year", value_column]].dropna().drop_duplicates("year")
        if observed.empty:
            continue

        if exact_endpoints:
            first_rows = observed.loc[observed["year"].eq(start_year)]
            last_rows = observed.loc[observed["year"].eq(end_year)]
            if first_rows.empty or last_rows.empty:
                continue
            first = first_rows.iloc[0]
            last = last_rows.iloc[0]
        else:
            if len(observed) < 2:
                continue
            first = observed.iloc[0]
            last = observed.iloc[-1]

        actual_start = int(first["year"])
        actual_end = int(last["year"])
        span_years = actual_end - actual_start
        if span_years <= 0:
            continue

        start_value = float(first[value_column])
        end_value = float(last[value_column])
        absolute_change = end_value - start_value
        percent_change = (
            100 * absolute_change / start_value if start_value != 0 else np.nan
        )
        annualized_percent = (
            100 * ((end_value / start_value) ** (1 / span_years) - 1)
            if start_value > 0 and end_value > 0
            else np.nan
        )
        rows.append(
            {
                "country_code": country_code,
                "country": COUNTRIES[country_code],
                "start_year": actual_start,
                "end_year": actual_end,
                "start_value": start_value,
                "end_value": end_value,
                "span_years": span_years,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "annualized_percent": annualized_percent,
                "annualized_absolute_change": absolute_change / span_years,
                "exact_requested_endpoints": (
                    actual_start == start_year and actual_end == end_year
                ),
            }
        )

    return pd.DataFrame(rows, columns=output_columns)


def completeness_table(
    frame: pd.DataFrame, start_year: int, end_year: int
) -> pd.DataFrame:
    """Return percentage completeness for every country and metric."""
    expected_years = end_year - start_year + 1
    window = frame.loc[frame["year"].between(start_year, end_year)]
    rows: list[dict] = []
    for country_code, country in COUNTRIES.items():
        country_frame = window.loc[window["country_code"].eq(country_code)]
        if country_frame.empty:
            continue
        for metric in CORE_METRICS:
            available = int(country_frame[metric].notna().sum())
            rows.append(
                {
                    "country_code": country_code,
                    "country": country,
                    "metric": metric,
                    "metric_label": METRIC_META[metric]["short_label"],
                    "available_years": available,
                    "expected_years": expected_years,
                    "missing_years": expected_years - available,
                    "completeness_pct": 100 * available / expected_years,
                }
            )
    return pd.DataFrame(rows)


def latest_available_snapshot(
    frame: pd.DataFrame, value_column: str, end_year: int
) -> pd.DataFrame:
    """Select each country's most recent available observation up to end_year."""
    rows: list[dict] = []
    for country_code, group in frame.loc[frame["year"].le(end_year)].groupby(
        "country_code"
    ):
        observed = group[["country", "year", value_column]].dropna().sort_values("year")
        if observed.empty:
            continue
        latest = observed.iloc[-1]
        rows.append(
            {
                "country_code": country_code,
                "country": latest["country"],
                "observed_year": int(latest["year"]),
                "value": float(latest[value_column]),
            }
        )
    return pd.DataFrame(rows)


def _minmax_100(series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if valid.empty:
        return result
    spread = valid.max() - valid.min()
    result.loc[valid.index] = 50.0 if spread == 0 else 100 * (valid - valid.min()) / spread
    return result


def balanced_scores(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    exact_endpoints: bool = False,
) -> pd.DataFrame:
    """Create a documented strength-minus-imbalance composite score."""
    gdp = endpoint_changes(
        frame, GDP_LEVEL, start_year, end_year, exact_endpoints
    )[["country_code", "country", "annualized_percent", "span_years"]].rename(
        columns={"annualized_percent": "economy_rate", "span_years": "economy_span"}
    )
    life = endpoint_changes(
        frame, LIFE, start_year, end_year, exact_endpoints
    )[["country_code", "annualized_percent", "span_years"]].rename(
        columns={"annualized_percent": "health_rate", "span_years": "health_span"}
    )
    education = endpoint_changes(
        frame, EDUCATION, start_year, end_year, exact_endpoints
    )[["country_code", "annualized_absolute_change", "span_years"]].rename(
        columns={
            "annualized_absolute_change": "education_rate",
            "span_years": "education_span",
        }
    )

    combined = gdp.merge(life, on="country_code", how="inner").merge(
        education, on="country_code", how="inner"
    )
    if combined.empty:
        return combined

    minimum_span = min(5, max(1, end_year - start_year))
    combined = combined.loc[
        combined[["economy_span", "health_span", "education_span"]]
        .ge(minimum_span)
        .all(axis=1)
    ].copy()
    if combined.empty:
        return combined

    for raw_column, score_column in [
        ("economy_rate", "economy_score"),
        ("health_rate", "health_score"),
        ("education_rate", "education_score"),
    ]:
        combined[score_column] = _minmax_100(combined[raw_column])

    components = ["economy_score", "health_score", "education_score"]
    combined["mean_progress_score"] = combined[components].mean(axis=1)
    combined["imbalance_penalty"] = combined[components].std(axis=1, ddof=0)
    combined["balanced_score"] = (
        combined["mean_progress_score"] - combined["imbalance_penalty"]
    )
    return combined.sort_values("balanced_score", ascending=False).reset_index(drop=True)


def iqr_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag, rather than delete, within-country 1.5 x IQR observations."""
    rows: list[dict] = []
    for country_code, country_frame in frame.groupby("country_code"):
        for metric in CORE_METRICS:
            values = country_frame[["year", metric]].dropna()
            if len(values) < 4:
                continue
            q1, q3 = values[metric].quantile([0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flagged = values.loc[~values[metric].between(lower, upper)]
            for record in flagged.itertuples(index=False):
                rows.append(
                    {
                        "country_code": country_code,
                        "country": COUNTRIES[country_code],
                        "year": int(record.year),
                        "metric": metric,
                        "metric_label": METRIC_META[metric]["short_label"],
                        "value": float(getattr(record, metric)),
                        "lower_iqr_fence": float(lower),
                        "upper_iqr_fence": float(upper),
                    }
                )
    return pd.DataFrame(rows)


def compute_kpis(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    exact_endpoints: bool = False,
) -> dict:
    """Calculate filter-responsive headline indicators."""
    gdp = endpoint_changes(frame, GDP_LEVEL, start_year, end_year, exact_endpoints)
    life = endpoint_changes(frame, LIFE, start_year, end_year, exact_endpoints)
    education = endpoint_changes(
        frame, EDUCATION, start_year, end_year, exact_endpoints
    )
    completeness = completeness_table(frame, start_year, end_year)

    education_leader = None
    if not education.empty:
        education_leader = education.sort_values("absolute_change", ascending=False).iloc[0]

    return {
        "country_count": int(frame["country_code"].nunique()),
        "median_gdp_change_pct": (
            float(gdp["percent_change"].median()) if not gdp.empty else np.nan
        ),
        "median_life_change_years": (
            float(life["absolute_change"].median()) if not life.empty else np.nan
        ),
        "education_leader_country": (
            str(education_leader["country"]) if education_leader is not None else "No data"
        ),
        "education_leader_change_pp": (
            float(education_leader["absolute_change"])
            if education_leader is not None
            else np.nan
        ),
        "average_completeness_pct": (
            float(completeness["completeness_pct"].mean())
            if not completeness.empty
            else np.nan
        ),
        "gdp_changes": gdp,
        "life_changes": life,
        "education_changes": education,
    }


def executive_summary(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    exact_endpoints: bool = False,
) -> list[str]:
    """Generate concise, evidence-linked summary bullets for the active filters."""
    kpis = compute_kpis(frame, start_year, end_year, exact_endpoints)
    bullets: list[str] = []

    gdp = kpis["gdp_changes"]
    if not gdp.empty:
        leader = gdp.sort_values("percent_change", ascending=False).iloc[0]
        bullets.append(
            f"{leader['country']} had the largest increase in income per person after "
            f"adjusting for inflation: {leader['percent_change']:.1f}% between "
            f"{int(leader['start_year'])} and {int(leader['end_year'])}."
        )

    life = kpis["life_changes"]
    if not life.empty:
        bullets.append(
            f"For the middle country in the group, life expectancy changed by "
            f"{life['absolute_change'].median():.1f} years."
        )

    education = kpis["education_changes"]
    if not education.empty:
        leader = education.sort_values("absolute_change", ascending=False).iloc[0]
        bullets.append(
            f"Secondary school enrolment improved most in {leader['country']}, rising by "
            f"{leader['absolute_change']:.1f} percentage points."
        )

    paired = frame[[GDP_LEVEL, LIFE]].dropna()
    if len(paired) >= 3 and paired[GDP_LEVEL].nunique() > 1 and paired[LIFE].nunique() > 1:
        correlation = paired[GDP_LEVEL].corr(paired[LIFE])
        bullets.append(
            f"The relationship between income per person and life expectancy is "
            f"{correlation:.2f}, based on {len(paired)} records. This pattern does not prove "
            "that one measure caused the other."
        )

    bullets.append(
        f"The dataset contains {kpis['average_completeness_pct']:.1f}% of the values expected "
        "for your choices. Check missing data before comparing countries."
    )
    return bullets
