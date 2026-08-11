from pathlib import Path

import numpy as np

from data_processing import (
    EDUCATION,
    GDP_LEVEL,
    LIFE,
    balanced_scores,
    completeness_table,
    compute_kpis,
    endpoint_changes,
    filter_data,
    load_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "world_bank_eac_2010_2024.csv"


def test_snapshot_shape_and_key_uniqueness():
    frame = load_dataset(DATA_PATH)
    assert len(frame) == 120
    assert frame["country_code"].nunique() == 8
    assert frame["year"].min() == 2010
    assert frame["year"].max() == 2024
    assert not frame.duplicated(["country_code", "year"]).any()


def test_filters_and_kpis_are_responsive():
    frame = load_dataset(DATA_PATH)
    filtered = filter_data(frame, ["KEN", "RWA"], 2012, 2022)
    assert set(filtered["country_code"]) == {"KEN", "RWA"}
    assert filtered["year"].between(2012, 2022).all()
    kpis = compute_kpis(filtered, 2012, 2022, exact_endpoints=False)
    assert kpis["country_count"] == 2
    assert np.isfinite(kpis["median_gdp_change_pct"])
    assert np.isfinite(kpis["median_life_change_years"])


def test_endpoint_change_uses_disclosed_actual_years():
    frame = load_dataset(DATA_PATH)
    changes = endpoint_changes(frame, EDUCATION, 2010, 2024, exact_endpoints=False)
    assert not changes.empty
    assert changes["start_year"].between(2010, 2024).all()
    assert changes["end_year"].between(2010, 2024).all()
    assert changes["span_years"].gt(0).all()


def test_exact_endpoint_mode_never_substitutes_years():
    frame = load_dataset(DATA_PATH)
    exact = endpoint_changes(frame, LIFE, 2010, 2024, exact_endpoints=True)
    assert not exact.empty
    assert exact["start_year"].eq(2010).all()
    assert exact["end_year"].eq(2024).all()
    assert exact["exact_requested_endpoints"].all()


def test_quality_and_balance_tables_have_expected_ranges():
    frame = load_dataset(DATA_PATH)
    completeness = completeness_table(frame, 2010, 2024)
    assert completeness["completeness_pct"].between(0, 100).all()
    balance = balanced_scores(frame, 2010, 2024, exact_endpoints=False)
    if not balance.empty:
        for column in ["economy_score", "health_score", "education_score"]:
            assert balance[column].between(0, 100).all()
        assert balance["country_code"].is_unique


def test_core_metrics_retain_real_missingness():
    frame = load_dataset(DATA_PATH)
    assert frame[GDP_LEVEL].isna().sum() > 0
    assert frame[EDUCATION].isna().sum() > 0
