"""Streamlit dashboard for economic growth and quality of life in East Africa."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processing import (
    CORE_METRICS,
    COUNTRIES,
    EDUCATION,
    END_YEAR,
    GDP_GROWTH,
    GDP_LEVEL,
    LIFE,
    METRIC_META,
    START_YEAR,
    balanced_scores,
    completeness_table,
    compute_kpis,
    endpoint_changes,
    executive_summary,
    filter_data,
    iqr_outliers,
    latest_available_snapshot,
    load_dataset,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "world_bank_eac_2010_2024.csv"
METADATA_PATH = BASE_DIR / "data" / "snapshot_metadata.json"

COUNTRY_COLORS = {
    "BDI": "#A44A3F",
    "COD": "#3D5A80",
    "KEN": "#008E7A",
    "RWA": "#E59F32",
    "SOM": "#6A4C93",
    "SSD": "#D85C41",
    "TZA": "#2A9D8F",
    "UGA": "#7A8B38",
}
MEASURE_COLORS = {
    "Real GDP per capita": "#0B6E75",
    "Life expectancy": "#D97732",
}

st.set_page_config(
    page_title="EAC Growth & Quality of Life",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #173238;
        --muted: #60757a;
        --teal: #0b6e75;
        --cream: #f6f3ea;
        --gold: #d69b3a;
    }
    .stApp { background: linear-gradient(180deg, #f6f3ea 0%, #fbfaf7 34%, #ffffff 100%); }
    .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #102f35; }
    [data-testid="stSidebar"] * { color: #f7f4ea; }
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: #173f46;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.88);
        border: 1px solid rgba(11,110,117,.16);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(20,57,63,.06);
    }
    [data-testid="stMetricLabel"] { color: #60757a; }
    [data-testid="stMetricValue"] { color: #173238; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.72);
        border-color: rgba(11,110,117,.15);
        border-radius: 18px;
    }
    .hero-kicker {
        color: #0b6e75; font-weight: 800; letter-spacing: .12em;
        text-transform: uppercase; font-size: .78rem; margin-bottom: .35rem;
    }
    .hero-title {
        color: #173238; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.04;
        font-weight: 800; letter-spacing: -.04em; margin: 0 0 .5rem 0;
    }
    .hero-subtitle { color: #60757a; font-size: 1.05rem; max-width: 850px; }
    .pill {
        display: inline-block; padding: .28rem .65rem; margin: .3rem .35rem .3rem 0;
        border-radius: 999px; background: #e4f0ed; color: #0b6e75;
        font-size: .78rem; font-weight: 700;
    }
    .summary-title { color: #173238; font-weight: 800; font-size: 1.05rem; }
    .small-note { color: #60757a; font-size: .84rem; }
    h1, h2, h3 { color: #173238; letter-spacing: -.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def fmt(value: float, pattern: str, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "No data"
    return f"{value:{pattern}}{suffix}"


def style_figure(fig: go.Figure, height: int = 470) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=64, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.72)",
        font=dict(family="Arial, sans-serif", color="#173238"),
        title_font=dict(size=20, color="#173238"),
        hoverlabel=dict(bgcolor="white", font_color="#173238"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(23,50,56,.10)", zerolinecolor="rgba(23,50,56,.25)")
    return fig


data = get_data()
metadata = get_metadata()

with st.sidebar:
    st.markdown("## EAC explorer")
    st.caption("Use the controls to update every KPI, chart, map, and summary.")

    country_names = list(COUNTRIES.values())
    selected_names = st.multiselect(
        "Partner States",
        options=country_names,
        default=country_names,
        help="Select one or more countries for regional comparison.",
    )
    selected_codes = [code for code, name in COUNTRIES.items() if name in selected_names]

    year_range = st.slider(
        "Study period",
        min_value=START_YEAR,
        max_value=END_YEAR,
        value=(START_YEAR, END_YEAR),
        step=1,
    )
    selected_start, selected_end = year_range

    exact_endpoints = st.toggle(
        "Require exact endpoint years",
        value=False,
        help=(
            "When enabled, a country is excluded from change rankings if either selected "
            "endpoint year is missing. Otherwise, the first and last available values in "
            "the selected period are used and disclosed."
        ),
    )

    map_label_to_metric = {
        METRIC_META[GDP_LEVEL]["label"]: GDP_LEVEL,
        METRIC_META[LIFE]["label"]: LIFE,
        METRIC_META[EDUCATION]["label"]: EDUCATION,
        METRIC_META[GDP_GROWTH]["label"]: GDP_GROWTH,
    }
    map_label = st.selectbox("Map indicator", options=list(map_label_to_metric))
    map_metric = map_label_to_metric[map_label]

    st.divider()
    st.markdown("**Source**")
    st.caption("World Bank World Development Indicators API")
    retrieved_at = metadata.get("retrieved_at_utc", "Not recorded")
    st.caption(f"Snapshot retrieved: {retrieved_at[:10] if retrieved_at != 'Not recorded' else retrieved_at}")
    st.link_button("World Bank API", "https://api.worldbank.org/v2")

if not selected_codes:
    st.warning("Select at least one Partner State in the sidebar to continue.")
    st.stop()

filtered = filter_data(data, selected_codes, selected_start, selected_end)
if filtered.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

kpis = compute_kpis(filtered, selected_start, selected_end, exact_endpoints)
endpoint_caption = (
    "Exact selected-year endpoints required"
    if exact_endpoints
    else "First and last available observations within the selected period"
)

st.markdown(
    """
    <div class="hero-kicker">East African Community policy dashboard</div>
    <div class="hero-title">Growth that people can feel?</div>
    <div class="hero-subtitle">
      Explore whether real economic growth across EAC Partner States has moved together with
      longer lives and broader access to secondary education.
    </div>
    <div>
      <span class="pill">World Bank WDI</span>
      <span class="pill">2010-2024</span>
      <span class="pill">8 Partner States</span>
      <span class="pill">Interactive evidence</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Active comparison method: {endpoint_caption}.")

kpi_columns = st.columns(4)
with kpi_columns[0]:
    st.metric("Partner States in view", kpis["country_count"])
with kpi_columns[1]:
    st.metric(
        "Median real GDP/capita change",
        fmt(kpis["median_gdp_change_pct"], ".1f", "%"),
        help="Median total percentage change among countries with comparable endpoints.",
    )
with kpi_columns[2]:
    st.metric(
        "Median life-expectancy change",
        fmt(kpis["median_life_change_years"], ".1f", " years"),
    )
with kpi_columns[3]:
    leader_change = kpis["education_leader_change_pp"]
    st.metric(
        "Top enrolment improver",
        kpis["education_leader_country"],
        None if not np.isfinite(leader_change) else f"{leader_change:+.1f} pp",
    )

with st.container(border=True):
    st.markdown('<div class="summary-title">Executive summary</div>', unsafe_allow_html=True)
    for bullet in executive_summary(
        filtered, selected_start, selected_end, exact_endpoints
    ):
        st.markdown(f"- {bullet}")
    st.markdown(
        '<div class="small-note">Rankings depend on available endpoint years. '
        "Use the Data quality tab before drawing policy conclusions.</div>",
        unsafe_allow_html=True,
    )

overview_tab, relationship_tab, drill_tab, quality_tab = st.tabs(
    ["Regional overview", "Growth & well-being", "Country drill-down", "Data quality"]
)

with overview_tab:
    map_column, trend_column = st.columns([0.92, 1.35])

    with map_column:
        map_data = latest_available_snapshot(filtered, map_metric, selected_end)
        st.subheader("Geographic view")
        st.caption(
            f"Latest available {METRIC_META[map_metric]['short_label'].lower()} value "
            f"up to {selected_end}; hover to see each observation year."
        )
        if map_data.empty:
            st.info("No map observations are available for the current filters.")
        else:
            map_fig = px.choropleth(
                map_data,
                locations="country_code",
                locationmode="ISO-3",
                color="value",
                hover_name="country",
                hover_data={
                    "country_code": False,
                    "value": ":,.2f",
                    "observed_year": True,
                },
                color_continuous_scale=["#e9e4d8", "#8ec2b9", "#0b6e75", "#173238"],
                labels={"value": METRIC_META[map_metric]["short_label"]},
                scope="africa",
                title=f"{METRIC_META[map_metric]['label']} across selected states",
            )
            map_fig.update_geos(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="rgba(23,50,56,.35)",
                bgcolor="rgba(0,0,0,0)",
                landcolor="#ece8df",
            )
            style_figure(map_fig, 515)
            map_fig.update_layout(coloraxis_colorbar_title=METRIC_META[map_metric]["unit"])
            st.plotly_chart(map_fig, width="stretch", config={"displaylogo": False})

    with trend_column:
        st.subheader("Economic trajectory")
        st.caption("Interactive lines preserve the annual pattern behind the endpoint ranking.")
        gdp_line_data = filtered.dropna(subset=[GDP_LEVEL])
        if gdp_line_data.empty:
            st.info("No GDP-per-capita observations are available.")
        else:
            gdp_fig = px.line(
                gdp_line_data,
                x="year",
                y=GDP_LEVEL,
                color="country_code",
                line_group="country_code",
                markers=True,
                hover_name="country",
                color_discrete_map=COUNTRY_COLORS,
                labels={
                    "year": "Year",
                    GDP_LEVEL: "Constant 2015 US$ per person",
                    "country_code": "Country",
                },
                title="Real GDP per capita, 2010-2024",
            )
            style_figure(gdp_fig, 515)
            gdp_fig.update_layout(hovermode="x unified")
            st.plotly_chart(gdp_fig, width="stretch", config={"displaylogo": False})

    ranking_column, balance_column = st.columns(2)
    with ranking_column:
        st.subheader("Education improvement")
        education_changes = kpis["education_changes"].sort_values(
            "absolute_change", ascending=True
        )
        if education_changes.empty:
            st.info("No countries have two usable enrolment endpoints for this period.")
        else:
            education_fig = px.bar(
                education_changes,
                x="absolute_change",
                y="country",
                orientation="h",
                color="country_code",
                color_discrete_map=COUNTRY_COLORS,
                custom_data=["start_year", "end_year", "start_value", "end_value"],
                labels={
                    "absolute_change": "Percentage-point change",
                    "country": "",
                    "country_code": "Country",
                },
                title="Secondary-school enrolment change",
            )
            education_fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>Change: %{x:.1f} pp"
                    "<br>Period: %{customdata[0]}-%{customdata[1]}"
                    "<br>Start: %{customdata[2]:.1f}%"
                    "<br>End: %{customdata[3]:.1f}%<extra></extra>"
                )
            )
            style_figure(education_fig, 455)
            education_fig.add_vline(x=0, line_width=1, line_color="rgba(23,50,56,.45)")
            education_fig.update_layout(showlegend=False)
            st.plotly_chart(
                education_fig, width="stretch", config={"displaylogo": False}
            )

    with balance_column:
        st.subheader("Balanced progress")
        balance = balanced_scores(
            filtered, selected_start, selected_end, exact_endpoints
        )
        st.caption(
            "Mean normalised progress minus cross-dimension imbalance; an analytical score, "
            "not an official World Bank index."
        )
        if balance.empty:
            st.info("Not enough overlapping economic, health, and education endpoints.")
        else:
            balance_plot = balance.sort_values("balanced_score")
            balance_fig = px.bar(
                balance_plot,
                x="balanced_score",
                y="country",
                orientation="h",
                color="balanced_score",
                color_continuous_scale=["#e9e4d8", "#8ec2b9", "#0b6e75"],
                range_color=[0, 100],
                hover_data={
                    "economy_score": ":.1f",
                    "health_score": ":.1f",
                    "education_score": ":.1f",
                    "balanced_score": ":.1f",
                },
                labels={"balanced_score": "Balance-adjusted score", "country": ""},
                title="Economic, health, and education balance",
            )
            style_figure(balance_fig, 455)
            balance_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(balance_fig, width="stretch", config={"displaylogo": False})

with relationship_tab:
    st.subheader("Has quality of life kept pace with economic growth?")
    relationship_left, relationship_right = st.columns(2)

    with relationship_left:
        comparison_country = st.selectbox(
            "Country for indexed comparison",
            options=[COUNTRIES[code] for code in selected_codes],
            key="indexed_country",
        )
        comparison_code = next(
            code for code, name in COUNTRIES.items() if name == comparison_country
        )
        comparison_data = filtered.loc[
            filtered["country_code"].eq(comparison_code)
        ].sort_values("year")
        indexed_rows: list[dict] = []
        for metric, label in [
            (GDP_LEVEL, "Real GDP per capita"),
            (LIFE, "Life expectancy"),
        ]:
            observed = comparison_data[["year", metric]].dropna()
            if observed.empty or observed.iloc[0][metric] == 0:
                continue
            base_year = int(observed.iloc[0]["year"])
            base_value = float(observed.iloc[0][metric])
            for row in observed.itertuples(index=False):
                indexed_rows.append(
                    {
                        "year": int(row.year),
                        "measure": label,
                        "index_value": 100 * getattr(row, metric) / base_value,
                        "base_year": base_year,
                    }
                )
        indexed = pd.DataFrame(indexed_rows)
        if indexed.empty:
            st.info("No indexed comparison is available for this country.")
        else:
            indexed_fig = px.line(
                indexed,
                x="year",
                y="index_value",
                color="measure",
                line_dash="measure",
                markers=True,
                color_discrete_map=MEASURE_COLORS,
                labels={
                    "year": "Year",
                    "index_value": "Index (first observed value = 100)",
                    "measure": "",
                },
                title=f"Relative progress in {comparison_country}",
            )
            style_figure(indexed_fig, 470)
            indexed_fig.add_hline(y=100, line_dash="dot", line_color="rgba(23,50,56,.45)")
            st.plotly_chart(indexed_fig, width="stretch", config={"displaylogo": False})

    with relationship_right:
        paired = filtered[["country_code", "country", "year", GDP_LEVEL, LIFE]].dropna()
        st.caption("Each point is a country-year observation; click legend items to isolate states.")
        if len(paired) < 3:
            st.info("At least three paired observations are required for this chart.")
        else:
            scatter_fig = px.scatter(
                paired,
                x=GDP_LEVEL,
                y=LIFE,
                color="country_code",
                hover_name="country",
                hover_data={"year": True, GDP_LEVEL: ":,.0f", LIFE: ":.1f"},
                log_x=True,
                trendline="ols",
                trendline_scope="overall",
                color_discrete_map=COUNTRY_COLORS,
                labels={
                    GDP_LEVEL: "GDP per capita (constant 2015 US$, log scale)",
                    LIFE: "Life expectancy (years)",
                    "country_code": "Country",
                },
                title="GDP per capita and life expectancy",
            )
            style_figure(scatter_fig, 500)
            st.plotly_chart(scatter_fig, width="stretch", config={"displaylogo": False})
            correlation = paired[GDP_LEVEL].corr(paired[LIFE])
            st.caption(
                f"Pooled Pearson correlation: {correlation:.2f} across {len(paired)} paired "
                "observations. Association does not establish causation."
            )

    st.subheader("Annual economic growth pattern")
    growth_pivot = (
        filtered.pivot(index="country", columns="year", values=GDP_GROWTH)
        .reindex([COUNTRIES[code] for code in selected_codes])
    )
    if growth_pivot.notna().any().any():
        growth_heatmap = px.imshow(
            growth_pivot,
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            labels={"x": "Year", "y": "", "color": "Annual growth (%)"},
            title="GDP-per-capita growth rates reveal shocks hidden by endpoint totals",
        )
        style_figure(growth_heatmap, 440)
        st.plotly_chart(growth_heatmap, width="stretch", config={"displaylogo": False})

with drill_tab:
    st.subheader("Country profile")
    st.caption("Drill from the regional view into one Partner State and export its evidence.")
    drill_country = st.selectbox(
        "Partner State",
        options=[COUNTRIES[code] for code in selected_codes],
        key="drill_country",
    )
    drill_code = next(code for code, name in COUNTRIES.items() if name == drill_country)
    country_data = filtered.loc[filtered["country_code"].eq(drill_code)].sort_values("year")

    drill_metrics = st.columns(3)
    for column, metric, label in [
        (drill_metrics[0], GDP_LEVEL, "Latest real GDP per capita"),
        (drill_metrics[1], LIFE, "Latest life expectancy"),
        (drill_metrics[2], EDUCATION, "Latest secondary enrolment"),
    ]:
        observed = country_data[["year", metric]].dropna()
        with column:
            if observed.empty:
                st.metric(label, "No data")
            else:
                latest = observed.iloc[-1]
                value_format = METRIC_META[metric]["format"]
                st.metric(
                    label,
                    f"{latest[metric]:{value_format}} {METRIC_META[metric]['unit']}",
                    help=f"Most recent observation in {int(latest['year'])}.",
                )

    profile_long = country_data.melt(
        id_vars=["country_code", "country", "year"],
        value_vars=[GDP_LEVEL, LIFE, EDUCATION],
        var_name="metric",
        value_name="value",
    ).dropna()
    profile_long["indicator"] = profile_long["metric"].map(
        {metric: METRIC_META[metric]["short_label"] for metric in [GDP_LEVEL, LIFE, EDUCATION]}
    )
    if profile_long.empty:
        st.info("No profile data are available for this country and period.")
    else:
        profile_fig = px.line(
            profile_long,
            x="year",
            y="value",
            facet_row="indicator",
            color="indicator",
            markers=True,
            color_discrete_sequence=["#0b6e75", "#d97732", "#6a4c93"],
            labels={"year": "Year", "value": "Observed value", "indicator": ""},
            title=f"Economic, health, and education profile: {drill_country}",
        )
        profile_fig.update_yaxes(matches=None, showticklabels=True)
        profile_fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
        style_figure(profile_fig, 720)
        profile_fig.update_layout(showlegend=False)
        st.plotly_chart(profile_fig, width="stretch", config={"displaylogo": False})

    country_changes = []
    for metric in [GDP_LEVEL, LIFE, EDUCATION]:
        change = endpoint_changes(
            country_data, metric, selected_start, selected_end, exact_endpoints
        )
        if change.empty:
            continue
        row = change.iloc[0]
        country_changes.append(
            {
                "Indicator": METRIC_META[metric]["label"],
                "Actual start": int(row["start_year"]),
                "Actual end": int(row["end_year"]),
                "Start value": row["start_value"],
                "End value": row["end_value"],
                "Absolute change": row["absolute_change"],
                "Percent change": row["percent_change"],
            }
        )
    st.dataframe(pd.DataFrame(country_changes), width="stretch", hide_index=True)
    st.download_button(
        "Download this country profile (CSV)",
        data=country_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{drill_code.lower()}_quality_of_life_{selected_start}_{selected_end}.csv",
        mime="text/csv",
    )

with quality_tab:
    st.subheader("Data quality and audit trail")
    st.caption(
        "Missing observations remain missing; gross enrolment may legitimately exceed 100%; "
        "IQR outliers are flagged for review and are not automatically removed."
    )
    completeness = completeness_table(filtered, selected_start, selected_end)
    quality_left, quality_right = st.columns([1.3, 0.7])

    with quality_left:
        if completeness.empty:
            st.info("No completeness results are available.")
        else:
            completeness_pivot = completeness.pivot(
                index="country", columns="metric_label", values="completeness_pct"
            ).reindex([COUNTRIES[code] for code in selected_codes])
            completeness_fig = px.imshow(
                completeness_pivot,
                text_auto=".0f",
                color_continuous_scale=["#efe8d8", "#8ec2b9", "#0b6e75"],
                range_color=[0, 100],
                aspect="auto",
                labels={"x": "Indicator", "y": "", "color": "Complete (%)"},
                title="Completeness by country and indicator",
            )
            style_figure(completeness_fig, 500)
            st.plotly_chart(
                completeness_fig, width="stretch", config={"displaylogo": False}
            )

    with quality_right:
        expected_rows = len(selected_codes) * (selected_end - selected_start + 1)
        duplicate_count = int(filtered.duplicated(["country_code", "year"]).sum())
        missing_count = int(filtered[CORE_METRICS].isna().sum().sum())
        outliers = iqr_outliers(filtered)
        st.metric("Expected country-year rows", expected_rows)
        st.metric("Duplicate country-year rows", duplicate_count)
        st.metric("Missing indicator cells", missing_count)
        st.metric("IQR flags for review", len(outliers))

    missing_table = (
        filtered.groupby(["country_code", "country"])[CORE_METRICS]
        .agg(lambda series: int(series.isna().sum()))
        .reset_index()
        .rename(columns={metric: METRIC_META[metric]["short_label"] for metric in CORE_METRICS})
    )
    with st.expander("Missing-value counts", expanded=False):
        st.dataframe(missing_table, width="stretch", hide_index=True)
    with st.expander("IQR observations requiring review", expanded=False):
        if outliers.empty:
            st.success("No within-country 1.5 x IQR flags were detected for the active filters.")
        else:
            st.dataframe(outliers, width="stretch", hide_index=True)
    with st.expander("Indicator definitions and preprocessing decisions", expanded=False):
        definitions = pd.DataFrame(
            [
                {
                    "Indicator": METRIC_META[metric]["label"],
                    "Unit": METRIC_META[metric]["unit"],
                    "Missing-value rule": "Retained as missing; never replaced with zero",
                    "Outlier rule": "Flagged by within-country 1.5 x IQR; retained for review",
                }
                for metric in CORE_METRICS
            ]
        )
        st.dataframe(definitions, width="stretch", hide_index=True)

    st.download_button(
        "Download filtered dashboard data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"eac_dashboard_filtered_{selected_start}_{selected_end}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Source: World Bank World Development Indicators API. Real GDP per capita uses constant "
    "2015 US dollars. Gross secondary enrolment can exceed 100%. Dashboard results update with "
    "the selected countries, years, endpoint method, and map indicator."
)
