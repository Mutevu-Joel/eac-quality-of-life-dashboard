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
    "BDI": "#9F4A35",
    "COD": "#315D72",
    "KEN": "#176B55",
    "RWA": "#D4912A",
    "SOM": "#4E7590",
    "SSD": "#C85A38",
    "TZA": "#168C82",
    "UGA": "#687A38",
}
MEASURE_COLORS = {
    "Real GDP per capita": "#176B55",
    "Life expectancy": "#C85A38",
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
        --ink: #1f3028;
        --muted: #5c6c62;
        --forest: #176b55;
        --lake: #247a8a;
        --sand: #f3e5c7;
        --cream: #fbf6e8;
        --ochre: #d4912a;
        --clay: #a84f35;
    }
    .stApp {
        background:
          radial-gradient(circle at 92% 3%, rgba(212,145,42,.13), transparent 24rem),
          linear-gradient(180deg, #fbf6e8 0%, #fffdf8 44%, #f8f3e9 100%);
    }
    .block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #183d34; }
    [data-testid="stSidebar"] * { color: #fffaf0; }
    [data-testid="stSidebar"] a { color: #f2c46d !important; }
    [data-testid="stHeader"] { background: rgba(251,246,232,.92); }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,253,247,.93);
        border: 1px solid rgba(80,78,51,.20);
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(73,58,34,.07);
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background: #fffaf0 !important;
        border: 2px solid #40705f !important;
        border-radius: 12px !important;
        min-height: 3rem;
        color: #1f3028 !important;
        box-shadow: none !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] input,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] input {
        color: #1f3028 !important;
        -webkit-text-fill-color: #1f3028 !important;
        opacity: 1 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] svg {
        fill: #176b55 !important;
    }
    div[data-testid="stSelectbox"]:focus-within div[data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"]:focus-within div[data-baseweb="select"] > div {
        border-color: #a84f35 !important;
        box-shadow: 0 0 0 3px rgba(168,79,53,.20) !important;
    }
    div[role="listbox"] {
        background: #fffaf0 !important;
        color: #1f3028 !important;
    }
    div[data-testid="stWidgetLabel"] p {
        color: #263e33;
        font-weight: 750;
        font-size: .93rem;
    }
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #fffaf0 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,250,240,.08);
        border: 1px solid rgba(255,250,240,.20);
        border-radius: 10px;
        padding: .42rem .55rem;
        margin: .08rem 0;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(240,186,79,.16);
        border-color: #f0ba4f;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #fffaf0 !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255,253,247,.94);
        border: 1px solid rgba(23,107,85,.20);
        border-top: 4px solid #d4912a;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(73,58,34,.07);
    }
    [data-testid="stMetricLabel"] { color: #5c6c62; }
    [data-testid="stMetricValue"] { color: #1f3028; }
    .eac-hero {
        position: relative;
        overflow: hidden;
        padding: 1.55rem 1.8rem 1.45rem;
        border-radius: 22px;
        border: 1px solid rgba(23,107,85,.20);
        background:
          linear-gradient(105deg, rgba(255,250,240,.98) 0%, rgba(243,229,199,.90) 67%, rgba(212,145,42,.17) 100%);
        box-shadow: 0 14px 34px rgba(73,58,34,.08);
        margin-bottom: 1rem;
    }
    .eac-hero::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 8px;
        background: repeating-linear-gradient(
            90deg,
            #176b55 0 42px,
            #f0ba4f 42px 62px,
            #a84f35 62px 88px,
            #247a8a 88px 126px,
            #1f3028 126px 140px
        );
    }
    .eac-hero::after {
        content: "";
        position: absolute;
        width: 190px;
        height: 190px;
        right: -65px;
        bottom: -105px;
        border: 24px solid rgba(23,107,85,.09);
        transform: rotate(45deg);
    }
    .hero-kicker {
        color: #176b55; font-weight: 800; letter-spacing: .12em;
        text-transform: uppercase; font-size: .78rem; margin: .3rem 0 .35rem;
    }
    .hero-title {
        color: #1f3028; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1.04;
        font-weight: 800; letter-spacing: -.04em; margin: 0 0 .5rem 0;
    }
    .hero-subtitle { color: #52665b; font-size: 1.05rem; max-width: 850px; }
    .pill {
        display: inline-block; padding: .28rem .65rem; margin: .3rem .35rem .3rem 0;
        border-radius: 999px; background: #e4eee7; color: #176b55;
        font-size: .78rem; font-weight: 700;
    }
    .filter-title {
        color: #176b55; font-weight: 850; font-size: 1.08rem;
        letter-spacing: .01em; margin-bottom: .05rem;
    }
    .filter-note { color: #5c6c62; font-size: .86rem; margin-bottom: .55rem; }
    .summary-title { color: #1f3028; font-weight: 800; font-size: 1.05rem; }
    .small-note { color: #5c6c62; font-size: .84rem; }
    h1, h2, h3 { color: #1f3028; letter-spacing: -.02em; }
    button[data-baseweb="tab"] { font-weight: 750; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #176b55 !important;
        border-bottom-color: #d4912a !important;
    }
    @media (max-width: 700px) {
        .block-container { padding-top: .7rem; }
        .eac-hero { padding: 1.3rem 1.1rem 1.15rem; }
    }
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
        font=dict(family="Arial, sans-serif", color="#1f3028"),
        title_font=dict(size=20, color="#1f3028"),
        hoverlabel=dict(bgcolor="#fffaf0", font_color="#1f3028"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(31,48,40,.10)", zerolinecolor="rgba(31,48,40,.25)")
    return fig


data = get_data()
metadata = get_metadata()

country_names = list(COUNTRIES.values())
map_label_to_metric = {
    "Income per person": GDP_LEVEL,
    "Life expectancy": LIFE,
    "Secondary school enrolment": EDUCATION,
    "Annual income growth": GDP_GROWTH,
}

with st.sidebar:
    st.markdown("## Explore the region")
    st.caption("Choose countries and years. The dashboard on the right updates automatically.")

    selected_names = st.multiselect(
        "Partner States",
        options=country_names,
        default=country_names,
        help="Choose one or more countries to compare.",
    )
    year_range = st.slider(
        "Study period",
        min_value=START_YEAR,
        max_value=END_YEAR,
        value=(START_YEAR, END_YEAR),
        step=1,
    )
    map_label = st.radio(
        "Map indicator",
        options=list(map_label_to_metric),
        help="Choose what the map colours represent.",
        key="map_indicator_sidebar",
    )
    exact_endpoints = st.toggle(
        "Use only the exact start and end years",
        value=False,
        help=(
            "Turn this on to compare only countries that have data for both selected years. "
            "Leave it off to use the nearest available years."
        ),
    )

    st.divider()
    st.markdown("**About this dashboard**")
    st.caption(
        "Economic growth, health, and education evidence for East African Community "
        "Partner States."
    )
    st.markdown("**Source**")
    st.caption("World Bank World Development Indicators API")
    retrieved_at = metadata.get("retrieved_at_utc", "Not recorded")
    retrieved_date = (
        retrieved_at[:10] if retrieved_at != "Not recorded" else retrieved_at
    )
    st.caption(f"Snapshot retrieved: {retrieved_date}")
    st.link_button(
        "World Bank API guide",
        "https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation",
    )

selected_codes = [code for code, name in COUNTRIES.items() if name in selected_names]
selected_start, selected_end = year_range
map_metric = map_label_to_metric[map_label]

st.markdown(
    """
    <div class="eac-hero">
      <div class="hero-kicker">Ukuaji na ustawi · East African Community policy dashboard</div>
      <div class="hero-title">Growth that people can feel?</div>
      <div class="hero-subtitle">
        Explore whether real economic growth across EAC Partner States has moved together with
        longer lives and broader access to secondary education.
      </div>
      <div>
        <span class="pill">World Bank WDI</span>
        <span class="pill">2010–2024</span>
        <span class="pill">8 Partner States</span>
        <span class="pill">Interactive evidence</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not selected_codes:
    st.warning("Select at least one Partner State in the sidebar to continue.")
    st.stop()

filtered = filter_data(data, selected_codes, selected_start, selected_end)
if filtered.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

kpis = compute_kpis(filtered, selected_start, selected_end, exact_endpoints)
endpoint_caption = (
    "only countries with data for both selected years"
    if exact_endpoints
    else "the nearest available years within your selected period"
)

st.caption(f"How changes are compared: {endpoint_caption}.")

kpi_columns = st.columns(4)
with kpi_columns[0]:
    st.metric("Partner States in view", kpis["country_count"])
with kpi_columns[1]:
    st.metric(
        "Median real GDP/capita change",
        fmt(kpis["median_gdp_change_pct"], ".1f", "%"),
        help="The middle percentage change among the selected countries with enough data.",
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
        '<div class="small-note">Some countries do not have data for every year. '
        "Check the Data quality tab before making conclusions.</div>",
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
            f"This map shows the latest available {METRIC_META[map_metric]['short_label'].lower()} "
            f"up to {selected_end}. Point to a country to see its value and year."
        )
        if map_data.empty:
            st.info("No map data is available for these choices.")
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
                color_continuous_scale=["#f3e5c7", "#e2b45b", "#4d9277", "#176b55"],
                labels={"value": METRIC_META[map_metric]["short_label"]},
                scope="africa",
                title=f"{METRIC_META[map_metric]['label']} across selected states",
            )
            map_fig.update_geos(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="rgba(31,48,40,.45)",
                bgcolor="rgba(0,0,0,0)",
                landcolor="#eadbbb",
                showocean=True,
                oceancolor="#dcecef",
                showlakes=True,
                lakecolor="#c9e3e7",
                countrycolor="rgba(31,48,40,.30)",
            )
            style_figure(map_fig, 515)
            map_fig.update_layout(coloraxis_colorbar_title=METRIC_META[map_metric]["unit"])
            st.plotly_chart(map_fig, width="stretch", config={"displaylogo": False})

    with trend_column:
        st.subheader("Economic trajectory")
        st.caption("Each line shows how income per person changed from year to year.")
        gdp_line_data = filtered.dropna(subset=[GDP_LEVEL])
        if gdp_line_data.empty:
            st.info("No income data is available for these choices.")
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
            st.info("There is not enough school enrolment data for this period.")
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
            education_fig.add_vline(x=0, line_width=1, line_color="rgba(31,48,40,.45)")
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
            "This score combines changes in income, life expectancy, and school enrolment. "
            "It was created for this dashboard and is not a World Bank score."
        )
        if balance.empty:
            st.info("There is not enough data to calculate this score.")
        else:
            balance_plot = balance.sort_values("balanced_score")
            balance_fig = px.bar(
                balance_plot,
                x="balanced_score",
                y="country",
                orientation="h",
                color="balanced_score",
                color_continuous_scale=["#f3e5c7", "#d4912a", "#176b55"],
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
            st.info("There is not enough data to compare income and life expectancy.")
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
            indexed_fig.add_hline(y=100, line_dash="dot", line_color="rgba(31,48,40,.45)")
            st.plotly_chart(indexed_fig, width="stretch", config={"displaylogo": False})

    with relationship_right:
        paired = filtered[["country_code", "country", "year", GDP_LEVEL, LIFE]].dropna()
        st.caption("Each dot shows one country in one year. Click a country name to show or hide it.")
        if len(paired) < 3:
            st.info("This chart needs at least three records with both measures available.")
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
                f"The correlation is {correlation:.2f} using {len(paired)} records. "
                "A relationship between the measures does not prove that one caused the other."
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
            title="Year-by-year change in income per person",
        )
        style_figure(growth_heatmap, 440)
        st.plotly_chart(growth_heatmap, width="stretch", config={"displaylogo": False})

with drill_tab:
    st.subheader("Country profile")
    st.caption("Choose a country to see its details and download its data.")
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
        st.info("No data is available for this country and period.")
    else:
        profile_fig = px.line(
            profile_long,
            x="year",
            y="value",
            facet_row="indicator",
            color="indicator",
            markers=True,
            color_discrete_sequence=["#176b55", "#c85a38", "#247a8a"],
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
        "Blank values stay blank. School enrolment can be above 100%. Unusual values are "
        "marked for checking but are not deleted."
    )
    completeness = completeness_table(filtered, selected_start, selected_end)
    quality_left, quality_right = st.columns([1.3, 0.7])

    with quality_left:
        if completeness.empty:
            st.info("There is no data to check for missing values.")
        else:
            completeness_pivot = completeness.pivot(
                index="country", columns="metric_label", values="completeness_pct"
            ).reindex([COUNTRIES[code] for code in selected_codes])
            completeness_fig = px.imshow(
                completeness_pivot,
                text_auto=".0f",
                color_continuous_scale=["#f3e5c7", "#e2b45b", "#176b55"],
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
    "Source: World Bank World Development Indicators. Income per person is measured in constant "
    "2015 US dollars. School enrolment can be above 100%. All results change when you use the filters."
)
