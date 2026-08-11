# EAC Growth and Quality of Life Dashboard

An interactive Streamlit dashboard for the MIT 8334 Data Analytics and Visualization capstone project. It examines whether real GDP-per-capita growth across East African Community Partner States has coincided with improvements in life expectancy and secondary-school enrolment from 2010 to 2024.

## Dashboard features

- Four filter-responsive KPI cards
- Country and year slicers
- Exact-endpoint or first/last-available comparison modes
- Interactive EAC choropleth map
- Real GDP-per-capita trend chart
- Secondary-enrolment improvement ranking
- Balanced economic, health, and education score
- Indexed GDP and life-expectancy comparison
- GDP-life-expectancy scatterplot with a fitted relationship
- GDP growth-rate heatmap
- Country-level drill-down and CSV download
- Data-quality completeness heatmap, missing-value audit, and IQR review table
- Dynamic executive summary

## Project structure

```text
eac_quality_of_life_dashboard/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── snapshot_metadata.json
│   ├── world_bank_eac_2010_2024.csv
│   └── world_bank_eac_2010_2024_raw.json
├── scripts/
│   └── fetch_world_bank_data.py
├── tests/
│   ├── test_app.py
│   └── test_data_processing.py
├── app.py
├── data_processing.py
├── requirements.txt
└── README.md
```



- The balanced-progress score is an analyst-designed comparison, not an official World Bank index.
- Pooled association charts do not establish causation.
