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

## Run locally

Use Python 3.12, which is also the current default on Streamlit Community Cloud.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Run automated checks with:

```bash
pytest -q
```

## Refresh the World Bank snapshot

The hosted app uses the checked-in CSV so it remains available even when the World Bank API is temporarily slow. To retrieve a fresh snapshot:

```bash
python scripts/fetch_world_bank_data.py
```

Review the generated CSV and raw JSON before committing the refresh.

## Host free on Streamlit Community Cloud

1. Create a new **public GitHub repository**, for example `eac-quality-of-life-dashboard`.
2. Upload or push the **contents of this directory** to the repository. `app.py` and `requirements.txt` should be at the repository root.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
4. Select **Create app** and then **Yup, I have an app**.
5. Choose the repository and the `main` branch.
6. Set the entrypoint to `app.py`.
7. Open **Advanced settings** and select Python 3.12. No secrets are required.
8. Choose an available app URL and select **Deploy**.

After deployment, every push to the GitHub repository automatically updates the app.

## Data and methodology notes

- Source: World Bank World Development Indicators API v2, source ID 2.
- Countries: Burundi, Democratic Republic of the Congo, Kenya, Rwanda, Somalia, South Sudan, Tanzania, and Uganda.
- Real GDP per capita uses constant 2015 US dollars.
- Secondary enrolment is a gross ratio and can legitimately exceed 100%.
- Missing values are retained; they are never converted to zero.
- Within-country 1.5 x IQR observations are flagged for review rather than deleted.
- The balanced-progress score is an analyst-designed comparison, not an official World Bank index.
- Pooled association charts do not establish causation.
