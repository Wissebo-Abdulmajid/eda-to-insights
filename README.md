# EDA-to-Insights

A stakeholder-friendly **EDA-to-Insights** framework for tabular datasets.  
Upload a dataset → run analysis → download an interactive HTML report + artifacts.

---

## What you get (in one run)

This system automatically produces:

- **Dataset overview**: rows, columns, types, duplicates, missing cells
- **Data quality diagnostics**: missingness hotspots, potential issues
- **Distributions & outliers**: histograms, boxplots, risk map
- **Relationships**: correlations, top pairs, scatter matrix, 3D view (when applicable)
- **Interactive HTML report** (works **offline** when configured to embed Plotly.js)
- **Artifacts ZIP** (CSV/JSON outputs used to generate the report)

---

## Quickstart (Streamlit App)

### 1) Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1


Windows (CMD):

python -m venv .venv
.venv\Scripts\activate


macOS / Linux:

python -m venv .venv
source .venv/bin/activate

2) Install the project

From the project root:

pip install -e .

3) Run the Streamlit UI
streamlit run src/edainsights/app.py


Open the local URL shown in your terminal (usually http://localhost:8501), then:

Upload your dataset (CSV / Excel / Parquet)

Click Generate Report

Download:

eda_report.html (interactive report)

eda_package.zip (report + artifacts)

Advanced usage (CLI)

For analysts who want reproducible runs via config:

edainsights run --data path/to/data.csv --config configs/default.yml

Supported file formats

CSV (.csv) — delimiter is auto-detected in the Streamlit UI

Excel (.xlsx, .xls)

Parquet (.parquet)

Outputs

A typical run generates:

reports/report.html — interactive report

artifacts/ folder containing:

profile_summary.json

quality_summary.json

issues.csv

column_profile.csv

correlation_numeric.csv (when correlation is enabled)

When using the Streamlit UI, you download:

eda_report.html

eda_package.zip (contains report + artifacts)

Project structure
eda-to-insights/
├─ configs/
│  └─ default.yml
├─ src/
│  └─ edainsights/
│     ├─ app.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ io.py
│     ├─ profiling/
│     ├─ quality/
│     └─ reporting/
│        ├─ html_report.py
│        └─ templates/
│           └─ report.html.j2
├─ pyproject.toml
└─ README.md

Deployment (Streamlit Community Cloud)

Push this repository to GitHub

Go to Streamlit Community Cloud and click New app

Select your repository + branch (main)

Set Main file path to:

src/edainsights/app.py


Deploy

After deployment, add your public URL to the “Live App” section below.

Live App

Streamlit URL: (add here after deployment)

Notes for stakeholders

This app is designed to be simple and safe: upload → generate → download.

Configuration is intentionally hidden in the UI to avoid overwhelming non-technical users.

The downloadable HTML report can be shared by email or opened locally.

License

MIT

Author

Wissebo Abdulmajid