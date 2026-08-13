# ResidenceAlpha — Multifamily Asset Performance & Renovation Strategy

Multifamily private equity analytics — resident renewal prediction, NLP on resident feedback, renovation ROI modeling for asset management decisions.

**🔗 Live app:** [residencial-multifamily-analytics.streamlit.app](https://residencial-multifamily-analytics.streamlit.app)

## Business problem

A multifamily owner or asset manager overseeing dozens of properties needs to know: which residents are unlikely to renew, which properties need operational versus capital intervention, which renovations would produce a sustainable rent premium, and where capital should be prioritized.

## What this project builds

- A synthetic 40-property, 1,249-lease multifamily dataset
- A resident renewal prediction model (with honest, non-cherry-picked metrics)
- An NLP classifier on resident feedback (theme + sentiment)
- Unsupervised property segmentation into 5 asset-management categories
- A transparent renovation ROI calculator — every formula shown, not a black box
- An interactive Streamlit application — **[try it live](https://residencial-multifamily-analytics.streamlit.app)**

## Data

All property, lease, and resident data is synthetic — generated to match realistic multifamily operating patterns. No real property or resident data is used. Full provenance: **[data_dictionary.md](reports/data_dictionary.md)**

## Modeling

Renewal prediction, resident feedback NLP, and property segmentation — each reported with real limitations stated plainly, including a segmentation labeling bug caught and fixed during development. Full detail: **[model_card.md](reports/model_card.md)**

## Deliverables

| Deliverable | Link |
|---|---|
| 🟢 Live Streamlit app | [residencial-multifamily-analytics.streamlit.app](https://residencial-multifamily-analytics.streamlit.app) |
| 📄 Asset Management Report (PDF) | [reports/residencealpha_report.pdf](reports/residencealpha_report.pdf) |
| 📋 Model Card | [reports/model_card.md](reports/model_card.md) |
| 📖 Data Dictionary | [reports/data_dictionary.md](reports/data_dictionary.md) |
| 🐍 Data generator script | [scripts/generate_residencealpha_data.py](scripts/generate_residencealpha_data.py) |
| 🧠 Model training script | [scripts/train_models.py](scripts/train_models.py) |
| 💻 Streamlit app source | [app/app.py](app/app.py) |
| 🎥 Demo video | *(link coming soon)* |
| 📈 Power BI dashboard | *(coming soon)* |

## Stack

Python (pandas, scikit-learn, SHAP, TF-IDF) · Streamlit · Plotly

## Disclaimer

All property, lease, and resident data in this repository is synthetic and generated for demonstration purposes. No real property, resident, or company data is used or represented.
