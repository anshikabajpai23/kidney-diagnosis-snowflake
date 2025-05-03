# Kidney Disease Diagnosis using Snowflake & Streamlit

This project demonstrates how to use the UCI Chronic Kidney Disease dataset to build a machine learning pipeline for disease prediction. It showcases data preprocessing, model building, and deployment through a Streamlit app, all integrated with Snowflake using Snowpark Python.

## 💾 Dataset

- Source: [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- File used: `kidney_disease.csv`
- Features include blood pressure, specific gravity, albumin, sugar, red blood cells, and more.
- Target variable: `classification` (ckd or notckd)

## 🏗️ Project Structure

```bash
.
├── kidney_disease.csv          # Raw dataset
├── creds.json                  # Snowflake credentials
├── report.doc                  # Summary of findings and insights
├── streamlit_app/
│   └── app_kidney.py           # Streamlit UI
├── preprocessing/
│   └── preprocess.py           # Feature engineering and cleaning
├── snowflake_integration/
│   └── load_to_snowflake.py    # Uploading to Snowflake using Snowpark
└── README.md
