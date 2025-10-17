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
├── requirements.txt            # Dependencies
├── creds.json                  # Snowflake credentials
├── report.doc                  # Summary of findings and insights
├── streamlit_app/
│   └── app_kidney.py           # Streamlit UI
├── preprocessing/
│   └── preprocess.ipynb           # Feature engineering and cleaning
├── snowflake_integration/
│   └── load_to_snowflake.ipynb    # Uploading to Snowflake using Snowpark
└── README.md
```

---

## 💾 Dataset

- **Source**: [UCI Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **File**: `kidney_disease.csv`
- Includes 24 features such as blood pressure, albumin, sugar, etc.
- **Target**: `classification` (either `ckd` or `notckd`)

---

## 🧠 ML Overview

- **Models**: Random Forest, XGBoost, Logistic Regression (customizable)
- **Preprocessing**: Missing value imputation, label encoding, feature scaling
- **Output**: Cleaned train/test datasets for modeling and Snowflake loading

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone [https://github.com/your-username/kidney-diagnosis-snowflake.git](https://github.com/anshikabajpai23/kidney-diagnosis-snowflake.git)
   cd kidney-diagnosis-snowflake
   ```

2. **Install dependencies**

   - Make sure you're using **Python 3.8 or higher**.
   - Then install all required packages:

     ```bash
     pip install -r requirements.txt
     ```

3. **Set up Snowflake credentials**

   - Create a file named `creds.json` in the project root.
   - Add your Snowflake account information like this:

     ```json
     {
       "account": "<your_account>",
       "user": "<your_username>",
       "password": "<your_password>",
       "role": "<your_role>",
       "warehouse": "<your_warehouse>",
       "database": "<your_database>",
       "schema": "<your_schema>"
     }
     ```

4. **Preprocess the dataset**

   - This will clean the raw `kidney_disease.csv` file and generate train/test splits under the `data/` folder.

     ```bash
     python preprocessing/preprocess.py
     ```

5. **Upload data to Snowflake**

   - Use the following script to upload the cleaned datasets to your Snowflake table:

     ```bash
     python snowflake_integration/load_to_snowflake.py
     ```

6. **Run the Streamlit app**

   - Launch the app in your browser:

     ```bash
     streamlit run streamlit_app/app_kidney.py
     ```

---

## 📊 Streamlit UI

- Interactive form for entering patient data
- Real-time kidney disease prediction
- Optional: Extend with SHAP/LIME for model interpretability

---

## 📄 Project Report

The `report.doc` file includes:
- Dataset exploration
- Preprocessing pipeline
- Modeling techniques
- Performance metrics
- Key takeaways

---

## 🚀 Future Improvements

- Add UDF support to run models directly inside Snowflake
- Use SHAP or LIME for interpretability
- Deploy the app on Streamlit Cloud or Hugging Face Spaces

---
