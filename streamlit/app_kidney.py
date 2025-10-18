
# Snowpark
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit, as_double
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import json
import os

# Create Session object
@st.cache_resource
def create_session_object():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(BASE_DIR, '..','creds.json')
    with open(creds_path) as f:
        connection_parameters = json.load(f)

     
    session = Session.builder.configs(connection_parameters).create()
    print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
    return session


def train (session, table, model, cwh, cwh_size, use_optimized, use_zero_copy_cloning):

    if (use_optimized):
        cmd = "alter warehouse " + cwh + " suspend"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set warehouse_size = '2X-Large'"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED'"
        session.sql(cmd).collect()
    
    model_name = str.replace(model, ' ', '_')
    session.call('sf_train',model, table, '@models', model_name, use_zero_copy_cloning)

    if (use_optimized):
        cmd = "alter warehouse " + cwh + " suspend"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set WAREHOUSE_TYPE = 'STANDARD'"
        session.sql(cmd).collect()

        cmd = "alter warehouse " + cwh + " set warehouse_size = '" + cwh_size + "'"
        session.sql(cmd).collect()


def score (session, table_orig, model_name, target_table, cwh, cwh_size, size_wh):

    cmd = "alter warehouse " + cwh + " set warehouse_size = '" + size_wh + "'"
    session.sql(cmd).collect()

    # in your Streamlit `score` function, before session.call:
# table_orig is currently "DATA.DEFAULT", so:
    full_name  = table_orig 
    short_name = table_orig.split('.')[-1]   # => "DEFAULT"
    session.call('sf_score', full_name, target_table, '@models', model_name)


 
    cmd = "alter warehouse " + cwh + " set warehouse_size = '" + cwh_size + "'"
    session.sql(cmd).collect()

    
def copy_into (session, list_files, table_name):

    session.call('copy_into', list_files, table_name)

    
def to_pct(value):
    
    val1= (float(value) * 100)
    val2 = f'{val1:.2f}'
    
    return val2 + " %"


def as_scalar(val):
    """
    Take either a length-1 pandas Series or a raw Python value,
    and return either an int/float/str or None.
    """
    # If it’s a pandas Series, grab its first element
    if hasattr(val, "iat"):
        scalar = val.iat[0]
    else:
        scalar = val

    # If it’s NaN (float) or pd.NA, normalize to None
    if scalar is pd.NA or (isinstance(scalar, float) and pd.isna(scalar)):
        return None

    # Otherwise, if it’s a NumPy scalar, convert to a native Python type
    try:
        return scalar.item()
    except:
        return scalar
#########################################
##### MAIN STREAMLIT APP STARTS HERE ####
#########################################


st.set_page_config(page_title="KD Classification",page_icon="❄️")

# Add header and a subheader
st.header("Early Kidney Disease Detection")

session = create_session_object()

with st.sidebar:
    option = option_menu("", ["Data", "Analyze Data", "Data Visualization","Training", "Models and Inference","Prediction / Scoring"
                                                             ],
                            icons=['upload','graph-up', 'play-circle','list-task', 'boxes', 'speedometer2'],
                            menu_icon="snow", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "white","font-color": "#249dda"},
            "icon": {"color": "#31c0e7", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "white"},
            "nav-link-selected": {"background-color": "7734f9"},
        })

if option == "Data":

    # List files in Snowflake stage
    data_load = session.sql('ls @load_data').collect()

    st.markdown("## 📂 Data Loading")
    st.markdown("Upload, preview, and load staged data into your Snowflake table.")
    st.markdown("---")

    # Create columns for layout
    col_files, col_name_table = st.columns(2)

    with st.container():
        with col_files:
            st.markdown("### 🔍 Select File")
            list_files = []
            files_available = session.sql("ls @load_data").collect()
            
            for f in files_available:
                list_files.append(f["name"])

            files = st.selectbox(
                "Choose a file from your Snowflake stage:",
                list_files,
                index=0 if list_files else None,
                help="Select the staged file you want to load into your Snowflake table."
            )

            st.success(f"✅ File selected: `{files}`")

        with col_name_table:
            st.markdown("### 🧾 Create Table")
            table_name_input = st.text_input(
                "Enter a table name to be created:",
                value="DEFAULT",
                placeholder="Enter table name..."
            )
            table_name = "DATA." + table_name_input.strip().upper()
            st.info(f"Table to be created: `{table_name}`")

        # Prepare file path for COPY INTO
        files = "@" + files

        st.markdown("---")
        st.markdown("###  Load Data into Snowflake")

        st.button(
            "Load Data",
            on_click=copy_into,
            args=(session, files, table_name),
            type="primary",
            use_container_width=True
        )

    st.markdown("---")
    st.caption("Tip: Ensure your stage and table schema match before loading data.")
elif option == "Analyze Data":

    st.markdown("## 📊 Analyze Data")
    st.markdown("View tables, explore data samples, and examine descriptive statistics.")
    st.markdown("---")

    with st.container():
        df_tables = (
            session.table("information_schema.tables")
            .filter(col("table_schema") == "DATA")
            .select(col("table_name"), col("row_count"), col("created"))
        )
        pd_tables = df_tables.to_pandas()

        st.markdown("### 🧾 Available Tables in `DATA` Schema")
        st.dataframe(pd_tables, use_container_width=True, hide_index=True)

    with st.container():
        st.markdown("---")
        list_tables_names = pd_tables["TABLE_NAME"].values.tolist()

        table_to_print = st.selectbox(
            "🔍 Select a table to describe statistics:",
            list_tables_names,
            index=0 if list_tables_names else None,
            help="Choose a table to preview and analyze basic statistics.",
        )

        if table_to_print:
            table_to_print = "DATA." + table_to_print

            df_table = session.table(table_to_print)
            pd_table = df_table.limit(3).to_pandas()
            pd_describe = df_table.describe().to_pandas()

            # Summary metrics
            col1, col2 = st.columns(2)
            with col1:
                positive = df_table.filter(col('"CLASSIFICATION"') == 1).count()
                st.metric(label="🟢 Positive", value=positive)

            with col2:
                negative = df_table.filter(col('"CLASSIFICATION"') == 0).count()
                st.metric(label="🔴 Negative", value=negative)

            st.markdown("---")

            with st.container():
                st.markdown(f"### 📋 Preview: `{table_to_print}`")
                st.dataframe(pd_table, use_container_width=True)

            with st.container():
                st.markdown("### 📈 Data Description")
                st.dataframe(pd_describe, use_container_width=True)

    st.markdown("---")
    st.caption("Tip: Metrics above are based on the `CLASSIFICATION` column if present.")
elif option == "Training":

    st.markdown("##  Model Training")
    st.markdown("Select a dataset, choose a model, and configure training options.")
    st.markdown("---")

    with st.container():
        df_tables = (
            session.table("information_schema.tables")
            .filter(col("table_schema") == "DATA")
            .select(col("table_name"))
        )
        pd_tables = df_tables.to_pandas()

        list_tables_names = pd_tables["TABLE_NAME"].values.tolist()
        table_to_train = st.selectbox(
            "📦 Select table to train model:",
            list_tables_names,
            index=0 if list_tables_names else None,
            help="Select a dataset from your Snowflake `DATA` schema for training.",
        )

        if table_to_train:
            table_to_train = "DATA." + table_to_train
            st.info(f"✅ Table selected for training: `{table_to_train}`")

            st.markdown("---")

            with st.container():
                df_models = session.table("models").select(col("model_name"))
                pd_models = df_models.to_pandas()

                model_option = st.selectbox(
                    " Choose a training model:",
                    pd_models,
                    index=0 if not pd_models.empty else None,
                    help="Select one of the registered models from the `models` table.",
                )

                if model_option is not None:
                    st.success(f"Model selected: `{model_option}`")

                    cwh = session.sql("select current_warehouse()").collect()
                    cwh = str(cwh[0])
                    cwh = (
                        cwh.replace("CURRENT_WAREHOUSE", "")
                        .replace(")", "")
                        .replace("Row((=", "")
                        .replace("'", "")
                    )

                    cmd = "show warehouses like '" + cwh + "'"
                    cwh_size = session.sql(cmd).collect()
                    cwh_size = cwh_size[0]["size"]

                    st.markdown("---")
                    st.markdown("### ⚙️ Training Configuration")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        use_zero_copy_cloning = st.checkbox(
                            "Keep zero-copy clone of training data"
                        )
                    with col2:
                        use_optimized = st.checkbox(
                            "Use optimized warehouse for large trainings"
                        )
                    with col3:
                        st.button(
                            "Train Model",
                            on_click=train,
                            args=(
                                session,
                                table_to_train,
                                model_option,
                                cwh,
                                cwh_size,
                                use_optimized,
                                use_zero_copy_cloning,
                            ),
                            type="primary",
                            use_container_width=True,
                        )

    st.markdown("---")
    st.caption("Tip: Zero-copy cloning preserves your training data snapshot without duplication.")

elif option == "Models and Inference":
    
    st.markdown("## 🧠 Models Catalog")
    st.markdown("View performance metrics, accuracy rankings, and model details.")
    st.markdown("---")

    # --- SECTION 1: Model Catalog Table ---
    with st.container():
        df_accuracy = session.table("accuracy_sum_v")
        pd_accuracy = df_accuracy.to_pandas()
        st.markdown("### 📋 All Trained Models")
        st.dataframe(pd_accuracy, use_container_width=True, hide_index=True)

    # --- SECTION 2: Top 5 Models by Accuracy ---
    with st.container():
        st.markdown("---")
        st.markdown("### 🏆 Top 5 Models by Accuracy")

        df_top = (
            df_accuracy
            .select(col("MODEL_NAME"), as_double(col("ACCURACY")).alias("ACCURACY"))
            .sort(col("ACCURACY"), ascending=False)
            .limit(5)
        )
        pd_top = df_top.to_pandas()
        pd_top.set_index("MODEL_NAME", inplace=True)

        st.bar_chart(pd_top, use_container_width=True)
        st.caption("Higher bars indicate better-performing models based on accuracy.")

    # --- SECTION 3: Detailed Model Metrics ---
    with st.container():
        st.markdown("---")
        st.markdown("### 🔍 View Detailed Model Statistics")

        list_models = pd_accuracy["MODEL_NAME"].tolist()
        model = st.selectbox(
            "Select a model to view details:",
            list_models,
            index=0 if list_models else None,
            help="Select a model from the catalog above to view its performance metrics.",
        )

        if model:
            pd_model = (
                session.table("class_report_sumary_v")
                .filter(col("MODEL_NAME") == model)
                .to_pandas()
            )

            # --- Header info ---
            st.markdown("---")
            st.markdown("#### 🧾 Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Model Name: {pd_model['MODEL_NAME'].values[0]}")
            with col2:
                st.text(f"Training Data: {pd_model['DATA_TRAINING'].values[0]}")

            st.markdown("---")

            # --- Confusion Matrix Summary ---
            st.markdown("### 🔢 Confusion Matrix Summary")

            col1, col2 = st.columns(2)
            with col1:
                tp_val = as_scalar(pd_model["TP"])
                st.metric(label="🟢 True Positive", value=tp_val)
            with col2:
                fp_val = as_scalar(pd_model["FP"])
                st.metric(label="🔴 False Positive", value=fp_val)

            col3, col4 = st.columns(2)
            with col3:
                fn_val = as_scalar(pd_model["FN"])
                st.metric(label="🟠 False Negative", value=fn_val)
            with col4:
                tn_val = as_scalar(pd_model["TN"])
                st.metric(label="🔵 True Negative", value=tn_val)

            st.markdown("---")

            # --- Negative Class Metrics ---
            st.markdown("### ⚖️ Negative Class Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="F1 Score (Negative)",
                    value=to_pct(pd_model["NEG_F1_SCORE"].values[0]),
                )
            with col2:
                st.metric(
                    label="Precision (Negative)",
                    value=to_pct(pd_model["NEG_PRECISION"].values[0]),
                )
            with col3:
                st.metric(
                    label="Recall (Negative)",
                    value=to_pct(pd_model["NEG_RECALL"].values[0]),
                )

            # --- Positive Class Metrics ---
            st.markdown("### 🌟 Positive Class Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="F1 Score (Positive)",
                    value=to_pct(pd_model["POS_F1_SCORE"].values[0]),
                )
            with col2:
                st.metric(
                    label="Precision (Positive)",
                    value=to_pct(pd_model["POS_PRECISION"].values[0]),
                )
            with col3:
                st.metric(
                    label="Recall (Positive)",
                    value=to_pct(pd_model["POS_RECALL"].values[0]),
                )

            st.markdown("---")

            # --- Overall Accuracy ---
            st.markdown("### 🎯 Overall Accuracy")
            st.metric(
                label="Model Accuracy",
                value=to_pct(pd_model["ACCURACY"].values[0]),
                delta_color="normal",
            )

            st.caption("All values shown are aggregated summaries from the evaluation phase.")
elif option == "Data Visualization":

    st.markdown("## 📊 Data Visualization")
    st.markdown("Explore feature distributions, correlations, and relationships.")
    st.markdown("---")

    # Fetch available tables
    df_tables = (
        session.table("information_schema.tables")
        .filter(col("table_schema") == "DATA")
        .select(col("table_name"))
    )
    pd_tables = df_tables.to_pandas()

    list_tables_names = pd_tables["TABLE_NAME"].values.tolist()
    table_selected = st.selectbox(
        "📦 Select a table for visualization:",
        list_tables_names,
        index=0 if list_tables_names else None,
        help="Select a dataset from your Snowflake `DATA` schema to visualize.",
    )

    if table_selected:
        table_selected_full = "DATA." + table_selected
        df_table = session.table(table_selected_full)
        pd_table = df_table.limit(5000).to_pandas()  # limit to prevent lag

        st.success(f"✅ Loaded table `{table_selected_full}` for visualization.")

        st.markdown("---")
        st.subheader("📈 Numeric Column Distribution")

        numeric_cols = pd_table.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if numeric_cols:
            col_to_plot = st.selectbox("Select a numeric column:", numeric_cols)
            st.bar_chart(pd_table[col_to_plot])
        else:
            st.info("No numeric columns found for histogram display.")

        st.markdown("---")
        st.subheader("📉 Correlation Heatmap")

        import seaborn as sns
        import matplotlib.pyplot as plt

        if len(numeric_cols) >= 2:
            corr = pd_table[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns to compute correlation heatmap.")

        st.markdown("---")
        # st.subheader("🧮 Pairwise Relationships")

        # import plotly.express as px
        # selected_cols = st.multiselect(
        #     "Choose 2–4 numeric columns to plot:",
        #     numeric_cols,
        #     default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
        # )
        # if len(selected_cols) >= 2:
        #     fig = px.scatter_matrix(pd_table, dimensions=selected_cols)
        #     st.plotly_chart(fig, use_container_width=True)
        # else:
        #     st.caption("Select at least two numeric columns to view relationships.")

        st.markdown("---")
        st.caption("Tip: Sampling limited to 5000 rows for performance.")
elif option == "Prediction / Scoring":

    st.markdown("## 🧮 Model Prediction / Scoring")
    st.markdown("Run model inference on new data or existing Snowflake tables.")
    st.markdown("---")

    # Load available models
    df_models = session.table("models").select(col("model_name"))
    pd_models = df_models.to_pandas()

    df_accuracy = session.table("accuracy_sum_v")
    pd_accuracy = df_accuracy.to_pandas()
    list_models = pd_accuracy["MODEL_NAME"].tolist()

    # list_models = pd_models["MODEL_NAME"].tolist()

    model_selected = st.selectbox(
        "Select a model for prediction:",
        list_models,
        index=0 if list_models else None,
        help="Choose one of your trained models for inference.",
    )

    if model_selected:
        st.success(f"✅ Model selected: `{model_selected}`")
        st.markdown("---")

        prediction_mode = st.radio(
            "Choose prediction mode:",
            ["Manual Input", "Table Scoring"],
            horizontal=True,
        )

        # ------------------ MANUAL INPUT MODE ------------------
        if prediction_mode == "Manual Input":
            st.markdown("### ✍️ Enter feature values manually")

            input_str = st.text_area(
                "Enter comma-separated feature values:",
                placeholder="e.g., 45, 1.2, 3.6, 0, 85",
                help="Provide feature values in the same order as training data columns.",
            )

            if st.button("Run Prediction", type="primary", use_container_width=True):
                if input_str.strip():
                    try:
                        result = session.call("sf_predict_manual", model_selected, input_str)
                        st.success(f"Predicted Output: `{result}`")
                    except Exception as e:
                        st.error(f"⚠️ Prediction failed: {e}")
                else:
                    st.warning("Please enter valid feature values.")

        # ------------------ TABLE SCORING MODE ------------------
        elif prediction_mode == "Table Scoring":
            st.markdown("### 📋 Score an Entire Table")

            df_tables = (
                session.table("information_schema.tables")
                .filter(col("table_schema") == "DATA")
                .select(col("table_name"))
            )
            pd_tables = df_tables.to_pandas()
            list_tables = pd_tables["TABLE_NAME"].tolist()

            table_input = st.selectbox(
                "Select input table for scoring:",
                list_tables,
                index=0 if list_tables else None,
                help="Choose a Snowflake table to score using the selected model.",
            )

            if table_input:
                table_full = "DATA." + table_input
                target_table = table_full + "_SCORED"

                cwh = session.sql("select current_warehouse()").collect()
                cwh = str(cwh[0])
                cwh = (
                    cwh.replace("CURRENT_WAREHOUSE", "")
                    .replace(")", "")
                    .replace("Row((=", "")
                    .replace("'", "")
                )

                cmd = "show warehouses like '" + cwh + "'"
                cwh_size = session.sql(cmd).collect()
                cwh_size = cwh_size[0]["size"]

                if st.button("Score Table", type="primary", use_container_width=True):
                    with st.spinner("Scoring table..."):
                        try:
                            score(
                                session,
                                table_full,
                                model_selected,
                                target_table,
                                cwh,
                                cwh_size,
                                size_wh="LARGE",
                            )
                            st.success(f"✅ Scoring completed! Output table: `{target_table}`")
                        except Exception as e:
                            st.error(f"⚠️ Error during scoring: {e}")

    st.markdown("---")
    st.caption("Tip: You can use Snowflake stored procedures like `sf_score` or `sf_predict_manual` to execute model predictions.")
