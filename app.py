import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from chronos_integrated import chronos
from lstm_integrated import lstm
from prophet_integrated import prophet_model
from holtwinter_integrated import holt_winters_univariate_model
from sarimax_integrated import sarimax_multivariate
# from ucm_integrated import ucm
from patchtst_integrated import patchtst_fast
from patch_multi_integrated import patchtst_forecast_multi
# from toto.toto_integrated import toto_model
from toto.toto_multi_integrated import toto_multivariate_model
from svr_integrated import svr_univariate
from catboost_integrated import catboost_forecast
from rnn_integrated import rnn
from lstm_multi_integrated import lstm_forecast
from ann_integrated import ann_forecast_model
from randomforest_integrated import random_forest_forecast
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.vector_ar.var_model import VAR
from arima_intrgrated import arima
from var_integrated import var_model_multivariate
# Streamlit app configuration
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("Time Series Forecasting Dashboard")

# File uploader for CSV
st.subheader("Upload your dataset (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv","xlsx","xls"])

if uploaded_file is not None:
    # Read the uploaded CSV
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    else:  # xlsx or xls
        df = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Ensure 'date' column exists and convert to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        st.error("The dataset must contain a 'date' column.")
        st.stop()

    # Select target column
    columns = [col for col in df.columns if col != 'date']
    target = st.selectbox("Select target column", columns)

    # Select date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    # Select analysis type
    analysis_type = st.selectbox("Select Analysis Type", ["Univariate", "Multivariate"])

    # Model category and model selection
    if analysis_type == "Univariate":
        model_category = st.selectbox("Select Model Category", ["Traditional", "ML", "DL", "Gen AI"])
        if model_category == "Traditional":
            model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "Holt-Winters"])
        elif model_category == "ML":
            model_choice = st.selectbox("Select Forecasting Model", ["SVR", "Prophet"])
        elif model_category == "DL":
            model_choice = st.selectbox("Select Forecasting Model", ["LSTM", "RNN"])
        elif model_category == "Gen AI":
            model_choice = st.selectbox("Select Forecasting Model", ["Chronos", "PatchTST"])
    else:  # Multivariate
        model_category = st.selectbox("Select Model Category", ["Traditional", "ML", "DL", "Gen AI"])
        if model_category == "Traditional":
            model_choice = st.selectbox("Select Forecasting Model", ["VAR", "SARIMA"])
        elif model_category == "ML":
            model_choice = st.selectbox("Select Forecasting Model", ["Random Forest", "CatBoost"])
        elif model_category == "DL":
            model_choice = st.selectbox("Select Forecasting Model", ["LSTM", "ANN"])
        elif model_category == "Gen AI":
            model_choice = st.selectbox("Select Forecasting Model", ["Toto", "PatchTST"])

    if st.button("Run Model"):
        try:
            # Run the selected model
            test_size = 0.3
            if analysis_type == "Univariate":
                if model_choice == "ARIMA":
                    model_results, data_frames = arima(df, start_date, end_date, target, test_size)
                if model_choice == "Holt-Winters":
                    model_results, data_frames = holt_winters_univariate_model(df, start_date, end_date, target, test_size)
                elif model_choice == "SVR":
                    model_results, data_frames = svr_univariate(df, start_date, end_date, target, test_size)
                elif model_choice == "Prophet":
                    model_results, data_frames = prophet_model(df, start_date, end_date, target, test_size)
                elif model_choice == "LSTM":
                    model_results, data_frames = lstm(df, start_date, end_date, target, test_size)
                elif model_choice == "RNN":
                    model_results, data_frames = rnn(df, start_date, end_date, target, test_size)
                elif model_choice == "Chronos":
                    model_results, data_frames = chronos(df, start_date, end_date, target, test_size)
                elif model_choice == "PatchTST":
                    model_results, data_frames = patchtst_fast(df, start_date, end_date, target, test_size)
            else:  # Multivariate
                if model_choice == "VAR":
                    model_results, data_frames = var_model_multivariate(df, start_date, end_date, target, test_size)
                if model_choice == "SARIMA":
                    model_results, data_frames = sarimax_multivariate(df, start_date, end_date, target, test_size)
                elif model_choice == "Random Forest":
                    model_results, data_frames = random_forest_forecast(df, start_date, end_date, target, test_size)
                elif model_choice == "CatBoost":
                    model_results, data_frames = catboost_forecast(df, start_date, end_date, target, test_size)
                elif model_choice == "LSTM":
                    model_results, data_frames = lstm_forecast(df, start_date, end_date, target, test_size)
                elif model_choice == "ANN":
                    model_results, data_frames = ann_forecast_model(df, start_date, end_date, target, test_size, lags=10)
                elif model_choice == "Toto":
                    model_results, data_frames =toto_multivariate_model(df, start_date, end_date, target, test_size)
                elif model_choice == "PatchTST":
                    model_results, data_frames = patchtst_forecast_multi(df, start_date, end_date, target, test_size)

            # Display metrics
            st.subheader("Model Performance Metrics")
            metrics = model_results["metrics"]
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Training Metrics**")
                st.write(f"Train MAPE: {metrics['train-mape']:.2f}%")
                st.write(f"Train Accuracy: {metrics['train-accuracy']:.2f}%")
            with col2:
                st.write("**Test Metrics**")
                st.write(f"Test MAPE: {metrics['test-mape']:.2f}%")
                st.write(f"Test Accuracy: {metrics['test-accuracy']:.2f}%")

            # Plot graphs
            st.subheader("Forecast Visualizations")
            
            # Training plot
            train_graph = model_results["graphs"]["train"]
            train_df = data_frames["train"]
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(
                x=train_graph["x-series"],
                y=train_df[target],
                mode='lines',
                name='Actual'
            ))
            fig_train.add_trace(go.Scatter(
                x=train_graph["x-series"],
                y=train_graph["y-series"],
                mode='lines',
                name='Predicted'
            ))
            fig_train.update_layout(
                title=train_graph["graph-label"],
                xaxis_title=train_graph["x-label"],
                yaxis_title=train_graph["y-label"],
                template="plotly_white"
            )
            st.plotly_chart(fig_train, use_container_width=True)

            # Test plot
            test_graph = model_results["graphs"]["test"]
            test_df = data_frames["test"]
            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(
                x=test_graph["x-series"],
                y=test_df[target],
                mode='lines',
                name='Actual'
            ))
            fig_test.add_trace(go.Scatter(
                x=test_graph["x-series"],
                y=test_graph["y-series"],
                mode='lines',
                name='Predicted'
            ))
            fig_test.update_layout(
                title=test_graph["graph-label"],
                xaxis_title=test_graph["x-label"],
                yaxis_title=test_graph["y-label"],
                template="plotly_white"
            )
            st.plotly_chart(fig_test, use_container_width=True)

        except Exception as e:
            st.error(f"Error running model: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")