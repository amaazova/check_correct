# streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="ARIMA + Portfolio Frontend", layout="wide")

st.title("Stock Forecasting (ARIMA + Portfolio)")

# =============================================================================
# Блок ввода базового URL для FastAPI-сервера
# (Например: "http://127.0.0.1:8000" или "https://<your-deployed-fastapi>/")
# =============================================================================
base_url = st.sidebar.text_input("FastAPI base URL", "http://127.0.0.1:8000")

# Для удобства небольшая функция, которая будет формировать полный путь
def endpoint_url(path: str) -> str:
    return base_url.rstrip("/") + path

# =============================================================================
# Табы в Streamlit
# =============================================================================
tabs = st.tabs(["Home", "Upload dataset", "Fit user data", "Predict", "Build portfolio", "Models / Set Active"])

# -----------------------------------------------------------------------------
# Tab 1: Home
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Home (Ping server)")

    if st.button("Check server status"):
        try:
            resp = requests.get(endpoint_url("/"))
            if resp.status_code == 200:
                data = resp.json()
                st.write("Server response:", data)
            else:
                st.error(f"Status code: {resp.status_code}, content: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# Tab 2: Upload dataset
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Upload CSV dataset for user ARIMA model")
    uploaded_file = st.file_uploader("Select CSV file with columns: ticker, date, adjusted_close", type=["csv"])

    if uploaded_file is not None:
        if st.button("Upload to server"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(endpoint_url("/upload_dataset"), files=files)
                if resp.status_code == 200:
                    st.success(f"Uploaded. Server response: {resp.json()}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# -----------------------------------------------------------------------------
# Tab 3: Fit user data (user_arima)
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Train on user_data.csv => user_arima")

    horizon = st.number_input("Horizon (days)", value=30, min_value=1, step=1)
    max_wait = st.number_input("Max wait (sec)", value=10, min_value=1, step=1)

    if st.button("Fit user_arima"):
        try:
            params = {"horizon": horizon, "max_wait": max_wait}
            resp = requests.post(endpoint_url("/fit_user_data"), params=params)
            if resp.status_code == 200:
                st.success(resp.json())
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# Tab 4: Predict
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Predict endpoint")
    st.write("Можно передать `model_id` (иначе возьмётся active_model), а также `ticker`")

    model_id_input = st.text_input("model_id (optional, e.g. user_arima)", value="")
    ticker_input = st.text_input("ticker (optional, e.g. AAPL)", value="")
    horizon_pred = st.number_input("Horizon to show", min_value=1, max_value=100, value=10)

    if st.button("Call /predict"):
        params = {}
        if model_id_input.strip():
            params["model_id"] = model_id_input.strip()
        if ticker_input.strip():
            params["ticker"] = ticker_input.strip()
        params["horizon"] = horizon_pred

        try:
            resp = requests.get(endpoint_url("/predict"), params=params)
            if resp.status_code == 200:
                data = resp.json()
                st.write("Response:", data)
                # Можем попробовать отобразить в табличном виде:
                df = pd.DataFrame([vars(x) for x in data["forecast"]]) if data.get("forecast") else pd.DataFrame()
                st.dataframe(df)
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# Tab 5: Build portfolio
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Build portfolio endpoint")
    use_user_data = st.checkbox("Use user_data (user_arima)?", value=False)
    momentum_period = st.number_input("Momentum period", min_value=1.0, value=5.0)
    top_quantile = st.number_input("Top quantile fraction", min_value=0.01, max_value=1.0, value=0.1)

    if st.button("Build portfolio"):
        params = {
            "use_user_data": str(use_user_data).lower(),
            "momentum_period": momentum_period,
            "top_quantile": top_quantile
        }
        # Преобразуем bool -> "true"/"false" если нужно.
        try:
            resp = requests.get(endpoint_url("/build_portfolio"), params=params)
            if resp.status_code == 200:
                data = resp.json()
                st.write("Response:", data)
                if "cumprod" in data and data["cumprod"]:
                    # data["cumprod"] - это dict с ключами = названия столбцов
                    # Можем попробовать построить DataFrame
                    df_cumprod = pd.DataFrame(data["cumprod"])
                    st.dataframe(df_cumprod)
                    st.line_chart(df_cumprod.set_index("date"))
                else:
                    st.info("cumprod is empty or missing.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# Tab 6: Models / Set Active
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("List all models on server & set active model")

    if st.button("Refresh model list"):
        try:
            resp = requests.get(endpoint_url("/models"))
            if resp.status_code == 200:
                data = resp.json()
                st.write("Models:", data)
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    model_to_set = st.text_input("Model ID to set active", value="user_arima")
    if st.button("Set model"):
        try:
            resp = requests.post(endpoint_url("/set"), json={"model_id": model_to_set})
            if resp.status_code == 200:
                st.success(resp.json())
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")
