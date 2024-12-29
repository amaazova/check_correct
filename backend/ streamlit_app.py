import streamlit as st
import pandas as pd
import requests   # <-- Для запросов к серверу
from builder.config import PATH
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(
    page_icon="💣",
    layout="wide"
)

# -------------------------------------------------------------------
# 1. Демонстрация чтения локального CSV и EDA (как было в исходном коде)
# -------------------------------------------------------------------
data = pd.read_csv(f"{PATH}/daily_bars.csv")
data['price_date'] = pd.to_datetime(data['price_date'], format='%Y-%m-%d')

ticker = st.sidebar.selectbox("Choose ticker", data['ticker'].unique())

if ticker:
    ticker_data = data.loc[data['ticker'] == ticker]
    ticker_data = ticker_data[['price_date', 'open', 'high', 'low',
                               'close', 'volume', 'turnover', 'adjusted_close']]
    ticker_data.set_index('price_date', inplace=True)

    st.markdown(f"**Selected ticker:** {ticker}")

    timeframe = st.selectbox("Choose timeframe", ['daily', 'monthly', 'yearly'])
    reset = {'daily': '1D', 'monthly': 'M', 'yearly': 'Y'}.get(timeframe, '1D')

    # Ресемплируем
    resampled_data = ticker_data.resample(reset).agg({
        'open': 'last',
        'high': 'last',
        'low': 'last',
        'close': 'last',
        'volume': 'sum',
        'turnover': 'sum',
        'adjusted_close': 'last',
    })

    # Фильтруем для примера за 2024+ год
    # Если не нужно, можно убрать
    resampled_data = resampled_data.loc[resampled_data.index.year >= 2024]

    # Добавим простые Moving Averages
    resampled_data['MA20'] = resampled_data['close'].rolling(window=20).mean()
    resampled_data['MA50'] = resampled_data['close'].rolling(window=50).mean()

    # Создадим OHLC-график
    fig = go.Figure()

    # Ohlc trace
    fig.add_trace(go.Ohlc(
        x=resampled_data.index,
        open=resampled_data['open'],
        high=resampled_data['high'],
        low=resampled_data['low'],
        close=resampled_data['close'],
        name='OHLC',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # MA20
    fig.add_trace(go.Scatter(
        x=resampled_data.index,
        y=resampled_data['MA20'],
        mode='lines',
        name='20-Day MA',
        line=dict(color='blue', width=1.5)
    ))
    # MA50
    fig.add_trace(go.Scatter(
        x=resampled_data.index,
        y=resampled_data['MA50'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='orange', width=1.5)
    ))

    # Volume
    fig.add_trace(go.Bar(
        x=resampled_data.index,
        y=resampled_data['volume'],
        name='Volume',
        marker_color='lightgray',
        opacity=0.3,
        yaxis='y2'
    ))

    # Оформление
    fig.update_layout(
        title=f'{timeframe.upper()} OHLC DATA',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.header("First rows")
        st.dataframe(resampled_data.head(8))

    with col2:
        st.header("Basic stats")
        st.dataframe(resampled_data.describe())

    box_plot = go.Figure(go.Box(y=resampled_data['close'], name='Box Plot'))
    box_plot.update_layout(title='Box Plot based on close',
                           yaxis_title='Values')

    st.plotly_chart(box_plot)

# -------------------------------------------------------------------
# 2. Пример взаимодействия с Вашим FastAPI-сервером (необязательная часть)
# -------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Server interaction")

# Поле ввода URL вашего сервера
# Если сервер крутится локально: http://127.0.0.1:8000
server_url = st.sidebar.text_input("Backend server URL", value="http://127.0.0.1:8000")

if st.sidebar.button("Get models from server"):
    try:
        r = requests.get(f"{server_url}/models", timeout=5)
        if r.status_code == 200:
            models_data = r.json()
            st.write("**List of models from server**:")
            st.json(models_data)
        else:
            st.error(f"Request error. Status code: {r.status_code}, Detail: {r.text}")
    except Exception as e:
        st.error(f"Could not connect to server: {e}")

if st.sidebar.button("Get prediction for the chosen ticker"):
    if not ticker:
        st.error("Please select ticker first.")
    else:
        try:
            # Пример обращения к /predict эндпоинту
            params = {"ticker": ticker, "horizon": 5}
            r = requests.get(f"{server_url}/predict", params=params, timeout=5)
            if r.status_code == 200:
                prediction_data = r.json()
                st.write("**Server returned forecast**:")
                st.json(prediction_data)
            else:
                st.error(f"Prediction error. Status code: {r.status_code}, Detail: {r.text}")
        except Exception as e:
            st.error(f"Could not get prediction: {e}")
