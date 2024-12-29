# team74_stock_price_forecasting/backend/pipelines/arima_pipeline.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error


def _arima_for_single_ticker(
    df_single: pd.DataFrame,
    horizon: int = 30,
    test_ratio: float = 0.2,
    remove_outliers: bool = True,
    min_data_points: int = 50
):
    """
    Логика ARIMA(1,1,1) для одного тикера (одномерная временная серия).
    df_single: DataFrame c колонками ["date","adjusted_close"] (только 1 ticker).
    Возвращает словарь: { status, order, aic, mape_test, forecast, ... }
    """

    # Проверка объёма
    if len(df_single) < min_data_points:
        raise ValueError(f"Single-ticker dataset too small ({len(df_single)})")

    # Заполнение пропусков
    df_single["adjusted_close"] = df_single["adjusted_close"].ffill().bfill()

    # Удаление выбросов
    if remove_outliers:
        low_q, high_q = df_single["adjusted_close"].quantile([0.01, 0.99])
        df_single = df_single[
            (df_single["adjusted_close"] >= low_q) &
            (df_single["adjusted_close"] <= high_q)
        ]

    df_single = df_single.sort_values("date").reset_index(drop=True)
    if len(df_single) < min_data_points:
        raise ValueError(f"After outliers removal, single ticker data too small: {len(df_single)}")

    # Делим на train / test
    n = len(df_single)
    train_size = int(n * (1 - test_ratio))
    train_df = df_single.iloc[:train_size]
    test_df = df_single.iloc[train_size:]

    series_train = train_df["adjusted_close"].values
    series_test = test_df["adjusted_close"].values

    # Строим ARIMA(1,1,1) на train
    # Если нужно выключать предупреждения, можно catch_warnings
    model = ARIMA(series_train, order=(1,1,1)).fit()
    order = (1,1,1)  
    aic = model.aic

    # Прогноз на тест
    test_preds = model.forecast(steps=len(series_test))
    mape_test = None
    if len(series_test) > 0:
        mape_test = mean_absolute_percentage_error(series_test, test_preds)

    # Финальный прогноз (обновляем модель на всём df_single)
    # Либо заново fit на всём df_single
    model_full = ARIMA(df_single["adjusted_close"].values, order=(1,1,1)).fit()
    final_pred = model_full.forecast(steps=horizon)

    last_date = df_single["date"].max()
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
    forecast_list = []
    for i in range(horizon):
        forecast_list.append({
            "date": str(future_dates[i].date()),
            "predicted": float(final_pred[i])
        })

    return {
        "status": "ok",
        "train_size": train_size,
        "test_size": len(series_test),
        "order": order,
        "aic": aic,
        "mape_test": mape_test,
        "horizon": horizon,
        "forecast": forecast_list
    }


def process_and_forecast_arima(
    df: pd.DataFrame,
    horizon: int = 30,
    test_ratio: float = 0.2,
    max_p: int = 5,  # <-- теперь не используется, но оставим для совместимости
    max_q: int = 5,  # <-- теперь не используется
    remove_outliers: bool = True,
    min_data_points: int = 50
):
    """
    Обработка многотикерного df:
    - Группируем по 'ticker'
    - Для каждого вызываем _arima_for_single_ticker(...) (жёстко (1,1,1))
    - Собираем результаты в список [{ ticker, status, order, forecast... }, ...]
    """
    results = []
    # Группируем по тикеру
    for tck, grp in df.groupby("ticker"):
        single_df = grp.copy()
        if "date" not in single_df.columns or "adjusted_close" not in single_df.columns:
            raise ValueError(f"Ticker {tck} missing required columns.")

        single_result = _arima_for_single_ticker(
            df_single=single_df,
            horizon=horizon,
            test_ratio=test_ratio,
            remove_outliers=remove_outliers,
            min_data_points=min_data_points
        )
        single_result["ticker"] = tck
        results.append(single_result)

    return results
