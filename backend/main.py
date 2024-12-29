# team74_stock_price_forecasting/backend/main.py

import os
import shutil
import time
import chardet
import pandas as pd
from typing import Annotated, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
import logging
from logging.handlers import RotatingFileHandler
from datetime import date
import concurrent.futures  # <-- добавили для ThreadPoolExecutor

from pipelines.arima_pipeline import process_and_forecast_arima
from pipelines.portfolio_pipeline import build_portfolio_pipeline

def get_custom_logger(logger_name: str = "my_app_logger") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)
    os.makedirs("logs", exist_ok=True)
    logfile_path = os.path.join("logs", f"{date.today():%Y-%m-%d}.log")
    # Здесь можно было бы использовать RotatingFileHandler, но пока что оставим FileHandler
    file_handler = logging.FileHandler(logfile_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = get_custom_logger("my_app_logger")

ALL_FORECASTS_CSV = "data/all_forecast_results.csv"
USER_DATA_PATH = "user_data.csv"

df_pretrained = pd.DataFrame()
model_store = {}                # {model_id: {"description":..., "df_forecast": DataFrame}, ...}
active_model_id: Optional[str] = None

from pydantic import BaseModel, Field

class BasicResponse(BaseModel):
    status: str
    message: Optional[str] = None

class FitResponse(BaseModel):
    status: str
    model_id: Optional[str] = None
    time_spent: Optional[float] = None
    message: Optional[str] = None

class PredictItem(BaseModel):
    ticker: str
    date: str
    predicted: float

class PredictResponse(BaseModel):
    status: str
    model_id: str
    ticker: Optional[str] = None
    rows: int
    forecast: List[PredictItem]
    message: Optional[str] = None

class ModelsListItem(BaseModel):
    model_id: str
    description: str

class ModelsResponse(BaseModel):
    models: List[ModelsListItem]

class SetModelResponse(BaseModel):
    status: str
    active_model: str

class PortfolioResponse(BaseModel):
    status: str
    cumprod: dict = Field(default_factory=dict)
    message: Optional[str] = None

class ModelID(BaseModel):
    model_id: str

app = FastAPI(title="ARIMA + Portfolio")

# При старте загружаем "pretrained_arima"
try:
    df_pretrained = pd.read_csv(ALL_FORECASTS_CSV)
    df_pretrained["date"] = pd.to_datetime(df_pretrained["date"], errors="coerce")
    df_pretrained = df_pretrained.sort_values(["Ticker", "date"]).reset_index(drop=True)
    model_store["pretrained_arima"] = {
        "description": "Pretrained ARIMA (auto parameters)",
        "df_forecast": df_pretrained.copy()
    }
    active_model_id = "pretrained_arima"  # По умолчанию используем pretrained
    logger.info(f"Loaded pretrained model at startup, shape={df_pretrained.shape}")
except Exception as e:
    logger.error(f"Cannot load {ALL_FORECASTS_CSV}: {e}")

def read_csv_with_autodetect(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read(8192)
    detect_result = chardet.detect(raw)
    encoding = detect_result["encoding"]
    if not encoding or detect_result["confidence"] < 0.5:
        encoding = "utf-8"
    df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    return df

def validate_arima_csv(df: pd.DataFrame):
    df.columns = [c.lower().strip() for c in df.columns]
    required = ["ticker","date","adjusted_close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing col {col}")
    return df

@app.get("/", response_model=BasicResponse)
async def root() -> BasicResponse:
    return BasicResponse(status="ok", message="Server up with a pretrained ARIMA.")

@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """
    Динамически показываем все модели из model_store.
    """
    out_list = []
    for mid, info in model_store.items():
        desc = info.get("description", "")
        out_list.append(ModelsListItem(model_id=mid, description=desc))
    return ModelsResponse(models=out_list)

@app.post("/set", response_model=SetModelResponse)
async def set_active_model(payload: ModelID) -> SetModelResponse:
    global active_model_id
    mid = payload.model_id
    if mid not in model_store:
        raise HTTPException(status_code=404, detail=f"No such model {mid}")
    active_model_id = mid
    return SetModelResponse(status="ok", active_model=mid)

class UploadDatasetResponse(BaseModel):
    status: str
    rows: int
    tickers_count: int

@app.post("/upload_dataset", response_model=UploadDatasetResponse)
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> UploadDatasetResponse:
    try:
        temp_path = "temp_user.csv"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = read_csv_with_autodetect(temp_path)
        df = validate_arima_csv(df)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df.to_csv(USER_DATA_PATH, index=False)

        return UploadDatasetResponse(
            status="success",
            rows=len(df),
            tickers_count=df["ticker"].nunique()
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fit", response_model=FitResponse)
async def fit_pretrained_arima(max_wait: int=10) -> FitResponse:
    start_time = time.time()
    # Имитируем какое-то «обучение»
    # Если нужно параллельное — здесь также можно ThreadPoolExecutor, но пока не трогаем
    time.sleep(3)
    elapsed = time.time() - start_time
    if elapsed > max_wait:
        return FitResponse(status="timeout", message="Fit took too long.")
    if df_pretrained.empty:
        raise HTTPException(status_code=500, detail="df_pretrained is empty, cannot proceed")

    model_store["pretrained_arima"] = {
        "description": "Pretrained ARIMA (auto parameters) updated",
        "df_forecast": df_pretrained.copy()
    }
    return FitResponse(
        status="ok",
        model_id="pretrained_arima",
        time_spent=round(elapsed,2),
        message="Pretrained ARIMA updated successfully."
    )

class FitResponseUser(BaseModel):
    status: str
    model_id: Optional[str] = None
    processed_tickers_count: Optional[int] = None
    forecast_rows: Optional[int] = None
    time_spent: Optional[float] = None
    message: Optional[str] = None

@app.post("/fit_user_data", response_model=FitResponseUser)
async def fit_user_data(horizon: int=30, max_wait: int=10) -> FitResponseUser:
    """
    Собираем прогноз user_arima, НО не делаем её активной автоматически.
    Если пользователь хочет ею пользоваться, пусть вызывает /set user_arima
    или явно укажет model_id=user_arima в /predict.
    """
    if not os.path.exists(USER_DATA_PATH):
        raise HTTPException(status_code=404, detail="No user_data.csv found.")
    try:
        df_user = pd.read_csv(USER_DATA_PATH)
        if df_user.empty:
            raise ValueError("user_data.csv is empty.")
        if not {"ticker","date","adjusted_close"}.issubset(df_user.columns):
            raise ValueError("Missing columns in user_data.csv")

        df_user = df_user.sort_values(["ticker","date"]).reset_index(drop=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.time()
    all_rows = []

    tickers = df_user["ticker"].unique()
    total = len(tickers)

    # -----------------------------
    # Параллелим обработку тикеров
    # -----------------------------
    def process_one_ticker(tck):
        """
        Обработка одного тикера (обёртка над process_and_forecast_arima).
        Если всё ок — возвращает список предсказаний, иначе пустой список.
        """
        df_tck = df_user[df_user["ticker"] == tck].copy()
        try:
            res_list = process_and_forecast_arima(
                df_tck,
                horizon=horizon,
                test_ratio=0.2,
                max_p=1,
                max_q=1,
                remove_outliers=False,
                min_data_points=10
            )
            out = []
            if res_list and len(res_list) == 1:
                fc = res_list[0]["forecast"]
                for fcrow in fc:
                    out.append({
                        "Ticker": tck,
                        "date": fcrow["date"],
                        "predicted": fcrow["predicted"]
                    })
            return out
        except:
            return []

    processed_count = 0

    # Запускаем через ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_ticker = {
            executor.submit(process_one_ticker, tck): tck
            for tck in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            # Проверяем таймаут
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                return FitResponseUser(
                    status="timeout",
                    model_id="user_arima",
                    processed_tickers_count=processed_count,
                    forecast_rows=len(all_rows),
                    time_spent=round(elapsed,2),
                    message=f"Time limit exceeded after {processed_count} / {total} tickers"
                )

            tck = future_to_ticker[future]
            try:
                result = future.result()
                all_rows.extend(result)
            except Exception as e:
                logger.warning(f"Error processing ticker={tck}: {e}")
            processed_count += 1

    # После выхода из PoolExecutor собираем общий результат
    df_fc = pd.DataFrame(all_rows)
    if df_fc.empty:
        elapsed = time.time() - start_time
        return FitResponseUser(
            status="ok",
            model_id="user_arima",
            processed_tickers_count=0,
            forecast_rows=0,
            time_spent=round(elapsed,2),
            message="No forecast rows for user_data"
        )

    df_fc = df_fc.sort_values(["Ticker","date"]).reset_index(drop=True)
    model_store["user_arima"] = {
        "description": f"User ARIMA(1,1,1), horizon={horizon}",
        "df_forecast": df_fc
    }

    elapsed = time.time() - start_time
    return FitResponseUser(
        status="ok",
        model_id="user_arima",
        processed_tickers_count=total,
        forecast_rows=len(df_fc),
        time_spent=round(elapsed,2),
        message="User ARIMA partial fit done."
    )

@app.get("/predict", response_model=PredictResponse)
async def predict_endpoint(
    model_id: Optional[str] = None,
    ticker: Optional[str] = None,
    horizon: int = 10
) -> PredictResponse:
    """
    Если model_id не передан и нет active_model, ошибка.
    Если model_id не передан, берём active_model.
    По умолчанию active_model = pretrained_arima.
    """
    if not model_id and not active_model_id:
        raise HTTPException(status_code=500, detail="No model_id and no active model.")
    if not model_id:
        model_id = active_model_id

    if model_id not in model_store:
        raise HTTPException(status_code=404, detail=f"No model {model_id} in store")

    df_fc = model_store[model_id]["df_forecast"]
    if df_fc.empty:
        raise HTTPException(status_code=500, detail="Selected model forecast is empty.")

    items = []
    if ticker:
        sub = df_fc[df_fc["Ticker"] == ticker].head(horizon)
        for _, row in sub.iterrows():
            items.append(PredictItem(
                ticker=row["Ticker"],
                date=str(row["date"]),
                predicted=float(row["predicted"])
            ))
    else:
        # Предсказываем для всех тикеров, берём первые `horizon` дат
        for tck, grp in df_fc.groupby("Ticker"):
            grp2 = grp.head(horizon)
            for _, row in grp2.iterrows():
                items.append(PredictItem(
                    ticker=tck,
                    date=str(row["date"]),
                    predicted=float(row["predicted"])
                ))

    return PredictResponse(
        status="ok",
        model_id=model_id,
        ticker=ticker,
        rows=len(items),
        forecast=items
    )

@app.get("/build_portfolio", response_model=PortfolioResponse)
async def build_portfolio_endpoint(
    use_user_data: bool=False,
    momentum_period: float=5,
    top_quantile: float=0.1
) -> PortfolioResponse:
    try:
        mid = "user_arima" if use_user_data else "pretrained_arima"
        if mid not in model_store:
            raise ValueError(f"No model {mid} found in store")

        df_fc = model_store[mid]["df_forecast"]
        if df_fc.empty:
            raise ValueError(f"Model {mid} forecast is empty.")

        df_port = df_fc.rename(columns={"predicted": "close"}).copy()
        df_port["date"] = df_port["date"].astype(str).str.replace(".", "-")
        df_port["date"] = pd.to_datetime(df_port["date"], format="%Y-%m-%d", errors="coerce")
        if df_port["date"].isna().any():
            raise ValueError("Some dates are invalid after parse.")

        base_dict_p = {}
        for tck, grp in df_port.groupby("Ticker"):
            grp = grp.sort_values("date").set_index("date")
            base_dict_p[tck] = grp[["close"]]

        pipeline_res = build_portfolio_pipeline(
            base_dict=base_dict_p,
            momentum_period=momentum_period,
            top_quantile=top_quantile
        )
        cum_df = pipeline_res["cumprod"]
        if cum_df.empty:
            return PortfolioResponse(status="ok", message="Portfolio built, but cumprod is empty.")

        out_dict = cum_df.reset_index().to_dict(orient="list")
        return PortfolioResponse(status="success", cumprod=out_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

