# data_loader.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from .config import PATH
import os

def get_forecasted_data() -> dict:
    csv_path = os.path.join(PATH, "all_forecast_results.csv")
    data = pd.read_csv(csv_path)
    base_dict = {}
    for ticker in tqdm(data['Ticker'].unique(), desc="Loading Tickers"):
        ticker_data = data.loc[data['Ticker'] == ticker, ['date', 'predicted']]
        ticker_data.columns = ['date', 'close']
        ticker_data.set_index('date', inplace=True)
        ticker_data.index = pd.to_datetime(ticker_data.index, format='%Y-%m-%d')
        base_dict[ticker] = ticker_data
    logger.success("FORECASTED DATA HAS BEEN UPLOADED")
    return base_dict

def get_test_data() -> dict:
    csv_path = os.path.join(PATH, "test_data.csv")
    data = pd.read_csv(csv_path)
    base_dict = {}
    for ticker in tqdm(data['ticker'].unique(), desc="Loading Tickers"):
        ticker_data = data.loc[data['ticker'] == ticker, ['price_date', 'adjusted_close']]
        ticker_data.columns = ['date', 'close']
        ticker_data.set_index('date', inplace=True)
        ticker_data.index = pd.to_datetime(ticker_data.index, format='%Y-%m-%d')
        base_dict[ticker] = ticker_data
    logger.success("TESTED DATA HAS BEEN UPLOADED")
    return base_dict

if __name__ == "__main__":
    forecasted_data = get_forecasted_data()
    tested_data = get_test_data()
    print(forecasted_data)
    print(tested_data)

