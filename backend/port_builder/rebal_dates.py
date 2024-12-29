# rebal_dates.py
import pandas as pd
import os
from .config import PATH

def index_dates(base_dict: dict) -> pd.DatetimeIndex:
    max_len = 0
    chosen_ticker = None
    for tck, df in base_dict.items():
        if len(df) > max_len:
            max_len = len(df)
            chosen_ticker = tck
    return base_dict[chosen_ticker].index

def rebal_dates(data: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return data[data.dayofweek == 4]
