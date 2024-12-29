# rater.py
import pandas as pd
import numpy as np

class Rater:
    def __init__(self, prices: dict, momentum_period: int = 5):
        self.prices = prices
        self.momentum_period = momentum_period

    def momentum(self, rating_dates: pd.DatetimeIndex) -> dict:
        all_rating = {}
        for ticker, df in self.prices.items():
            df = df.reindex(rating_dates, method='ffill').dropna()
            df['return'] = df['close'].pct_change(self.momentum_period)
            all_rating[ticker] = df['return']
        return all_rating
