# main.py  (пример, лежит в папке port_builder)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import yfinance as yf

# Если все модули (data_loader.py, rebal_dates.py, rater.py, weights.py, portfolio_builder.py)
# находятся в той же папке port_builder, то делаем относительные импорты:
from .data_loader import get_forecasted_data
from .rebal_dates import index_dates, rebal_dates as get_rebal_dates
from .rater import Rater  # (или rater_local, если файл так назван)
from .weights import TopQuantileEqual
from .portfolio_builder import PortfolioWithFees


def run_portfolio_example():
    # для эндпоинта
    base_dict = get_forecasted_data()

    ind_dates = index_dates(base_dict)

    # для эндпоинта
    rating_builder = Rater(prices=base_dict, momentum_period=5)
    price_momentum_rating = rating_builder.momentum(rating_dates=ind_dates)

    # переименуем переменную, чтобы не конфликтовать с названием функции
    reb_dates = get_rebal_dates(ind_dates)

    weight = TopQuantileEqual(
        base_dict, price_momentum_rating, reb_dates,
        top_quantile=0.1,
        filter_value=None,
        filter_currency=None,
        filter_column=None
    )

    # для эндпоинта
    leaders_weights, losers_weights = weight.build_weights(top=True), weight.build_weights(top=False)

    ruonia_sample = pd.Series(index=ind_dates, dtype='float64').fillna(0.00001)
    leaders_pf = PortfolioWithFees(
        ind_dates, base_dict, leaders_weights,
        rebalance_for_days=1,
        slippage=0.000001,
        multiplicator_rate=ruonia_sample
    )
    leaders_pf.build_portfolio()

    # для эндпоинта
    leaders = pd.Series(leaders_pf.index_bars["close"], name='Close').to_frame()

    # для эндпоинта
    benchmark = yf.download("^BVSP", start=leaders.index[0], end=leaders.index[-1])
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark = benchmark.droplevel(0, axis=1)

    overal_data = pd.concat([leaders, benchmark['Close']], axis=1)
    overal_data.columns = ['portfolio', 'index']

    # для эндпоинта
    cumprod = (overal_data.pct_change().dropna() + 1).cumprod() - 1
    print(cumprod.tail())

if __name__ == "__main__":
    run_portfolio_example()
