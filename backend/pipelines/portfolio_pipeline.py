# team74_stock_price_forecasting/backend/pipelines/portfolio_pipeline.py

import pandas as pd
import yfinance as yf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Импортируем из пакета port_builder (должен лежать в backend/port_builder/)
# Предполагаем, что у вас есть:
#   backend/port_builder/
#      __init__.py
#      rater.py
#      weights.py
#      portfolio_builder.py
#      rebal_dates.py
# и т.п.

from port_builder.rater import Rater
from port_builder.weights import TopQuantileEqual
from port_builder.portfolio_builder import PortfolioWithFees
from port_builder.rebal_dates import index_dates, rebal_dates


def build_portfolio_pipeline(
    base_dict: dict,
    momentum_period: int = 5,
    top_quantile: float = 0.1,
    slippage: float = 0.000001,
    rebalance_for_days: int = 1,
    multiplicator_rate_value: float = 0.00001,
    use_benchmark: bool = True
):
    """
    1) index_dates -> rater -> rebal_dates
    2) Weights -> PortfolioWithFees
    3) Скачиваем бенчмарк (yf.download("^BVSP")) (опционально)
    4) Возвращаем { "portfolio_df", "benchmark_df", "combined", "cumprod" }
    """
    # 1) Определяем даты и строим рейтинг
    ind_dates = index_dates(base_dict)
    rating_builder = Rater(prices=base_dict, momentum_period=momentum_period)
    price_momentum_rating = rating_builder.momentum(rating_dates=ind_dates)
    rebal_ds = rebal_dates(ind_dates)

    # 2) Формируем веса
    weight_obj = TopQuantileEqual(
        prices=base_dict,
        all_rating=price_momentum_rating,
        rebalance_dates=rebal_ds,
        top_quantile=top_quantile
    )
    leaders_weights = weight_obj.build_weights(top=True)

    # 3) Собираем портфель
    ruonia_sample = pd.Series(index=ind_dates, dtype='float64').fillna(multiplicator_rate_value)
    pf = PortfolioWithFees(
        index_dates=ind_dates,
        prices=base_dict,
        weights=leaders_weights,
        rebalance_for_days=rebalance_for_days,
        slippage=slippage,
        multiplicator_rate=ruonia_sample
    )
    pf.build_portfolio()

    portfolio_df = pf.index_bars
    portfolio_df.rename(columns={"close": "portfolio_value"}, inplace=True)

    # 4) Скачиваем бенчмарк (если нужно)
    bench_df = None
    if use_benchmark and not portfolio_df.empty:
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        bench = yf.download("^BVSP", start=start_date, end=end_date)
        if not bench.empty:
            if isinstance(bench.columns, pd.MultiIndex):
                bench = bench.droplevel(0, axis=1)
            if "Close" in bench.columns:
                bench_df = bench[["Close"]].copy()
                bench_df.rename(columns={"Close": "index_value"}, inplace=True)
                bench_df = bench_df.reindex(portfolio_df.index, method='ffill')
        else:
            bench_df = pd.DataFrame(index=portfolio_df.index, columns=["index_value"])

    # 5) Кумулятивная доходность
    if bench_df is not None:
        combined = pd.concat([portfolio_df["portfolio_value"], bench_df["index_value"]], axis=1)
        ret_df = combined.pct_change().dropna() + 1
        cumprod_df = ret_df.cumprod() - 1
    else:
        combined = portfolio_df[["portfolio_value"]].copy()
        ret_df = combined.pct_change().dropna() + 1
        cumprod_df = ret_df.cumprod() - 1

    return {
        "portfolio_df": portfolio_df,
        "benchmark_df": bench_df,
        "combined": combined,
        "cumprod": cumprod_df
    }
