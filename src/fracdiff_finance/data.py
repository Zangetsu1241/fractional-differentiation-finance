"""Synthetic financial series and strict adjusted-price ingestion."""
from __future__ import annotations
import numpy as np
import pandas as pd

def synthetic_prices(days: int=1400, seed: int=42) -> pd.DataFrame:
    if days<180:raise ValueError("days must be >= 180")
    rng=np.random.default_rng(seed)
    returns=np.zeros(days)
    state=0
    for t in range(1,days):
        if rng.random()<.012:state=1-state
        sigma=.008 if state==0 else .020
        returns[t]=.12*returns[t-1]+(.0003 if state==0 else -.0001)+sigma*rng.standard_normal()
    return pd.DataFrame({'close':100*np.exp(np.cumsum(returns))}, index=pd.bdate_range('2018-01-01',periods=days))

def load_prices(path: str) -> pd.DataFrame:
    df=pd.read_csv(path)
    if not {'date','close'}.issubset(df.columns):raise ValueError('CSV needs date,close columns')
    dates=pd.to_datetime(df['date'],utc=True,errors='raise')
    close=pd.to_numeric(df['close'],errors='raise').to_numpy(float)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('dates must be valid, unique and increasing')
    if len(close)<180 or not np.isfinite(close).all() or (close<=0).any():
        raise ValueError('need >=180 positive finite adjusted closing prices')
    return pd.DataFrame({'close':close},index=dates)
