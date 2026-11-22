"""Causal finite-width fractional differencing for log-prices."""
from __future__ import annotations
import numpy as np
import pandas as pd

def weights_fractional(d: float, size: int) -> np.ndarray:
    """Lag-ordered fractional weights [w_0,...,w_(size-1)]."""
    if not np.isfinite(d) or d < 0 or d > 1:
        raise ValueError("d must be finite and between 0 and 1")
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    w=np.empty(size,dtype=float)
    w[0]=1.0
    for k in range(1,size):
        w[k]=-w[k-1]*(d-k+1)/k
    return w

def fixed_width_weights(d: float, threshold: float=1e-3, max_size: int=10000) -> np.ndarray:
    """Truncate once the next absolute coefficient is below threshold."""
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if not isinstance(max_size, int) or max_size < 2:
        raise ValueError("max_size must be integer >= 2")
    if not np.isfinite(d) or not 0<=d<=1:
        raise ValueError("d must be finite and in [0, 1]")
    if d==0:return np.array([1.0])
    if d==1:return np.array([1.0,-1.0])
    w=[1.0]
    for k in range(1,max_size):
        new=-w[-1]*(d-k+1)/k
        if abs(new)<threshold:
            return np.asarray(w)
        w.append(new)
    raise ValueError("max_size insufficient for threshold; increase max_size or threshold")

def fracdiff(series: pd.Series, d: float, threshold: float=1e-3, max_size: int=10000) -> pd.Series:
    """Finite-width causal fractional difference with leading NaNs for warmup."""
    if not isinstance(series,pd.Series):
        raise TypeError("series must be a pandas Series")
    x=series.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("series must contain finite values")
    weights=fixed_width_weights(d,threshold,max_size)
    width=len(weights)
    result=np.full(len(x),np.nan)
    if len(x)>=width:
        lagged=np.lib.stride_tricks.sliding_window_view(x,width)
        result[width-1:]=lagged@weights[::-1]
    return pd.Series(result,index=series.index,name=f'fracdiff_d{d:g}')
