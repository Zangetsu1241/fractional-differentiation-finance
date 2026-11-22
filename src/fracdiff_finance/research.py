"""Train-only fractional-order selection, stationarity tests and out-of-sample ML."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from .fractional import fracdiff,fixed_width_weights

@dataclass(frozen=True)
class Config:
    train_fraction: float=.60
    val_fraction: float=.20
    threshold: float=.005
    orders: tuple[float,...]=(0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)
    adf_alpha: float=.05
    ridge_alpha: float=10.
    cost_bps: float=5.

def _adf_pvalue(x: np.ndarray) -> float:
    x=np.asarray(x,dtype=float)
    if len(x)<30 or not np.isfinite(x).all() or np.ptp(x)<1e-12:return float('nan')
    try:return float(adfuller(x,regression='c',autolag='AIC')[1])
    except (ValueError,np.linalg.LinAlgError):return float('nan')

def _corr(a,b):
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return float('nan')
    return float(pearsonr(a,b).statistic)

def select_order(log_prices: pd.Series, train_end: int, cfg: Config):
    """Search candidate fractional orders using training observations only."""
    if not isinstance(log_prices,pd.Series) or not np.isfinite(log_prices.to_numpy(float)).all():
        raise ValueError('log_prices must be finite Series')
    if train_end<80 or train_end>len(log_prices):raise ValueError('invalid train_end')
    if not 0<cfg.adf_alpha<1 or not 0<cfg.threshold<1:
        raise ValueError('invalid threshold/alpha')
    if not cfg.orders or any(not np.isfinite(d) or d<0 or d>1 for d in cfg.orders):
        raise ValueError('orders must be between 0 and 1')
    train=log_prices.iloc[:train_end]
    records=[]
    for d in cfg.orders:
        fd=fracdiff(train,float(d),threshold=cfg.threshold)
        valid=fd.notna().to_numpy()
        a=fd.to_numpy()[valid]
        b=train.to_numpy()[valid]
        p=_adf_pvalue(a)
        records.append({'d':float(d),'width':len(fixed_width_weights(float(d),cfg.threshold)),
                        'train_observations':len(a),'adf_pvalue':p,
                        'correlation_to_log_price':_corr(a,b),
                        'train_stationary_at_alpha':bool(np.isfinite(p) and p<cfg.adf_alpha)})
    table=pd.DataFrame(records).sort_values('d').reset_index(drop=True)
    candidates=table[table['train_stationary_at_alpha']]
    if candidates.empty:
        choice=float(max(cfg.orders))
        policy='fallback_max_d_no_stationary_candidate'
    else:
        choice=float(candidates.iloc[0]['d'])
        policy='smallest_train_stationary_d'
    return choice,table,policy

def _features(log_prices,d,threshold):
    r=log_prices.diff()
    fd=fracdiff(log_prices,d,threshold)
    return pd.DataFrame({'fractional_log_price':fd,
                         'return_1d':r,
                         'return_5d':r.rolling(5).sum(),
                         'volatility_10d':r.rolling(10).std()},index=log_prices.index)

def _score(y,pred,cost_bps):
    position=np.sign(pred)
    turnover=np.abs(position-np.r_[0.,position[:-1]])
    net=position*y-turnover*cost_bps/10000
    return {'mse':float(mean_squared_error(y,pred)),
            'mae':float(mean_absolute_error(y,pred)),
            'direction_accuracy':float(np.mean(np.sign(y)==np.sign(pred))),
            'total_net_return_illustrative':float(np.exp(np.sum(net))-1),
            'turnover_units':float(turnover.sum())}

def run_experiment(prices: pd.DataFrame, cfg: Config=Config()):
    """Train-only order selection and Ridge fitting; shared chronological dates."""
    if not .4<=cfg.train_fraction<=.8 or not .05<=cfg.val_fraction<=.3 or cfg.train_fraction+cfg.val_fraction>=.9:
        raise ValueError('invalid chronological split fractions')
    if cfg.ridge_alpha<0 or cfg.cost_bps<0:raise ValueError('invalid ridge_alpha or costs')
    if not isinstance(prices,pd.DataFrame) or 'close' not in prices or len(prices)<180:
        raise ValueError('need a DataFrame with >=180 daily adjusted prices')
    x=prices['close'].to_numpy(float)
    if not np.isfinite(x).all() or np.any(x<=0) or not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError('prices must be positive finite, dated, ordered and unique')
    series=pd.Series(np.log(x),index=prices.index,name='log_price')
    n=len(series)
    train_end=int(n*cfg.train_fraction)
    val_end=int(n*(cfg.train_fraction+cfg.val_fraction))
    selected,diagnostics,policy=select_order(series,train_end,cfg)
    frames={'Raw log price':_features(series,0.,cfg.threshold),
            'Integer diff':_features(series,1.,cfg.threshold),
            'Fractional diff':_features(series,selected,cfg.threshold),
            'Returns only':_features(series,1.,cfg.threshold).drop(columns=['fractional_log_price'])}
    target=series.diff().shift(-1)
    mask=target.notna()
    for frame in frames.values():mask &= frame.notna().all(axis=1)
    positions=np.flatnonzero(mask.to_numpy())
    splits={'train':positions[positions<train_end],
            'validation':positions[(positions>=train_end)&(positions<val_end)],
            'test':positions[positions>=val_end]}
    if min(len(v) for v in splits.values())<20:
        raise ValueError('not enough usable observations in every split; raise threshold or supply more data')
    predictions=[]
    reports={}
    for name,frame in frames.items():
        scaler=StandardScaler()
        fit=scaler.fit_transform(frame.iloc[splits['train']].to_numpy(float))
        model=Ridge(alpha=cfg.ridge_alpha).fit(fit,target.iloc[splits['train']].to_numpy(float))
        reports[name]={'feature_count':int(frame.shape[1]),'splits':{}}
        for split,idx in splits.items():
            testx=scaler.transform(frame.iloc[idx].to_numpy(float))
            truth=target.iloc[idx].to_numpy(float)
            forecast=model.predict(testx)
            reports[name]['splits'][split]=_score(truth,forecast,cfg.cost_bps)
            for time,t,pred,actual in zip(frame.index[idx],idx,forecast,truth):
                predictions.append({'signal_date':time,'target_date':frame.index[t+1],
                                    'split':split,'method':name,
                                    'predicted_next_log_return':float(pred),'actual_next_log_return':float(actual)})
    details={'selected_order':selected,'selection_policy':policy,
             'split_sizes':{k:len(v) for k,v in splits.items()},
             'train_end_date':str(prices.index[train_end-1]),
             'validation_end_date':str(prices.index[val_end-1]),
             'metrics':reports}
    return details, diagnostics, pd.DataFrame(predictions)
