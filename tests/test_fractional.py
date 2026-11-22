import json
import numpy as np
import pandas as pd
import pytest
from fracdiff_finance.fractional import weights_fractional, fixed_width_weights, fracdiff
from fracdiff_finance.data import synthetic_prices,load_prices
from fracdiff_finance.research import Config,select_order,run_experiment
from fracdiff_finance.cli import main

def test_integer_and_identity_weights():
    assert np.array_equal(fixed_width_weights(0),[1.])
    assert np.array_equal(fixed_width_weights(1),[1.,-1.])
    assert np.array_equal(weights_fractional(1,5),[1.,-1.,0.,0.,0.])

def test_binomial_recursion():
    weights=weights_fractional(.4,40)
    for k in range(1,len(weights)):
        assert np.isclose(weights[k],-weights[k-1]*(1.4-k)/k)

def test_fractional_diff_matches_integer_first_diff():
    s=pd.Series(np.arange(100,dtype=float)**2)
    z=fracdiff(s,1.)
    assert np.isnan(z.iloc[0])
    assert np.allclose(z.iloc[1:],s.diff().iloc[1:])
    assert np.array_equal(fracdiff(s,0.),s.astype(float))

def test_finite_width_hand_convolution():
    s=pd.Series(np.linspace(1,3,100))
    weights=fixed_width_weights(.5,threshold=.02)
    v=fracdiff(s,.5,threshold=.02)
    assert v.isna().sum()==len(weights)-1
    for i in (len(weights)-1,50,99):
        assert np.isclose(v.iloc[i],np.dot(weights,s.iloc[i-len(weights)+1:i+1].to_numpy()[::-1]))

def test_future_modifications_cannot_change_earlier_features():
    s=pd.Series(np.random.default_rng(4).normal(size=600).cumsum())
    changed=s.copy();changed.iloc[420:]+=500.
    a=fracdiff(s,.5,.005);b=fracdiff(changed,.5,.005)
    assert np.allclose(a.iloc[:420],b.iloc[:420],equal_nan=True)

def test_invalid_inputs():
    for d in (-.1,1.1,float('nan')):
        with pytest.raises(ValueError):fixed_width_weights(d)
    with pytest.raises(ValueError):fixed_width_weights(.5,threshold=0)
    with pytest.raises(ValueError):weights_fractional(.5,0)
    with pytest.raises(ValueError):fracdiff(pd.Series([1.,float('nan')]),.4)
    with pytest.raises(ValueError):fixed_width_weights(.01,1e-15,max_size=5)

def test_order_selection_only_uses_train_prefix():
    p=synthetic_prices(480)
    x=pd.Series(np.log(p.close),index=p.index)
    y=x.copy();y.iloc[330:]+=100
    cfg=Config(threshold=.02)
    a,tab,_=select_order(x,300,cfg)
    b,tab2,_=select_order(y,300,cfg)
    assert a==b
    pd.testing.assert_frame_equal(tab,tab2)

def test_stationary_series_selects_identity_when_adf_accepts():
    series=pd.Series(np.random.default_rng(77).standard_normal(500))
    d,tab,policy=select_order(series,400,Config(threshold=.02))
    assert d==0
    assert tab.iloc[0]['train_stationary_at_alpha']
    assert policy=='smallest_train_stationary_d'

def test_experiment_chronology_and_comparison():
    p=synthetic_prices(410)
    result,table,preds=run_experiment(p,Config(threshold=.02))
    assert result['split_sizes']['test']>=20
    assert set(result['metrics'])=={'Raw log price','Integer diff','Fractional diff','Returns only'}
    assert set(preds.split)=={'train','validation','test'}
    assert (preds.signal_date<preds.target_date).all()
    assert len(preds)==sum(result['split_sizes'].values())*4
    assert table.d.is_monotonic_increasing
    assert all(np.isfinite(v['splits']['test']['mse']) for v in result['metrics'].values())

def test_test_suffix_invariance():
    p=synthetic_prices(410)
    q=p.copy();q.loc[q.index[340:],'close']*=2
    cfg=Config(threshold=.02)
    a,da,pa=run_experiment(p,cfg)
    b,db,pb=run_experiment(q,cfg)
    assert a['selected_order']==b['selected_order']
    pd.testing.assert_frame_equal(da,db)
    initial=pa[pa.signal_date<p.index[340]]
    modified=pb[pb.signal_date<p.index[340]]
    assert np.allclose(initial.predicted_next_log_return,modified.predicted_next_log_return)

def test_same_train_and_validation_predictions_when_test_prices_change():
    p=synthetic_prices(420)
    q=p.copy();q.loc[q.index[350:],'close']*=2.0
    a,_,pa=run_experiment(p,Config(threshold=.02))
    b,_,pb=run_experiment(q,Config(threshold=.02))
    for split in ('train','validation'):
        x=pa[pa.split==split].predicted_next_log_return
        y=pb[pb.split==split].predicted_next_log_return
        assert np.allclose(x,y)

def test_invalid_prices_and_bad_csv(tmp_path):
    p=synthetic_prices(180)
    p.iloc[3,0]=-2
    with pytest.raises(ValueError):run_experiment(p)
    f=tmp_path/'prices.csv';f.write_text('date,close\n2020-01-03,100\n2020-01-03,105\n')
    with pytest.raises(ValueError):load_prices(f)

def test_deterministic_seeded_experiment():
    p=synthetic_prices(330,seed=11)
    a,da,pa=run_experiment(p,Config(threshold=.02))
    b,db,pb=run_experiment(p,Config(threshold=.02))
    assert a==b
    pd.testing.assert_frame_equal(da,db)
    pd.testing.assert_frame_equal(pa,pb)

def test_cli_outputs(tmp_path):
    main(['--days','350','--threshold','.02','--output',str(tmp_path)])
    for name in ('metrics.json','order_selection.csv','predictions.csv','research.png'):
        assert (tmp_path/name).exists()
    assert json.loads((tmp_path/'metrics.json').read_text())['data']=='synthetic'
