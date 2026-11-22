"""Run fractional differentiation vs standard financial ML research benchmarks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import synthetic_prices,load_prices
from .research import Config,run_experiment

def main(argv=None):
    p=argparse.ArgumentParser(description='Fractional differentiation financial ML research')
    p.add_argument('--csv',help='CSV with ascending date,close columns')
    p.add_argument('--days',type=int,default=1200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threshold',type=float,default=.005)
    p.add_argument('--cost-bps',type=float,default=5.)
    p.add_argument('--output',default='results')
    args=p.parse_args(argv)
    prices=load_prices(args.csv) if args.csv else synthetic_prices(args.days,args.seed)
    cfg=Config(threshold=args.threshold,cost_bps=args.cost_bps)
    report,diag,pred=run_experiment(prices,cfg)
    result=Path(args.output);result.mkdir(parents=True,exist_ok=True)
    (result/'metrics.json').write_text(json.dumps({'data':'external CSV' if args.csv else 'synthetic',
         'configuration':vars(args),'research':report},indent=2))
    diag.to_csv(result/'order_selection.csv',index=False)
    pred.to_csv(result/'predictions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(diag['d'],diag['adf_pvalue'],marker='o',label='ADF p-value')
    axes[0].axhline(cfg.adf_alpha,color='red',linestyle='--',label='Significance threshold')
    axes[0].set(xlabel='Fractional order d',ylabel='Train-only ADF p-value',title='Stationarity across fractional orders')
    axes[0].legend()
    methods=list(report['metrics'])
    errors=[report['metrics'][m]['splits']['test']['mse'] for m in methods]
    axes[1].bar(methods,errors)
    axes[1].set(title='Held-out forecasting MSE',ylabel='MSE')
    axes[1].tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(result/'research.png',dpi=160);plt.close(fig)
    print(json.dumps({'selected_order':report['selected_order'],
                      'test_mse':dict(zip(methods,errors))},indent=2))
    return report

if __name__=='__main__':main()
