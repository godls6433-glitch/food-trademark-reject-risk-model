"""Validate team row order and pool exactly one OOF prediction per filing row."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

parser = argparse.ArgumentParser()
parser.add_argument('cache',type=Path)
args=parser.parse_args()
root=Path.cwd()
here=Path(__file__).resolve().parent
out=here/'artifacts'
data=pd.read_parquet(args.cache/'rows.parquet')
base=root/'trademark_xgb_team_20261012/artifacts'
old=pd.read_csv(base/'XGB_OOF_predictions_257400.csv',dtype={'순번':str,'출원번호':str}).rename(columns={'출원번호':'app'})
ids=['순번','app','fold']
pd.testing.assert_frame_equal(data[ids],old[ids],check_dtype=False)
lab=pd.read_csv(next(root.glob('02_*/label_folds.csv')),usecols=['순번','fold','registered'],dtype={'순번':str})
pd.testing.assert_frame_equal(data[['순번','fold']],lab[['순번','fold']],check_dtype=False)
y=data.y.to_numpy()
assert np.array_equal(y,1-lab.registered)
p=np.full(len(y),np.nan)
folds=[]
for k in range(1,6):
    arr=np.load(out/f'fold{k}.npz')
    idx=arr['idx']
    assert np.array_equal(idx,np.flatnonzero(data.fold==k))
    rest=np.flatnonzero(data.fold!=k)
    mask=np.random.default_rng(42).random(len(rest))<0.2
    assert np.array_equal(arr['train'],rest[~mask]) and np.array_equal(arr['cal'],rest[mask])
    p[idx]=arr['prob']
    folds.append(dict(fold=k,train=len(rest[~mask]),calibration=len(rest[mask]),test=len(idx),
                      ROC=roc_auc_score(y[idx],p[idx]),PR=average_precision_score(y[idx],p[idx]),Brier=brier_score_loss(y[idx],p[idx])))
assert len(p)==257400 and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
data[ids].assign(y=y,v9_47=p).to_parquet(out/'paired_ids_and_predictions.parquet',index=False)
folds=pd.DataFrame(folds)
folds.to_csv(here/'fold_metrics.csv',index=False)
team=pd.read_csv(root/'food-trademark-reject-risk-model/evaluation/results/three_results.csv')
team=team[team['test'].eq('5-fold')&team.model.str.startswith('2 RF')].copy()
assert len(team)==1
v9=dict(model='V9 (47 shared)',test='5-fold',ROC=roc_auc_score(y,p),PR=average_precision_score(y,p),Brier=brier_score_loss(y,p),fold_sd=folds.ROC.std(ddof=1))
summary=pd.concat([team,pd.DataFrame([v9])],ignore_index=True)
summary.to_csv(here/'metrics.csv',index=False)
print(summary.to_string(index=False))
print(folds.to_string(index=False))
