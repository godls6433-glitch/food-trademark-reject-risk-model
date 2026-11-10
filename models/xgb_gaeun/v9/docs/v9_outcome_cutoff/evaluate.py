"""Evaluate unchanged predictions for the fixed 65,590-row temporal holdout."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
here=Path(__file__).resolve().parent
out=here/'artifacts'
cache,prior=map(Path,sys.argv[1:])
d=pd.read_parquet(cache/'rows.parquet')
eligible=np.load(out/'eligible.npy')
oof=np.full(len(d),np.nan)
for k in range(1,6):
    z=np.load(out/f'blend{k}.npz')
    idx=np.flatnonzero(eligible & d.fold.eq(k).to_numpy())
    assert np.array_equal(idx,z['idx'])
    oof[idx]=z['raw']
assert np.array_equal(np.isfinite(oof),eligible)
logits=lambda p:logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1)
cal=LogisticRegression(C=1e6,max_iter=1000,random_state=42).fit(logits(oof[eligible]),d.y[eligible])
assert cal.coef_[0,0]>0
z=np.load(out/'blend0.npz'); idx=np.flatnonzero(d.date.ge('20230101'))
assert len(idx)==65590 and np.array_equal(idx,z['idx'])
test=d.iloc[idx][['순번','app','fold','date','y']].reset_index(drop=True)
prev=pd.read_parquet(prior/'paired_predictions.parquet').reset_index(drop=True)
pd.testing.assert_frame_equal(test,prev[test.columns])
test['previous']=prev.v9_47
test['dated_event']=cal.predict_proba(logits(z['raw']))[:,1]
test.to_parquet(out/'paired_predictions.parquet',index=False)
scores=[]
for period,g in [('2023–24',test),*test.groupby(test.date.str[:4])]:
    for model in ['previous','dated_event']:
        y,p=g.y,g[model]
        scores.append(dict(period=period,model=model,n=len(y),refused=y.sum(),roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),brier=brier_score_loss(y,p),mean_pred=p.mean()))
scores=pd.DataFrame(scores)
assert abs(scores.roc_auc.iloc[0]-0.7199216873758786)<1e-10
scores.to_csv(here/'metrics.csv',index=False)
print(scores.to_string(index=False))
