"""Cross-fit sigmoid calibration, verify team rows and score the two OOF vectors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

parser = argparse.ArgumentParser()
parser.add_argument('cache',type=Path)
args = parser.parse_args()
root = Path.cwd()
here = Path(__file__).resolve().parent
out = here/'artifacts'
data = pd.read_parquet(args.cache/'rows.parquet')
team = root/'trademark_xgb_team_20261012/artifacts'
baseline = pd.read_csv(team/'XGB_OOF_predictions_257400.csv',dtype={'순번':str,'출원번호':str})
ids = ['순번','app','fold']
pd.testing.assert_frame_equal(data[ids],baseline.rename(columns={'출원번호':'app'})[ids],check_dtype=False)
labels = pd.read_csv(next(root.glob('02_*/label_folds.csv')),usecols=['순번','registered','fold'],dtype={'순번':str})
pd.testing.assert_frame_equal(data[['순번','fold']],labels[['순번','fold']],check_dtype=False)
y = data.y.to_numpy()
assert np.array_equal(y,1-labels.registered) and len(y)==257400 and y.sum()==60642 and data['순번'].is_unique
fold = data.fold.to_numpy()
oof = np.full(len(y),np.nan)
def load_raw(excluded):
    key = ''.join(str(k) for k in sorted(excluded))
    values = np.load(out/f'raw_{key}.npz')
    idx = values['idx']
    assert np.array_equal(idx,np.flatnonzero(np.isin(fold,excluded)))
    return idx,values['raw']

def logits(raw):
    return logit(np.clip(raw,1e-7,1-1e-7)).reshape(-1,1)

for outer in range(1,6):
    cal_raw = np.full(len(y),np.nan)
    for inner in sorted(set(range(1,6))-{outer}):
        idx,p = load_raw([outer,inner])
        keep = fold[idx]==inner
        cal_raw[idx[keep]] = p[keep]
    fit = fold!=outer
    assert np.array_equal(np.isfinite(cal_raw),fit)
    cal = LogisticRegression(C=1e6,max_iter=1000,solver='lbfgs',random_state=42)
    cal.fit(logits(cal_raw[fit]),y[fit])
    assert cal.coef_[0,0]>0
    idx,p = load_raw([outer])
    oof[idx] = cal.predict_proba(logits(p))[:,1]
assert np.isfinite(oof).all()
frame = data[ids+['y']].assign(xgb219=baseline['거절확률'].to_numpy(),v9_47=oof)
frame.to_parquet(out/'paired_oof.parquet',index=False)
def metrics(rows,model):
    target, p = rows.y, rows[model]
    return dict(n=len(p),rejected=int(target.sum()),ROC=roc_auc_score(target,p),PR=average_precision_score(target,p),
                Brier=brier_score_loss(target,p),mean_pred=p.mean())
summary, detail = [], []
for model in ['xgb219','v9_47']:
    splits = [dict(model=model,fold=k,**metrics(g,model)) for k,g in frame.groupby('fold')]
    aucs = [s['ROC'] for s in splits]
    summary.append(dict(model=model,**metrics(frame,model),fold_mean=np.mean(aucs),fold_sd=np.std(aucs,ddof=1)))
    detail.extend(splits)
summary = pd.DataFrame(summary)
reference = pd.read_csv(root/'food-trademark-reject-risk-model/evaluation/results/metrics.csv').set_index('model').loc['가은 XGB']
cols = ['ROC','PR','Brier','fold_mean','fold_sd']
np.testing.assert_allclose(summary.loc[0,cols].to_numpy(float),reference[cols].to_numpy(float),atol=1e-10,rtol=0)
summary.to_csv(here/'metrics.csv',index=False)
pd.DataFrame(detail).to_csv(here/'fold_metrics.csv',index=False)
print(summary.to_string(index=False))
print(pd.DataFrame(detail).to_string(index=False))
# Match the team comparison's paired row bootstrap (fixed OOF predictions).
rng = np.random.default_rng(42)
delta = []
for _ in range(200):
    i = rng.integers(0,len(y),len(y))
    delta.append(roc_auc_score(y[i],oof[i])-roc_auc_score(y[i],frame.xgb219.to_numpy()[i]))
low, high = np.quantile(delta,[0.025,0.975])
print(f'paired ROC-AUC difference 95% interval: [{low:.8f}, {high:.8f}]')
