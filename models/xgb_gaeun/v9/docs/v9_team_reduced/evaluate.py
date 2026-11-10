"""Compare three fixed prediction vectors on identical cases and labels."""
import argparse
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('cache', type=Path, help='Baseline v9_team_holdout/artifacts')
args = parser.parse_args()
here = Path(__file__).resolve().parent
out = here/'artifacts'
data = pd.read_parquet(args.cache/'rows.parquet')
test = data.loc[data.date.ge('20230101')].reset_index(drop=True)
base = pd.read_parquet(args.cache/'paired_predictions.parquet').reset_index(drop=True)
ids = ['순번','app','fold']
pd.testing.assert_frame_equal(base[ids+['date','y']],test[ids+['date','y']])
assert len(test) == 65590 and test.y.sum() == 13262 and test['순번'].is_unique
base = base.drop(columns='xgb219').rename(columns={'v9':'v9_53'})
models = ['v9_53','v9_48','v9_47']
for size, name in zip([48,47],models[1:]):
    pred = pd.read_parquet(out/str(size)/'v9_predictions.parquet').reset_index(drop=True)
    pd.testing.assert_frame_equal(pred[ids], base[ids])
    assert pred.v9.between(0,1).all()
    base[name] = pred.v9.to_numpy()
base.to_parquet(out/'paired_predictions.parquet', index=False)
rows = []
for period, frame in [('2023–24',base),*base.groupby(base.date.str[:4])]:
    for model in models:
        y, score = frame.y, frame[model]
        rows.append(dict(period=period,model=model,n=len(frame),roc_auc=roc_auc_score(y,score),
                         pr_auc=average_precision_score(y,score),brier=brier_score_loss(y,score)))
metrics = pd.DataFrame(rows)
for metric in ['roc_auc','pr_auc','brier']:
    ref = metrics.groupby('period')[metric].transform('first')
    metrics['delta_'+metric] = metrics[metric]-ref
metrics.to_csv(here/'metrics.csv',index=False)
print(metrics.to_string(index=False))

# For fixed predictions, resample normalized-name groups together in all models.
# Pre-sort once and accumulate equal-score groups to award tied pairs half credit.
def ranking(score):
    order = np.argsort(score,kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(score[order]))+1]
    return order, starts

def auc(weights, labels, rank):
    order, starts = rank
    positive = np.add.reduceat((weights*labels)[order], starts)
    negative = np.add.reduceat((weights*(1-labels))[order], starts)
    return np.sum(positive*(np.cumsum(negative)-0.5*negative))/(positive.sum()*negative.sum())

def group_name(s):
    return ''.join(c for c in unicodedata.normalize('NFC',s).lower() if c.isalnum())

group = pd.factorize(test.name.fillna('').map(group_name))[0]
count = group.max()+1
y = test.y.to_numpy()
ranks = [ranking(base[name].to_numpy()) for name in models]
np.testing.assert_allclose([auc(np.ones(len(y)),y,r) for r in ranks], metrics.roc_auc.iloc[:3], atol=1e-12)
rng = np.random.default_rng(42)
scores = []
for _ in range(1000):
    weights = np.bincount(rng.integers(count,size=count),minlength=count)[group]
    scores.append([auc(weights,y,r) for r in ranks])
scores = np.array(scores)
intervals = []
for left,right in [(1,0),(2,0),(2,1)]:
    low,high = np.quantile(scores[:,left]-scores[:,right],[0.025,0.975])
    intervals.append(dict(model=models[left],reference=models[right],roc_auc_diff=metrics.roc_auc.iloc[left]-metrics.roc_auc.iloc[right],
                          lower_95=low,upper_95=high,bootstrap_replicates=len(scores),name_groups=count))
pd.DataFrame(intervals).to_csv(here/'auc_intervals.csv',index=False)
print(pd.DataFrame(intervals).to_string(index=False))
