"""Compare fixed predictions using exactly the labels/IDs used for the team comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
OUT = HERE/'artifacts'
TEAM = ROOT/'trademark_xgb_team_20261012/time_holdout_2022_2023_2024'
test = pd.read_parquet(OUT/'rows.parquet').query('date >= "20230101"').reset_index(drop=True)
old = pd.read_csv(next(TEAM.glob('*65590.csv')),dtype={'순번':str,'출원번호':str}).rename(columns={'출원번호':'app'})
v9 = pd.read_parquet(OUT/'v9_predictions.parquet')
ids = ['순번','app','fold']
for frame in [v9, old]:
    pd.testing.assert_frame_equal(frame[ids],test[ids],check_dtype=False)
labels = pd.read_csv(next(ROOT.glob('02_*/label_folds.csv')),dtype={'순번':str}).set_index('순번',verify_integrity=True)
assert np.array_equal(1-labels.loc[test['순번'],'registered'],test.y)
pred = test[ids+['date','y']].assign(xgb219=old['거절확률'],v9=v9.v9)
assert len(pred)==65590 and pred['순번'].is_unique and pred.y.sum()==13262
assert pred[['xgb219','v9']].apply(lambda c:c.between(0,1).all()).all()
pred.to_parquet(OUT/'paired_predictions.parquet',index=False)
rows = []
for period, subset in [('2023–24',pred), *pred.groupby(pred.date.str[:4])]:
    for model in ['xgb219','v9']:
        y,p = subset.y,subset[model]
        rows.append(dict(period=period,model=model,n=len(y),rejected=y.sum(),roc_auc=roc_auc_score(y,p),
                         pr_auc=average_precision_score(y,p),brier=brier_score_loss(y,p),mean_probability=p.mean()))
result = pd.DataFrame(rows)
result.to_csv(HERE/'metrics.csv',index=False)
print(result.to_string(index=False))
# Cross-check all three periods against the 219-feature submission's archived metrics.
import json
archived = json.loads((TEAM/'metrics.json').read_text())['calibrated_temporal_holdout']
for r, expected in zip(result[result.model=='xgb219'].itertuples(), archived, strict=True):
    assert r.n == expected['rows']
    np.testing.assert_allclose([r.roc_auc,r.pr_auc,r.brier],
                               [expected[c] for c in ['roc_auc','pr_auc_ap','brier']],atol=1e-9,rtol=0)
