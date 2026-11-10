"""Fit fixed V9 branches on team folds and refit on all pre-2023 filings."""
from pathlib import Path
import gc
import sys
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT = Path.cwd()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('cache', type=Path, help='Baseline v9_team_holdout/artifacts')
parser.add_argument('count', type=int, choices=[48,47])
args = parser.parse_args()
CACHE = args.cache.resolve()
OUT = Path(__file__).resolve().parent/'artifacts'/str(args.count)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(ROOT/'food-trademark-reject-risk-model/models/xgb_gaeun/v9/src'))
from trademark_risk.config import XGB_PARAMS, LINEAR_C, TREE_WEIGHT

def fit_split(k, numeric, y):
    split = np.load(CACHE/f'text{k}.npz')
    fit, pred = split['fit'], split['pred']
    tree = XGBClassifier(**XGB_PARAMS)
    train_tree = np.column_stack((split['fit_svd'],numeric[fit])).astype('float32')
    assert train_tree.shape[1] == 64 + args.count
    tree.fit(train_tree, y[fit])
    tree_score = tree.predict_proba(np.column_stack((split['pred_svd'],numeric[pred])).astype('float32'))[:,1]
    train_linear = sparse.hstack((sparse.load_npz(CACHE/f'fit{k}.npz'),numeric[fit]),format='csr')
    pred_linear = sparse.hstack((sparse.load_npz(CACHE/f'pred{k}.npz'),numeric[pred]),format='csr')
    lr = LogisticRegression(C=LINEAR_C,solver='liblinear',max_iter=500,random_state=42)
    lr.fit(train_linear, y[fit])
    assert max(lr.n_iter_) < 500
    raw = TREE_WEIGHT*tree_score + (1-TREE_WEIGHT)*lr.predict_proba(pred_linear)[:,1]
    np.savez(OUT/f'blend{k}.npz', idx=pred,raw=raw)
    if k==0:
        joblib.dump((tree,lr),OUT/'final_branches.joblib')

if __name__=='__main__':
    data = pd.read_parquet(CACHE/'rows.parquet')
    train = data.date.lt('20230101').to_numpy()
    # Prevent accidental use of holdout labels by either branch or calibrator.
    y = data.y.where(train,-1).to_numpy()
    numeric = np.column_stack((np.load(CACHE/'structure.npy'), np.load(CACHE/'search.npy')))
    sys.path.insert(0, str(ROOT/'trademark_risk_v9/src'))
    from v9_common import V6_COLUMNS, V7_COLUMNS
    sys.path.insert(0, str(ROOT/'trademark_risk_v3/src'))
    from shared import ROLES
    basic = 'length hangul_ratio latin_ratio digit_ratio logo class_count goods_count class_29 class_30 class_32 class_43'.split()
    columns = basic + ['kb_'+role+'_context' for role in ROLES[:10]] + V6_COLUMNS + V7_COLUMNS
    assert len(set(columns)) == numeric.shape[1] == 53
    dropped = {'kb_EFFECT_context','kb_USE_context','kb_PRICE_context','kb_COMMON_EXPRESSION_context','v7_pair_exact_goods'}
    if args.count == 47:
        dropped.add('v7_char_max')
    keep = [i for i,c in enumerate(columns) if c not in dropped]
    assert len(keep) == args.count
    for column in sorted(dropped):
        values = numeric[train,columns.index(column)]
        print(column, 'nonzero_train', np.count_nonzero(values), flush=True)
    exact_a, exact_b = [columns.index(c) for c in ('v7_pair_exact_goods','prior_exact_name_goods_coverage')]
    print('different_exact_pair_train', np.count_nonzero(numeric[train,exact_a]!=numeric[train,exact_b]),flush=True)
    numeric = numeric[:,keep]
    for k in (1,2,3,4,5,0):
        fit_split(k,numeric,y)
        gc.collect()
        print('fitted',k,flush=True)
    oof = np.full(len(y),np.nan)
    for k in range(1,6):
        r = np.load(OUT/f'blend{k}.npz')
        assert np.array_equal(r['idx'],np.flatnonzero(train & (data.fold==k)))
        oof[r['idx']] = r['raw']
    assert np.array_equal(np.isfinite(oof),train)
    logits = lambda p:logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1)
    calibrator = LogisticRegression(C=1e6,max_iter=1000,solver='lbfgs',random_state=42)
    calibrator.fit(logits(oof[train]),y[train])
    assert calibrator.coef_[0,0]>0
    joblib.dump(calibrator,OUT/'calibrator.joblib')
    full = np.load(OUT/'blend0.npz')
    assert np.array_equal(full['idx'],np.flatnonzero(~train))
    submission = data.loc[~train,['순번','app','fold']].copy()
    submission['v9'] = calibrator.predict_proba(logits(full['raw']))[:,1]
    submission.to_parquet(OUT/'v9_predictions.parquet',index=False)
