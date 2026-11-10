"""Evaluate V9 with the final RF outer folds, seed-42 split and isotonic calibration."""
import argparse
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument('cache', type=Path)
parser.add_argument('search_cache', type=Path)
parser.add_argument('--shard', type=int, required=True, choices=[0,1])
args = parser.parse_args()
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent/'artifacts'
sys.path.insert(0,str(args.cache.resolve().parent))
from vectorize import encode
sys.path.insert(0,str(ROOT/'food-trademark-reject-risk-model/models/xgb_gaeun/v9/src'))
from trademark_risk.config import LINEAR_C, TREE_WEIGHT, XGB_PARAMS
from trademark_risk.model import SEARCH_COLUMNS, STRUCTURE_COLUMNS, CONTEXT_COLUMNS
data = pd.read_parquet(args.cache/'rows.parquet')
docs = pd.read_parquet(args.cache/'documents.parquet')
assert data.index.equals(docs.index)
fold = data.fold.to_numpy()
y = data.y.to_numpy()
assert len(y)==257400 and set(fold)==set(range(1,6)) and y.sum()==60642
names = list(STRUCTURE_COLUMNS)+list(CONTEXT_COLUMNS)+list(SEARCH_COLUMNS)
drop = {'kb_EFFECT_context','kb_USE_context','kb_PRICE_context','kb_COMMON_EXPRESSION_context',
        'v7_pair_exact_goods','v7_char_max'}
dense = np.column_stack([np.load(args.cache/'structure.npy'),np.load(args.search_cache/'search.npy')])
numeric = dense[:,[i for i,c in enumerate(names) if c not in drop]]
assert len(names)==dense.shape[1]==53 and numeric.shape==(len(y),47)
def fit_subset(k):
    test = np.flatnonzero(fold==k)
    rest = np.flatnonzero(fold!=k)
    mask = np.random.default_rng(42).random(len(rest)) < 0.2
    cal,train = rest[mask],rest[~mask]
    assert len(np.unique(np.r_[test,cal,train]))==len(fold)
    idx = np.r_[cal,test]
    _, x, z = encode(docs,train,idx)
    svd = TruncatedSVD(n_components=64,n_iter=4,random_state=61).fit(x)
    def tree_input(text,idx):
        return np.column_stack((svd.transform(text).astype(np.float32),numeric[idx])).astype(np.float32)
    tree = XGBClassifier(**XGB_PARAMS).fit(tree_input(x,train),y[train])
    p = tree.predict_proba(tree_input(z,idx))[:,1]
    lr = LogisticRegression(C=LINEAR_C,solver='liblinear',max_iter=500,random_state=42)
    lr.fit(sparse.hstack((x,numeric[train]),format='csr'),y[train])
    assert max(lr.n_iter_)<500
    q = lr.predict_proba(sparse.hstack((z,numeric[idx]),format='csr'))[:,1]
    raw = TREE_WEIGHT*p + (1-TREE_WEIGHT)*q
    iso = IsotonicRegression(out_of_bounds='clip').fit(raw[:len(cal)],y[cal])
    np.savez(OUT/f'fold{k}.npz',idx=test,prob=iso.predict(raw[len(cal):]),train=train,cal=cal)
    print('fold',k,'train/cal/test',len(train),len(cal),len(test),flush=True)

for k in list(range(1,6))[args.shard::2]:
    fit_subset(k)
