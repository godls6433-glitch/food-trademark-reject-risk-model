"""Time holdout, with dated events required in every fit and calibration fold."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
parser=argparse.ArgumentParser()
parser.add_argument('cache',type=Path)
parser.add_argument('--shard',type=int,choices=[0,1],required=True)
args=parser.parse_args()
root=Path.cwd()
out=Path(__file__).resolve().parent/'artifacts'
sys.path.insert(0,str(args.cache.resolve().parent))
from vectorize import encode
sys.path.insert(0,str(root/'food-trademark-reject-risk-model/models/xgb_gaeun/v9/src'))
from trademark_risk.config import LINEAR_C,TREE_WEIGHT,XGB_PARAMS
from trademark_risk.model import STRUCTURE_COLUMNS,CONTEXT_COLUMNS,SEARCH_COLUMNS
data=pd.read_parquet(args.cache/'rows.parquet')
docs=pd.read_parquet(args.cache/'documents.parquet')
eligible=np.load(out/'eligible.npy')
test=data.date.ge('20230101').to_numpy()
assert eligible.sum()==107303 and test.sum()==65590 and not (eligible&test).any()
assert docs.index.equals(data.index)
names=list(STRUCTURE_COLUMNS)+list(CONTEXT_COLUMNS)+list(SEARCH_COLUMNS)
drop={'kb_EFFECT_context','kb_USE_context','kb_PRICE_context','kb_COMMON_EXPRESSION_context','v7_pair_exact_goods','v7_char_max'}
numeric=np.column_stack([np.load(args.cache/'structure.npy'),np.load(args.cache/'search.npy')])[:,[i for i,c in enumerate(names) if c not in drop]]
assert numeric.shape==(len(data),47)
fold=data.fold.to_numpy()
y=data.y.where(eligible,-1).to_numpy()

def train(k):
    fit=np.flatnonzero(eligible & (fold!=k))
    pred=np.flatnonzero(test if k==0 else eligible & (fold==k))
    _,x,z=encode(docs,fit,pred)
    svd=TruncatedSVD(n_components=64,n_iter=4,random_state=61).fit(x)
    def tree_input(text,idx):
        return np.column_stack((svd.transform(text),numeric[idx])).astype(np.float32)
    tree=XGBClassifier(**XGB_PARAMS).fit(tree_input(x,fit),y[fit])
    p=tree.predict_proba(tree_input(z,pred))[:,1]
    lr=LogisticRegression(C=LINEAR_C,solver='liblinear',max_iter=500,random_state=42)
    lr.fit(sparse.hstack((x,numeric[fit]),format='csr'),y[fit])
    assert max(lr.n_iter_) < 500
    q=lr.predict_proba(sparse.hstack((z,numeric[pred]),format='csr'))[:,1]
    np.savez(out/f'blend{k}.npz',idx=pred,raw=TREE_WEIGHT*p+(1-TREE_WEIGHT)*q)
    print(k,len(fit),len(pred),flush=True)
for k in [1,2,3,4,5,0][args.shard::2]:
    train(k)
