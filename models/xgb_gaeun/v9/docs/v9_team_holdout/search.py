"""Recover identical pre-2023 cached features, compute all other V9 queries as of filing or 2022-12-31."""
from pathlib import Path
import sys
import multiprocessing as mp
import joblib
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent / 'artifacts'
sys.path.insert(0, str(ROOT/'trademark_risk_v9/src'))
from v9_common import V6_COLUMNS, V7_COLUMNS, norm
from retrieval import PriorSearch
from v7_search import Search
COLUMNS = V6_COLUMNS + V7_COLUMNS

def batch(numbers):
    values = []
    for i in numbers:
        q = queries[i].copy()
        # Original search uses strict inequalities on both filing and publication dates.
        q['date'] = min(q['date'], '20230101')
        if not norm(q['name']):
            f = dict.fromkeys(COLUMNS, 0.)
            f['prior_none'] = 1.
        else:
            a, _, aa = index.base.extract(q)
            b, _, ab = index.extract(q)
            assert not any(x[key] for x in [aa, ab] for key in ['future_count','own_family_count'])
            f = {**a, **b}
        values.append([f[c] for c in COLUMNS])
    np.save(OUT/f'chunk_{numbers[0]}.npy', np.asarray(values, dtype=np.float32))
    return len(numbers)

if __name__ == '__main__':
    qcols = ['app','name','classes','goods','date','logo']
    d = pd.read_parquet(OUT/'rows.parquet',columns=qcols)
    orig = pd.read_parquet(ROOT/'trademark_risk_v6/artifacts/cohort.parquet', columns=qcols)
    # All fields affecting retrieval must match; only training-period hits qualify.
    old = pd.read_parquet(ROOT/'trademark_risk_v9/artifacts/features.parquet')
    assert old.app.equals(orig.app) and old.app.is_unique
    lookup = orig.reset_index().rename(columns={'index':'cache_row'})
    joined = d.merge(lookup,on=qcols,how='left',validate='m:1')
    reuse = joined.cache_row.notna() & d.date.lt('20230101')
    result = np.zeros((len(d),len(COLUMNS)),np.float32)
    result[reuse] = old.iloc[joined.loc[reuse,'cache_row'].astype(int)][COLUMNS]
    ids = np.flatnonzero(~reuse)
    chunks = [ids[i:i+400] for i in range(0,len(ids),400)]
    todo = [chunk for chunk in chunks if not (OUT/f'chunk_{chunk[0]}.npy').exists()]
    print('cache hits',reuse.sum(),'fresh rows',len(ids),'chunks pending',len(todo),flush=True)
    queries = d.to_dict('records')
    del old, orig, joined, lookup
    index = joblib.load(ROOT/'trademark_risk_v9/artifacts/search.joblib')
    with mp.get_context('fork').Pool(5) as pool:
        for i, size in enumerate(pool.imap_unordered(batch,todo),1):
            if i%10==0: print('chunks',i,'/',len(todo),flush=True)
    for chunk in chunks:
        result[chunk] = np.load(OUT/f'chunk_{chunk[0]}.npy')
    assert np.isfinite(result).all()
    np.save(OUT/'search.npy',result)
