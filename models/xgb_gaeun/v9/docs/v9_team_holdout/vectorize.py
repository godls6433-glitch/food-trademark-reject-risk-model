"""Refit the V9 text vocabulary, IDF and latent representation inside every training partition."""
from pathlib import Path
import time
import gc
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path(__file__).resolve().parent/'artifacts'
def encode(docs, fit_indices, predict_indices):
    channels = [
        TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=30000),
        TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=20000),
        TfidfVectorizer(tokenizer=str.split, token_pattern=None, max_features=12000),
    ]
    fit_blocks, predict_blocks = [], []
    for c, vectorizer in zip(['name','jamo','goods'],channels):
        vectorizer.set_params(min_df=5, lowercase=False, sublinear_tf=True)
        fit_blocks.append(vectorizer.fit_transform(docs[c].iloc[fit_indices]))
        predict_blocks.append(vectorizer.transform(docs[c].iloc[predict_indices]))
    return channels, sparse.hstack(fit_blocks,format='csr'), sparse.hstack(predict_blocks,format='csr')

def run_fold(k, d, docs):
    train = d.date.lt('20230101').to_numpy()
    fit = np.flatnonzero(train & ((d.fold!=k) if k else True))
    pred = np.flatnonzero(train & (d.fold==k)) if k else np.flatnonzero(~train)
    channels, x, z = encode(docs,fit,pred)
    svd = TruncatedSVD(n_components=64,n_iter=4,random_state=61).fit(x)
    prefix = OUT/f'text{k}'
    np.savez(prefix.with_suffix('.npz'), fit=fit,pred=pred,fit_svd=svd.transform(x).astype(np.float32),
             pred_svd=svd.transform(z).astype(np.float32))
    sparse.save_npz(OUT/f'fit{k}.npz', x, compressed=False)
    sparse.save_npz(OUT/f'pred{k}.npz', z, compressed=False)
    if k==0:
        joblib.dump((channels,svd),OUT/'final_transform.joblib')
    (OUT/f'ready{k}').touch()

if __name__=='__main__':
    d = pd.read_parquet(OUT/'rows.parquet',columns=['fold','date'])
    docs = pd.read_parquet(OUT/'documents.parquet')
    for k in (1,2,3,4,5,0):
        if not (OUT/f'ready{k}').exists():
            run_fold(k,d,docs)
            gc.collect()
        print('ready',k,time.strftime('%T'),flush=True)
