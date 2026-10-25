import pandas as pd, numpy as np, glob, re, unicodedata
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
S='work'
f=pd.read_csv(S+'/dp2/features_final_v8.csv',encoding='utf-8-sig')
for t in ['한글상표','영문상표','복합문자','도형복합']: f['nt_'+t]=(f.name_type==t).astype(int)
base=[c for c in f.columns[6:] if c!='name_type']
pool=pd.concat([pd.read_csv(x,encoding='utf-8-sig',usecols=['순번','출원(국제등록)일자']) for x in sorted(glob.glob('data/raw/tm_pool_*.csv'))])
f=f.merge(pool,on='순번',how='left'); f['dt']=pd.to_datetime(f['출원(국제등록)일자'].astype(str).str[:10])
y=1-f.registered.values; f['y']=y; P0=y.mean(); A=20
def grams(s):
    s=unicodedata.normalize('NFKC',str(s)).lower(); s=re.sub(r'\(.*?\)','',s) or s
    s=re.sub(r'[^0-9a-z가-힣]','',s)
    return list({s[i:i+2] for i in range(len(s)-1)}) or ([s] if s else [])
g=f[['dt','y','fold']].copy(); g['gr']=f.상표명칭.map(grams); g['row']=np.arange(len(f))
e=g.explode('gr').dropna(subset=['gr'])
def agg(rates,rows):
    t=pd.DataFrame({'row':rows,'r':rates}).groupby('row').r.agg(['max','mean','min','count'])
    out=np.full((len(f),4),np.nan); out[t.index.values]=t.values; return out
# fold-based (예환 style)
def foldTE(test_mask_fold_rows, train_rows_mask):
    st=e[train_rows_mask].groupby('gr').y.agg(['sum','count'])
    return st
feat_fold={}
for k in range(1,6):
    out=np.full((len(f),4),np.nan)
    trf=[j for j in range(1,6) if j!=k]
    st=e[e.fold.isin(trf)].groupby('gr').y.agg(['sum','count'])
    te=e[e.fold==k].join(st,on='gr'); r=((te['sum'].fillna(0)+A*P0)/(te['count'].fillna(0)+A)).values
    o=agg(r,te.row.values); out[f.fold.values==k]=o[f.fold.values==k]
    for j in trf:
        st2=e[e.fold.isin([x for x in trf if x!=j])].groupby('gr').y.agg(['sum','count'])
        tj=e[e.fold==j].join(st2,on='gr'); r=((tj['sum'].fillna(0)+A*P0)/(tj['count'].fillna(0)+A)).values
        o=agg(r,tj.row.values); out[f.fold.values==j]=o[f.fold.values==j]
    feat_fold[k]=out
# past-only: grams of applications filed strictly earlier
c=e.groupby(['gr','dt']).y.agg(['sum','count']).sort_index()
cs=c.groupby(level=0).cumsum()-c
ep=e.join(cs,on=['gr','dt']); r=((ep['sum']+A*P0)/(ep['count']+A)).values
past=agg(r,ep.row.values)
cols4=['te_max','te_mean','te_min','te_n']
def run(name,extra=None,per_fold=None):
    p=np.zeros(len(f))
    for k in range(1,6):
        X=f[base].copy()
        if extra is not None: X[cols4]=extra
        if per_fold is not None: X[cols4]=per_fold[k]
        tr=f.fold.values!=k
        m=H(max_iter=400,learning_rate=0.05,random_state=42).fit(X[tr],y[tr]); p[~tr]=m.predict_proba(X[~tr])[:,1]
    print(name,'ROC %.4f PR %.4f'%(AUC(y,p),AP(y,p)))
run('해인 28')
run('+단어 거절률(fold 기준, 미래 포함)',per_fold=feat_fold)
run('+단어 거절률(먼저 출원된 것만)',extra=past)
