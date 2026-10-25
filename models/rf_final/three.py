import pandas as pd, numpy as np, glob, re, unicodedata, json
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR
from xgboost import XGBClassifier
S='work'
f=pd.read_csv(S+'/dp2/features_final_v8.csv',encoding='utf-8-sig')
for t in ['한글상표','영문상표','복합문자','도형복합']: f['nt_'+t]=(f.name_type==t).astype(int)
base=[c for c in f.columns[6:] if c!='name_type']
pool=pd.concat([pd.read_csv(x,encoding='utf-8-sig',usecols=['순번','출원(국제등록)일자']) for x in sorted(glob.glob('data/raw/tm_pool_*.csv'))])
f=f.merge(pool,on='순번'); f['dt']=pd.to_datetime(f['출원(국제등록)일자'].astype(str).str[:10])
y=1-f.registered.values; f['y']=y; A=20
def grams(s):
    s=unicodedata.normalize('NFKC',str(s)).lower(); s=re.sub(r'\(.*?\)','',s) or s
    s=re.sub(r'[^0-9a-z가-힣]','',s)
    return list({s[i:i+2] for i in range(len(s)-1)}) or ([s] if s else [])
g=f[['dt','y']].copy(); g['gr']=f.상표명칭.map(grams); g['row']=np.arange(len(f))
e=g.explode('gr').dropna(subset=['gr'])
W=['w_max','w_mean','w_min','w_n']
def agg(r,rows):
    t=pd.DataFrame({'row':rows,'r':r}).groupby('row').r.agg(['max','mean','min','count'])
    out=np.full((len(f),4),np.nan); out[t.index.values]=t.values; return out
def past_rates(src_mask, P0):
    # 각 행: src_mask 안에서 자기보다 먼저(다른 날) 출원된 상표만으로 계산
    es=e[src_mask[e.row.values]]
    c=es.groupby(['gr','dt']).y.agg(['sum','count']).sort_index(); cs=c.groupby(level=0).cumsum()
    # 각 (gr, dt)에 대해 dt보다 이전 날짜까지 누적값: merge_asof
    cs=cs.reset_index().sort_values('dt'); q=e[['gr','dt','row']].sort_values('dt')
    m=pd.merge_asof(q,cs,on='dt',by='gr',allow_exact_matches=False)
    r=((m['sum'].fillna(0)+A*P0)/(m['count'].fillna(0)+A)).values
    return agg(r,m.row.values)
def fit_pred(kind,Xtr,ytr,Xte):
    rng=np.random.default_rng(42); cal=rng.random(len(ytr))<0.2
    if kind=='RF':
        m=RF(n_estimators=300,min_samples_leaf=20,max_samples=0.7,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1)
    else:
        m=XGBClassifier(n_estimators=600,learning_rate=0.05,max_depth=6,min_child_weight=5,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,tree_method='hist',random_state=42,n_jobs=4)
    m.fit(Xtr[~cal],ytr[~cal])
    iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(Xtr[cal])[:,1],ytr[cal])
    return iso.predict(m.predict_proba(Xte)[:,1])
def met(yy,p): return dict(ROC=round(AUC(yy,p),4),PR=round(AP(yy,p),4),Brier=round(BR(yy,p),4))
models=[('1 해인 RF (28개)','RF',False),('2 RF + 단어별 거절률 (32개)','RF',True),('3 XGB + 단어별 거절률 (32개)','XGB',True)]
res=[]; oof={}
# 5-fold: 단어 거절률은 전체 라벨 중 먼저 출원된 상표로 계산 (자기 자신·같은 날 제외)
Wall=past_rates(np.ones(len(f),bool), y.mean())
for name,kind,uw in models:
    X=f[base].copy()
    if uw: X[W]=Wall
    X=X.fillna(-1).values; p=np.zeros(len(f)); fa=[]
    for k in range(1,6):
        tr=f.fold.values!=k; p[~tr]=fit_pred(kind,X[tr],y[tr],X[~tr]); fa.append(AUC(y[~tr],p[~tr]))
    oof[name]=p; r=dict(model=name,test='5-fold',**met(y,p),fold_sd=round(float(np.std(fa,ddof=1)),4)); print(r,flush=True); res.append(r)
# 시간 분할: 학습 행은 학습기간 안에서 먼저 출원된 것만, 시험 행은 2022년까지 데이터만
tr=(f.dt.dt.year<=2022).values; te=~tr
Wtr=past_rates(tr, y[tr].mean())
st=e[tr[e.row.values]].groupby('gr').y.agg(['sum','count']); P0=y[tr].mean()
j=e[te[e.row.values]].join(st,on='gr'); Wte=agg(((j['sum'].fillna(0)+A*P0)/(j['count'].fillna(0)+A)).values,j.row.values)
Wts=np.where(tr[:,None],Wtr,Wte); ts={}
for name,kind,uw in models:
    X=f[base].copy()
    if uw: X[W]=Wts
    X=X.fillna(-1).values; p=fit_pred(kind,X[tr],y[tr],X[te]); ts[name]=p
    r=dict(model=name,test='2023~24',**met(y[te],p)); print(r,flush=True); res.append(r)
pd.DataFrame(res).to_csv(S+'/three_results.csv',index=False)
o=f[['순번','출원(국제등록)번호','fold']].copy()
for k,v in oof.items(): o[k]=v
o.to_csv(S+'/three_oof.csv',index=False)
t=f.loc[te,['순번','출원(국제등록)번호']].copy()
for k,v in ts.items(): t[k]=v
t.to_csv(S+'/three_timesplit.csv',index=False)
# paired bootstrap vs model 2
rng=np.random.default_rng(42)
for lab,P,yy in [('5-fold',oof,y),('2023~24',ts,y[te])]:
    ks=list(P); n=len(yy); D={k:[] for k in ks if k!=ks[1]}
    for b in range(200):
        i=rng.integers(0,n,n); a=AUC(yy[i],P[ks[1]][i])
        for k in D: D[k].append(AUC(yy[i],P[k][i])-a)
    for k,v in D.items(): print(lab,k,'- 모델2',round(np.mean(v),4),np.percentile(v,[2.5,97.5]).round(4))
