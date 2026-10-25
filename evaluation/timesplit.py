import pandas as pd, numpy as np, glob, re, unicodedata
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR
S='work'
U='team_preds'
f=pd.read_csv(S+'/dp2/features_final_v8.csv',encoding='utf-8-sig')
for t in ['한글상표','영문상표','복합문자','도형복합']: f['nt_'+t]=(f.name_type==t).astype(int)
base=[c for c in f.columns[6:] if c!='name_type']
pool=pd.concat([pd.read_csv(x,encoding='utf-8-sig',usecols=['순번','출원(국제등록)일자']) for x in sorted(glob.glob('data/raw/tm_pool_*.csv'))])
og=pd.read_csv(S+'/dp1/owner_group_labels.csv',encoding='utf-8-sig')
f=f.merge(pool,on='순번').merge(og,on='순번'); f['dt']=pd.to_datetime(f['출원(국제등록)일자'].astype(str).str[:10])
y=1-f.registered.values; f['y']=y
tr=(f.dt.dt.year<=2022).values; te=~tr
print('train',tr.sum(),'test',te.sum(),'test rej',y[te].mean().round(4))
P0=y[tr].mean(); A=20
def grams(s):
    s=unicodedata.normalize('NFKC',str(s)).lower(); s=re.sub(r'\(.*?\)','',s) or s
    s=re.sub(r'[^0-9a-z가-힣]','',s)
    return list({s[i:i+2] for i in range(len(s)-1)}) or ([s] if s else [])
g=f[['dt','y','fold']].copy(); g['gr']=f.상표명칭.map(grams); g['row']=np.arange(len(f)); g['tr']=tr
e=g.explode('gr').dropna(subset=['gr'])
def agg(r,rows):
    t=pd.DataFrame({'row':rows,'r':r}).groupby('row').r.agg(['max','mean','min','count'])
    out=np.full((len(f),4),np.nan); out[t.index.values]=t.values; return out
def rate(sub,stats):
    j=sub.join(stats,on='gr'); return ((j['sum'].fillna(0)+A*P0)/(j['count'].fillna(0)+A)).values, j.row.values
W_te=np.full((len(f),4),np.nan)
r,rows=rate(e[~e.tr], e[e.tr].groupby('gr').y.agg(['sum','count'])); o=agg(r,rows); W_te[te]=o[te]
# 예환식: 학습 행은 무작위 fold 교차 (미래 포함)
W_fold=W_te.copy()
for k in range(1,6):
    st=e[e.tr&(e.fold!=k)].groupby('gr').y.agg(['sum','count']); r,rows=rate(e[e.tr&(e.fold==k)],st); o=agg(r,rows)
    m=tr&(f.fold.values==k); W_fold[m]=o[m]
# 먼저 출원된 것만
W_past=W_te.copy()
et=e[e.tr]; c=et.groupby(['gr','dt']).y.agg(['sum','count']).sort_index(); cs=c.groupby(level=0).cumsum()-c
j=et.join(cs,on=['gr','dt']); o=agg(((j['sum']+A*P0)/(j['count']+A)).values,j.row.values); W_past[tr]=o[tr]
# 출원인 경험 (출원 시점에 알 수 있는 이전 출원 수, 같은 날 출원 수)
c=f.groupby(['출원인그룹','dt']).size().rename('sd'); cum=(c.groupby(level=0).cumsum()-c).rename('pr')
x=f[['출원인그룹','dt']].join(c,on=['출원인그룹','dt']).join(cum,on=['출원인그룹','dt'])
f['ap_prior']=np.log1p(x.pr.values); f['ap_sameday']=x.sd.values-1
W=['w_max','w_mean','w_min','w_n']
def run(name,cols,Wm=None):
    X=f[[c for c in cols if c not in W]].copy()
    if Wm is not None: X[W]=Wm
    X=X.fillna(-1)
    idx=np.where(tr)[0]; rng=np.random.default_rng(42); cal=rng.random(len(idx))<0.2
    m=RF(n_estimators=300,min_samples_leaf=20,max_samples=0.7,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1).fit(X.iloc[idx[~cal]],y[idx[~cal]])
    iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(X.iloc[idx[cal]])[:,1],y[idx[cal]])
    p=iso.predict(m.predict_proba(X[te])[:,1])
    res=dict(model=name,ROC=AUC(y[te],p),PR=AP(y[te],p),Brier=BR(y[te],p))
    print(res,flush=True); return res
R=[run('해인 28',base),
   run('+단어거절률 예환식(학습행 미래포함)',base+W,W_fold),
   run('+단어거절률 먼저출원만',base+W,W_past),
   run('+출원인경험',base+['ap_prior','ap_sameday']),
   run('+단어거절률 먼저출원만+출원인경험',base+W+['ap_prior','ap_sameday'],W_past)]
# 제출된 예측값의 2023~2024 성적 (참고: 무작위 fold라 미래 데이터로 학습됨)
for k,fn in {'해인 RF':'oof_haein_rf.csv','예환 XGB':'oof_yehwan_xgb.csv','가은 XGB':'oof_gaeun_xgb.csv'}.items():
    p=pd.read_csv(glob.glob(U+'/'+fn)[0],encoding='utf-8-sig').거절확률.values[te]
    R.append(dict(model='제출 OOF 2023-24: '+k,ROC=AUC(y[te],p),PR=AP(y[te],p),Brier=BR(y[te],p))); print(R[-1])
pd.DataFrame(R).to_csv(S+'/timesplit_results.csv',index=False)
