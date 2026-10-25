import pandas as pd, numpy as np, glob, json
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR
S='work'
U='team_preds'
lab=pd.read_csv(S+'/dp2/features_final_v8.csv',encoding='utf-8-sig',usecols=['순번','출원(국제등록)번호','registered','fold','cls_29','cls_30','cls_32','cls_43','상품분류'])
lab['y']=1-lab.registered
files={'해인 RF':'oof_haein_rf.csv','예환 XGB':'oof_yehwan_xgb.csv','가은 XGB':'oof_gaeun_xgb.csv'}
P={}
for k,f in files.items():
    d=pd.read_csv(glob.glob(U+'/'+f)[0],encoding='utf-8-sig')
    chk=dict(rows=len(d), same_order=bool((d.순번.values==lab.순번.values).all()),
      appno=bool((d.출원번호.astype(str).values==lab['출원(국제등록)번호'].astype(str).values).all()),
      fold=bool((d.fold.values==lab.fold.values).all()), nan=int(d.거절확률.isna().sum()),
      rng=(float(d.거절확률.min()),float(d.거절확률.max())), nuniq=int(d.거절확률.nunique()))
    print(k,chk)
    P[k]=d.거절확률.values
y=lab.y.values
print('reject rate',y.mean())
rows=[]
for k,p in P.items():
    fa=[AUC(y[lab.fold==f],p[lab.fold==f]) for f in range(1,6)]
    q=pd.qcut(pd.Series(p).rank(method='first'),10,labels=False)
    dec=pd.Series(y).groupby(q).mean()
    rows.append(dict(model=k,ROC=AUC(y,p),PR=AP(y,p),Brier=BR(y,p),fold_mean=np.mean(fa),fold_sd=np.std(fa,ddof=1),
      mean_pred=p.mean(),dec_lo=dec.iloc[0],dec_hi=dec.iloc[-1]))
R=pd.DataFrame(rows); print(R.round(4).to_string())
# paired bootstrap of differences
rng=np.random.default_rng(42); ks=list(P); B=200; diffs={}
n=len(y)
for b in range(B):
    i=rng.integers(0,n,n)
    a={k:AUC(y[i],P[k][i]) for k in ks}
    for k in ks[1:]: diffs.setdefault(k,[]).append(a[k]-a[ks[0]])
for k,v in diffs.items(): print('AUC diff',k,'-',ks[0],np.mean(v).round(4),np.percentile(v,[2.5,97.5]).round(4))
print('pred corr (spearman)'); print(pd.DataFrame(P).corr('spearman').round(3))
# calibration by predicted decile
for k,p in P.items():
    q=pd.qcut(pd.Series(p).rank(method='first'),10,labels=False)
    t=pd.DataFrame({'p':p,'y':y}).groupby(q).mean()
    print(k,'calib max |pred-actual| %.3f'%(t.p-t.y).abs().max())
# applicant history subgroup
og=pd.read_csv(S+'/dp1/owner_group_labels.csv',encoding='utf-8-sig')
pool=pd.concat([pd.read_csv(f,encoding='utf-8-sig',usecols=['순번','출원(국제등록)일자']) for f in sorted(glob.glob('data/raw/tm_pool_*.csv'))])
d=lab[['순번','y','상품분류']].merge(og,on='순번').merge(pool,on='순번',how='left')
assert len(d)==n and (d.순번.values==lab.순번.values).all()
d['dt']=pd.to_datetime(d['출원(국제등록)일자'].astype(str).str[:10],errors='coerce')
print('date NA',d.dt.isna().sum())
# prior labeled apps by same owner group strictly before date (labels only; approximation)
d=d.sort_values(['출원인그룹','dt'])
first_dt=d.groupby(['출원인그룹','dt']).size().groupby(level=0).cumsum()-d.groupby(['출원인그룹','dt']).size()
d=d.join(first_dt.rename('prior'),on=['출원인그룹','dt']).sort_values('순번')
for k in ks: d[k]=P[k]
d['ncls']=lab[['cls_29','cls_30','cls_32','cls_43']].sum(1).values
d['hist']=pd.cut(d.prior,[-1,0,2,10,1e9],labels=['0건(첫 출원)','1~2건','3~10건','11건+'])
print(d.groupby('hist',observed=True).agg(n=('y','size'),rej=('y','mean')).round(3))
out=[]
for h,g in d.groupby('hist',observed=True):
    out.append({'group':h,'n':len(g),'rej':g.y.mean(),**{k:AUC(g.y,g[k]) for k in ks}})
for c,g in d.groupby(d.ncls.clip(upper=2)):
    out.append({'group':f'류 {c}개{"이상" if c==2 else ""}','n':len(g),'rej':g.y.mean(),**{k:AUC(g.y,g[k]) for k in ks}})
O=pd.DataFrame(out); print(O.round(4).to_string())
# mean pred by n class
print(d.groupby(d.ncls.clip(upper=4))[['y']+ks].mean().round(3))
print(d.groupby('hist',observed=True)[['y']+ks].mean().round(3))
R.to_csv(S+'/metrics.csv',index=False); O.to_csv(S+'/subgroups.csv',index=False)
d[['순번','hist','ncls']].to_csv(S+'/sub_keys.csv',index=False)
