import pandas as pd, numpy as np, glob
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
S='work'
f=pd.read_csv(S+'/dp2/features_final_v8.csv',encoding='utf-8-sig')
for t in ['한글상표','영문상표','복합문자','도형복합']: f['nt_'+t]=(f.name_type==t).astype(int)
base=[c for c in f.columns[6:] if c!='name_type']
print(len(base))
og=pd.read_csv(S+'/dp1/owner_group_labels.csv',encoding='utf-8-sig')
pool=pd.concat([pd.read_csv(x,encoding='utf-8-sig',usecols=['순번','출원(국제등록)일자']) for x in sorted(glob.glob('data/raw/tm_pool_*.csv'))])
d=f[['순번']].merge(og,on='순번').merge(pool,on='순번')
d['dt']=pd.to_datetime(d['출원(국제등록)일자'].astype(str).str[:10])
c=d.groupby(['출원인그룹','dt']).size().rename('sameday')
cum=(c.groupby(level=0).cumsum()-c).rename('prior')
d=d.join(c,on=['출원인그룹','dt']).join(cum,on=['출원인그룹','dt']).sort_values('순번')
f['ap_prior']=np.log1p(d.prior.values); f['ap_sameday']=d.sameday.values-1
f['n_cls']=f[['cls_29','cls_30','cls_32','cls_43']].sum(axis=1)
y=1-f.registered.values
sets={'해인 28':base,'+n_cls':base+['n_cls'],'+출원인경험':base+['ap_prior','ap_sameday'],'+둘다':base+['n_cls','ap_prior','ap_sameday']}
for k,cols in sets.items():
    p=np.zeros(len(f))
    for fo in range(1,6):
        tr=f.fold!=fo; m=H(max_iter=400,learning_rate=0.05,random_state=42).fit(f.loc[tr,cols],y[tr]); p[~tr]=m.predict_proba(f.loc[~tr,cols])[:,1]
    print(k,'ROC %.4f PR %.4f'%(AUC(y,p),AP(y,p)))
