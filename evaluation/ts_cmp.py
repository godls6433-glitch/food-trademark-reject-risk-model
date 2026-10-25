import pandas as pd, numpy as np, glob
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss as BR
S='work'
U='team_preds'
g=pd.read_csv(S+'/ts_preds_haein.csv')
lab=pd.read_csv(S+'/dp2/features_final_v8.csv',encoding='utf-8-sig',usecols=['순번','fold'])
ga=pd.read_csv(glob.glob(U+'/ts_gaeun_xgb.csv')[0],encoding='utf-8-sig')
ye=pd.read_csv(glob.glob(U+'/ts_yehwan_xgb.csv')[0],encoding='utf-8-sig')
for k,d in [('가은',ga),('예환',ye)]:
    print(k,len(d),'순번같음',(d.순번.values==g.순번.values).all(),'출원번호같음',(d.출원번호.astype(str).values==g['출원(국제등록)번호'].astype(str).values).all(),
      'NaN',d.거절확률.isna().sum(),'range',d.거절확률.min(),d.거절확률.max())
print('가은 fold 일치',(ga.merge(lab,on='순번',suffixes=('','_l')).eval('fold==fold_l')).all())
print('예환 라벨 일치',(ye['실제(거절=1)'].values==g.y.values).all(), '연도',ye.출원연도.value_counts().to_dict())
P={'해인 RF':g['해인28'].values,'해인 RF+단어거절률':g['해인28+단어'].values,'예환 XGB':ye.거절확률.values,'가은 XGB':ga.거절확률.values}
y=g.y.values; rows=[]
for k,p in P.items():
    yr=ye.출원연도.values
    rows.append(dict(model=k,ROC=AUC(y,p),PR=AP(y,p),Brier=BR(y,p),mean_pred=p.mean(),ROC_2023=AUC(y[yr==2023],p[yr==2023]),ROC_2024=AUC(y[yr==2024],p[yr==2024])))
R=pd.DataFrame(rows); print('actual',y.mean().round(4)); print(R.round(4).to_string())
rng=np.random.default_rng(42); n=len(y); ref='해인 RF+단어거절률'; D={k:[] for k in P if k!=ref}
for b in range(300):
    i=rng.integers(0,n,n); a=AUC(y[i],P[ref][i])
    for k in D: D[k].append(AUC(y[i],P[k][i])-a)
for k,v in D.items(): print(k,'-',ref,np.mean(v).round(4),np.percentile(v,[2.5,97.5]).round(4))
R.to_csv(S+'/ts_compare.csv',index=False)
