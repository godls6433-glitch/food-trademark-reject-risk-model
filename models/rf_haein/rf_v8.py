import pandas as pd, numpy as np, sys, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
set_=sys.argv[1]; t0=time.time()
f=pd.read_pickle('features_v8.pkl'); y=1-f.registered.values
base=['place_ratio','logo','name_len','cls_29','cls_30','cls_32','cls_43']
for t in ['한글상표','영문상표','복합문자','도형복합']: f['nt_'+t]=(f.name_type==t).astype(int); base.append('nt_'+t)
desc=['desc_ratio2','goods_ratio2','own_goods_ratio','other_goods_ratio']
core=['core_sim','core_hits','core_n80','core_part_sim','core_n_contain','core_translit_sim','top_reg_c','top_code_jacc_c']
tok=['tok_hits','tok_hits_reg','tok_max_idf','tok_share']
S={'v7':base+desc+core+tok+['self_reg_core_sim'],'v7b':base+desc+core+tok+['self_reg_core_sim','self_reg_max_sim']}
cols=S[set_]; X=f[cols].astype(float).values
cal=np.zeros(len(f)); rng=np.random.RandomState(42); imps=[]
for k in range(1,6):
    tr=np.where(f.fold!=k)[0]; te=np.where(f.fold==k)[0]; rng.shuffle(tr); cut=int(len(tr)*.8)
    m=RandomForestClassifier(n_estimators=300,min_samples_leaf=20,max_samples=0.7,class_weight='balanced',n_jobs=-1,random_state=42).fit(X[tr[:cut]],y[tr[:cut]])
    iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(X[tr[cut:]])[:,1],y[tr[cut:]])
    cal[te]=iso.predict(m.predict_proba(X[te])[:,1]); imps.append(m.feature_importances_)
pd.DataFrame({'순번':f['순번'],'출원번호':f['출원(국제등록)번호'],'fold':f.fold,'거절확률':cal}).to_pickle(f'oof_rf_{set_}_g.pkl')
fa=[roc_auc_score(y[f.fold==k],cal[f.fold==k]) for k in range(1,6)]
print('RESULT',set_,len(cols),round(roc_auc_score(y,cal),4),round(average_precision_score(y,cal),4),round(brier_score_loss(y,cal),4),round(np.mean(fa),4),round(np.std(fa),4))
print(pd.Series(np.mean(imps,0),cols).sort_values(ascending=False).round(3).head(12).to_string())
