import pandas as pd, numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
src=open('rf_v8.py').read(); exec(src[:src.index("cols=S[set_]")].replace("set_=sys.argv[1]; t0=time.time()","set_='v7'; t0=time.time()"))
cols=S['v7']; X=f[cols].astype(float).values
P=dict(n_estimators=100,max_depth=14,min_samples_leaf=50,max_samples=0.7,class_weight='balanced',n_jobs=-1,random_state=42)
cal=np.zeros(len(f)); rng=np.random.RandomState(42)
for k in range(1,6):
    tr=np.where(f.fold!=k)[0]; te=np.where(f.fold==k)[0]; rng.shuffle(tr); cut=int(len(tr)*.8)
    m=RandomForestClassifier(**P).fit(X[tr[:cut]],y[tr[:cut]]); iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(X[tr[cut:]])[:,1],y[tr[cut:]]); cal[te]=iso.predict(m.predict_proba(X[te])[:,1])
print('compact CV',round(roc_auc_score(y,cal),4),round(average_precision_score(y,cal),4))
idx=np.arange(len(f)); rng=np.random.RandomState(42); rng.shuffle(idx); cut=int(len(idx)*.8)
m=RandomForestClassifier(**P).fit(X[idx[:cut]],y[idx[:cut]]); iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(X[idx[cut:]])[:,1],y[idx[cut:]])
F=[];T=[];L=[];R=[];V=[];off=[0]
for e in m.estimators_:
    t=e.tree_; v=t.value[:,0,:]; p=v[:,1]/v.sum(1)
    F+=t.feature.tolist(); T+=[round(float(x),6) for x in t.threshold]; L+=t.children_left.tolist(); R+=t.children_right.tolist(); V+=[round(float(x),5) for x in p]; off.append(len(F))
json.dump(dict(cols=cols,F=F,T=T,L=L,R=R,V=V,off=off,isoX=[float(x) for x in iso.X_thresholds_],isoY=[float(x) for x in iso.y_thresholds_]),open('demo_files/model.json','w'),separators=(',',':'))
s=f.sample(50,random_state=3); raw=m.predict_proba(s[cols].astype(float).values)[:,1]
s[['순번']+cols].assign(raw=raw,cal=iso.predict(raw)).to_json('demo_files/check_sample.json',orient='records',force_ascii=False)
print(cols)
