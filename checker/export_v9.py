# 점검기용 경량 RF (100그루, 깊이 14) + 단어별 거절률 4개 → model.json, wrate.json
import pandas as pd, numpy as np, json, re, unicodedata
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
S='work'
exec(open(S+'/three.py').read().split("def fit_pred")[0])   # f, base, y, e, W, grams, past_rates, agg, A
Wall=past_rates(np.ones(len(f),bool), y.mean())
f[W]=Wall; f[W]=f[W].fillna(-1)
cols=base+W; X=f[cols].astype(float).values
P=dict(n_estimators=100,max_depth=14,min_samples_leaf=50,max_samples=0.7,class_weight='balanced',n_jobs=-1,random_state=42)
cal=np.zeros(len(f)); rng=np.random.RandomState(42)
for k in range(1,6):
    tr=np.where(f.fold!=k)[0]; te=np.where(f.fold==k)[0]; rng.shuffle(tr); cut=int(len(tr)*.8)
    m=RandomForestClassifier(**P).fit(X[tr[:cut]],y[tr[:cut]]); iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(X[tr[cut:]])[:,1],y[tr[cut:]]); cal[te]=iso.predict(m.predict_proba(X[te])[:,1])
print('lite CV ROC %.4f PR %.4f Brier %.4f'%(roc_auc_score(y,cal),average_precision_score(y,cal),brier_score_loss(y,cal)))
g=np.where(cal<0.12,'낮음',np.where(cal<0.30,'중간','높음'))
t=pd.DataFrame({'g':g,'y':y}).groupby('g').y.agg(['size','mean']); t['share']=t['size']/len(y); print(t.round(4))
print('높음 재현율',round(y[g=='높음'].sum()/y.sum(),4),'낮음 놓친 비율',round(y[g=='낮음'].sum()/y.sum(),4))
idx=np.arange(len(f)); rng=np.random.RandomState(42); rng.shuffle(idx); cut=int(len(idx)*.8)
m=RandomForestClassifier(**P).fit(X[idx[:cut]],y[idx[:cut]]); iso=IsotonicRegression(out_of_bounds='clip').fit(m.predict_proba(X[idx[cut:]])[:,1],y[idx[cut:]])
F=[];T=[];L=[];R=[];V=[];off=[0]
for est in m.estimators_:
    tt=est.tree_; v=tt.value[:,0,:]; p=v[:,1]/v.sum(1)
    F+=tt.feature.tolist(); T+=[round(float(x),6) for x in tt.threshold]; L+=tt.children_left.tolist(); R+=tt.children_right.tolist(); V+=[round(float(x),5) for x in p]; off.append(len(F))
json.dump(dict(cols=cols,F=F,T=T,L=L,R=R,V=V,off=off,isoX=[float(x) for x in iso.X_thresholds_],isoY=[float(x) for x in iso.y_thresholds_]),open(S+'/new/model.json','w'),separators=(',',':'))
st=e.groupby('gr').y.agg(['sum','count'])
json.dump({'P0':float(y.mean()),'A':A,'g':{k:[int(a),int(b)] for k,a,b in zip(st.index,st['sum'],st['count'])}},open(S+'/new/wrate.json','w'),ensure_ascii=False,separators=(',',':'))
# 검증 표본: 이름별 단어 거절률(전체 라벨 기준)과 모델 원점수
smp=f.sample(300,random_state=3)
def wr(name):
    gs=grams(name)
    if not gs: return [-1]*4
    r=[(st.loc[x,'sum']+A*y.mean())/(st.loc[x,'count']+A) if x in st.index else y.mean() for x in gs]
    return [max(r),float(np.mean(r)),min(r),len(r)]
smp['wr']=smp.상표명칭.map(wr)
raw=m.predict_proba(smp[cols].astype(float).values)[:,1]
json.dump([{'name':str(n),'wr':w,'x':list(map(float,xx)),'raw':float(r),'cal':float(c)} for n,w,xx,r,c in zip(smp.상표명칭,smp.wr,smp[cols].astype(float).values,raw,iso.predict(raw))],open(S+'/new/check.json','w'),ensure_ascii=False)
print(len(st),'grams')
