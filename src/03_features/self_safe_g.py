# 본인 선행상표 유사도의 시점 안전 버전: 라벨 출원일 전에 이미 등록된 본인(계열사 포함) 상표만
import pandas as pd, numpy as np, pickle, time
from rapidfuzz import fuzz
t0=time.time()
G=pickle.load(open('prep.pkl','rb')); globals().update(G); globals().update(pickle.load(open('prep2.pkl','rb')))
V=pickle.load(open('prep_v3.pkl','rb')); l_sets,l_has3=V['l_sets'],V['l_has3']
exec(open('owner_patch.py').read())
from collections import defaultdict
by=defaultdict(list)
for j in range(len(p_cust)):
    by[('c',p_cust[j])].append(j)
    if p_co[j]>=0: by[('o',p_co[j])].append(j)
    if p_rep[j]>=0: by[('r',p_rep[j])].append(j)
ld=Lr.date.values.astype('datetime64[D]'); out=np.zeros((n,2),dtype=np.float32)
for i in range(n):
    keys=[('c',Lr.cust_id.values[i])]
    if Lr.co_id.values[i]>=0: keys.append(('o',Lr.co_id.values[i]))
    if Lr.rep_id.values[i]>=0: keys.append(('r',Lr.rep_id.values[i]))
    cand=set()
    for k in keys: cand.update(by.get(k,()))
    if not cand: continue
    c=np.fromiter(cand,dtype=np.int64)
    c=c[(p_date[c]<ld[i])&(p_regdate[c]<ld[i])&(p_alive_until[c]>ld[i])]
    if not len(c): continue
    ok=np.array([ (not p_codes[j].isdisjoint(l_sets[i])) if (p_hascode[j] and l_has3[i]) else bool(p_bits[j]&Lr.cls_bits.values[i]) for j in c])
    c=c[ok]
    if not len(c): continue
    nm=Lr.nm.values[i]; jm=Lr.jm.values[i]; cj=Lr.cj.values[i]
    out[i,0]=max(max(fuzz.ratio(nm,p_nm[j]),fuzz.ratio(jm,p_jm[j])) for j in c)/100
    out[i,1]=max(fuzz.ratio(cj,p_cj[j]) if p_clen[j]>=2 else 0 for j in c)/100
pd.DataFrame({'순번':Lr['순번'].values.astype(int),'self_reg_max_sim':out[:,0],'self_reg_core_sim':out[:,1]}).to_pickle('feat_selfsafe_g.pkl')
print('done',round(time.time()-t0))
