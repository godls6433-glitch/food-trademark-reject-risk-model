# v3 포함 유사도 (기준 C, 라벨 쪽 식품 코드만, 타인 상표)
import pandas as pd, numpy as np, time, pickle, ahocorasick
t0=time.time()
G=pickle.load(open('prep.pkl','rb')); globals().update(G)
V=pickle.load(open('prep_v3.pkl','rb')); l_sets,l_has3,p_sets=V['l_sets'],V['l_has3'],V['p_sets']
CAP=2000; pos={}
for j,w in enumerate(p_nm):
    if len(w)>=2: pos.setdefault(w,[]).append(j)
A1=ahocorasick.Automaton()
for w in pos: A1.add_word(w,w)
A1.make_automaton()
A2=ahocorasick.Automaton()
for w in set(w for w in Lr.nm.values if len(w)>=2): A2.add_word(w,w)
A2.make_automaton()
rev={}
for j,w in enumerate(p_nm):
    if len(w)>=2:
        for _,lw in A2.iter(w):
            if lw!=w:
                l=rev.setdefault(lw,[])
                if len(l)<CAP: l.append(j)
mp=np.zeros(n,dtype=np.float32); nc=np.zeros(n,dtype=np.float32)
ld=Lr.date.values.astype('datetime64[D]'); lc=Lr.cust_id.values; lco=Lr.co_id.values; lrep=Lr.rep_id.values; lb=Lr.cls_bits.values
for i,w in enumerate(Lr.nm.values):
    if len(w)<2: continue
    cand=[]
    for _,pw in A1.iter(w): cand.extend(pos[pw][:CAP])
    cand.extend(rev.get(w,[]))
    if not cand: continue
    cand=np.unique(np.array(cand))
    m=(p_date[cand]<ld[i])&(p_alive_until[cand]>ld[i])&(p_cust[cand]!=lc[i])
    if lco[i]>=0: m&=p_co[cand]!=lco[i]
    if lrep[i]>=0: m&=p_rep[cand]!=lrep[i]
    if not m.any(): continue
    clsov=(p_bits[cand]&lb[i])>0
    cov=np.array([(not p_sets[j].isdisjoint(l_sets[i])) if (p_hascode[j] and l_has3[i]) else clsov[k] for k,j in enumerate(cand)])
    m&=cov
    if m.any():
        ratio=np.minimum(p_len[cand],len(w))/np.maximum(p_len[cand],len(w))
        mp[i]=ratio[m].max(); nc[i]=m.sum()
pd.DataFrame({'max_part_sim':mp,'n_contain':nc}).to_pickle('contain_v3.pkl'); print('done',round(time.time()-t0))
