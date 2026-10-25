# 주요부 기준 유사도 변수 (기준 C, 라벨 식품 코드, 타인 상표)
import pandas as pd, numpy as np, time, pickle, os, sys
from rapidfuzz import fuzz, process
t0=time.time()
G=pickle.load(open('prep.pkl','rb')); globals().update(G); globals().update(pickle.load(open('prep2.pkl','rb')))
V=pickle.load(open('prep_v3.pkl','rb')); CODE_L3,CODE_P3,l_has3,l_sets=V['CODE_L3'],V['CODE_P3'],V['l_has3'],V['l_sets']
T=pickle.load(open('core_translit.pkl','rb'))
exec(open('owner_patch.py').read())
p_rom=np.array(T['p_rom'],dtype=object); p_en=np.array(T['p_en'],dtype=object); l_rom=np.array(T['l_rom'],dtype=object); l_en=np.array(T['l_en'],dtype=object)
p_has_en=p_en!=''; p_has_rom=p_rom!=''
BUDGET=float(sys.argv[1]); OUT='partsC3'; os.makedirs(OUT,exist_ok=True)
done=set(int(f.split('_')[1].split('.')[0]) for f in os.listdir(OUT)); cnt=0; K=5
def compute(s0,s1):
    Q=Lr.iloc[s0:s1]; qi=np.arange(s0,s1); R=np.arange(len(Q))
    kk=np.searchsorted(p_date_sorted,Q.date.values.astype('datetime64[D]').max()); cols=p_order[:kk]
    cs=process.cdist(Q.cj.tolist(),[p_cj[j] for j in cols],scorer=fuzz.ratio,dtype=np.uint8,workers=-1)
    cs=np.where(p_clen[cols][None,:]>=2,cs,0)
    qd=Q.date.values.astype('datetime64[D]')[:,None]
    tm=(p_date[cols][None,:]<qd)&(p_alive_until[cols][None,:]>qd)
    own=p_cust[cols][None,:]==Q.cust_id.values[:,None]
    qco=Q.co_id.values[:,None]; own|=(qco>=0)&(p_co[cols][None,:]==qco)
    qrep=Q.rep_id.values[:,None]; own|=(qrep>=0)&(p_rep[cols][None,:]==qrep)
    cls_ov=(p_bits[cols][None,:]&Q.cls_bits.values[:,None])>0
    ov=(CODE_L3[qi]@CODE_P3[:,cols]).toarray()>0
    nocode=(~l_has3[qi][:,None])|(~p_hascode[cols][None,:])
    mo=tm&np.where(nocode,cls_ov,ov)&~own
    csm=np.where(mo,cs,0)
    hit=(p_core[cols][None,:]==Q.core_id.values[:,None])&(Q.core.str.len().values[:,None]>=2)
    o={'core_sim':csm.max(1)/100,'core_hits':(hit&mo).sum(1),'core_n80':(csm>=80).sum(1)}
    a1=csm.argmax(1); t1=cols[a1]; has=csm[R,a1]>0
    reg=p_regdate[t1]<qd[:,0]
    o['top_reg_c']=np.where(has,reg,0).astype(np.int8)
    o['top_code_jacc_c']=np.array([len(l_sets[i]&p_codes[j])/max(1,len(l_sets[i]|p_codes[j])) if h else 0 for i,j,h in zip(qi,t1,has)],dtype=np.float32)
    tr=np.zeros(len(Q),dtype=np.float32)
    for qs,ps,pmask in ((l_rom[qi],p_en,p_has_en),(l_en[qi],p_rom,p_has_rom)):
        rows=np.where(qs!='')[0]; sel=pmask[cols]
        if len(rows)==0 or not sel.any(): continue
        sim=process.cdist(list(qs[rows]),list(ps[cols[sel]]),scorer=fuzz.ratio,dtype=np.uint8,workers=-1)
        v=np.where(mo[rows][:,sel],sim,0).max(1)/100; tr[rows]=np.maximum(tr[rows],v)
    o['core_translit_sim']=tr
    df=pd.DataFrame(o,index=qi)
    c16=csm.astype(np.int16); top=np.argpartition(-c16,K,axis=1)[:,:K]
    tv=np.take_along_axis(c16,top,1); od=np.argsort(-tv,1); top=np.take_along_axis(top,od,1); tv=np.take_along_axis(tv,od,1)
    for r in range(K): df[f'ctop{r+1}_idx']=np.where(tv[:,r]>0,cols[top[:,r]],-1); df[f'ctop{r+1}_sim']=tv[:,r]/100
    return df
for s in range(0,n,CH):
    if s in done: continue
    if time.time()-t0>BUDGET: break
    pd.concat([compute(x,min(x+100,s+CH,n)) for x in range(s,min(s+CH,n),100)]).to_pickle(f'{OUT}/part_{s}.pkl'); cnt+=1
print(f'this run {cnt}, remaining {(n+CH-1)//CH-len(done)-cnt}, {round(time.time()-t0)}s',flush=True)
