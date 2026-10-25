# 결합상표 단어별 비교: 설명어를 뺀 단어(띄어쓰기·영문/한글 경계·camelCase로 분리)를 선행상표 단어와 정확히 일치 비교
import pandas as pd, numpy as np, re, pickle, time, sys
t0=time.time()
G=pickle.load(open('prep.pkl','rb')); globals().update(G); globals().update(pickle.load(open('prep2.pkl','rb')))
V=pickle.load(open('prep_v3.pkl','rb')); CODE_L3,CODE_P3,l_has3=V['CODE_L3'],V['CODE_P3'],V['l_has3']
exec(open('owner_patch.py').read())
E=pickle.load(open('en_dict.pkl','rb')); GE,Q,FL=E['GOODS_EN'],E['QUAL'],E['FILL']
gw=pd.read_csv('goods_words_v2.csv'); pl=pd.read_csv('place_dict_v2.csv')
gw=gw[gw.ambiguous==0]; places=set(pl[pl.ambiguous==0].place.str.lower()); goods=set(gw[gw.one_char==0].word); one=set(gw[gw.one_char==1].word)
rx=re.compile('|'.join(map(re.escape,sorted(goods|places,key=len,reverse=True))))
def stem(t):
    for suf in ('ies','es','s'):
        if t.endswith(suf) and len(t)-len(suf)>=3: return t[:-len(suf)]+('y' if suf=='ies' else '')
    return t
def toks(name):
    s=str(name); o=re.sub(r'\(.*?\)','',s)
    base=o if re.sub(r'[^0-9a-zA-Z가-힣]','',o) else ' '.join(re.findall(r'\((.*?)\)',s))
    base=re.sub(r'([a-z])([A-Z])',r'\1 \2',base)
    base=re.sub(r'([가-힣])([A-Za-z0-9])',r'\1 \2',base); base=re.sub(r'([A-Za-z0-9])([가-힣])',r'\1 \2',base)
    out=set()
    for t in re.split(r'[^0-9A-Za-z가-힣]+',base):
        t=t.lower()
        if not t: continue
        if re.fullmatch(r'[a-z]+',t):
            if len(t)>=3 and stem(t) not in GE and t not in GE and stem(t) not in Q and t not in Q and t not in FL: out.add(t)
        elif re.search('[가-힣]',t):
            r=rx.sub('',t)
            if r and r[-1] in one: r=r[:-1]
            if len(r)>=2: out.add(r)     # 단어 안의 설명어를 떼고 남은 부분
        elif len(t)>=2 and not t.isdigit(): out.add(t)
    return out
if len(sys.argv)>1 and sys.argv[1]=='index':
    pt=[toks(x) for x in p_name]; pickle.dump(pt,open('tok_pool.pkl','wb')); print('pool toks',round(time.time()-t0)); sys.exit()
pt=pickle.load(open('tok_pool.pkl','rb'))
idx={}
for j,ts in enumerate(pt):
    for t in ts: idx.setdefault(t,[]).append(j)
idx={t:np.array(v,dtype=np.int64) for t,v in idx.items()}
tdates={t:np.sort(p_date[v]) for t,v in idx.items()}
all_sorted=np.sort(p_date)
N=len(pt); CAP=30000
lt=[toks(x) for x in Lr['상표명칭'].values]
ld=Lr.date.values.astype('datetime64[D]'); lc=Lr.cust_id.values; lco=Lr.co_id.values; lrep=Lr.rep_id.values; lb=Lr.cls_bits.values
CL=CODE_L3.tocsr(); CP=CODE_P3.tocsc()
R=np.zeros((n,4),dtype=np.float32)
for i,ts in enumerate(lt):
    ts=[t for t in ts if t in idx and len(idx[t])<=CAP]
    if not ts: continue
    dl=sum(len(t) for t in lt[i]) or 1
    hits=set(); best_idf=0; best_share=0
    for t in ts:
        c=idx[t]
        m=(p_date[c]<ld[i])&(p_alive_until[c]>ld[i])&(p_cust[c]!=lc[i])
        if lco[i]>=0: m&=p_co[c]!=lco[i]
        if lrep[i]>=0: m&=p_rep[c]!=lrep[i]
        if not m.any(): continue
        c=c[m]
        if l_has3[i]:
            ov=np.asarray((CL[i]@CP[:,c]).todense()).ravel()>0
            ov=np.where(p_hascode[c],ov,(p_bits[c]&lb[i])>0)
        else: ov=(p_bits[c]&lb[i])>0
        c=c[ov]
        if not len(c): continue
        nb=np.searchsorted(all_sorted,ld[i]); dfb=np.searchsorted(tdates[t],ld[i])
        hits.update(c.tolist()); best_idf=max(best_idf,np.log(max(nb,1)/max(dfb,1))); best_share=max(best_share,len(t)/dl)
    if hits:
        h=np.fromiter(hits,dtype=np.int64)
        R[i]=[len(h),(p_regdate[h]<ld[i]).sum(),best_idf,best_share]
    if i%50000==0: print(i,round(time.time()-t0),flush=True)
out=pd.DataFrame(R,columns=['tok_hits','tok_hits_reg','tok_max_idf','tok_share']); out.insert(0,'순번',Lr['순번'].values.astype(int))
out['n_tok']=[len(t) for t in lt]
out.to_pickle('feat_tok.pkl' if len(sys.argv)<2 else sys.argv[1]); y=1-Lr.registered.values
for c in ['tok_hits','tok_hits_reg']: print(c,'>0',(out[c]>0).mean().round(3),'거절률',y[out[c]>0].mean().round(3),'vs',y[out[c]==0].mean().round(3))
m=(out.n_tok>=2)&(out.tok_hits>0); print('단어 2개 이상 & 공유 단어 있음',m.mean().round(3),'거절률',y[m].mean().round(3))
print('done',round(time.time()-t0))
