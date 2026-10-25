import pandas as pd, numpy as np, re, pickle, json, os
G=pickle.load(open('prep.pkl','rb')); Lr=G['Lr']; p_name=G['p_name']
# 선행상표 풀 상세 (feat_prep_x와 같은 순서로 재구성)
T=pd.read_pickle('tm_full.pkl').drop(columns=['src_end','출원공고번호']).reset_index(drop=True); T['src']='T'
P=pd.read_pickle('p_full_dedup.pkl'); P['src']='P'; D=pd.read_pickle('d_full.pkl').drop(columns=['src_end']); D['src']='D'
G23=set("G020101+G020102+G0202+G0204+G0207+G0208+G0211+G0301+G0302+G0406+G0502+G0602+G0702+G0704+G0707+G0801+G0802+G0803+G1001+G100401+G1006+G1009+G4601".split('+'))
old=pd.concat([T,P,D]); key=lambda d: d['출원(국제등록)번호'].astype(str)+'|'+d['상품분류'].fillna('')+'|'+d['상표명칭'].fillna('')
okk=set(key(old)); X=pd.concat([pd.read_pickle('N_full.pkl'),pd.read_pickle('E_full.pkl')],ignore_index=True).drop(columns=['src_end'])
X=X[~key(X).isin(okk)]; X=X[~X.drop(columns=['순번']).duplicated()]; X=X[X['유사군코드'].fillna('').str.split('|').apply(lambda s: bool(set(s)&G23))]; X['src']='X'
U=pd.concat([T,P,D,X],ignore_index=True)
U['status']=U['법적상태'].str.replace(r'<.*','',regex=True).str.strip(); U['regdate']=pd.to_datetime(U['등록일자'],format='%Y.%m.%d',errors='coerce')
def norm(s):
    s=str(s); o=re.sub(r'\(.*?\)','',s); c=lambda x: re.sub(r'[^0-9a-zA-Z가-힣]','',x).lower()
    return c(o) or c(' '.join(re.findall(r'\((.*?)\)',s)))
U['nm']=U['상표명칭'].apply(norm)
ok=((U.src=='T')&U.status.isin(['출원','공고','등록','소멸']))|(U.src=='P')|((U.src=='D')&U.regdate.notna())|((U.src=='X')&(U.status.isin(['출원','공고','등록'])|((U.status=='소멸')&U.regdate.notna())))
ok&=U.nm.str.len()>0
Pf=U[ok].reset_index(drop=True); a=Pf['상표명칭'].astype(str).values; b=np.asarray(p_name).astype(str); print('len',len(Pf),len(p_name),'mismatch',(a!=b).sum()); assert len(Pf)==len(p_name) and (a!=b).sum()<=5
# 라벨 쪽
o=pd.read_pickle('oof_rf_features_v5_v6.pkl')[['순번','거절확률']]
f=pd.read_pickle('features_v5.pkl').merge(pd.read_pickle('feat_selfsafe.pkl'),on='순번').merge(pd.read_pickle('feat_desc2.pkl'),on='순번').merge(pd.read_pickle('feat_tok.pkl'),on='순번').merge(o,on='순번')
parts=pd.concat([pd.read_pickle('parts4/'+x) for x in os.listdir('parts4')]).sort_index(); parts['순번']=Lr['순번'].values.astype(int)

T2=T.copy(); T2['순번']=np.arange(1,len(T2)+1)
f=f.merge(T2[['순번','지정상품','유사군코드','출원(국제등록)일자']],on='순번')
r=pd.read_csv('reject_reasons.csv',encoding='utf-8-sig',dtype=str).drop_duplicates('출원번호')
f['출원번호s']=f['출원(국제등록)번호'].astype(str); f=f.merge(r[['출원번호','거절사유']],left_on='출원번호s',right_on='출원번호',how='left')
Lcore=Lr[['순번','core']].copy(); Lcore['순번']=Lcore.순번.astype(int); f=f.merge(Lcore,on='순번')
# 표본: 예측 10구간 × 거절/등록 고르게, 거절 사유 있는 건 우선
f['dec']=pd.qcut(f.거절확률.rank(method='first'),10,labels=False)
rng=np.random.RandomState(11); pick=[]
for d in range(10):
    g=f[f.dec==d]
    rej=g[(g.registered==0)&g.거절사유.notna()]; reg=g[g.registered==1]
    nr=min(len(rej),20); pick+=list(rej.sample(nr,random_state=d).index)+list(reg.sample(40-nr,random_state=d).index)
S=f.loc[pick].copy()
gmap=pd.read_csv('goods_code_map.csv',encoding='utf-8-sig'); gm=dict(zip(gmap.지정상품,gmap.유사군코드))
FOOD=pickle.load(open('prep_v3.pkl','rb'))['FOOD']
def grade(p): return '낮음' if p<0.12 else ('높음' if p>=0.30 else '중간')
def rtype(x):
    sim=max(x.max_sim, x.core_sim if len(x.core)>=2 else 0)
    s_hi=sim>=0.9 or (x.tok_hits_reg>0 and x.tok_share>=0.5); d_hi=x.desc_ratio2>=0.5
    return '복합' if (s_hi and d_hi) else ('유사' if s_hi else ('식별력' if d_hi else '뚜렷한 유형 없음'))
out=[]
for _,x in S.iterrows():
    lcodes=set(str(x.유사군코드).split('|'))&FOOD
    goods=[g.strip() for g in str(x.지정상품).split('|') if g.strip()]
    pri=[]
    for k in range(1,6):
        j=int(x[f'top{k}_idx'])
        if j<0: continue
        q=Pf.iloc[j]; pc=set(str(q['유사군코드']).split('|')) if isinstance(q['유사군코드'],str) else set()
        reg=pd.notna(q.regdate) and q.regdate<pd.to_datetime(x['출원(국제등록)일자'],format='%Y.%m.%d')
        pri.append(dict(name=q['상표명칭'],cls=q['상품분류'],status=q.status,sim=round(float(x[f'top{k}_sim']),2),app=str(q['출원(국제등록)번호']),date=q['출원(국제등록)일자'],reg=bool(reg),codes=sorted(pc&lcodes)))
    gb=[]
    for g in goods[:40]:
        cs=set(str(gm.get(g,'')).split('|'))-{''}
        if not cs: continue
        hit=[p['name'] for p in pri if p['sim']>=0.8 and set(p['codes'])&cs]
        gb.append(dict(g=g,codes=sorted(cs),hit=hit))
    why=[]
    if x.desc_ratio2>=0.5: why.append(f"상표명의 {round(x.desc_ratio2*100)}%가 상품명·지명·품질표현 (식별력 약함)")
    if x.own_goods_ratio>0: why.append("자기 지정상품 이름이 상표명에 들어 있음")
    if x.other_goods_ratio>0: why.append("지정하지 않은 상품 이름이 들어 있음 (품질오인 주의)")
    if x.core_hits>0: why.append(f"주요부가 같은 선행상표 {int(x.core_hits)}건")
    elif x.core_sim>=0.9 and len(x.core)>=2: why.append(f"주요부가 매우 비슷한 선행상표 있음 (유사도 {x.core_sim:.2f})")
    if x.n_exact>0: why.append(f"이름이 완전히 같은 선행상표 {int(x.n_exact)}건")
    elif x.max_sim>=0.9: why.append(f"가장 비슷한 선행상표 유사도 {x.max_sim:.2f}")
    if x.tok_hits_reg>0: why.append(f"같은 단어를 쓰는 등록상표 {int(x.tok_hits_reg)}건")
    if x.n_contain>0: why.append(f"이름 안에 통째로 들어 있는 선행상표 {int(x.n_contain)}건")
    if x.self_reg_max_sim>=0.8: why.append("본인(계열사)의 비슷한 등록상표가 있음 (위험을 낮춤)")
    if x.logo==1: why.append("로고(도형)와 함께 출원 (위험을 약간 낮춤)")
    if not why: why.append("눈에 띄는 위험 신호 없음")
    out.append(dict(id=int(x.순번),name=x['상표명칭'],cls=x['상품분류'],date=x['출원(국제등록)일자'],type=x.name_type,logo=int(x.logo),
        p=round(float(x.거절확률),3),grade=grade(x.거절확률),rtype=rtype(x),actual='거절' if x.registered==0 else '등록',reason=(x.거절사유 if isinstance(x.거절사유,str) else ''),
        why=why,pri=pri,goods=gb,ngoods=len(goods),
        f=dict(desc=round(float(x.desc_ratio2),2),max_sim=round(float(x.max_sim),2),core_sim=round(float(x.core_sim),2),core=x.core,self=round(float(x.self_reg_max_sim),2),tok=int(x.tok_hits_reg))))
json.dump(out,open('demo_data.json','w'),ensure_ascii=False)
s=pd.DataFrame(out); print(len(s), s.grade.value_counts().to_dict(), s.actual.value_counts().to_dict(), round(os.path.getsize('demo_data.json')/1e6,2),'MB')
print(pd.crosstab(s[s.reason!=''].rtype,s[s.reason!=''].reason))
