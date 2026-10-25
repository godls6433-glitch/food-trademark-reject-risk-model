# 출원인 같은 그룹 판별 (선행상표 풀 + 학습 라벨 전체)
import pandas as pd, numpy as np, re, pickle, collections
P=pickle.load(open('pool_owner.pkl','rb')); L=pickle.load(open('label_owner.pkl','rb'))
G=pickle.load(open('prep.pkl','rb')); Lr=G['Lr']; p_nm=G['p_nm']
lab=L.set_index('순번').loc[Lr['순번'].astype(int).values].reset_index()
SUF=r'주식회사|유한회사|유한책임회사|합자회사|합명회사|농업회사법인|영농조합법인|어업회사법인|협동조합|사단법인|재단법인|\(주\)|\(유\)|㈜|\bco\b|\bcorp\b|\bcorporation\b|\bcompany\b|\bltd\b|\blimited\b|\binc\b|\bllc\b|\bgmbh\b|\bag\b|\bsa\b|\bs\.a\b|\bspa\b|\bpte\b|\bpty\b|\bbv\b|\bnv\b|\bplc\b|\bkk\b|컴퍼니|코퍼레이션|리미티드|엘티디|인코포레이티드|피티이|피티와이|게엠베하|악티엔게젤샤프트'
def key(n):
    s=str(n).lower(); s=re.sub(SUF,' ',s); s=re.sub(r'[^0-9a-z가-힣]','',s); return s
names=pd.concat([P[['name','cust','kind','rep']],lab[['name','cust','kind','rep']]],ignore_index=True)
names['key']=names.name.apply(key)
# 출원인 단위: 고객번호
own=names.drop_duplicates('cust').reset_index(drop=True)
own=own[own.cust!='']
idx={c:i for i,c in enumerate(own.cust)}
par=list(range(len(own)))
def find(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
size=[1]*len(par)
def union(a,b,why,cap=20):
    ra,rb=find(a),find(b)
    if ra==rb: return
    if size[ra]+size[rb]>cap: cnt['skip(size) '+why]+=1; return
    par[ra]=rb; size[rb]+=size[ra]; cnt[why]+=1
cnt=collections.Counter()
pool_cust=P.cust.values; lab_cust=lab.cust.values
omarks=collections.defaultdict(set)
for c,n in zip(pool_cust,p_nm):
    if c in idx and len(n)>=2: omarks[idx[c]].add(n)
for c,n in zip(lab_cust,Lr.nm.values):
    if c in idx and len(n)>=2: omarks[idx[c]].add(n)
share=lambda a,b: len(omarks[a]&omarks[b])>0
# 규칙 2: 같은 이름 키 (2글자 이상)
by=collections.defaultdict(list)
for i,(k,kd) in enumerate(zip(own.key,own.kind)):
    if len(k)>=2 and kd!='개인': by[k].append(i)
for k,v in by.items():
    for x in v[1:]: union(v[0],x,'R2 같은 이름')
# 규칙 3: 회사명 포함 (짧은 쪽 3글자 이상, 법인·외국만)
corp=[(i,k) for i,(k,kd) in enumerate(zip(own.key,own.kind)) if len(k)>=3 and kd!='개인']
import ahocorasick
A=ahocorasick.Automaton()
ks=collections.defaultdict(list)
for i,k in corp: ks[k].append(i)
for k in ks: A.add_word(k,k)
A.make_automaton()
for i,k in corp:
    for _,sub in A.iter(k):
        if sub!=k and len(sub)>=3:
            for j in ks[sub]:
                if share(i,j): union(i,j,'R3 회사명 포함+같은 상표')
# 규칙 4: 대표자명
def clean_rep(r): r=re.sub(r'\s','',str(r)); return '' if r in ('','추후제출') else r
rep=collections.defaultdict(list)
for i,(r,kd) in enumerate(zip(own.rep,own.kind)):
    r=clean_rep(r)
    if len(r)>=2 and kd=='법인': rep[r].append(i)
for r,v in rep.items():
    for a in range(len(v)):
        for b in range(a+1,min(len(v),a+30)):
            if share(v[a],v[b]): union(v[a],v[b],'R4 같은 대표자+같은 상표')
indiv=collections.defaultdict(list)
for i,(k,kd) in enumerate(zip(own.name,own.kind)):
    if kd=='개인': indiv[re.sub(r'\s','',str(k))].append(i)
for r,v in rep.items():
    for y in v:
        for x in indiv.get(r,[]):
            if share(y,x): union(y,x,'R4 대표자=개인+같은 상표')
# 규칙 5: 3글자 이상 상표명을 2개 이상 똑같이 가진 출원인끼리
mk=collections.defaultdict(set)
for c,n in zip(pool_cust,p_nm):
    if c in idx and len(n)>=3: mk[n].add(idx[c])
for c,n in zip(lab_cust,Lr.nm.values):
    if c in idx and len(n)>=3: mk[n].add(idx[c])
pair=collections.Counter()
for n,s in mk.items():
    if len(s)==2:
        s=sorted(s)
        for a in range(len(s)):
            for b in range(a+1,len(s)): pair[(s[a],s[b])]+=1
for (a,b),c in pair.items():
    if c>=2: union(a,b,'R5 같은 상표 2개 이상')
print(dict(cnt))
gid={c:find(i) for c,i in idx.items()}
pg=np.array([gid.get(c,-1-k) for k,c in enumerate(pool_cust)])
lg=np.array([gid.get(c,-10_000_000-k) for k,c in enumerate(lab_cust)])
pickle.dump({'p_grp':pg,'l_grp':lg},open('owner_grp.pkl','wb'))
sz=collections.Counter(find(i) for i in range(len(own))); print('owners',len(own),'groups',len(sz),'largest',sz.most_common(3))
big=sz.most_common(1)[0][0]; print('largest group sample',own.name.values[[i for i in range(len(own)) if find(i)==big][:15]].tolist())
