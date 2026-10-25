# v3 준비: 라벨 쪽 유사군코드를 29·30·32·43류 코드(13판 심사기준)로 제한한 코드 행렬
import pandas as pd, numpy as np, pickle, json
from scipy import sparse
G=pickle.load(open('prep.pkl','rb')); Lr=G['Lr']; Pf_codes=G['Pf_codes']
c={int(k):set(v) for k,v in json.load(open('sim_std/codes_by_class.json')).items()}
FOOD=c[29]|c[30]|c[32]|c[43]
p_sets=[frozenset(x for x in s.split('|') if x) for s in Pf_codes]
l_all=[frozenset(x for x in str(s).split('|') if x) if isinstance(s,str) else frozenset() for s in Lr['유사군코드'].values]
l_sets=[s&FOOD for s in l_all]
vocab={v:i for i,v in enumerate(sorted(set().union(*p_sets)|set().union(*l_sets)))}
def mat(sets):
    r,cc=[],[]
    for i,s in enumerate(sets):
        for x in s: r.append(i); cc.append(vocab[x])
    return sparse.csr_matrix((np.ones(len(r),dtype=np.float32),(r,cc)),shape=(len(sets),len(vocab)))
CODE_L3=mat(l_sets); CODE_P3=mat(p_sets).T.tocsc()
l_has3=np.array([len(s)>0 for s in l_sets])
print('labels losing codes:',np.mean([len(a)!=len(b) for a,b in zip(l_all,l_sets)]).round(3),'labels with no food code:',(~l_has3).mean().round(4))
pickle.dump(dict(CODE_L3=CODE_L3,CODE_P3=CODE_P3,l_has3=l_has3,l_sets=l_sets,p_sets=p_sets,FOOD=FOOD),open('prep_v3.pkl','wb'),protocol=4)
