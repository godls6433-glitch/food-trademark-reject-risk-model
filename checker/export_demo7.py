import pandas as pd, numpy as np, re, pickle, json, os
src=open('demo_data.py').read(); exec(src[:src.index('# 라벨 쪽')])     # Pf 재구성 (선행상표 풀, prep과 같은 순서)
Pf=Pf[['상표명칭','상품분류','status','출원(국제등록)일자','regdate','nm']].copy()
import gc; del U,X,T,P,D,old,okk,G; gc.collect()
CS=pickle.load(open('core_str.pkl','rb'))['p_core_str']; CT=pickle.load(open('core_translit.pkl','rb'))
G=pickle.load(open('prep.pkl','rb')); P2=pickle.load(open('prep2.pkl','rb')); TP=pickle.load(open('translit_prep.pkl','rb')); TK=pickle.load(open('tok_pool.pkl','rb'))
FOOD=sorted(pickle.load(open('prep_v3.pkl','rb'))['FOOD']); fid={c:i for i,c in enumerate(FOOD)}
alive=Pf.status.isin(['출원','공고','등록']).values
idx=np.where(alive)[0]; print('alive pool',len(idx))
nm=Pf.nm.values; jm=np.array(G['p_jm'],dtype=object); cj=np.array(G['p_cj'],dtype=object)
rows=[]
st_code={'출원':0,'공고':1,'등록':2}
for j in idx:
    name=str(Pf['상표명칭'].iat[j]).replace('\t',' ').replace('\n',' ')
    pc=P2['p_codes'][j]; food=[str(fid[c]) for c in pc if c in fid]
    c=CS[j] if CS[j]!=nm[j] else ''
    rows.append('\t'.join([name, str(Pf['상품분류'].iat[j] or ''), str(st_code[Pf.status.iat[j]]), Pf['출원(국제등록)일자'].iat[j].replace('.',''),
        '1' if pd.notna(Pf.regdate.iat[j]) else '0', ','.join(food), str(len(pc)), str(int(G['p_bits'][j])), c,
        ' '.join(sorted(TK[j])), CT['p_rom'][j], CT['p_en'][j]]))
os.makedirs('demo_files',exist_ok=True)
# 크기 15MB 이하로 나누기
parts=[]; buf=[]; size=0; k=0
for r in rows:
    b=len(r.encode())+1
    if size+b>14_500_000:
        open(f'demo_files/pool_{k}.tsv','w').write('\n'.join(buf)); parts.append(f'pool_{k}.tsv'); k+=1; buf=[]; size=0
    buf.append(r); size+=b
open(f'demo_files/pool_{k}.tsv','w').write('\n'.join(buf)); parts.append(f'pool_{k}.tsv')
for p in parts: print(p, round(os.path.getsize('demo_files/'+p)/1e6,1),'MB')
json.dump(parts,open('demo_files/parts.json','w'))
