# 지정상품 → 유사군코드 대응표 (29·30·32·43류). 상품이 들어간 상표에 그 코드가 95% 이상 함께 붙어 있으면 대응으로 봄
import pandas as pd, numpy as np, json
from collections import Counter, defaultdict
FOOD=set(); c=json.load(open('sim_std/codes_by_class.json'))
for k in ['29','30','32','43']: FOOD|=set(c[k])
code_cls={x:k for k in ['29','30','32','43'] for x in c[k]}
t=pd.concat([pd.read_pickle('tm_full.pkl'),pd.read_pickle('p_full_dedup.pkl')])
t=t[pd.to_datetime(t['출원(국제등록)일자'],format='%Y.%m.%d')>='2012-01-01']
t=t[t['지정상품'].notna()&t['유사군코드'].notna()]
item_n=Counter(); co=defaultdict(Counter); icls=defaultdict(Counter)
for g,cd,cl in zip(t['지정상품'],t['유사군코드'],t['상품분류'].fillna('')):
    cs=set(cd.split('|'))&FOOD
    if not cs: continue
    for it in set(x.strip() for x in g.split('|') if x.strip()):
        item_n[it]+=1; co[it].update(cs)
        if '|' not in cl: icls[it][cl]+=1
rows=[]
for it,n in item_n.items():
    if n<5: continue
    best=[(cd,k/n) for cd,k in co[it].items() if k/n>=0.95]
    if not best: continue
    # 상품 하나에 코드가 너무 많이 붙으면(다른 상품 코드가 우연히 항상 같이 붙은 경우) 빈도 1위만
    best.sort(key=lambda z:-z[1])
    codes=[b[0] for b in best][:3]
    if icls[it] and icls[it].most_common(1)[0][0] not in ('29','30','32','43'): continue
    rows.append(dict(지정상품=it,유사군코드='|'.join(codes),류=(icls[it].most_common(1)[0][0] if icls[it] else ''),출현수=n,동반비율=round(best[0][1],3)))
m=pd.DataFrame(rows).sort_values('출현수',ascending=False)
m.to_csv('goods_code_map.csv',index=False,encoding='utf-8-sig')
print(len(m)); print(m.head(25).to_string(index=False))
for q in ['김치','김밥','커피','맥주','생수','떡볶이','한식점업','카페업','과자','라면']:
    print(q, m[m.지정상품==q][['유사군코드','류','출현수']].values.tolist())
