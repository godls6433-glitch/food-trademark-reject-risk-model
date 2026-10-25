import pandas as pd, re
from collections import Counter
from kiwipiepy import Kiwi
d=pd.concat([pd.read_pickle(f) for f in ['tm_full.pkl','p_full_dedup.pkl','d_full.pkl']],ignore_index=True)
d=d[d['상품분류'].isin(['29','30','32','43'])]          # 단일 류(식음료)만 → 다른 류 상품이 섞이지 않게
print('single-class marks',len(d))
items=Counter()
for g in d['지정상품'].dropna():
    for it in re.split(r'[|;]', g):
        it=re.sub(r'\(.*?\)|\[.*?\]','',it); it=re.sub(r'[^가-힣]','',it)
        if 1<=len(it)<=8: items[it]+=1
kiwi=Kiwi(); nouns=Counter()
for w,c in items.most_common(10000):
    if c<5: break
    for t in kiwi.tokenize(w):
        if t.tag in('NNG','NNP') and 1<=len(t.form)<=6: nouns[t.form]+=c
stop=set(open('build_goods_words.py',encoding='utf-8').read().split("stop = set('''")[1].split("'''")[0].split())
manual={}
exec(open('build_goods_words.py',encoding='utf-8').read().split('manual = ')[1].split('\n}\n')[0].join(['manual = ','\n}\n']) if False else '')
src=open('build_goods_words.py',encoding='utf-8').read(); ns={}; exec('manual = '+src.split('manual = ')[1].split('\n}')[0]+'\n}',ns); manual=ns['manual']
modifier=re.compile(r'.+(된|하는|위한|않은|있는)|.*(가공한|조리한|건조한|냉동한|가열한|발효한)')     # '건조된미역' 같은 수식형 품목 제외(된장은 유지)
manual['상품명']+=' 탕후루 크로플 크루키 약과 쫀득쿠키 휘낭시에 까눌레 스콘 브라우니 그릭요거트 하이볼 칵테일 커피빈 콜드브루 수제버거 닭강정 꿔바로우 훠궈 쌀국수 포케 샐러드 브런치'
rows=[]
for cat,ws in manual.items():
    for w in ws.split(): rows.append((w,cat,'직접 추가'))
for w,c in items.most_common():
    if c>=30 and 2<=len(w)<=6 and w not in stop and not modifier.match(w): rows.append((w,'상품명','지정상품 원문'))
for w,c in nouns.most_common():
    # 명사 조각은 그 자체가 지정상품 품목으로도 10번 이상 쓰인 것만 (유지·수유 같은 조각 제외)
    if c>=50 and len(w)>=2 and w not in stop and items.get(w,0)>=10: rows.append((w,'상품명','지정상품 명사(Kiwi)'))
g=pd.DataFrame(rows,columns=['word','category','source']).drop_duplicates('word')
g['one_char']=(g.word.str.len()==1).astype(int)
g.to_csv('goods_words_v2.csv',index=False,encoding='utf-8-sig')
old=set(pd.read_csv('goods_words.csv').word); new=set(g.word)
print('v2',len(g),g.category.value_counts().to_dict(),g.source.value_counts().to_dict())
print('removed',len(old-new),'added',len(new-old))
print('removed sample',sorted(old-new)[:60]); print('added sample',sorted(new-old)[:60])
print('check', {w:(w in new) for w in ['수유','유지','김치','라떼','감귤','녹차','커피','버스','베레모','된장','국밥','김밥','어묵','마라탕','탕후루','크로플','소금빵','두바이']})
