# 영문 설명어 사전 + 식별력 변수 v2 + 지정상품-상표명 관계 변수
import pandas as pd, numpy as np, re, pickle, collections
pairs=pickle.load(open('goods_pairs_13.pkl','rb'))
STOP={'for','of','and','in','not','with','based','use','as','or','a','an','the','from','other','than','being','used','made','making','consisting','containing','purposes','services','service','providing','rental','management','facilities','information','temporary','accommodation','consumption','human','animal','animals','live','preparation','preparations','mainly','principally','predominating','by','on','into','to','be','its','their','such','non','pre','type','types','etc','products','goods','included','including','others','agents','additives','industrial','medical','pharmaceutical','purpose','supply','supplying','reservation','reservations','booking','bookings','boarding','residential','house','houses','hotel','hotels','persons','person','child','children','elderly','care','day','nursery','camps','camp','rest','homes','home'}
def stem(t):
    for suf in ('ies','es','s'):
        if t.endswith(suf) and len(t)-len(suf)>=3: return t[:-len(suf)]+('y' if suf=='ies' else '')
    return t
en2ko=collections.defaultdict(set)
for ko,en in pairs:
    for t in re.findall(r'[a-z]+',en):
        if len(t)>=3 and t not in STOP: en2ko[stem(t)].add(ko)
GOODS_EN=set(en2ko)
GOODS_EN|={stem(x) for x in ['cafe','coffee','bakery','bake','baker','deli','kitchen','bistro','grill','bar','pub','restaurant','dining','diner','eatery','brewery','brew','food','foods','burger','pizza','chicken','donut','doughnut','bagel','waffle','toast','pasta','sushi','taco','ramen','noodle','bbq','steak','tea','juice','smoothie','beer','wine','soju','makgeolli','kimchi','oat','granola','cereal','snack','dessert','cake','cookie','bread','pie','tart','muffin','croissant','macaron','gelato','icecream','cream','yogurt','milk','cheese','butter','chocolate','candy','jelly','honey','jam','sauce','spice','salt','sugar','oil','rice','meat','pork','beef','fish','seafood','salad','soup','sandwich','water','soda','cola','latte','espresso','mocha','matcha','wrap','roll','bowl','bento','lunch','brunch','dinner','meal','market','shop','store','mart','house','table','lab','factory','farm','company','co','corp','brand','gourmet','catering']}
QUAL={stem(x) for x in ['premium','fresh','organic','natural','best','good','great','tasty','delicious','yummy','healthy','health','pure','real','true','original','classic','special','signature','homemade','handmade','home','daily','simple','simply','easy','quick','fast','hot','cold','sweet','spicy','crispy','crunchy','soft','rich','mild','light','zero','free','sugarfree','low','high','extra','super','mega','mini','big','new','fine','prime','golden','gold','royal','grand','vital','vegan','plant','bio','eco','green','clean','well','wellness','protein','diet','fit','energy','power','vitamin','gluten','instant','frozen','dried','roasted','smoked','grilled','fried','baked','cooked','raw','whole','mix','plus','pro','max','choice','select','selected','finest','tradition','traditional','korean','korea','local','world','global','international','grade','quality','authentic','famous','favorite','amazing','perfect','happy','love','lovely','yum','delight','taste','flavor','flavour','recipe','style','story','time','day','life']}
FILL={'the','a','an','our','my','your','you','we','i','and','of','in','with','for','to','by','on','at','it','is','its','all','one','me','us','le','la','de','du','des','el','los','las','das','der','die'}
pickle.dump(dict(en2ko=dict(en2ko),GOODS_EN=GOODS_EN,QUAL=QUAL,FILL=FILL),open('en_dict.pkl','wb'))
print('영문 상품어',len(GOODS_EN),'품질어',len(QUAL))

G=pickle.load(open('prep.pkl','rb')); Lr=G['Lr']
T=pd.read_pickle('tm_full.pkl').reset_index(drop=True); T['순번']=np.arange(1,len(T)+1)
goods_txt=T.set_index('순번').loc[Lr['순번'].values,'지정상품'].fillna('').values
gw=pd.read_csv('goods_words_v2.csv'); pl=pd.read_csv('place_dict_v2.csv')
gw=gw[gw.ambiguous==0]; places=pl[pl.ambiguous==0].place.str.lower().tolist()
one=set(gw[gw.one_char==1].word); goods=set(gw[gw.one_char==0].word); qual_ko=set(gw[gw.category=='품질표현'].word)
rx_all=re.compile('|'.join(map(re.escape,sorted(goods|set(places),key=len,reverse=True))))
rx_g=re.compile('|'.join(map(re.escape,sorted(goods-qual_ko,key=len,reverse=True))))
def tokens(name):
    s=str(name); o=re.sub(r'\(.*?\)','',s)
    base=o if re.sub(r'[^0-9a-zA-Z가-힣]','',o) else ' '.join(re.findall(r'\((.*?)\)',s))
    base=re.sub(r'([a-z])([A-Z])',r'\1 \2',base)
    return [t.lower() for t in re.findall(r'[A-Za-z]+',base)]
rows=[]
is43=Lr['상품분류'].fillna('').str.split('|').apply(lambda c:'43' in c).values
for nm,name,gtxt,c43 in zip(Lr.nm.values,Lr['상표명칭'].values,goods_txt,is43):
    n=len(nm) or 1
    # 한글: 기존 규칙
    r=rx_all.sub('',nm)
    if r and r[-1] in one: r=r[:-1]
    if r in one: r=''
    ko_cov=len(nm)-len(r)
    kg=rx_g.findall(nm); ko_own=sum(len(w) for w in kg if (w in gtxt or c43)); ko_other=sum(len(w) for w in kg if not (w in gtxt or c43))
    # 영문
    tk=tokens(name); en_g=en_q=en_f=0; en_own=en_other=0
    for t in tk:
        st=stem(t)
        if st in GOODS_EN or t in GOODS_EN:
            en_g+=len(t)
            kos=en2ko.get(st,set())
            if c43 or (kos and any(k in gtxt for k in kos)): en_own+=len(t)
            elif kos: en_other+=len(t)
            else: en_own+=len(t)          # 업종어(cafe, bakery 등)는 자기 업종으로 봄
        elif st in QUAL or t in QUAL: en_q+=len(t)
        elif t in FILL: en_f+=len(t)
    en_cov=en_g+en_q+(en_f if (en_g+en_q)>0 else 0)
    rows.append((min(1,(ko_cov+en_cov)/n), min(1,(sum(map(len,kg))+en_g)/n), (ko_own+en_own)/n, (ko_other+en_other)/n, en_cov/n))
F=pd.DataFrame(rows,columns=['desc_ratio2','goods_ratio2','own_goods_ratio','other_goods_ratio','en_desc_ratio']).clip(upper=1)
F['has_own_goods']=(F.own_goods_ratio>0).astype(int); F['has_other_goods']=(F.other_goods_ratio>0).astype(int)
F.insert(0,'순번',Lr['순번'].values.astype(int)); F.to_pickle('feat_desc2.pkl')
y=1-Lr.registered.values
print(F.describe().T.round(3).to_string())
for c in ['has_own_goods','has_other_goods']: print(c,'비율',F[c].mean().round(3),'거절률',y[F[c]==1].mean().round(3),'vs',y[F[c]==0].mean().round(3))
ex=Lr['상표명칭'].values
for q in ['OUR Bakery','SIMPLY COOK','YOU NEED CAKE','THE BAKE Deli','A!MAZING OAT','bonaparte','MAYNARDS']:
    i=np.where(ex==q)[0][:1]
    if len(i): print(q,F.iloc[i[0]][['desc_ratio2','own_goods_ratio','other_goods_ratio']].round(2).to_dict())
