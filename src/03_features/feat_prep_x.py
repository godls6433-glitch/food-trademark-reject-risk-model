"""상표 거절 위험 예측 — 변수 계산 (비교 기준 3가지 동시 계산)
기준 A: 같은 류만 / 기준 B: 4개 류 통합 / 기준 C: 유사군코드가 하나라도 겹치는 상표만
"""
import pandas as pd, numpy as np, re, time, sys
from rapidfuzz import fuzz, process
from jamo import h2j
from scipy import sparse

t0 = time.time()
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None      # 시험용: 라벨 일부만
CH = 200

# ---------- 1. 데이터 ----------
T = pd.read_pickle('tm_full.pkl').drop(columns=['src_end', '출원공고번호']).reset_index(drop=True)
T['순번'] = np.arange(1, len(T) + 1); T['src'] = 'T'
P = pd.read_pickle('p_full_dedup.pkl'); P['src'] = 'P'
D = pd.read_pickle('d_full.pkl').drop(columns=['src_end']); D['src'] = 'D'
G23=set("G020101+G020102+G0202+G0204+G0207+G0208+G0211+G0301+G0302+G0406+G0502+G0602+G0702+G0704+G0707+G0801+G0802+G0803+G1001+G100401+G1006+G1009+G4601".split('+'))
old=pd.concat([T,P,D]); key=lambda d: d['출원(국제등록)번호'].astype(str)+'|'+d['상품분류'].fillna('')+'|'+d['상표명칭'].fillna('')
okk=set(key(old))
X=pd.concat([pd.read_pickle('N_full.pkl'),pd.read_pickle('E_full.pkl')],ignore_index=True).drop(columns=['src_end'])
X=X[~key(X).isin(okk)]
X=X[~X.drop(columns=['순번']).duplicated()]
X=X[X['유사군코드'].fillna('').str.split('|').apply(lambda s: bool(set(s)&G23))]
X['src']='X'
U = pd.concat([T, P, D, X], ignore_index=True)
U['status'] = U['법적상태'].str.replace(r'<.*', '', regex=True).str.strip()
U['date'] = pd.to_datetime(U['출원(국제등록)일자'], format='%Y.%m.%d')
U['regdate'] = pd.to_datetime(U['등록일자'], format='%Y.%m.%d', errors='coerce')
ap = U.iloc[:, 5].fillna('').str.split('|').str[0].str.split('/')
U['cust'] = ap.str[2].fillna('')
U['corp'] = ap.str[3].fillna('').str.contains('법인')
corename = ap.str[1].fillna('').str.replace(r'주식회사|유한회사|\(주\)|㈜|농업회사법인|영농조합법인|협동조합|\(유\)|\s', '', regex=True).str.lower()
U['core_co'] = np.where(U['corp'], corename, '')
U['rep'] = np.where(U['corp'], ap.str[5].fillna('').str.strip(), '')

# ---------- 2. 상표명 정규화·주요부 ----------
def norm(s):
    s = str(s); o = re.sub(r'\(.*?\)', '', s)
    c = lambda x: re.sub(r'[^0-9a-zA-Z가-힣]', '', x).lower()
    return c(o) or c(' '.join(re.findall(r'\((.*?)\)', s)))
U['nm'] = U['상표명칭'].apply(norm)
gw = pd.read_csv('goods_words_v2.csv'); pl = pd.read_csv('place_dict_v2.csv')
gw = gw[gw.ambiguous == 0]; places = pl[(pl.ambiguous == 0)].place.str.lower().tolist()
one = set(gw[gw.one_char == 1].word)
goods = set(gw[gw.one_char == 0].word)
quality = set(gw[gw.category.isin(['품질표현'])].word); biz = set(gw[gw.category == '업종명'].word)
rx_all = re.compile('|'.join(map(re.escape, sorted(goods | set(places), key=len, reverse=True))))
rx_goods = re.compile('|'.join(map(re.escape, sorted(goods, key=len, reverse=True))))
rx_place = re.compile('|'.join(map(re.escape, sorted(set(places), key=len, reverse=True))))
def strip_desc(s):
    r = rx_all.sub('', s)
    if r and r[-1] in one: r = r[:-1]
    if r in one: r = ''
    return r
U['rest'] = U['nm'].apply(strip_desc)
# 영문 설명어(상품어·품질어·관사 등)도 주요부에서 제외
import pickle as _pk
_E=_pk.load(open('en_dict.pkl','rb')); _GE,_Q,_F=_E['GOODS_EN'],_E['QUAL'],_E['FILL']
def _stem(t):
    for suf in ('ies','es','s'):
        if t.endswith(suf) and len(t)-len(suf)>=3: return t[:-len(suf)]+('y' if suf=='ies' else '')
    return t
def _toks(name):
    s=str(name); o=re.sub(r'\(.*?\)','',s)
    base=o if re.sub(r'[^0-9a-zA-Z가-힣]','',o) else ' '.join(re.findall(r'\((.*?)\)',s))
    base=re.sub(r'([a-z])([A-Z])',r'\1 \2',base)
    return [t.lower() for t in re.findall(r'[A-Za-z]+',base)]
def _strip_en(rest,name):
    tk=_toks(name)
    hit=[t for t in tk if (_stem(t) in _GE or t in _GE or _stem(t) in _Q or t in _Q)]
    if not hit: return rest
    for t in sorted(set(hit+[t for t in tk if t in _F]),key=lambda z:(-len(z),z)): rest=rest.replace(t,'',1)
    return rest
U['rest'] = [_strip_en(r,n) for r,n in zip(U['rest'].values,U['상표명칭'].values)]
L = U['nm'].str.len().replace(0, np.nan)
U['desc_ratio'] = (1 - U['rest'].str.len() / L).fillna(0)
cov = lambda rx: U['nm'].apply(lambda s: min(1.0, sum(map(len, rx.findall(s))) / len(s)) if s else 0)
U['goods_word_ratio'] = cov(rx_goods); U['place_ratio'] = cov(rx_place)
U['core'] = np.where(U['rest'].str.len() >= 2, U['rest'], U['nm'])
U['jm'] = U['nm'].apply(lambda s: ''.join(h2j(s)))
U['cj'] = U['core'].apply(lambda s: ''.join(h2j(s)))
U['core_id'] = U['core'].astype('category').cat.codes
U['cust_id'] = pd.factorize(U['cust'])[0]
U['co_id'] = np.where(U['core_co'] != '', pd.factorize(U['core_co'])[0], -1)
U['rep_id'] = np.where(U['rep'] != '', pd.factorize(U['rep'])[0], -1)
print('normalize', round(time.time() - t0), 's', flush=True)

# ---------- 3. 류·유사군코드 ----------
bit = {'29': 1, '30': 2, '32': 4, '43': 8}
U['cls_bits'] = U['상품분류'].fillna('').apply(lambda x: sum(bit.get(c, 0) for c in x.split('|')))
codes = U['유사군코드'].fillna('').str.split('|')
vocab = {c: i for i, c in enumerate(sorted({c for cs in codes for c in cs if c}))}
r_, c_ = [], []
for i, cs in enumerate(codes):
    for c in cs:
        if c: r_.append(i); c_.append(vocab[c])
CODE = sparse.csr_matrix((np.ones(len(r_), dtype=np.float32), (r_, c_)), shape=(len(U), len(vocab)))
U['has_code'] = np.asarray(CODE.sum(1)).ravel() > 0

# ---------- 4. 비교 대상(풀) ----------
pool_ok = ((U.src == 'T') & U.status.isin(['출원', '공고', '등록', '소멸'])) | (U.src == 'P') | ((U.src == 'D') & U.regdate.notna()) | ((U.src=='X') & (U.status.isin(['출원','공고','등록']) | ((U.status=='소멸') & U.regdate.notna())))
pool_ok &= U['nm'].str.len() > 0
PI = np.where(pool_ok)[0]
Pf = U.iloc[PI].reset_index(drop=True)
p_nm, p_jm, p_cj = Pf.nm.tolist(), Pf.jm.tolist(), Pf.cj.tolist()
p_len = Pf.nm.str.len().values; p_clen = Pf.core.str.len().values
p_date = Pf.date.values.astype('datetime64[D]')
p_alive_until = np.where((Pf.src.values == 'D') | ((Pf.src.values=='X') & (Pf.status.values=='소멸')), (Pf.regdate + pd.DateOffset(years=10)).values.astype('datetime64[D]'), np.datetime64('2100-01-01'))
p_cust = Pf.cust_id.values; p_co = Pf.co_id.values; p_rep = Pf.rep_id.values
p_bits = Pf.cls_bits.values; p_hascode = Pf.has_code.values; p_core = Pf.core_id.values
p_name = Pf['상표명칭'].values
CODE_P = CODE[PI].T.tocsc()
p_order = np.argsort(p_date, kind='stable'); p_date_sorted = p_date[p_order]
print('pool', len(Pf), round(time.time() - t0), 's', flush=True)

# ---------- 5. 라벨 ----------
lab = pd.read_pickle('label_folds.pkl')
Lr = U[U.src == 'T'].set_index('순번').loc[lab['순번']].reset_index()
Lr = Lr.join(lab[['registered', 'fold']].reset_index(drop=True))
if LIMIT: Lr = Lr.sort_values('date').tail(LIMIT).reset_index(drop=True)
LI = U.index[(U.src == 'T')].values[Lr['순번'].values - 1]
CODE_L = CODE[LI]
Lr = Lr.sort_values('date').reset_index(drop=True)
LI = U.index[(U.src == 'T')].values[Lr['순번'].values - 1]
CODE_L = CODE[LI]
n = len(Lr)
crit = ['A', 'B', 'C']
Pf_codes = Pf['유사군코드'].fillna('').tolist()
F = {f'{k}_{c}': np.zeros(n, dtype=np.float32) for c in crit for k in ['max_char_sim', 'max_phon_sim', 'max_part_sim', 'core_sim', 'n_similar', 'n_contain', 'core_hits']}
BEST = {c: np.empty(n, dtype=object) for c in crit}


import pickle
lr_cols = ['순번','출원(국제등록)번호','상품분류','상표명칭','registered','fold','desc_ratio','goods_word_ratio','place_ratio','명칭구분','nm','jm','cj','core','core_id','date','cust_id','co_id','rep_id','cls_bits','has_code','유사군코드']
Lr = Lr[lr_cols]
pickle.dump({k: globals()[k] for k in ['Lr', 'n', 'crit', 'CH', 'p_nm', 'p_jm', 'p_cj', 'p_len', 'p_clen', 'p_date', 'p_alive_until', 'p_cust', 'p_co', 'p_rep', 'p_bits', 'p_hascode', 'p_core', 'p_name', 'CODE_P', 'CODE_L', 'p_order', 'p_date_sorted', 'Pf_codes']}, open('prep.pkl','wb'), protocol=4)
pickle.dump({'p_regdate': Pf.regdate.values.astype('datetime64[D]'), 'p_isD': ((Pf.src=='D')|((Pf.src=='X')&(Pf.status=='소멸'))).values,
 'p_corp': Pf.corp.values, 'p_codes': [frozenset(x.split('|')) if isinstance(x,str) else frozenset() for x in Pf['유사군코드']], 'p_src': Pf.src.values}, open('prep2.pkl','wb'), protocol=4)
pickle.dump({'p_core_str':Pf['core'].tolist(),'l_core_str':Lr['core'].tolist()},open('core_str.pkl','wb'),protocol=4)
print('prep saved', round(time.time()-t0),'s', 'pool', len(Pf), Pf.src.value_counts().to_dict())
