import pandas as pd, numpy as np, re
from sklearn.model_selection import StratifiedGroupKFold
d=pd.read_pickle('tm_full.pkl').drop(columns=['src_end','출원공고번호']).reset_index(drop=True)
d['순번']=np.arange(1,len(d)+1)
st=d['법적상태'].str.replace(r'<.*','',regex=True).str.strip()
yr=d['출원(국제등록)일자'].str[:4].astype(int)
lab=d[(yr<=2024)&st.isin(['등록','공고','소멸','거절'])&d['공존동의상표'].isna()].copy()
lab['registered']=(st[lab.index]!='거절').astype(int)
# 최근 3개월 거절결정 제외 (거절결정서 발송일을 아는 건만)
r=pd.read_csv('rej_reasons_v2.csv',dtype=str)[['app_no','reject_send_date']].dropna().drop_duplicates('app_no')
r['sd']=pd.to_datetime(r['reject_send_date'].astype(str).str[:8],format='%Y%m%d',errors='coerce')
recent=set(r.loc[r.sd>='2026-07-01','app_no'])
n0=len(lab); lab=lab[~((lab.registered==0)&lab['출원(국제등록)번호'].isin(recent))]
print('recent reject removed',n0-len(lab))
def norm(s):
    o=re.sub(r'\(.*?\)','',str(s)); c=lambda x: re.sub(r'[^0-9a-zA-Z가-힣]','',x).lower()
    return c(o) or c(' '.join(re.findall(r'\((.*?)\)',str(s))))
cust=lab.iloc[:,5].str.split('/').str[2].fillna('')
lab['group']=(cust+'|'+lab['상표명칭'].apply(norm)).astype('category').cat.codes
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
lab['fold']=0
for k,(_,te) in enumerate(sgkf.split(lab,lab.registered,lab.group),1): lab.iloc[te,lab.columns.get_loc('fold')]=k
print(len(lab),'groups',lab.group.nunique(),'reject rate',round(1-lab.registered.mean(),4))
print(lab.groupby('fold').agg(n=('fold','size'),reject=('registered',lambda v:round(1-v.mean(),4))))
print('groups split across folds',(lab.groupby('group').fold.nunique()>1).sum())
out=lab[['순번','출원(국제등록)번호','상품분류','출원(국제등록)일자','상표명칭','registered','fold']].rename(columns={'출원(국제등록)번호':'출원번호','출원(국제등록)일자':'출원일자'})
out.to_csv('label_folds.csv',index=False,encoding='utf-8-sig'); out.to_pickle('label_folds.pkl')
