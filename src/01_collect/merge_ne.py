import pandas as pd, glob, re, sys
from concurrent.futures import ProcessPoolExecutor
ref=list(pd.read_pickle('tm_full.pkl').drop(columns=['src_end','출원공고번호']).columns)
def rd(f):
    x=pd.read_excel(f,header=7,engine='calamine',dtype=str); return f,x
for tag in ['N','E']:
    fs=sorted(glob.glob(f'newdata/{tag}/{tag}_*.xlsx'),key=lambda f:int(re.search(r'_(\d+)\.',f).group(1)))
    with ProcessPoolExecutor(2) as ex: res=list(ex.map(rd,fs))
    parts=[]; bad=[]
    for f,x in res:
        end=int(re.search(r'_(\d+)\.',f).group(1)); seq=pd.to_numeric(x['순번'],errors='coerce')
        if list(x.columns)!=ref: bad.append((end,'cols',list(x.columns)))
        if seq.min()!=end-4999 or (seq.max()!=end and len(x)!=5000): bad.append((end,'seq',int(seq.min()),int(seq.max()),len(x)))
        x['src_end']=end; parts.append(x)
    d=pd.concat(parts,ignore_index=True); seq=pd.to_numeric(d['순번'])
    print(tag,'files',len(fs),'rows',len(d),'max 순번',int(seq.max()),'unique',seq.nunique(),'missing',len(set(range(1,int(seq.max())+1))-set(seq)))
    print(' bad',bad[:5])
    a=pd.to_datetime(d['출원(국제등록)일자'],format='%Y.%m.%d',errors='coerce')
    print(' date',a.min(),a.max(),'sorted',a.is_monotonic_increasing,'NaT',a.isna().sum())
    print(' dup app_no',d['출원(국제등록)번호'].duplicated().sum(),'dup rows',d.drop(columns=['순번','src_end']).duplicated().sum())
    print(' status',d['법적상태'].fillna('').str.replace(r'<.*','',regex=True).str.strip().value_counts().to_dict())
    print(' class',d['상품분류'].fillna('').str.split('|').explode().value_counts().head(10).to_dict())
    print(' 명칭구분',d['명칭구분'].value_counts().to_dict())
    d.to_pickle(f'{tag}_full.pkl')
