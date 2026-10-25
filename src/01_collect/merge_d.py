import pandas as pd, glob, re
fs=glob.glob('data/raw/*D_*.xlsx')
fs=sorted(fs,key=lambda f:int(re.search(r'D_(\d+)',f).group(1)))
ref=list(pd.read_pickle('tm_full.pkl').drop(columns=['src_end']).columns)
parts=[]
for f in fs:
    end=int(re.search(r'D_(\d+)',f).group(1))
    x=pd.read_excel(f,header=7,engine='calamine',dtype=str)
    x['src_end']=end
    seq=pd.to_numeric(x['순번'],errors='coerce')
    print(end,len(x),'순번',int(seq.min()),int(seq.max()),'출원일',x['출원(국제등록)일자'].min(),x['출원(국제등록)일자'].max(),'cols_same',list(x.columns[:-1])==ref)
    parts.append(x)
d=pd.concat(parts,ignore_index=True)
seq=pd.to_numeric(d['순번']); print('total',len(d),'unique 순번',seq.nunique(),'missing',sorted(set(range(1,93293))-set(seq))[:10])
print('status',d['법적상태'].str.replace(r'<.*','',regex=True).str.strip().value_counts().to_dict())
print('dup rows',d.drop(columns=['순번','src_end']).duplicated().sum(),'dup app_no',d['출원(국제등록)번호'].duplicated().sum())
a=pd.to_datetime(d['출원(국제등록)일자'],format='%Y.%m.%d'); print('date',a.min(),a.max(),'monotonic',a.is_monotonic_increasing)
print('등록일 없음',d['등록일자'].isna().sum())
print('class',d['상품분류'].str.split('|').explode().value_counts().head(6).to_dict())
d.to_pickle('d_full.pkl')
