"""Construct the dated-event sensitivity cohort; mailing does not identify first public access."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
ROOT=Path.cwd()
OUT=Path(__file__).resolve().parent/'artifacts'
cache=Path(sys.argv[1])
d=pd.read_parquet(cache/'rows.parquet')
files=sorted(next(ROOT.glob('02_*')).rglob('tm_pool_*.csv'))
r=pd.concat([pd.read_csv(p,usecols=['순번','출원(국제등록)번호','등록공고번호','등록일자','법적상태','등록번호'],dtype=str,keep_default_na=False) for p in files])
z=d.merge(r,on='순번',how='left',validate='1:1')
assert z.app.equals(z['출원(국제등록)번호']) and z['순번'].equals(d['순번'])
assert (z.y.eq(1)==z['법적상태'].eq('거절')).all()
r=pd.read_csv(next(ROOT.glob('02_*/reject_reasons.csv')),dtype=str,keep_default_na=False)
dt='거절결정서_발송일'
r[dt]=pd.to_datetime(r[dt],format='%Y%m%d',errors='raise')
z['mail']=pd.to_datetime(z.app.map(r.groupby('출원번호')[dt].max()))
filed=pd.to_datetime(z.date,format='%Y%m%d')
reg=pd.to_datetime(z['등록일자'],format='%Y.%m.%d',errors='coerce')
year=pd.to_numeric(z['등록공고번호'].str.extract(r'^\d{2}((?:19|20)\d{2})\d{7}$',expand=False),errors='coerce')
early=filed.le('2022-12-31')
registered=z['법적상태'].isin(['등록','소멸']) & z['등록번호'].str.fullmatch(r'\d+') & reg.between(filed,'2022-12-31') & year.between(reg.dt.year,2022)
refused=z.y.eq(1) & z.mail.between(filed,'2022-12-31')
eligible=early & (registered|refused)
assert len(d)==257400 and (~early).sum()==65590 and early.sum()==191810
np.save(OUT/'eligible.npy',eligible)
print('train label counts before/after',d[early].y.value_counts().to_dict(),d[eligible].y.value_counts().to_dict())
print('send-file',len(r),'dated apps',r['출원번호'].nunique(),'early rejects with date',int((early & z.y.eq(1) & z.mail.notna()).sum()))
print('undated/late',int((early & z.y.eq(1)&z.mail.isna()).sum()),int((early & z.y.eq(1)&z.mail.gt('2022-12-31')).sum()))
audit=z.loc[early,['순번','app','date','y','등록공고번호','등록일자','mail']].copy()
audit['eligible']=eligible[early]
audit.to_parquet(OUT/'eligibility_audit.parquet',index=False)
print(pd.crosstab(z.loc[eligible,'fold'],z.loc[eligible,'y']))
