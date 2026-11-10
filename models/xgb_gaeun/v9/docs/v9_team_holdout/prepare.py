"""Run with the original version directories as cwd; write only to this experiment."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent / 'artifacts'
TEAM = ROOT / 'trademark_xgb_team_20261012'
t = pd.read_parquet(TEAM / 'artifacts/cohort.parquet')
data = next(ROOT.glob('02_*'))
manifest = json.loads((TEAM / 'artifacts/data_audit.json').read_text())
for path, digest in manifest['exact_source_hashes_sha256'].items():
    with (ROOT / path).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
raw = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False,
                usecols=['순번','출원(국제등록)번호','지정상품','명칭구분']) for p in sorted(data.rglob('tm_pool_*.csv'))])
t = t.merge(raw, on='순번', validate='1:1')
assert t['출원번호'].equals(t['출원(국제등록)번호'])
t = t.rename(columns={'출원번호':'app','상표명칭':'name','상품분류':'classes','지정상품':'goods'})
t['date'] = t['출원일자'].str.replace('-','',regex=False)
t['logo'] = t['명칭구분'].eq('도형복합').astype(int)
t['y'] = 1-t.registered
test = t.date.ge('20230101')
assert test.sum()==65590 and (~test).sum()==191810
ids = ['순번','app','fold']
baseline = pd.read_csv(next((TEAM/'time_holdout_2022_2023_2024').glob('*65590.csv')),
                       dtype={'순번':str,'출원번호':str}).rename(columns={'출원번호':'app'})
pd.testing.assert_frame_equal(t.loc[test, ids].reset_index(drop=True), baseline[ids],check_dtype=False)
t[['순번','app','name','classes','goods','date','logo','fold','y']].fillna('').to_parquet(OUT/'rows.parquet')
