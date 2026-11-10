"""Build V9 text and 21 non-search numeric inputs, without outcomes."""
from pathlib import Path
import sys
import json
import unicodedata
import numpy as np
import pandas as pd
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'trademark_risk_v3/src'))
from shared import norm, goods_tokens, numeric, ROLES
OUT = Path(__file__).resolve().parent/'artifacts'
lexicon = json.loads((ROOT/'trademark_risk_v3/artifacts/safe_lexicon.json').read_text())
basic = 'length hangul_ratio latin_ratio digit_ratio logo class_count goods_count class_29 class_30 class_32 class_43'.split()
columns = basic + ['kb_'+r+'_context' for r in ROLES[:10]]
d = pd.read_parquet(OUT/'rows.parquet',columns=['name','goods','classes','logo'])
values = np.empty((len(d),21))
for i,row in enumerate(d.itertuples(index=False)):
    f,_ = numeric(row._asdict(),lexicon)
    values[i] = [f[c] for c in columns]
np.save(OUT/'structure.npy',values)
n = d.name.map(norm)
pd.DataFrame({'name':n, 'jamo':n.map(lambda s:unicodedata.normalize('NFD',s)),
              'goods':d.goods.map(lambda s:' '.join(goods_tokens(s)))}).to_parquet(OUT/'documents.parquet')
