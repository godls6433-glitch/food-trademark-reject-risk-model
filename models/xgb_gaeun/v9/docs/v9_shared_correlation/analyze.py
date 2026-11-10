"""Usage: python analyze.py PATH_TO_VERSION_DIRECTORIES OUTPUT_DIRECTORY"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def main(root, output):
    v6 = root / 'trademark_risk_v6/artifacts'
    v9 = root / 'trademark_risk_v9/artifacts'
    cache = v6 / 'text_cache'
    meta = json.loads((cache / 'complete.json').read_text())
    frozen = json.loads((v9 / 'frozen.json').read_text())
    cohort_path = v6 / 'cohort.parquet'
    search_path = v9 / 'training_features.parquet'
    for path, expected in [(cohort_path, meta['development_apps_sha']),
                           (search_path, frozen['hashes'][str(search_path.relative_to(root))])]:
        with path.open('rb') as file:
            assert hashlib.file_digest(file, 'sha256').hexdigest() == expected
    cohort = pd.read_parquet(cohort_path, columns=['app', 'date', 'period'])
    cohort = cohort[cohort.period.isin(['fit', 'calibration', 'validation'])].reset_index(drop=True)
    fit = cohort.period.eq('fit')
    dates = cohort.loc[fit, 'date'].astype(str)
    assert dates.str[:4].between('2017', '2021').all() and fit.sum() == frozen['fit']
    basic = 'length hangul_ratio latin_ratio digit_ratio logo class_count goods_count class_29 class_30 class_32 class_43'.split()
    roles = 'LOCATION INGREDIENT PRODUCT SERVICE_CONTENT QUALITY EFFECT USE METHOD PRICE COMMON_EXPRESSION'.split()
    names = basic + [f'kb_{role}_context' for role in roles]
    legacy = sparse.load_npz(cache / 'legacy.npz').tocsr()
    assert legacy.shape == (len(cohort), len(meta['legacy_feature_names']))
    positions = [meta['legacy_feature_names'].index(name) for name in names]
    data = pd.DataFrame(legacy[:, positions][fit.to_numpy()].toarray(), columns=names)
    search = pd.read_parquet(search_path).set_index('app', verify_integrity=True)
    assert search.index.equals(pd.Index(cohort.app)) and cohort.app.is_unique
    prior = [c for c in search if c.startswith(('prior_', 'v7_'))]
    assert len(prior) == 32
    data = pd.concat([data, search.loc[cohort.loc[fit, 'app'], prior].astype(np.float32).reset_index(drop=True)], axis=1)
    assert data.shape == (fit.sum(), 53) and np.isfinite(data.values).all()
    output.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame({'nunique': data.nunique(), 'nonzero': data.ne(0).sum(),
                          'min': data.min(), 'max': data.max()})
    stats.to_csv(output / 'feature_summary.csv', index_label='feature')
    pearson = data.corr()
    spearman = data.corr(method='spearman')
    rows = []
    for a, b in combinations(data, 2):
        r, rho = pearson.loc[a, b], spearman.loc[a, b]
        rows.append((a, b, r, rho, int(data[a].ne(data[b]).sum()), min(abs(r), abs(rho))))
    pairs = pd.DataFrame(rows, columns=['a', 'b', 'pearson', 'spearman', 'unequal_rows', 'min_abs_corr'])
    pairs.sort_values('min_abs_corr', ascending=False).to_csv(output / 'pairs.csv', index=False)
    print(dates.str[:4].value_counts().sort_index().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
