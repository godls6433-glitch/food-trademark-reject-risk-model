# 최종 거절 유형 모델: 전체 16,834건으로 RF 두 개(유사·식별력) 학습, 5-fold에서 가장 많이 고른 설정 사용
import sys, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
FEAT, REJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
f = pd.read_csv(FEAT, encoding='utf-8-sig'); r = pd.read_csv(REJ, encoding='utf-8-sig')
d = f.merge(r[['app_no','reason']], left_on='출원(국제등록)번호', right_on='app_no')
d = d[(d.registered == 0) & d.reason.isin(['유사','식별력','복합'])]
base = ['place_ratio','logo','name_len','cls_29','cls_30','cls_32','cls_43']
for t in ['한글상표','영문상표','복합문자','도형복합']: d['nt_'+t] = (d.name_type == t).astype(int); base.append('nt_'+t)
cols = base + ['desc_ratio2','goods_ratio2','own_goods_ratio','other_goods_ratio','core_sim','core_hits','core_n80','core_part_sim','core_n_contain','core_translit_sim','top_reg_c','top_code_jacc_c','tok_hits','tok_hits_reg','tok_max_idf','tok_share','self_reg_core_sim']
X = d[cols].astype(float).fillna(0).values
mk = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42)
ms = mk().fit(X, d.reason.isin(['유사','복합'])); md = mk().fit(X, d.reason.isin(['식별력','복합']))
joblib.dump({'cols': cols, 'sim': ms, 'dist': md, 'thr_sim': 0.425, 'thr_dist': 0.475}, OUT + '/rej_type_rf.joblib', compress=3)
imp = pd.DataFrame({'변수': cols, '유사_중요도': ms.feature_importances_, '식별력_중요도': md.feature_importances_}).round(4).sort_values('유사_중요도', ascending=False)
imp.to_csv(OUT + '/feature_importance.csv', index=False, encoding='utf-8-sig'); print(imp.head(10).to_string()); print(imp.sort_values('식별력_중요도', ascending=False).head(6).to_string())
