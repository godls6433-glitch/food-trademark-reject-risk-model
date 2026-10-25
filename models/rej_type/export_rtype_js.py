# 점검기용 경량 거절 유형 모델: RF 100그루(유사·식별력 각각), 같은 5-fold에서 성능 확인 후 전체로 학습해 JSON 내보내기
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
FEAT, REJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
f = pd.read_csv(FEAT, encoding='utf-8-sig'); r = pd.read_csv(REJ, encoding='utf-8-sig')
d = f.merge(r[['app_no','reason']], left_on='출원(국제등록)번호', right_on='app_no')
d = d[(d.registered == 0) & d.reason.isin(['유사','식별력','복합'])].reset_index(drop=True)
base = ['place_ratio','logo','name_len','cls_29','cls_30','cls_32','cls_43']
for t in ['한글상표','영문상표','복합문자','도형복합']: d['nt_'+t] = (d.name_type == t).astype(int); base.append('nt_'+t)
cols = base + ['desc_ratio2','goods_ratio2','own_goods_ratio','other_goods_ratio','core_sim','core_hits','core_n80','core_part_sim','core_n_contain','core_translit_sim','top_reg_c','top_code_jacc_c','tok_hits','tok_hits_reg','tok_max_idf','tok_share','self_reg_core_sim']
X = d[cols].astype(float).fillna(0).values
ys = d.reason.isin(['유사','복합']).astype(int).values; yd = d.reason.isin(['식별력','복합']).astype(int).values
yc = d.reason.map({'유사':0,'식별력':1,'복합':2}).values; fold = d.fold.values
TS, TD = 0.425, 0.475
mk = lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features='sqrt', max_depth=14, class_weight='balanced', n_jobs=-1, random_state=42)
to_type = lambda ps, pd_: np.where((ps >= TS) & (pd_ >= TD), 2, np.where(ps - TS >= pd_ - TD, 0, 1))
ps = np.zeros(len(d)); pdd = np.zeros(len(d))
for k in range(1, 6):
    tr = fold != k; te = fold == k
    ps[te] = mk().fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]; pdd[te] = mk().fit(X[tr], yd[tr]).predict_proba(X[te])[:, 1]
pr = to_type(ps, pdd)
print('lite acc', round(accuracy_score(yc, pr), 4), 'macroF1', round(f1_score(yc, pr, average='macro'), 4), 'auc', round(roc_auc_score(ys, ps), 4), round(roc_auc_score(yd, pdd), 4))
def flat(m):
    off = [0]; F = []; T = []; L = []; R = []; V = []
    for e in m.estimators_:
        t = e.tree_; F += t.feature.clip(0).tolist(); T += t.threshold.tolist(); L += t.children_left.tolist(); R += t.children_right.tolist()
        v = t.value[:, 0, :]; V += np.round(v[:, 1] / v.sum(1), 6).tolist(); off.append(off[-1] + t.node_count)
    return {'off': off, 'F': F, 'T': T, 'L': L, 'R': R, 'V': V}
out = {'cols': cols, 'thr_sim': TS, 'thr_dist': TD, 'sim': flat(mk().fit(X, ys)), 'dist': flat(mk().fit(X, yd)),
       'cv': {'acc': round(accuracy_score(yc, pr), 4), 'macro_f1': round(f1_score(yc, pr, average='macro'), 4)}}
json.dump(out, open(OUT, 'w'), separators=(',', ':'))
# 브라우저 계산 검증용 표본
d.assign(ps=ps, pd=pdd).head(0)
