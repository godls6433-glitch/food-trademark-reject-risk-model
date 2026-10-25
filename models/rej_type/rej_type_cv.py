# 예상 거절 유형 보조 모델: 거절 사유가 매칭된 거절 건만으로 "유사 / 식별력" 두 라벨(다중 라벨)을 예측
# 라벨: 유사 -> 유사만, 식별력 -> 식별력만, 복합 -> 둘 다, 지정상품만(90건) -> 제외
# fold는 features_final_v8.csv의 fold 그대로, 튜닝은 학습 fold 안에서만(안쪽 4-fold = 남은 fold), random_state=42
import sys, json, itertools, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb
FEAT, REJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
f = pd.read_csv(FEAT, encoding='utf-8-sig'); r = pd.read_csv(REJ, encoding='utf-8-sig')
f = f.merge(r[['app_no', 'reason']], left_on='출원(국제등록)번호', right_on='app_no', how='left')
d = f[(f.registered == 0) & f.reason.isin(['유사', '식별력', '복합'])].reset_index(drop=True)
base = ['place_ratio','logo','name_len','cls_29','cls_30','cls_32','cls_43']
for t in ['한글상표','영문상표','복합문자','도형복합']: d['nt_'+t] = (d.name_type == t).astype(int); base.append('nt_'+t)
desc = ['desc_ratio2','goods_ratio2','own_goods_ratio','other_goods_ratio']
core = ['core_sim','core_hits','core_n80','core_part_sim','core_n_contain','core_translit_sim','top_reg_c','top_code_jacc_c']
tok = ['tok_hits','tok_hits_reg','tok_max_idf','tok_share']
cols = base + desc + core + tok + ['self_reg_core_sim']
X = d[cols].astype(float).fillna(0).values
ys = d.reason.isin(['유사','복합']).astype(int).values; yd = d.reason.isin(['식별력','복합']).astype(int).values
CL = np.array(['유사','식별력','복합']); yc = d.reason.map({'유사':0,'식별력':1,'복합':2}).values
fold = d.fold.values
print('rows', len(d), d.reason.value_counts().to_dict(), 'features', len(cols))

def to_type(ps, pd_, ts, td):
    # 둘 다 기준 이상이면 복합, 아니면 기준 대비 더 높은 쪽
    both = (ps >= ts) & (pd_ >= td)
    return np.where(both, 2, np.where(ps - ts >= pd_ - td, 0, 1))

def best_thr(ps, pd_, y):
    g = np.round(np.arange(0.2, 0.81, 0.025), 3); best = (-1, .5, .5)
    for ts in g:
        for td in g:
            s = f1_score(y, to_type(ps, pd_, ts, td), average='macro')
            if s > best[0]: best = (s, ts, td)
    return best[1], best[2]

def rf(p): return lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=p[0], max_features=p[1], class_weight='balanced', n_jobs=-1, random_state=42)
def gb(p): return lambda: lgb.LGBMClassifier(n_estimators=p[0], learning_rate=0.03, num_leaves=p[1], min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight='balanced', random_state=42, verbose=-1)
def lr(p): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=p[0], class_weight='balanced', max_iter=2000))
GRID = {'rf': (rf, list(itertools.product([5, 20, 50], ['sqrt', 0.5]))),
        'lgbm': (gb, list(itertools.product([200, 500], [15, 31]))),
        'logit': (lr, [(0.1,), (1.0,)])}

def two_label(mk, Xtr, ytr_s, ytr_d, Xte):
    return mk().fit(Xtr, ytr_s).predict_proba(Xte)[:, 1], mk().fit(Xtr, ytr_d).predict_proba(Xte)[:, 1]

res = {}; oof_all = {}
for name, (fac, grid) in GRID.items():
    ps = np.zeros(len(d)); pdd = np.zeros(len(d)); pred = np.zeros(len(d), int); chosen = []
    for k in range(1, 6):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        # 안쪽 CV: 학습 fold 4개를 하나씩 검증으로
        best = None
        for p in grid:
            ips = np.zeros(len(d)); ipd = np.zeros(len(d))
            for j in set(fold[tr]):
                a = tr[fold[tr] != j]; b = tr[fold[tr] == j]
                ips[b], ipd[b] = two_label(fac(p), X[a], ys[a], yd[a], X[b])
            sc = roc_auc_score(ys[tr], ips[tr]) + roc_auc_score(yd[tr], ipd[tr])
            if best is None or sc > best[0]: best = (sc, p, ips[tr].copy(), ipd[tr].copy())
        ts, td = best_thr(best[2], best[3], yc[tr])
        ps[te], pdd[te] = two_label(fac(best[1]), X[tr], ys[tr], yd[tr], X[te])
        pred[te] = to_type(ps[te], pdd[te], ts, td); chosen.append({'fold': k, 'param': str(best[1]), 'thr_sim': ts, 'thr_dist': td})
    oof_all[name] = (ps, pdd, pred)
    res[name] = {'params': chosen}
    print(name, chosen, flush=True)

# 기준선: 지금 점검기(index.html rtype)의 간이 규칙
s_hi = (d.core_hits > 0) | (d.core_sim >= 0.9) | ((d.tok_hits_reg > 0) & (d.tok_share >= 0.5)); d_hi = d.desc_ratio2 >= 0.5
base_pred = np.where(s_hi & d_hi, 2, np.where(s_hi, 0, np.where(d_hi, 1, -1)))

def summ(pred, ps=None, pdd=None):
    o = {'exact_acc': accuracy_score(yc, pred), 'macro_f1': f1_score(yc, pred, labels=[0,1,2], average='macro'),
         'sim_label_acc': float(np.mean((np.isin(pred, [0,2])).astype(int) == ys)),
         'dist_label_acc': float(np.mean((np.isin(pred, [1,2])).astype(int) == yd)),
         'f1_by_class': dict(zip(CL, np.round(f1_score(yc, pred, labels=[0,1,2], average=None), 3).tolist())),
         'fold_exact_acc': [round(accuracy_score(yc[fold==k], pred[fold==k]), 3) for k in range(1, 6)]}
    if ps is not None:
        o.update({'auc_sim': roc_auc_score(ys, ps), 'auc_dist': roc_auc_score(yd, pdd), 'ap_sim': average_precision_score(ys, ps), 'ap_dist': average_precision_score(yd, pdd)})
    return {a: (round(b, 4) if isinstance(b, float) else b) for a, b in o.items()}

res['baseline_rule'] = summ(base_pred); res['baseline_rule']['no_type_share'] = round(float(np.mean(base_pred == -1)), 4)
res['majority_유사'] = summ(np.zeros(len(d), int))
for name in GRID: res[name].update(summ(*[oof_all[name][2], oof_all[name][0], oof_all[name][1]]))
cm = {}
for name in ['baseline_rule'] + list(GRID):
    p = base_pred if name == 'baseline_rule' else oof_all[name][2]
    cm[name] = pd.crosstab(pd.Series(CL[yc], name='실제'), pd.Series(np.where(p < 0, '유형 없음', CL[np.clip(p, 0, 2)]), name='예측')).to_dict()
res['confusion'] = cm
json.dump(res, open(OUT + '/cv_results.json', 'w'), ensure_ascii=False, indent=1)
o = d[['순번','출원(국제등록)번호','fold','reason']].copy()
for name in GRID: o[name+'_p유사'], o[name+'_p식별력'], o[name+'_예측'] = np.round(oof_all[name][0],4), np.round(oof_all[name][1],4), CL[oof_all[name][2]]
o['기준선_예측'] = np.where(base_pred < 0, '유형 없음', CL[np.clip(base_pred, 0, 2)])
o.to_csv(OUT + '/oof_rej_type.csv', index=False, encoding='utf-8-sig')
for k, v in res.items():
    if k != 'confusion': print(k, {a: b for a, b in v.items() if a != 'params'})
