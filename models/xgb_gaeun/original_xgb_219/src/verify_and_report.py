"""Read exported bytes, verify the submission contract, write reproducible report."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from train import ROOT, OUT, MODELS, metrics, xgb, logit
from features import class_values, CLASSES


def run():
    data = next(p for p in ROOT.parent.iterdir() if p.name.startswith("02_"))
    label = pd.read_csv(data/"label_folds.csv", dtype={"순번": str, "출원번호": str})
    saved = pd.read_csv(OUT/"XGB_OOF_predictions_257400.csv", dtype={"순번": str, "출원번호": str})
    assert saved.columns.tolist() == ["순번", "출원번호", "fold", "거절확률"]
    assert len(saved) == len(label) == 257400 and saved["순번"].is_unique
    assert saved["순번"].equals(label["순번"])
    assert saved.fold.equals(label.fold)
    assert saved["거절확률"].notna().all() and saved["거절확률"].between(0, 1).all()
    originals = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["순번", "출원(국제등록)번호"])
                           for f in data.rglob("tm_pool_*.csv")]).set_index("순번")["출원(국제등록)번호"]
    assert saved["출원번호"].eq(saved["순번"].map(originals)).all()
    assert not saved["출원번호"].str.contains("[Ee][+]", regex=True).any()
    y = (1-label.registered).to_numpy()
    recalculated = metrics(y, saved["거절확률"].to_numpy())
    reported = json.loads((OUT/"metrics.json").read_text())
    for k, v in recalculated.items():
        assert abs(v-reported["calibrated_oof"][k]) < 1e-10, (k, v)
    audit = json.loads((OUT/"data_audit.json").read_text())
    for filename, expected in audit["exact_source_hashes_sha256"].items():
        assert hashlib.sha256((ROOT.parent/filename).read_bytes()).hexdigest() == expected, filename
    split = json.loads((OUT/"split_audit.json").read_text())
    assert len(split) == 20
    for s in split:
        assert s["outer_test_fold"] not in s["model_training_folds"]
        assert s["calibration_fold"] not in s["model_training_folds"]
        assert s["outer_test_fold"] != s["calibration_fold"]
        assert len(s["model_training_folds"]) == 3
    X = pd.read_parquet(OUT/"features.parquet")
    assert len(X) == len(saved)
    # Saved models and calibrators must actually reproduce the exported values.
    max_error = 0.
    for fold in range(1, 6):
        positions = np.flatnonzero(saved.fold.to_numpy() == fold)[::250]
        model = xgb.XGBClassifier()
        model.load_model(MODELS/f"outer{fold}.ubj")
        cal = joblib.load(MODELS/f"outer{fold}_sigmoid.joblib")
        # OOF arrays are float64. Preserve that exact arithmetic during replay.
        actual = cal.predict_proba(logit(model.predict_proba(X.iloc[positions])[:, 1].astype(np.float64)))[:, 1]
        max_error = max(max_error, float(np.max(np.abs(actual-saved.iloc[positions]["거절확률"].to_numpy()))))
    assert max_error < 1e-10
    neighbor_rows = 0
    ref = pd.read_parquet(OUT/"reference_safe.parquet")
    cohort = pd.read_parquet(OUT/"cohort.parquet")
    def mask(value):
        return sum(1 << CLASSES.index(c) for c in class_values(value))
    prior_class = pd.Series(ref["상품분류"].map(mask).to_numpy(), index=ref.source_key)
    target_class = pd.Series(cohort["상품분류"].map(mask).to_numpy(), index=cohort["순번"])
    for p in pd.read_csv(OUT/"neighbors_internal.csv", dtype={"순번": str}, chunksize=100000):
        assert (p.prior_day < p.query_day).all()
        assert p.similarity.between(0, 1).all()
        assert ((p.prior_key.map(prior_class).to_numpy() & p["순번"].map(target_class).to_numpy()) > 0).all()
        assert p["rank"].between(1, 5).all()
        assert p.groupby("순번", sort=False).similarity.diff().dropna().le(1e-12).all()
        neighbor_rows += len(p)
    aux = pd.read_csv(OUT/"XGB_OOF_expected_refusal_type.csv", dtype={"순번": str, "출원번호": str})
    assert len(aux) == len(saved) and aux["순번"].equals(saved["순번"]) and aux.fold.equals(saved.fold)
    assert set(aux["예상거절유형"]) <= {"유사", "식별력"}
    checks = {"rows": len(saved), "all_sequences_unique_and_original_order": True,
              "all_folds_unchanged": True, "all_application_ids_match_original_pool": True,
              "all_probabilities_finite_and_in_range": True, "source_hashes_unchanged": True,
              "nested_split_leakage_checks_pass": True, "model_replay_max_abs_error": max_error,
              "all_prior_dates_strictly_before_query": True, "neighbor_rows": neighbor_rows,
              "all_prior_classes_overlap": True, "neighbor_similarity_descending": True,
              "all_auxiliary_rows_present": True, "csv_recomputed_metrics": recalculated}
    (OUT/"submission_validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    summary_rows = [{"scope": "all_OOF", **recalculated},
                    {"scope": "fold_mean", "roc_auc": reported["roc_auc_fold_mean"]},
                    {"scope": "fold_std_ddof0", "roc_auc": reported["roc_auc_fold_std_population"]},
                    {"scope": "constant_baseline", "roc_auc": .5, "pr_auc_ap": float(y.mean()), "brier": float(y.mean()*(1-y.mean()))}]
    pd.DataFrame(summary_rows).to_csv(OUT/"XGB_metrics_summary.csv", index=False, encoding="utf-8-sig")
    packages = ["pandas", "numpy", "scipy", "scikit-learn", "xgboost", "rapidfuzz", "pyarrow", "joblib"]
    (ROOT/"requirements.txt").write_text("\n".join(f"{p}=={importlib.metadata.version(p)}" for p in packages)+"\n")
    config = json.loads((ROOT/"model_config.json").read_text())
    reasonconfig = json.loads((ROOT/"reason_model_config.json").read_text())
    report = f"""독립 XGBoost 모델 결과 — 팀 공통 규칙 2026-10-08 적용
작성 기준일: 2026-10-12

1. 제출 파일
artifacts/XGB_OOF_predictions_257400.csv
UTF-8 BOM, 257,400행. 열: 순번, 출원번호, fold, 거절확률.
거절확률은 보정된 0~1 값이다. 0.30은 거절확률 30%를 뜻한다.
label_folds.csv 순서/순번/fold/정답을 그대로 사용했으며 행 추가·제외가 없다.
출원번호는 순번으로 원본 학습풀에서 가져와 과학적 표기 손상을 피했다.

2. 비교 지표 (거절=1, 등록=0, 전체 OOF 확률 기준)
ROC-AUC: {recalculated['roc_auc']:.8f}
PR-AUC (Average Precision): {recalculated['pr_auc_ap']:.8f}
PR 곡선 사다리꼴 면적: {recalculated['pr_auc_trapezoidal']:.8f}
Brier: {recalculated['brier']:.8f}
fold ROC-AUC 평균: {reported['roc_auc_fold_mean']:.8f}
fold ROC-AUC 표준편차(ddof=0): {reported['roc_auc_fold_std_population']:.8f}
fold ROC-AUC 표준편차(ddof=1): {reported['roc_auc_fold_std_sample']:.8f}
거절 60,642 / 전체 257,400 = {y.mean():.8%}
상수 거절확률 기준 Brier: {reported['constant_prevalence_brier']:.8f}
팀의 PR-AUC 구현 차이에 대비해 Average Precision과 사다리꼴 면적을 함께 기록했다.
원시 확률 지표는 metrics.json에 진단용으로만 기록했으며, 그 점수로 보정 방식을 다시 고르지 않았다.

3. 독립성 및 학습
기존 trademark_risk_v1~v7의 코드, 특징 파일, 예측값, 학습 가중치는 사용하지 않았다.
전용 가상환경과 새 코드로 원본 3종 CSV부터 재계산했다.
XGBClassifier, random_state=42, objective=binary:logistic, tree_method=hist.
{json.dumps(config['parameters'], ensure_ascii=False)}
파라미터는 외부 fold 평가 전에 고정했다. 시험 fold 점수에 따른 튜닝/모델 선택이 없다.
각 outer fold k: 나머지 4개 fold 안에서 3개로 학습, 남은 1개 예측을 4번 수행한다.
그 내부 OOF 예측의 logit에 sigmoid 보정기를 학습한다.
그 후 4개 fold 전체로 XGB를 재학습하고, k의 확률에 해당 보정기를 적용한다.
outer 시험 fold의 정답은 XGB 학습과 보정 모두에서 제외했다.
전체 5개 outer fold 예측을 이어붙인 결과만 제출 CSV에 썼다.
배포용 final_xgb.ubj는 전체 라벨 학습 모델이다. 제출 CSV와 용도를 분리했다.

4. 비교 대상과 정보 시점
학습풀 395,982건 + P 141,494건 + D 93,292건만 사용. 외부 상표 데이터 없음.
P는 제공 파일에서 이미 파일 경계 중복 9건이 제거되어 있어 다시 행을 제거하지 않았다.
선행 후보는 29·30·32·43류 중 적어도 한 류가 겹치고, 대상 출원일보다 엄격하게 앞선 출원이다.
동일 날짜 출원과 동일 출원번호는 유사도 목록에서 제외한다.
법적상태로 후보를 거르거나 P/D 출처를 특징으로 넣지 않는다.
제공 데이터로 과거 권리의 유효 상태를 복원할 수 없으므로, 목록은 과거 출원 명칭의 유사 후보이며 유효 권리 충돌 판정은 아니다.
본인 소유 여부는 제외 조건에 쓰지 않는다. 현재 출원인 정보가 당시 출원인임을 보장할 이력이 없기 때문이다.
현재 법적상태, 등록일/번호, 공고일/번호, 심사진행상태, TM5, 공존동의, 거절사유, fold/순번/출원번호는 거절확률 입력에서 제외.
지정상품과 유사군코드도 변경 이력 및 최종 사용자 입력의 제약을 고려해 이번 모델에는 사용하지 않는다.
출원일은 과거 후보를 제한할 때만 쓰고, 예측 변수에는 넣지 않았다.
명칭구분의 도형복합=로고 있음, 다른 알려진 구분=없음, 구분 누락=결측으로 처리했다.
이 값은 이미지 실물을 확인한 로고 유무가 아닌 메타데이터 대용값이다.

5. 상표명과 변수
NFKC 정규화, 소문자화, 문자·숫자 보존, 구두점·공백 제거.
전체 이름 / 한글만 / 영문만 별칭 사이의 최대 정규화 Indel 유사도(RapidFuzz ratio)를 쓴다.
동일 류의 과거 기록만 들어간 색인에서 2·3글자 조각 및 앞/뒤 조각으로 후보를 검색한다.
류마다 희귀 조각 8개, 조각별 최대 1,200건(앞 300+최근 900), 가중 후보 상위 240개 및 정확 일치 최근 300건을 재정렬한다.
따라서 top 5는 근사 검색 후보 내 상위 5개이며 전체 선행 상표를 완전 탐색한 보장은 없다.
한글·영문 음역, 의미/관념, 로고 이미지 자체의 유사도는 구현하지 않았다.
동일 출원번호 후보는 한 번만 표시하고, 유사도·출원번호·안정된 행 순서로 동점 처리한다.
전체 변수 {X.shape[1]}개: 이름 길이/문자 종류/혼합/공백/괄호/로고, 4개 류, 수동 상품·평가·지명·업종 단어,
과거 동일 명칭 빈도, 과거 문자 조각 빈도, 선행 top 5 유사도/간격/평균/임계 유사도 개수/선행 경과일 등.
정확한 전체 변수 목록은 artifacts/feature_list.csv, 고정 사전은 dictionary.json.
사전은 이번 작업에서 사전 지정했다. 거절사유 문서나 기존 모델의 사전을 보고 만든 사전이 아니다.

6. 데이터 검수
label_folds 출원번호 과학적 표기 행: {audit['label_scientific_application_rows']:,}.
label_folds의 손상된 명칭과 원본이 다른 행: {audit['label_name_source_differences']}건. 원본 이름을 사용하고 라벨/fold는 유지.
류·출원일·상태 기반 정답은 원본과 전건 일치한다.
수정 내역: artifacts/label_name_source_differences.csv.
입력 파일 SHA-256, 행 수, 결합 검수는 data_audit.json. 제출물 재검증은 submission_validation.json.
저장 모델을 다시 불러온 5개 fold 표본 예측과 CSV의 최대 차이: {max_error:.3g}.
선행 목록 {neighbor_rows:,}행의 날짜가 대상 출원일보다 앞서는지 전건 검사했다.

7. 보조 출력
artifacts/XGB_prior_top5.csv.gz: 유사도 순 선행상표명·류·출원번호·출원일·법적상태·문자유사도.
법적상태는 2026-10 수집 당시 정보이며 대상 출원일 당시 상태가 아니다. 표시 목적으로만 사후 결합했다.
적합한 문자 후보가 없으면 5건보다 적거나 0건일 수 있다. 억지로 5건을 채우지 않았다.
artifacts/XGB_OOF_expected_refusal_type.csv: 전체 행의 예상 거절 유형(유사/식별력) 및 두 보조 점수.
reject_reasons의 라벨 코호트 내 알려진 거절 {reasonconfig['known_training_rows']:,}행만 보조 모델 정답으로 사용.
복합은 두 유형 모두 양성, 지정상품만은 보조 학습에서만 제외했다. 거절확률 행은 여전히 전건 유지한다.
보조 모델도 각 outer 시험 fold를 제외한 알려진 사유만 학습한다.
이 점수는 이미 거절된 사유확보 사례의 조건부 유형 점수이며 실제 전체 거절확률이 아니다. 별도 확률 보정은 하지 않았다.
위험 등급은 확정하지 않았다. risk_threshold_candidates.csv와 risk_grade_candidate_rates.csv로 팀 결정 가능.
고정 fold는 무작위 그룹 검증이므로 미래 연도 성능을 직접 검증한 결과로 해석하지 않는다.

8. 재현 실행 (프로젝트 상위 폴더에서)
sh trademark_xgb_team_20261012/run.sh trademark_xgb_team_20261012/src/test_contract.py
sh trademark_xgb_team_20261012/run.sh trademark_xgb_team_20261012/src/prepare.py
sh trademark_xgb_team_20261012/run.sh trademark_xgb_team_20261012/src/train.py
sh trademark_xgb_team_20261012/run.sh trademark_xgb_team_20261012/src/auxiliary.py
sh trademark_xgb_team_20261012/run.sh trademark_xgb_team_20261012/src/verify_and_report.py
기존 모델 파일이 있으면 train.py는 동일 설정의 모델을 재사용한다. 완전 재학습은 새 출력 폴더를 사용한다.
새 이름 추론 예시:
sh trademark_xgb_team_20261012/run.sh trademark_xgb_team_20261012/src/predict.py --name '가은별빛카페' --classes 43 --logo 0
조회기준일 기본값은 이번 분석일 2026-10-12로 고정되어 있으며 --as-of로 변경 가능하다.
이 데이터의 최신 출원 범위는 2026-09-30까지이므로 이후 신규 상표는 비교 대상에 없다.

9. 사용 라이브러리 방법 문서
XGBoost API: https://xgboost.readthedocs.io/en/stable/python/python_api.html
확률 보정: https://scikit-learn.org/stable/modules/calibration.html
Average Precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
"""
    (ROOT/"결과_설명서.txt").write_text(report, encoding="utf-8")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(ROOT.rglob("*")) if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts
                and p.suffix not in (".log",) and p.name != "manifest_sha256.json"}
    (ROOT/"manifest_sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
