# V9 공통 변수 축소 실험

팀 라벨과 시간 분할을 적용한 원본 V9(공통 변수 53개)의 2023–24년 예측을 기준으로, 아래 두 실험을 진행했다.

- 48개 버전은 kb_EFFECT_context, kb_USE_context, kb_PRICE_context, kb_COMMON_EXPRESSION_context 및 v7_pair_exact_goods를 뺐다. 팀 학습 191,810건에서도 앞의 4개는 모두 0이었으며 마지막 변수는 prior_exact_name_goods_coverage와 전 행에서 일치했다.
- 47개 버전은 여기에 v7_char_max를 추가로 제외하고 prior_char_max를 유지했다.

기존 V9의 공통 열 순서를 유지한 채 XGBoost와 로지스틱 회귀 양쪽에서 지정한 열을 제거했다. XGBoost 입력은 각각 117, 112, 111열이다. 동일한 팀 fold별 텍스트 행렬·SVD, 선행상표 검색 시점, 하이퍼파라미터, 시드, 혼합 비율을 사용했다. 각 축소 버전은 2017–2022년 자료에서 5-fold OOF 예측으로 별도의 sigmoid 보정기를 학습한 뒤, 해당 기간 전체로 재학습한 두 갈래의 예측에 적용했다.

2023년 33,070건과 2024년 32,520건의 순번·출원번호·fold·정답이 원본 비교와 일치하는지 검사했다. PR-AUC는 average precision으로 계산했다. 전체 ROC-AUC 차이의 95% 구간은 정규화한 상표명 단위로 1,000회 짝지어 복원추출한 백분위 구간이다. 이 구간은 이미 학습한 세 모델의 평가 표본 변동을 나타낸다. 학습 시드를 바꿨을 때의 변동은 포함하지 않는다. 기존 V9 개발과 평가에 일부 사용한 자료를 대상으로 한 사후 비교다.

## 비교 결과

| 공통 변수 수 | ROC-AUC | PR-AUC | Brier |
|---:|---:|---:|---:|
| 53 | 0.719895 | 0.406965 | 0.146858 |
| 48 | 0.720037 | 0.406247 | 0.146956 |
| 47 | 0.719922 | 0.406952 | 0.146881 |

47개 버전의 원본 대비 ROC-AUC 차이는 +0.000026(95% 구간 −0.000583~+0.000582)이었다. PR-AUC 차이는 −0.000013, Brier 차이는 +0.000023이었다. 이 표본에서는 공통 변수 6개를 줄여도 세 지표가 원본에 가까웠다. 연도별 수치는 metrics.csv, 짝지은 비교 구간은 auc_intervals.csv에 있다.

## 재현

기존 버전 폴더의 공통 상위 경로에서, 원본 실험의 prepare.py, search.py, text_input.py, vectorize.py로 생성한 캐시와 원본 평가 예측을 사용한다. 아래 CACHE는 원본 실험의 rows.parquet, paired_predictions.parquet, fold별 행렬이 있는 디렉터리다.

```sh
experiment=food-trademark-reject-risk-model/models/xgb_gaeun/v9/docs/v9_team_reduced
python "$experiment/fit_reduced.py" "$CACHE" 48
python "$experiment/fit_reduced.py" "$CACHE" 47
python "$experiment/evaluate.py" "$CACHE"
```
