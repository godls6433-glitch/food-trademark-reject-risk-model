# 팀 최종 RF의 5-fold 방식으로 평가한 47변수 V9

팀 비교표에는 5-fold 보정 방식이 두 가지 있다. 기존 가은 219변수 XGB는 외부 평가 fold를 제외한 내부 fold 예측으로 sigmoid 보정기를 학습했다. 이전 V9 0.717707은 이 219변수 제출본의 방식에 맞춘 값이다. 이번 비교의 기준은 최종 32변수 랜덤포레스트를 평가한 models/rf_final/three.py의 fit_pred 함수다.

- 팀 257,400행의 원래 순번·라벨·fold를 사용한다. 매 외부 fold에서 남은 네 fold의 원래 순서를 유지하고, 난수 발생기를 시드 42로 새로 초기화해 난수가 0.2 미만인 행을 보정용으로 둔다. 나머지 행으로만 예측 모델을 학습한다.
- V9는 동일 학습 행으로 TF-IDF·SVD·XGBoost·로지스틱 회귀를 적합한다. 보정용 행에서 두 갈래 예측을 반반 혼합한 점수에 isotonic 보정을 맞추고 외부 평가 fold를 예측한다. 팀 최종 RF와 같이 5개 평가 fold의 예측을 합쳐 ROC-AUC·PR-AUC(average precision)·Brier를 계산한다.
- 최종 RF의 5-fold 코드에는 2022년 말 결과 공개 여부 필터가 없으며, 연도를 섞는다. 단어별 과거 거절률을 계산할 때 출원일이 더 이른 사건으로 제한하지만 최초 결과 공개일은 검사하지 않는다. 또 fold를 나누기 전 전체 라벨을 사용해 단어별 거절률과 평활화용 전체 거절률을 만든다. 따라서 외부 평가 fold 라벨이 RF의 입력 특징에 반영될 수 있다. V9는 단어별 거절률을 입력하지 않는다. 같은 분할·보정 방식으로 얻은 점수도 과거 시점의 공개 정보를 완전히 재현한 성능으로 해석할 수 없다.
- RF 수치는 팀의 evaluation/results/three_results.csv에 기록된 집계값이다. V9는 새로 학습해 채점했다. 팀의 evaluation/cmp.py와 비교 문서에서 확인한 팀 순서, 기존 제출 예측 및 label_folds에 대조해 검증했다. RF의 비공개 개별 학습·보정 행 명세는 별도로 대조하지 못했다.

## 재현

기존 버전 폴더들의 공통 상위 경로에서 실행한다. CACHE는 원본 팀 시간 분할 실험의 artifacts 디렉터리(상위에 vectorize.py 필요), SEARCH는 팀 전체 5-fold용 출원일 기준 검색 특징 search.npy가 있는 디렉터리다.

```sh
exp=food-trademark-reject-risk-model/models/xgb_gaeun/v9/docs/v9_final_rf_protocol
mkdir -p "$exp/artifacts"
python "$exp/train.py" "$CACHE" "$SEARCH" --shard 0
python "$exp/train.py" "$CACHE" "$SEARCH" --shard 1
python "$exp/evaluate.py" "$CACHE"
```
