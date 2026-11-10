# 팀 전체 257,400건의 5-fold 비교

기존 가은 XGB 219변수 제출본과 공통 변수 47개로 줄인 V9를 팀이 지정한 동일한 fold와 라벨로 평가했다. 대상은 2017~2024년 출원 257,400행이며 거절은 60,642행이다. 출원번호가 여러 행에 걸치는 경우에도 팀 원래 순번을 유지했다.

## 전체 OOF 성능

| 모델 | ROC-AUC | PR-AUC | Brier | fold별 ROC-AUC 평균 ± 표준편차 |
|---|---:|---:|---:|---:|
| 기존 219변수 XGB | 0.654302 | 0.374203 | 0.169533 | 0.654344 ± 0.001897 |
| 공통 변수 47개 V9 | 0.717707 | 0.453160 | 0.159061 | 0.717713 ± 0.002240 |

다섯 fold 모두 V9의 ROC-AUC·PR-AUC가 높고 Brier는 낮았다. 전체 ROC-AUC 차이는 +0.063405였다. 팀 비교 방식처럼 동일한 행을 두 모델에서 함께 복원추출한 200회 부트스트랩의 95% 구간은 [0.06114695, 0.06573610]이다. 이 구간은 고정된 OOF 예측의 표본 변동을 나타낸다.

## 평가 방법

- 기존 219변수 모델은 제출된 건별 OOF 확률을 그대로 사용했다. 팀 원본 라벨 및 제출 파일의 순번·출원번호·fold를 점검하고, 팀 저장소 집계표의 ROC-AUC·PR-AUC·Brier와 fold 평균·표준편차를 재검산했다.
- V9는 공통 변수 47개를 양쪽 모델에 사용한다. XGBoost는 여기에 텍스트 SVD 64차원을 더해 111열을 입력받고, 로지스틱 회귀는 TF-IDF를 사용한다. 두 확률을 0.5씩 혼합한다. 6개 변수의 제외 목록, 모델 설정, 혼합 비율을 기존 47개 실험대로 고정했다.
- 평가 fold 하나를 제외한 네 fold로 두 모델을 다시 학습해 해당 fold의 점수를 냈다. 확률 보정기는 그 네 fold 안에서 다시 한 fold씩 제외하고 만든 혼합 예측과 정답만 사용했다. 두 fold를 제외하는 학습 부분이 같으면 한 번 학습해 양쪽의 내부 보정 예측에 활용했다. TF-IDF 어휘·IDF와 SVD는 매 학습 부분에서 새로 적합했다.
- 각 행의 선행상표는 해당 출원일보다 출원일·공고일이 모두 앞서야 한다. 2023·2024년에도 이 조건을 적용해 출원 시점의 검색 후보를 구성했다. 시간 분할 실험에서 2022년 말에 고정했던 평가용 검색 특징은 그대로 사용하지 않았다.
- 전체 지표는 5개 평가 fold에서 각 행이 한 번씩 얻은 OOF 확률을 합쳐 계산했다. PR-AUC는 average precision, fold 간 표준편차는 팀 비교표와 같은 표본 표준편차(ddof=1)다.
- 이 평가는 연도를 섞는 팀 fold를 사용한다. 이후 연도 출원에 대한 성능은 별도의 2023~24년 시간 분할 비교를 참조한다. V9 구조와 변수 축소안은 일부 겹치는 자료에서 개발한 뒤 고정했으므로 이번 결과도 사후 비교에 해당한다.

## 재현

기존 버전 폴더의 공통 상위 경로에서 실행한다. CACHE에는 앞선 v9_team_holdout 실험의 rows.parquet, documents.parquet, structure.npy, search.npy가 있어야 하고, 그 상위 폴더에는 vectorize.py가 필요하다. 새로운 입력으로 재현할 때는 이 실험의 빈 artifacts 폴더에서 시작한다. 원본 V9 코호트·검색 색인과 팀 219변수 모델의 제출 OOF 파일도 필요하다.

```sh
experiment=food-trademark-reject-risk-model/models/xgb_gaeun/v9/docs/v9_team_5fold
mkdir -p "$experiment/artifacts"
python "$experiment/search.py" "$CACHE"
python "$experiment/train.py" "$CACHE" --shard 0
python "$experiment/train.py" "$CACHE" --shard 1
python "$experiment/evaluate.py" "$CACHE"
```

search.py는 시점 조건이 같은 기존 검색 특징을 재사용하고 부족한 행을 계산한다. train.py는 학습 부분별 혼합 확률을 만들고, evaluate.py는 각 외부 fold의 확률을 보정하고 팀 제출본과 동일한 지표를 계산한다. metrics.csv는 전체 결과, fold_metrics.csv는 fold별 결과다. 행별 예측은 로컬 artifacts/paired_oof.parquet에 저장한다.
