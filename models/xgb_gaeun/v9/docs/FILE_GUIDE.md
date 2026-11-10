# 가은 모델 공유본 파일 안내

이번 변수 축소 결과는 [53·48·47개 비교](v9_team_reduced/README.md)에서 볼 수 있다. 기존 219변수 모델과 같은 조건에서 V9를 평가한 수치는 [팀 시간 분할 비교](v9_team_holdout/README.md)에, 축소 후보를 고른 근거는 [변수 상관분석](v9_shared_correlation/README.md)에 있다.

원본 V9의 0.7088은 자체 라벨로 평가한 2024년 결과다. 팀 라벨로 재학습한 원본 V9는 2023~24년 65,590건에서 0.719895, 공통 변수 47개 버전은 0.719922를 기록했다. 이 공유본의 모델 패키지는 원본 V9 구현이며, 축소 버전은 별도 비교 실험으로 제공한다.

```text
xgb_gaeun/
├── README.md                  두 모델 소개와 성능 비교
├── original_xgb_219/          팀 제출 모델
│   ├── README.md              219변수 모델 실행 방법
│   └── src/                   전처리·학습·예측 코드
└── v9/                        후속 V9 모델
    ├── README.md              모델 구조·검증 결과·실행 방법
    ├── src/trademark_risk/    검색·학습·예측 코드
    ├── docs/                  버전 이력·변수 축소 실험·평가표
    ├── data/ · resources/     개발 사례 목록과 사전
    ├── scripts/ · tests/      검증 도구와 자동 테스트
    └── pyproject.toml · requirements.txt
```

집계 결과는 CSV로 바로 열 수 있다. 실제 데이터로 재학습하려면 로컬 KIPRIS 원자료, 팀 라벨·fold, 연구 단계 산출물이 필요하다. 행별 예측과 학습 모델은 로컬에서 관리한다. 가상 자료 테스트는 연구 데이터 없이 실행할 수 있다.

## 최상위 · 시작 문서와 실행 환경

| 파일 | 용도 |
|---|---|
| [README.md](../../README.md) | 모델의 목적, 팀 제출 당시 219변수 모델과의 관계, 원본 V9 성능, 실행 순서를 설명한다. |
| [.gitignore](../../.gitignore) | 원자료, 건별 예측, 학습 모델, 캐시를 Git에서 제외하고 공유할 집계 CSV를 허용한다. |
| [pyproject.toml](../pyproject.toml) | 패키지 이름과 버전, Python 요구 버전, 빌드 방식, pytest와 Ruff 설정을 정의한다. |
| [requirements.txt](../requirements.txt) | 학습·검색·평가에 사용하는 Python 라이브러리의 버전을 고정한다. |

## 팀 제출 모델 · 219변수 XGBoost

[original_xgb_219/README.md](../../original_xgb_219/README.md)에서 원본 모델의 파일별 용도와 재현 순서를 확인할 수 있다.

## data · 평가에서 제외할 개발 사례

| 파일 | 용도 |
|---|---|
| [inspected_cases.json](../data/inspected_cases.json) | 이미 결과를 확인한 개발 사례 5건의 출원번호와 정규화 상표명을 기록한다. 새 평가자료를 검사할 때 이 목록과 대조한다. |

## docs · 실험 기록과 파일 안내

| 파일 | 용도 |
|---|---|
| [V9 README](../README.md) | 원본과 축소 버전의 입력 구성, 도식, 검증 조건과 실행 순서를 설명한다. |
| [FILE_GUIDE.md](FILE_GUIDE.md) | 공유본에 포함된 모든 폴더와 파일의 용도, 결과를 읽는 순서, 재현에 필요한 자료를 설명하는 이 문서다. |
| [VERSION_HISTORY.md](VERSION_HISTORY.md) | V1부터 V9까지 시도한 변경, 평가 결과, 채택 또는 보류 이유를 시간순으로 정리한다. |
| [CODE_REVIEW.md](CODE_REVIEW.md) | 연구 코드를 Python 패키지로 정리하면서 바꾼 구조와 원래 동작을 검증한 방법을 설명한다. |

## docs/v9_shared_correlation · 2017~2021년 V9 학습 148,646건의 변수 분석

| 파일 | 용도 |
|---|---|
| [README.md](v9_shared_correlation/README.md) | 공통 변수 53개에서 상수·완전 중복·강한 상관을 구분하고 축소 후보의 선정 근거를 제시한다. |
| [analyze.py](v9_shared_correlation/analyze.py) | 학습 행과 53개 변수를 복원해 Pearson·Spearman 상관계수와 값이 다른 행 수를 계산한다. |
| [feature_summary.csv](v9_shared_correlation/feature_summary.csv) | 변수별 고유값 수, 0이 아닌 행 수, 최솟값과 최댓값을 담은 53행 요약표다. |
| [pairs.csv](v9_shared_correlation/pairs.csv) | 가능한 1,378개 변수 쌍의 두 상관계수와 값이 서로 다른 행 수를 담는다. |

## docs/v9_team_holdout · 팀 공통 조건에서 기존 219변수 XGB와 원본 V9 비교

| 파일 | 용도 |
|---|---|
| [README.md](v9_team_holdout/README.md) | 팀 라벨, 191,810건 학습 및 65,590건 평가 조건, 검색 시점 제한, 재현 명령을 설명한다. |
| [prepare.py](v9_team_holdout/prepare.py) | 팀 자료의 해시를 확인하고 순번으로 질의·라벨을 연결한다. 평가 행 순서와 식별자를 기존 제출 파일과 대조한다. |
| [search.py](v9_team_holdout/search.py) | 기존 질의와 일치하는 학습 행의 검색 특징을 재사용하고 나머지는 다시 계산한다. 평가용 후보는 2022년 말까지 공개된 상표로 제한한다. |
| [text_input.py](v9_team_holdout/text_input.py) | 이름·자모·지정상품 텍스트와 구조·문맥 변수 21개를 준비한다. |
| [vectorize.py](v9_team_holdout/vectorize.py) | 각 fold의 학습 부분에서 TF-IDF와 SVD를 적합하고 최종 학습용 변환도 별도로 만든다. |
| [fit.py](v9_team_holdout/fit.py) | 텍스트와 검색 특징으로 XGB·로지스틱 회귀를 학습하고, fold 밖 예측을 사용해 혼합 확률을 보정한다. |
| [evaluate.py](v9_team_holdout/evaluate.py) | 기존 제출 예측과 V9 예측을 동일 평가 행에 맞춰 채점하고 219변수 모델의 보관 지표와 대조한다. |
| [metrics.csv](v9_team_holdout/metrics.csv) | 원본 V9와 219변수 모델의 전체·연도별 ROC-AUC, PR-AUC, Brier, 평균 예측 확률을 담는다. |

## docs/v9_team_reduced · 공통 변수 53개·48개·47개 비교

| 파일 | 용도 |
|---|---|
| [README.md](v9_team_reduced/README.md) | 두 축소안에서 제외한 변수와 동일 팀 표본에서의 성능 차이를 설명한다. |
| [fit_reduced.py](v9_team_reduced/fit_reduced.py) | 48 또는 47을 인수로 받아 지정한 열을 두 모델 갈래에서 제외한 뒤 각각 재학습하고 보정한다. |
| [evaluate.py](v9_team_reduced/evaluate.py) | 세 모델의 평가 행과 정답을 맞춘 뒤 지표 차이와 상표명 단위 부트스트랩 구간을 계산한다. |
| [metrics.csv](v9_team_reduced/metrics.csv) | 세 버전의 전체·연도별 지표와 원본 53개 버전 대비 차이를 담는다. |
| [auc_intervals.csv](v9_team_reduced/auc_intervals.csv) | 전체 ROC-AUC에 대한 세 쌍의 차이와 1,000회 짝지은 부트스트랩 95% 구간을 담는다. |

## docs/v9_team_5fold · 47변수 V9와 기존 219변수 모델의 팀 5-fold 비교

| 파일 | 용도 |
|---|---|
| [README.md](v9_team_5fold/README.md) | 평가 조건, 전체 결과와 재현 순서 |
| [search.py](v9_team_5fold/search.py) | 각 출원일 기준의 선행상표 특징 구성 |
| [train.py](v9_team_5fold/train.py) | 내부·외부 학습 부분별 47변수 혼합 모델 학습 |
| [evaluate.py](v9_team_5fold/evaluate.py) | 중첩 확률 보정, 팀 지표 대조와 부트스트랩 |
| [metrics.csv](v9_team_5fold/metrics.csv) | 두 모델의 전체 OOF 지표 및 fold 평균·표준편차 |
| [fold_metrics.csv](v9_team_5fold/fold_metrics.csv) | 각 fold의 건수와 두 모델 성능 |

## 추가 평가

| 실험 | 파일별 용도 |
|---|---|
| [최종 RF 방식 5-fold](v9_final_rf_protocol/README.md) | train.py는 fold별 학습과 isotonic 보정, evaluate.py는 OOF 채점을 수행한다. metrics.csv와 fold_metrics.csv에 전체·그룹별 결과가 있다. |
| [결과 시점 점검](v9_outcome_cutoff/README.md) | cohort.py는 날짜 근거로 학습 표본을 선별하고 fit.py와 evaluate.py는 재학습·보정·평가를 수행한다. metrics.csv에 시간 분할 결과가 있다. |

## resources · 모델에서 참조하는 사전

| 파일 | 용도 |
|---|---|
| [cmudict.dict](../resources/cmudict.dict) | 영어 철자를 발음 기호로 변환할 때 참조하는 CMU 발음사전이다. |
| [CMUDICT_LICENSE](../resources/CMUDICT_LICENSE) | 동봉한 CMU 발음사전의 저작권 표시와 배포 조건이다. |
| [role_lexicon.json](../resources/role_lexicon.json) | 원재료·상품명 등 역할별 용어 사전이다. 선행상표 이름에서 요부 후보를 추출할 때 일반 명칭을 판별하는 데 사용한다. |

## scripts · 기존 연구 모델과의 대조

| 파일 | 용도 |
|---|---|
| [check_against_frozen_v9.py](../scripts/check_against_frozen_v9.py) | 정리한 패키지의 색인·검색 특징·예측 확률을 동결한 원본 V9와 대조한다. 원본 모델 및 연구 작업자료가 필요하다. |

## src/trademark_risk · V9 모델 패키지

| 파일 | 용도 |
|---|---|
| [__init__.py](../src/trademark_risk/__init__.py) | 패키지 설명과 버전 9.0.0을 제공한다. |
| [__main__.py](../src/trademark_risk/__main__.py) | python -m trademark_risk 명령을 CLI로 연결한다. |
| [cli.py](../src/trademark_risk/cli.py) | 색인 생성, 특징 계산, 학습, 평가, 개별 예측 등의 명령과 인수를 처리한다. |
| [config.py](../src/trademark_risk/config.py) | 입출력 경로, 기간 구분, 허용 입력 필드, 고정 모델 설정을 정의한다. |
| [io_utils.py](../src/trademark_risk/io_utils.py) | 파일 해시 계산, JSON 읽기·쓰기, 원자적 저장, 원본 엑셀 읽기를 담당한다. |
| [textnorm.py](../src/trademark_risk/textnorm.py) | 이름·날짜·상품류·지정상품을 정규화하고 문자 조각과 자모를 만든다. |
| [text_features.py](../src/trademark_risk/text_features.py) | 이름·자모·지정상품 TF-IDF와 구조·사전 문맥 특징을 조립한다. |
| [phonetic.py](../src/trademark_risk/phonetic.py) | 영문과 한글 이름을 발음 코드로 변환하고 상표 이름의 요부 후보를 만든다. |
| [prior_search.py](../src/trademark_risk/prior_search.py) | 공고·출원 시점과 사건군을 검사해 선행상표를 찾고 이름·지정상품 검색 특징 22개를 계산한다. |
| [pair_search.py](../src/trademark_risk/pair_search.py) | 발음·요부 후보를 추가해 같은 선행상표 안에서 이름과 상품의 유사도를 결합한 검색 특징 10개를 만든다. |
| [publication_dates.py](../src/trademark_risk/publication_dates.py) | 공식 서지·원본 셀에서 확인한 공고일을 대조하고 충돌하는 근거를 걸러낸다. |
| [search_index.py](../src/trademark_risk/search_index.py) | 검증한 공고일을 선행상표 풀에 반영하고 검색 색인을 만든다. |
| [feature_table.py](../src/trademark_risk/feature_table.py) | 여러 출원의 검색 특징 32개를 묶음 단위로 계산하고 저장한다. |
| [model.py](../src/trademark_risk/model.py) | 선형·트리 입력을 조립하고 두 모델의 학습, 혼합, 확률 보정, 지표 계산을 수행한다. |
| [train.py](../src/trademark_risk/train.py) | V9 고유의 시간 분할을 검증하고 정해진 설정으로 학습한 모델을 동결한다. |
| [freeze.py](../src/trademark_risk/freeze.py) | 모델과 의존 파일의 해시를 기록하고 후속 평가 때 변경 여부를 확인한다. |
| [predict.py](../src/trademark_risk/predict.py) | 출원 한 건의 입력을 정리하고 거절 확률과 검색 근거 후보를 반환한다. |
| [evaluate.py](../src/trademark_risk/evaluate.py) | 동결된 모델을 기간별로 채점하고 기존 모델과의 성능 차이를 계산한다. |
| [fresh_evaluation.py](../src/trademark_risk/fresh_evaluation.py) | 새 평가 후보가 과거 개발 사건과 겹치는지, 결과 라벨이 요건을 충족하는지 확인하고 채점한다. |

## tests · 가상 자료 기반 자동 검증

| 파일 | 용도 |
|---|---|
| [conftest.py](../tests/conftest.py) | 여러 테스트가 공유하는 가상 상표·검색 풀·테스트 환경을 만든다. |
| [test_guards.py](../tests/test_guards.py) | 결과 필드 차단, 기존 사건의 새 평가 제외, 공고일 근거 검사를 확인한다. |
| [test_phonetic.py](../tests/test_phonetic.py) | 한영 발음 변환, 띄어쓰기 처리, 일반 명칭 제거, 발음사전 처리를 검증한다. |
| [test_pipeline.py](../tests/test_pipeline.py) | 가상 자료로 색인 → 특징 → 학습 → 동결 → 예측 → 평가 흐름을 검증한다. |
| [test_search.py](../tests/test_search.py) | 공개 시점, 동일 사건군 제외, 후보 부재와 발음 유사 상표 검색을 검증한다. |
| [test_text_features.py](../tests/test_text_features.py) | 지정상품 문맥 일치 조건과 겹치는 용어의 중복 집계 방지를 확인한다. |
| [test_textnorm.py](../tests/test_textnorm.py) | 문자 정규화, 지정상품 분리, 짧은 이름, 상품류 및 날짜 처리를 확인한다. |
