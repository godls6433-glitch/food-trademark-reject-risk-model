# 팀 제출 모델 · 219변수 XGBoost

팀에 처음 제출한 219변수 XGBoost의 연구 코드입니다. 전체 257,400건의 팀 5-fold ROC-AUC는 0.6543, 2022년까지 학습한 뒤 2023~24년 65,590건에서 평가한 ROC-AUC는 0.6637입니다. [후속 V9와 비교](../README.md)

## 재현 경로

원본 경로 규칙을 사용합니다. 이 폴더를 저장소 밖 별도 실험 디렉터리에 복사하고, 복사본과 같은 상위 디렉터리에 팀 원자료 폴더(이름이 `02_`로 시작)를 둡니다. 예측 모델과 출원별 자료는 로컬에서 생성합니다. 패키지 버전은 requirements.txt에 기록돼 있습니다.

실험 복사본에서 가상환경을 활성화한 뒤 순서대로 실행합니다.

```sh
python -m pip install -r requirements.txt
python src/prepare.py
python src/train.py
python src/time_holdout.py
```

## 파일별 역할

| 경로 | 역할 |
|---|---|
| `src/prepare.py` · `src/features.py` | 팀 원자료와 라벨을 연결하고 상표명·과거 상표 검색에서 219개 변수를 생성합니다. |
| `src/train.py` · `src/time_holdout.py` | 각각 팀 5-fold 학습·보정과 2023~24년 시간 분할 평가를 실행합니다. |
| `src/predict.py` · `src/auxiliary.py` | 학습 결과로 개별 출원을 예측하고 거절 유형 등 보조 결과를 생성합니다. |
| `src/test_contract.py` · `src/test_serving_parity.py` · `src/verify_and_report.py` | 데이터·모델 검증과 학습·예측 특징 대조, 결과 보고를 담당합니다. |
| `run.sh` · `requirements.txt` | 원본 실행 스크립트와 Python 라이브러리 버전입니다. |
