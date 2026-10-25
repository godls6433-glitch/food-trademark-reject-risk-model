# 최종 모델: 해인 RF + 단어별 거절률 (공통)

세 모델을 같은 기준으로 비교한 뒤 팀이 함께 정한 최종 모델입니다. 해인 RF 28변수에 예환 모델의 단어별 과거 거절률(2글자 조각, 먼저 출원된 상표만으로 계산) 4개를 더한 32변수, isotonic 보정.
`n_estimators=300, min_samples_leaf=20, max_samples=0.7, max_features='sqrt', class_weight='balanced'`

| 평가 | ROC-AUC |
|---|---|
| 5-fold 교차검증 | 0.704 |
| 2023~24 새 데이터 | 0.723 |

- `three.py`: 28변수 RF, 32변수 RF(최종), 32변수 XGB를 5-fold와 시간 분할로 비교·학습
