# 해인 RF (28변수)

랜덤포레스트, 변수 28개, isotonic 보정. 5-fold ROC-AUC 0.694 / 2023~24 0.713.
`n_estimators=300, min_samples_leaf=20, max_samples=0.7, max_features='sqrt', class_weight='balanced'`

- 비교 대상: 유사군코드가 겹치는 먼저 출원된 타인 상표(다른 류 5·31·33·1류 포함, 750,680건)
- 본인·계열사 제외: 출원인 판별 규칙 5개, 그룹 최대 20곳
- 유사 판단: 주요부(설명어를 뺀 부분) 기준 발음·음역·포함 유사도
- `rf_v8.py`: 학습·5-fold 예측 / 변수 계산 코드는 `src/03_features/`
- [수정 이력](process.md)
