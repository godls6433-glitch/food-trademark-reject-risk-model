# 점검기

[점검기 시연판](https://godls6433-glitch.github.io/food-trademark-reject-risk-model/) — 예시 상표 8개의 결과를 미리 계산해 보여 줍니다 (페이지 파일: `docs/index.html`).

실제 점검기는 상표명·지정상품·로고 유무를 입력받아 KIPRIS 현존 상표 579,256건과 비교합니다. 이 비교 데이터는 KIPRIS 이용약관 제16조에 따라 공개할 수 없어 시연판에는 넣지 않았습니다.

- `export_v9.py`: 경량 RF(100그루, 깊이 14) + 단어별 거절률 4개를 JSON으로 내보내기 (`models/rf_final/three.py`를 `work/`에 두고 실행)
- `val.js`: 내보낸 모델이 파이썬 결과와 같은지 확인
- `demo_data.py`, `export_demo7.py`, `export_model8.py`: 비슷한 선행 상표 목록·이전 버전 모델 내보내기
- 새 상표는 현존 상표(출원·공고·등록) 579,256건과 비교합니다.
