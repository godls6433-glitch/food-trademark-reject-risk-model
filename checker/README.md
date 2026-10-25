# 점검기

[상표 거절 위험 점검기](https://claude.ai/artifact/WWQhvbtfBTHPfRATmEwbzb) — 입력: 상표명·지정상품·로고 유무

- `export_v9.py`: 경량 RF(100그루, 깊이 14) + 단어별 거절률 4개를 JSON으로 내보내기 (`models/rf_final/three.py`를 `work/`에 두고 실행)
- `val.js`: 내보낸 모델이 파이썬 결과와 같은지 확인
- `demo_data.py`, `export_demo7.py`, `export_model8.py`: 비슷한 선행 상표 목록·이전 버전 모델 내보내기
- 새 상표는 현존 상표(출원·공고·등록) 579,256건과 비교합니다.
