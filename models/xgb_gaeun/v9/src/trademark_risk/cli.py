"""명령줄 진입점: `python -m trademark_risk <단계>`.

단계 순서
    repair-dates → build-index → build-features → train → evaluate
그 뒤에는 predict(개별 예측)와 fresh-eval(새 자료 1회 채점)을 쓴다.
"""

from __future__ import annotations

import argparse
import json

from .config import Paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trademark_risk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("repair-dates", help="근거가 확인된 출원공고일 보완 레지스트리 생성")
    commands.add_parser("build-index", help="선행상표 검색 색인 생성")
    features = commands.add_parser("build-features", help="라벨 표 전체의 검색 특징 계산(중단 후 이어 하기 가능)")
    features.add_argument("--workers", type=int, default=4)
    commands.add_parser("train", help="고정 설정으로 학습하고 모델 동결")
    commands.add_parser("evaluate", help="2024–2026년 개발 자료 사후 채점(1회)")

    predict = commands.add_parser("predict", help="출원 한 건의 거절 확률")
    for field in ("name", "goods", "classes", "date"):
        predict.add_argument(f"--{field}", required=True)
    predict.add_argument("--app", default="", help="자기 출원·관련 사건을 후보에서 빼려면 출원번호 입력")
    predict.add_argument("--logo", type=int, choices=[0, 1], default=0)

    fresh = commands.add_parser("fresh-eval", help="새 평가자료 입장 심사 후 1회 채점")
    fresh.add_argument("--csv", required=True)
    fresh.add_argument("--provenance", required=True)

    args = parser.parse_args(argv)
    paths = Paths.default()

    if args.command == "repair-dates":
        from . import publication_dates

        result = publication_dates.run(paths)
    elif args.command == "build-index":
        from . import search_index

        result = dict(indexed_rows=len(search_index.build(paths).base.pool))
    elif args.command == "build-features":
        from . import feature_table

        result = dict(rows=len(feature_table.build(paths, args.workers)))
    elif args.command == "train":
        from . import train

        result = dict(development_2023=train.run(paths), serving_parity=train.check_serving_parity(paths))
    elif args.command == "evaluate":
        from . import evaluate

        result = evaluate.run(paths)
    elif args.command == "predict":
        from .predict import Predictor

        query = {field: getattr(args, field) for field in ("app", "name", "goods", "classes", "date", "logo")}
        result = Predictor(paths).predict(query)
    else:
        from . import fresh_evaluation

        result = fresh_evaluation.run(paths, args.csv, args.provenance)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
