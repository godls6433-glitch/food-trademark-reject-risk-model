"""모델 동결.

학습이 끝나면 모델·입력 자료·코드의 SHA-256을 기록한다. 이후 예측이나 평가를 할 때마다
다시 대조해서, 평가 점수를 본 뒤 무언가를 슬쩍 고쳐 다시 재는 일을 구조적으로 막는다.
개선하려면 새 버전을 만들어 새 평가 계획으로 진행한다.
"""

from __future__ import annotations

from pathlib import Path

from .config import REPO_ROOT, Paths
from .io_utils import read_json, sha256, write_json


class FrozenModelError(RuntimeError):
    pass


def _tracked_files(paths: Paths) -> list[Path]:
    code = sorted((REPO_ROOT / "src/trademark_risk").glob("*.py"))
    return [paths.model, paths.search_index, paths.date_registry, paths.cohort, paths.text_bundle, *code]


def _label(path: Path) -> str:
    for root in (REPO_ROOT, REPO_ROOT.parent):
        if path.is_relative_to(root):
            return str(path.relative_to(root))
    return str(path)


def ensure_not_frozen(paths: Paths) -> None:
    if paths.manifest.exists():
        raise FrozenModelError("Model already frozen; create a new version for further changes")


def freeze(paths: Paths, **summary) -> dict:
    manifest = dict(hashes={_label(path): sha256(path) for path in _tracked_files(paths)}, **summary)
    write_json(paths.manifest, manifest)
    return manifest


def verify(paths: Paths) -> dict:
    manifest = read_json(paths.manifest)
    current = {_label(path): path for path in _tracked_files(paths)}
    for label, expected in manifest["hashes"].items():
        if label not in current or sha256(current[label]) != expected:
            raise FrozenModelError(f"Frozen dependency changed: {label}")
    return manifest
