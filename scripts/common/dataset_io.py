"""CSV/JSONL/JSON 입력과 파일 해시. 모델 입력과 정답 메타데이터를 분리한다."""

# Teacher 생성·양자화 보정·평가가 공유하는 입력 계층. label은 감사/평가용 메타데이터로 본문과 분리한다.
# 원문 text와 비교용 해시는 용도가 다르므로, 중복 검사에 쓴 정규화 문자열로 실제 입력을 덮어쓰지 않는다.

from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

DEFAULT_TEACHER_MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"


# 전사 본문·정답·출처 메타데이터를 분리한 공통 입력 한 건. frozen은 실행 중 값 변경을 막는다.
@dataclass(frozen=True)
class InputExample:
    example_id: str
    text: str
    gold_label: int | None
    source: dict[str, Any]


# 텍스트를 UTF-8 바이트로 해시한다. 공백 정규화 여부는 이 함수가 아니라 호출자가 결정한다.
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path):
    """대형 파일을 1MiB씩 해시한다. 파일 동일성은 학습 출처의 증명이 아니다."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_sha256(model):
    """모델 폴더의 내용으로 실행을 식별해, 이동한 모델도 검증하고 다른 체크포인트의 예측이 섞이지 않게 한다.

    기존 평가와 같은 파일 선택·정렬·바이트 연결 순서를 유지한다. JSON 토크나이저와 설정도 포함된다.
    """
    model = Path(model)
    weights = set(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("로컬 safetensors 모델 폴더가 필요합니다")
    digest = hashlib.sha256()
    for path in sorted(weights | set(model.glob("*.json")) | set(model.glob("*.model"))):
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1048576), b""):
                digest.update(block)
    return digest.hexdigest()


def read_json(path):
    """UTF-8 JSON 한 파일을 읽는다. JSONL이나 모델 유효성 검사는 호출자가 따로 처리한다."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value, *, ensure_ascii=False):
    """들여쓴 JSON과 마지막 개행을 저장한다. 경로 생성/덮어쓰기 허용 여부는 호출자가 결정한다."""
    path.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2) + "\n", encoding="utf-8")


# 외부 데이터의 알려진 라벨 표기만 0/1로 변환한다. None은 정상 판정과 구분해야 한다.
def parse_binary_label(value: Any) -> int | None:
    # 알려진 0/1 표기만 인정한다. 해석 불가한 값을 임의로 '정상(0)' 처리하지 않는다.
    value = str(value).strip().lower() if value is not None else ""
    if value in {"0", "normal", "non-phishing", "non_phishing", "benign"}:
        return 0
    if value in {"1", "phishing", "voice_phishing", "voice-phishing"}:
        return 1
    return None


def stable_example_id(row: Mapping[str, Any], number: int, id_column: str, text: str) -> str:
    # 제공된 ID를 우선하고, 없으면 원문 해시와 행 번호로 재현 가능한 ID를 만든다.
    supplied = str(row.get(id_column, "")).strip()
    return supplied or f"row-{number:06d}-{sha256_text(' '.join(text.split()))[:12]}"


def iter_source_rows(path: Path) -> Iterator[dict[str, Any]]:
    """CSV/JSONL/JSON을 모두 동일한 dict 스트림으로 바꾼다."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{number}")
                yield row
        return
    if suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("A .json input must be a list of objects")
        yield from rows
        return
    raise ValueError(f"Unsupported input type: {suffix}; use CSV, JSONL, or JSON")


def load_examples(
    path: Path, text_column: str, label_column: str, id_column: str, limit: int | None
) -> Iterator[InputExample]:
    """빈 본문을 건너뛰고 나머지 메타데이터는 source에 보존한다. limit은 시험 실행용이다."""
    emitted = 0
    for number, row in enumerate(iter_source_rows(path), 1):
        text = str(row.get(text_column, "")).strip()
        if not text:
            continue
        source = {k: v for k, v in row.items() if k not in {text_column, label_column}}
        yield InputExample(
            stable_example_id(row, number, id_column, text), text,
            parse_binary_label(row.get(label_column)), source,
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def read_jsonl(path):
    """완료·부분 예측을 같은 규칙으로 읽는다. 손상된 마지막 행은 버리지 않고 실패시켜 평가 분모를 보존한다."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path, rows):
    """한 표본을 한 줄로 저장해 Python/C++ 평가기가 같은 순서로 읽게 한다. 새 입력 저장용이며 재개 append는 별도다."""
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
