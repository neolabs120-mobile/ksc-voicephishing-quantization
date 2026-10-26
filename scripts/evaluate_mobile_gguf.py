"""GGUF 평가 입력 준비, HF 변환 기준값 계산, JSONL 결과 집계.

gguf_eval.cpp가 실제 llama.cpp 추론을 담당한다. 동일 전사·프롬프트·전체 어휘 판정을
사용하며 데이터는 자르지 않는다. 설명 생성 표본은 결과를 보기 전에 고정한다.
"""

# prepare는 공통 입력 생성, reference는 HF 기준 logits, report는 저장된 GGUF 예측 집계를 담당한다.
# GGUF의 실제 추론은 gguf_eval.cpp에서 수행하므로 Python과 C++ 양쪽을 함께 읽으면 전체 흐름이 연결된다.

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from common.dataset_io import read_jsonl as read_rows, write_jsonl as write_rows
from common.dataset_io import file_sha256, load_examples, sha256_text, write_json
from common.evaluation_metrics import classification_report, label_choice
from common.student_format import chat_ids, read_prompt, user_prompt, validate_student_target


# 로컬 원본 Student 토크나이저만 사용하며 온라인의 변경된 파일을 자동 다운로드하지 않는다.
def tokenizer_at(source):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer._student_architecture = "gemma3_text"
    return tokenizer


# 실제 입력 ID와 렌더링 문자열을 함께 고정해 HF/GGUF 토크나이저 불일치를 나중에 검출한다.
def prepare(a):
    tokenizer = tokenizer_at(a.source)
    prompt = read_prompt(a.source / "student_system_prompt.txt")
    label_ids = [tokenizer.encode(str(n), add_special_tokens=False) for n in (0, 1)]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("0/1은 각각 하나의 토큰이어야 합니다")
    rows, smoke, parity, stats = [], [], [], {}
    for split in ("validation", "test", "challenge"):
        path = a.data / f"E_{split}.csv"
        subset = []
        for example in load_examples(path, "text", "label", "id", None):
            messages = [{"role": "user", "content": prompt + "\n\n" + user_prompt(example.text)}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = chat_ids(tokenizer, messages, generation=True)
            if tokenizer.encode(rendered, add_special_tokens=False) != ids:
                raise ValueError("HF 프롬프트 렌더링/토큰 ID 불일치")
            subset.append(
                dict(
                    id=example.example_id, split=split, label=example.gold_label,
                    text_sha256=sha256_text(example.text), prompt=rendered, input_ids=ids,
                    label_ids=[ids[0] for ids in label_ids],
                )
            )
        if not subset or any(r["label"] not in (0, 1) for r in subset):
            raise ValueError(f"잘못된 평가 데이터: {split}")
        if len({r["id"] for r in subset}) != len(subset):
            raise ValueError(f"중복 ID: {split}")
        # 결과와 무관하게 클래스별 첫 3개를 설명 생성 표본으로 선정한다.
        for label in (0, 1):
            group = [r for r in subset if r["label"] == label]
            smoke.extend(dict(r, generate_tokens=256) for r in group[:3])
            if split == "validation":
                ordered = sorted(group, key=lambda r: len(r["input_ids"]))
                parity.extend(ordered[i] for i in (0, len(ordered) // 2, len(ordered) - 1))
        rows.extend(subset)
        stats[split] = dict(
            n=len(subset), labels=dict(Counter(r["label"] for r in subset)),
            max_tokens=max(len(r["input_ids"]) for r in subset), csv_sha256=file_sha256(path),
        )
    if max(len(r["input_ids"]) + r.get("generate_tokens", 0) for r in rows + smoke) > 32768:
        raise ValueError("문맥 초과; 평가 입력은 자동으로 자르지 않습니다")
    a.output.mkdir(parents=True, exist_ok=True)
    for name, items in (("evaluation", rows), ("smoke", smoke), ("parity", parity)):
        write_rows(a.output / f"{name}.inputs.jsonl", items)
    write_json(a.output / "datasets.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


# 양자화 전 HF 모델의 FP32 마지막 위치 logits를 변환 검증 기준으로 저장한다.
def reference(a):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        a.source, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).eval()
    result = []
    for row in read_rows(a.input):
        start = perf_counter()
        with torch.inference_mode():
            ids = torch.tensor([row["input_ids"]])
            logits = (
                model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, -1]
                .float()
            )
        if not torch.isfinite(logits).all():
            raise ValueError("HF 원본 출력 NaN/Inf")
        l0, l1 = [float(logits[i]) for i in row["label_ids"]]
        token = int(logits.argmax())
        result.append(
            dict(
                id=row["id"], split=row["split"], label=row["label"],
                next_token=token, logit_0=l0, logit_1=l1,
                conditional_p1=float(torch.sigmoid(torch.tensor(l1 - l0))),
                seconds=perf_counter() - start,
            )
        )
        print(row["id"], result[-1], flush=True)
    write_rows(a.output, result)


# 예상 ID·정답·전사 해시를 확인한 후 지표를 계산한다. 선택된 생성 표본은 출력 형식도 검사한다.
def report(a):
    rows = read_rows(a.input)
    expected = {(r["split"], r["id"]): r for r in read_rows(a.expected)}
    actual = {(r["split"], r["id"]): r for r in rows}
    if len(actual) != len(rows) or actual.keys() != expected.keys():
        raise ValueError("누락/중복 평가 행; 완료 결과로 집계할 수 없습니다")
    for key, row in actual.items():
        original = expected[key]
        if (
            any(row[k] != original[k] for k in ("label", "text_sha256"))
            or not row["tokenizer_match"]
        ):
            raise ValueError("평가 입력 불일치")
    summary = {}
    tokenizer = tokenizer_at(a.source) if any("generated_ids" in r for r in rows) else None
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = dict(
            native=classification_report(subset),
            label_choice=classification_report(
                [dict(r, prediction=label_choice(r)) for r in subset]
            ), mean_prefill_ms=sum(r["prefill_ms"] for r in subset) / len(subset),
        )
    if tokenizer:
        for row in rows:
            row["response"] = tokenizer.decode(row["generated_ids"], skip_special_tokens=True)
            row["format_valid"] = validate_student_target(row["response"]) is not None
        summary["generation"] = dict(
            samples=len(rows), valid_format=sum(r["format_valid"] for r in rows),
            length_stops=sum(r["finish_reason"] == "length" for r in rows),
        )
        write_rows(a.output.with_suffix(".responses.jsonl"), rows)
    write_json(a.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    """입력 고정·HF 기준 계산·GGUF 집계를 별도 단계로 실행한다. 변환 오차와 양자화 효과를 구분하기 위해서다."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "reference", "report"))
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--expected", type=Path)
    a = p.parse_args()
    if a.mode == "prepare" and not a.data:
        p.error("prepare에는 --data가 필요합니다")
    if a.mode != "prepare" and not a.input:
        p.error("--input이 필요합니다")
    if a.mode == "report" and not a.expected:
        p.error("report에는 --expected 입력 기록이 필요합니다")
    globals()[a.mode](a)


if __name__ == "__main__":
    main()
