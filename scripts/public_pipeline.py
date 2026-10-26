"""공개용 실행 연결부: 기존 연구의 변환·추론·지표 함수를 이식 가능한 CLI로 연결한다.

새 실행은 사용자 모델/CSV를 사용하며, 저장된 논문 결과를 새 추론 결과로 복사하지 않는다.
임베딩/공유 출력 헤드는 E8로 고정하고 블록 선형 가중치의 정밀도만 비교한다.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from common.dataset_io import file_sha256, load_examples, read_jsonl, sha256_text, write_json, write_jsonl
from common.evaluation_metrics import classification_report, label_choice
from evaluate_mobile_gguf import prepare
from prepare_mobile_gguf import inspect, q8_blocks, quantize_command

ROOT = Path(__file__).resolve().parents[1]


def new_directory(path):
    """이전 실행을 덮어쓰지 않도록 비어 있는 출력 폴더에서만 새 실험을 시작한다."""
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def execute(command, log):
    """외부 양자화·추론의 실제 명령과 로그를 남기고 실패 시 완료 기록 생성을 중단한다."""
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, check=True)


def selected_blocks(precision, blocks):
    """전체 Q4·전체 Q8·혼합 배정을 명시적으로 구분해 잘못된 텐서 선택을 예방한다."""
    if precision != "mixed" and blocks:
        raise ValueError("--q8-blocks is only valid with --precision mixed")
    return list(range(18)) if precision == "q8" else q8_blocks(blocks if precision == "mixed" else [])


def quantize(f32, quantizer, output, precision, blocks):
    """각 후보를 같은 F32 원본에서 새로 만들고 실제 텐서 형식이 배정과 일치할 때만 확정한다."""
    f32, output = f32.resolve(), output.resolve()
    if output.exists() or output.with_suffix(".incomplete.gguf").exists():
        raise ValueError(f"Model output already exists: {output}")
    selected = selected_blocks(precision, blocks)
    inspect(f32, False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".incomplete.gguf")
    command = quantize_command(str(quantizer), f32, pending, selected)
    if precision == "q8":
        # 전체 Q8은 별도 덮어쓰기 규칙 없이 양자화기의 Q8_0 형식을 직접 사용한다.
        command = [str(quantizer), "--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0",
                   str(f32), str(pending), "Q8_0", "8"]
    execute(command, output.with_suffix(".quantize.log"))
    spec = inspect(pending, True, selected)
    pending.rename(output)
    manifest = dict(precision=precision, q8_blocks=selected, embedding="Q8_0", norms="F32",
                    source_sha256=file_sha256(f32), quantizer_sha256=file_sha256(Path(quantizer)),
                    command=command, **spec)
    write_json(output.with_suffix(".model.json"), manifest)
    return manifest


def checked_rows(predictions, expected):
    """누락·중복·라벨/본문 변경·토큰 불일치를 검출해 다른 표본이나 미완료 결과가 지표에 섞이지 않게 한다."""
    source = {(r["split"], r["id"]): r for r in expected}
    actual = {(r["split"], r["id"]): r for r in predictions}
    if not source or len(source) != len(expected) or len(actual) != len(predictions) or source.keys() != actual.keys():
        raise ValueError("Missing, duplicate, empty, or unexpected evaluation rows")
    for key, row in actual.items():
        original = source[key]
        if (any(row[k] != original[k] for k in ("label", "text_sha256"))
                or row["tokenizer_match"] is not True or row["tokens"] != len(original["input_ids"])
                or any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k])
                       for k in ("conditional_p1", "logit_0", "logit_1", "prefill_ms"))
                or not 0 <= row["conditional_p1"] <= 1 or row["prefill_ms"] < 0
                or row["prediction"] not in (0, 1, None)):
            raise ValueError(f"Evaluation input/output mismatch: {key}")
    return predictions


def measures(rows):
    """논문의 0/1 선택 지표와 전체 어휘 argmax의 무효 출력을 나눠 판정 조건을 명확하게 기록한다."""
    chosen = classification_report([dict(r, prediction=label_choice(r)) for r in rows])
    return dict(n=len(rows), macro_f1=chosen["macro_f1"], recall=chosen["phishing_recall"],
                fpr=chosen["false_positive_rate"], fn=chosen["class_1"]["fn"],
                fp=chosen["class_1"]["fp"], invalid=classification_report(rows)["invalid_outputs"])


def evaluate(model, inputs, runner, output, gpu_layers, kv):
    """실제 C++ 평가기를 실행한 뒤 전체 입력 대응과 해시를 확인하고 분할별 성능을 집계한다."""
    output = new_directory(output)
    expected = read_jsonl(inputs)
    record = dict(status="running", model_sha256=file_sha256(model), input_sha256=file_sha256(inputs),
                  runner_sha256=file_sha256(runner), gpu_layers=gpu_layers, kv=kv,
                  code_sha256={str(p.relative_to(ROOT)): file_sha256(p)
                               for p in sorted((ROOT / "scripts").rglob("*")) if p.suffix in (".py", ".cpp")})
    write_json(output / "run.json", record)
    prediction_file = output / "predictions.jsonl"
    execute([runner, model, inputs, prediction_file, gpu_layers, kv], output / "inference.log")
    rows = checked_rows(read_jsonl(prediction_file), expected)
    metrics = {split: measures([r for r in rows if r["split"] == split])
               for split in sorted({r["split"] for r in rows})}
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", dict(record, status="complete", predictions_sha256=file_sha256(prediction_file)))
    return metrics


def convert_command(args):
    """양자화 전 Gemma 체크포인트를 F32 GGUF로 변환하고 학습 프롬프트와 변환 출처를 함께 보존한다."""
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Conversion output must be outside the original checkpoint")
    output = new_directory(output)
    config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config") or (args.source / "embedding_int8.json").exists():
        raise ValueError("An unquantized checkpoint is required")
    if config.get("model_type") != "gemma3_text" or config.get("num_hidden_layers") != 18:
        raise ValueError("This recipe requires the 18-block Gemma 3 270M text model")
    converter = args.llama_dir / "convert_hf_to_gguf.py"
    model = output / "Student_F32.gguf"
    command = [sys.executable, converter, args.source, "--outtype", "f32", "--outfile", model]
    execute(command, output / "conversion.log")
    write_json(output / "conversion.json", dict(command=list(map(str, command)),
               converter_sha256=file_sha256(converter), model=inspect(model, False),
               source_files={p.name: file_sha256(p) for p in sorted(source.iterdir())
                             if p.suffix in {".safetensors", ".json", ".model", ".jinja", ".txt"}}))
    shutil.copy2(args.source / "student_system_prompt.txt", output / "student_system_prompt.txt")


def validation_rows(data):
    """원 연구와 같이 연속 공백만 정규화해 Train과 겹친 Validation을 제외하고 원문은 보존한다."""
    splits = {s: list(load_examples(data / f"E_{s}.csv", "text", "label", "id", None))
              for s in ("train", "validation")}
    if any(not rows or len({r.example_id for r in rows}) != len(rows)
           or any(r.gold_label not in (0, 1) for r in rows) for rows in splits.values()):
        raise ValueError("Train/validation must contain unique IDs and binary labels")
    trained = {" ".join(r.text.split()) for r in splits["train"]}
    kept = [r for r in splits["validation"] if " ".join(r.text.split()) not in trained]
    if {r.gold_label for r in kept} != {0, 1}:
        raise ValueError("Validation must retain both labels after train overlap exclusion")
    ids = {r.example_id for r in kept}
    return kept, [r.example_id for r in splits["validation"] if r.example_id not in ids]


def prepare_command(args):
    """전체 평가 입력과 Train 중복을 제외한 탐색용 Validation을 분리해 같은 프롬프트로 고정한다."""
    kept, removed = validation_rows(args.data)
    args.output = new_directory(args.output)
    prepare(SimpleNamespace(source=args.source, data=args.data, output=args.output))
    rows = read_jsonl(args.output / "evaluation.inputs.jsonl")
    originals = {r.example_id: r for r in kept}
    for split in ("validation", "test", "challenge"):
        subset = [r for r in rows if r["split"] == split and (split != "validation" or r["id"] in originals)]
        if split == "validation" and (len(subset) != len(kept) or any(
                r["label"] != originals[r["id"]].gold_label
                or r["text_sha256"] != sha256_text(originals[r["id"]].text) for r in subset)):
            raise ValueError("Prepared validation inputs differ from original CSV")
        write_jsonl(args.output / f"{split}.inputs.jsonl", subset)
    write_json(args.output / "validation.selection.json", dict(
        train_csv_sha256=file_sha256(args.data / "E_train.csv"),
        validation_csv_sha256=file_sha256(args.data / "E_validation.csv"),
        n=len(kept), removed_ids=removed, normalization="whitespace collapse for overlap comparison only"))


def sweep_command(args):
    """동일 Validation 입력에서 Q4 기준과 단일 Q8 블록을 비교하며 Test/Challenge로 자동 선정하지 않는다."""
    output = new_directory(args.output)
    inputs = read_jsonl(args.inputs)
    if not inputs or any(r["split"] != "validation" for r in inputs):
        raise ValueError("The sweep accepts nonempty validation inputs only")
    if len(set(args.blocks)) != len(args.blocks):
        raise ValueError("Duplicate sweep block numbers")
    q8_blocks(args.blocks)
    reports = []
    for block in [None, *sorted(args.blocks)]:
        name = "W4_ALL_E8" if block is None else f"W4_B{block:02d}_W8_E8"
        model = output / f"{name}.gguf"
        spec = quantize(args.f32, args.quantizer, model, "mixed", [] if block is None else [block])
        metrics = evaluate(model, args.inputs, args.runner, output / name, args.gpu_layers, args.kv)["validation"]
        reports.append(dict(name=name, block=block, mb=spec["bytes"] / 1e6, **metrics))
        write_json(output / "validation.json", reports)
    with (output / "validation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)


def cumulative_order(reports):
    """원 연구의 Validation Recall→F1→낮은 FPR→블록 번호 순위를 그대로 고정한다."""
    rows = [r for r in reports if r["block"] is not None]
    if sorted(r["block"] for r in rows) != list(range(18)) or any(r["invalid"] for r in rows):
        raise ValueError("All 18 valid single-block validation results are required")
    for row in rows:
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ("recall", "fpr")):
            raise ValueError("Invalid validation ranking scores")
        row["f1"] = row.get("f1", row.get("macro_f1"))
        if not isinstance(row["f1"], (int, float)) or not math.isfinite(row["f1"]) or not 0 <= row["f1"] <= 1:
            raise ValueError("Invalid validation F1 score")
    return [r["block"] for r in sorted(rows, key=lambda r: (-r["recall"], -r["f1"], r["fpr"], r["block"]))]


def cumulative_plan(order):
    """고정된 순위의 상위 2~18개로 17개 후보를 만들어 선택 단계와 실제 실행을 분리한다."""
    if sorted(order) != list(range(18)):
        raise ValueError("The ranking must contain each of the 18 blocks once")
    return [dict(k=k, blocks=sorted(order[:k]), added=order[k - 1]) for k in range(2, 19)]


def select_cumulative(reports, max_fpr):
    """관측 Challenge FPR 조건에서 Recall 최댓값을 찾되 동률 후보를 전부 남겨 숨은 선정 규칙을 만들지 않는다."""
    if not 0 <= max_fpr <= 1:
        raise ValueError("--max-fpr must be between 0 and 1")
    eligible = [r for r in reports if r["challenge"]["invalid"] == 0 and r["challenge"]["fpr"] <= max_fpr]
    if not eligible:
        raise ValueError("No candidate meets the observed Challenge FPR and output-validity conditions")
    recall = max(r["challenge"]["recall"] for r in eligible)
    return dict(max_fpr=max_fpr, recall=recall,
                selected_candidates=[r for r in eligible if r["challenge"]["recall"] == recall],
                selection_uses_challenge=True, independent_final_evaluation=False)


def cumulative_command(args):
    """Validation 순위의 누적 후보를 실제 재양자화·평가하며 Challenge 재사용에 따른 선정 편향을 기록한다."""
    if not 0 <= args.max_fpr <= 1:
        raise ValueError("--max-fpr must be between 0 and 1")
    order = cumulative_order(json.loads(args.validation_results.read_text(encoding="utf-8")))
    rows = [r for r in read_jsonl(args.inputs) if r["split"] in ("test", "challenge")]
    if {r["split"] for r in rows} != {"test", "challenge"}:
        raise ValueError("Cumulative comparison requires both Test and Challenge inputs")
    output = new_directory(args.output)
    inputs = output / "evaluation.inputs.jsonl"
    write_jsonl(inputs, rows)
    plan = cumulative_plan(order)
    write_json(output / "plan.json", dict(order=order, candidates=plan,
               validation_results_sha256=file_sha256(args.validation_results),
               input_sha256=file_sha256(args.inputs), max_fpr=args.max_fpr,
               selection_uses_challenge=True, independent_final_evaluation=False))
    reports = []
    for candidate in plan:
        model = output / f"K{candidate['k']:02d}.gguf"
        spec = quantize(args.f32, args.quantizer, model, "mixed", candidate["blocks"])
        metrics = evaluate(model, inputs, args.runner, output / model.stem, args.gpu_layers, args.kv)
        reports.append(dict(candidate, mb=spec["bytes"] / 1e6, **metrics))
        write_json(output / "results.json", reports)
    write_json(output / "selection.json", select_cumulative(reports, args.max_fpr))


def parser():
    """이식 가능한 경로 인자만 노출하며 개인 컴퓨터의 실험 폴더를 기본값으로 요구하지 않는다."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Unquantized HF Gemma -> verified F32 GGUF")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--llama-dir", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    inputs = sub.add_parser("prepare-inputs", help="Freeze tokenizer IDs, prompts and data hashes")
    inputs.add_argument("--source", type=Path, required=True)
    inputs.add_argument("--data", type=Path, required=True)
    inputs.add_argument("--output", type=Path, required=True)
    quant = sub.add_parser("quantize", help="Q4/Q8/mixed linear weights; embedding always Q8")
    quant.add_argument("--f32", type=Path, required=True)
    quant.add_argument("--quantizer", type=Path, required=True)
    quant.add_argument("--output", type=Path, required=True)
    quant.add_argument("--precision", choices=("q4", "q8", "mixed"), required=True)
    quant.add_argument("--q8-blocks", type=int, nargs="*", default=[])
    evaluation = sub.add_parser("evaluate", help="Run the actual llama.cpp evaluator and report metrics")
    evaluation.add_argument("--model", type=Path, required=True)
    sweep = sub.add_parser("sweep", help="Validation-only Q4 baseline + single-block Q8 sweep")
    sweep.add_argument("--f32", type=Path, required=True)
    sweep.add_argument("--quantizer", type=Path, required=True)
    sweep.add_argument("--blocks", type=int, nargs="+", default=list(range(18)))
    cumulative = sub.add_parser("cumulative", help="Compare 17 cumulative candidates; Challenge is reused for selection")
    cumulative.add_argument("--f32", type=Path, required=True)
    cumulative.add_argument("--quantizer", type=Path, required=True)
    cumulative.add_argument("--validation-results", type=Path, required=True)
    cumulative.add_argument("--max-fpr", type=float, default=0.05)
    for command in (evaluation, sweep, cumulative):
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--runner", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--gpu-layers", type=int, default=0)
        command.add_argument("--kv", choices=("f32", "f16"), default="f32")
    return p


def main():
    """요청한 실험 단계만 실행하고 실제 완료된 지표를 출력해 준비 단계와 추론 결과를 구분한다."""
    args = parser().parse_args()
    if args.command == "quantize":
        result = quantize(args.f32, args.quantizer, args.output, args.precision, args.q8_blocks)
        print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.model, args.inputs, args.runner, args.output, args.gpu_layers, args.kv), indent=2))
    else:
        {"convert": convert_command, "prepare-inputs": prepare_command,
         "sweep": sweep_command, "cumulative": cumulative_command}[args.command](args)


if __name__ == "__main__":
    main()
