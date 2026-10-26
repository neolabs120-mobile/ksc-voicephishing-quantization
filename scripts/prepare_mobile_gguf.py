"""학습 원본 → F32 GGUF → 지정 블록 Q8 / 공유 임베딩 Q8 / 나머지 Q4_0.

GPTQ 체크포인트를 재양자화하지 않는다. llama.cpp b9010으로 검증한 배포 레시피다.
원본은 읽기만 하며, 출력 폴더에는 명령·해시·실제 텐서 형식을 함께 기록한다.
"""

# HF 원본을 F32 GGUF로 바꾼 뒤 지정 레시피로 새 GGUF를 만든다. Q4 파일의 일부를 되돌리는 방식이 아니다.
# quantize_command는 양자화기 인자만 구성하고 inspect는 실제 텐서 형식을 독립적으로 확인한다.

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

from common.dataset_io import file_sha256


# 긴 변환/양자화 로그를 파일로 남긴다. 하위 명령이 실패하면 완료 모델로 계속 진행하지 않는다.
def run(command, log):
    print("실행:", command, flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def q8_blocks(blocks):
    """단일 번호·여러 번호·None을 같은 형식으로 검사한다. None/빈 목록은 전체 Q4다."""
    blocks = [] if blocks is None else [blocks] if isinstance(blocks, int) else list(blocks)
    if any(type(b) is not int or not 0 <= b < 18 for b in blocks) or len(set(blocks)) != len(
        blocks
    ):
        raise ValueError("8비트 블록은 중복 없는 0~17 번호여야 합니다")
    return sorted(blocks)


def inspect(path, quantized, eight_bit_block=11):
    """파일명·설정값을 믿지 않고 실제 저장된 126개 선형 가중치의 형식을 검사한다."""
    from gguf import GGUFReader

    selected = q8_blocks(eight_bit_block)
    reader = GGUFReader(path)
    layout = {t.name: t.tensor_type.name for t in reader.tensors}
    linear = re.compile(r"blk\.(\d+)\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight")
    blocks = Counter()
    for name, kind in layout.items():
        match = linear.fullmatch(name)
        if match:
            block = int(match[1])
            blocks[block] += 1
            expected = ("Q8_0" if block in selected else "Q4_0") if quantized else "F32"
        elif name == "token_embd.weight":
            expected = "Q8_0" if quantized else "F32"
        else:
            expected = "F32"
        if kind != expected:
            raise ValueError(f"비트 배정 불일치: {name}: {kind} != {expected}")
    if blocks != Counter({i: 7 for i in range(18)}):
        raise ValueError(f"Gemma 18개 블록/126개 선형 가중치 불일치: {blocks}")
    if "token_embd.weight" not in layout or "output.weight" in layout:
        raise ValueError("입력 임베딩과 출력 헤드는 한 벌의 가중치를 공유해야 합니다")
    return dict(
        bytes=path.stat().st_size, sha256=file_sha256(path),
        tensor_counts=dict(Counter(layout.values())), tensors=layout,
    )


def quantize_command(quantizer, source, output, eight_bit_block=11):
    """지정 번호들의 선형 가중치만 Q8로 보존한다. 임베딩/헤드는 항상 공유 Q8이다."""
    selected = q8_blocks(eight_bit_block)
    command = [quantizer, "--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"]
    if selected:
        # 괄호와 양쪽 점으로 번호를 한정해 B3가 B13을, B1이 B10~17을 잡지 않게 한다.
        number = (
            str(selected[0]) if len(selected) == 1 else "(" + "|".join(map(str, selected)) + ")"
        )
        command += ["--tensor-type", rf"^blk\.{number}\..*weight$=q8_0"]
    # b9010에서는 --pure가 개별 텐서 지정까지 무시한다. 저장 뒤 inspect로 전체 배정을 확인한다.
    return command + [str(source), str(output), "Q4_0", "8"]


def main():
    """양자화 전 Student에서 GGUF를 만들고 지정 비트 배정을 검증한다. GPTQ 가중치를 다시 양자화하지 않는다."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--llama-dir", required=True, type=Path)
    p.add_argument("--quantizer", default="llama-quantize")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--blocks", nargs="*", type=int, default=[11], help="Q8 블록 번호(0부터)")
    a = p.parse_args()
    blocks = q8_blocks(a.blocks)
    source, output = a.source.resolve(), a.output.resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config") or (source / "embedding_int8.json").exists():
        p.error("양자화 전 Student 원본이 필요합니다")
    if config.get("num_hidden_layers") != 18 or config.get("model_type") != "gemma3_text":
        p.error("이 레시피는 Gemma 3 270M용입니다")
    if output == source or output.is_relative_to(source):
        p.error("원본 외부의 새 출력 폴더를 지정하세요")
    output.mkdir(parents=True, exist_ok=True)
    # 기존 B11 파일명은 유지한다. 최종 전달 조합은 --blocks 3 5 11로 명시한다.
    name = "Gemma_GGUF_B11Q8_E8" if blocks == [11] else (
        "Gemma_Q8_" + ("_".join(f"B{b:02d}" for b in blocks) or "NONE") + "_E8"
    )
    f32, mixed = output / "Student_F32.gguf", output / f"{name}.gguf"
    if f32.exists() or mixed.exists():
        p.error("완료/미완료 GGUF를 덮어쓰지 않습니다. 새 출력 폴더를 지정하세요")

    converter = a.llama_dir.resolve() / "convert_hf_to_gguf.py"
    commands = [
        [sys.executable, str(converter), str(source), "--outtype", "f32", "--outfile", str(f32)],
        quantize_command(a.quantizer, f32, mixed, blocks),
    ]
    manifest = dict(
        recipe="selected blocks=Q8_0; other linear=Q4_0; shared embedding/head=Q8_0; norms=F32",
        blocks=blocks, source=str(source), commands=commands,
        converter_sha256=file_sha256(converter),
        source_files={
            p.name: file_sha256(p)
            for p in sorted(source.iterdir())
            if p.suffix in {".safetensors", ".json", ".model", ".jinja", ".txt"}
        }, status="running",
    )
    record = output / "conversion.json"
    record.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for command, name in zip(commands, ("convert", "quantize")):
        run(command, output / f"{name}.log")
    manifest.update(
        status="verified",
        models={"f32": inspect(f32, False), "mixed": inspect(mixed, True, blocks)},
    )
    record.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "student_system_prompt.txt").write_bytes(
        (source / "student_system_prompt.txt").read_bytes()
    )
    print(
        json.dumps(
            {
                n: {k: v for k, v in spec.items() if k != "tensors"}
                for n, spec in manifest["models"].items()
            }, indent=2,
        )
    )


if __name__ == "__main__":
    main()
