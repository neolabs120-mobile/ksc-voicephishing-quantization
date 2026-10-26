"""모델 가중치 없이 양자화 배정·평가 분모·공개 CLI의 핵심 계약을 검사한다."""

import copy
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_mobile_gguf import inspect, q8_blocks, quantize_command
from public_pipeline import (checked_rows, cumulative_order, cumulative_plan, measures, new_directory,
                             select_cumulative, selected_blocks, validation_rows)


class PipelineTests(unittest.TestCase):
    """도구나 모델을 내려받지 않고도 틀린 배정과 불완전한 결과를 거부하는지 확인한다."""

    def fixtures(self):
        """ID가 분할 간 같아도 서로 다른 표본으로 유지하는 최소 입력·예측을 만든다."""
        inputs = [dict(id="same-id", split=s, label=1, text_sha256="abc", input_ids=[2, 3])
                  for s in ("test", "challenge")]
        rows = [dict(id=r["id"], split=r["split"], label=r["label"], text_sha256=r["text_sha256"],
                     tokens=2, tokenizer_match=True, conditional_p1=0.8, logit_0=0.1, logit_1=1.5,
                     prefill_ms=1.0, prediction=1) for r in inputs]
        return inputs, rows

    def test_block_assignment(self):
        """범위·중복·전체 정밀도와 혼합 옵션의 충돌을 막아 잘못된 레시피 실행을 예방한다."""
        self.assertEqual(selected_blocks("q4", []), [])
        self.assertEqual(selected_blocks("q8", []), list(range(18)))
        self.assertEqual(selected_blocks("mixed", [11, 3, 5]), [3, 5, 11])
        for blocks in ([18], [-1], [3, 3], [True]):
            with self.assertRaises(ValueError):
                q8_blocks(blocks)
        with self.assertRaises(ValueError):
            selected_blocks("q4", [3])

    def test_tensor_selector(self):
        """B3가 B13을 함께 선택하지 않도록 번호 경계와 공유 E8 옵션을 검사한다."""
        import re

        command = quantize_command("llama-quantize", Path("source.gguf"), Path("out.gguf"), [3, 5, 11])
        selector = command[command.index("--tensor-type") + 1].split("=", 1)[0]
        self.assertIsNotNone(re.search(selector, "blk.3.attn_q.weight"))
        self.assertIsNotNone(re.search(selector, "blk.11.ffn_down.weight"))
        self.assertIsNone(re.search(selector, "blk.13.attn_q.weight"))
        self.assertEqual(command[command.index("--token-embedding-type") + 1], "q8_0")
        self.assertNotIn("--pure", command)

    def test_actual_tensor_layout_is_checked(self):
        """양자화 명령이 맞아도 실제 저장 형식·126개 선형 가중치·임베딩 공유가 틀리면 거부한다."""
        names = [f"blk.{block}.{layer}.weight" for block in range(18)
                 for layer in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")]
        tensors = [SimpleNamespace(name=name, tensor_type=SimpleNamespace(
            name="Q8_0" if int(name.split(".")[1]) in (3, 5, 11) else "Q4_0")) for name in names]
        tensors.append(SimpleNamespace(name="token_embd.weight", tensor_type=SimpleNamespace(name="Q8_0")))
        tensors.append(SimpleNamespace(name="output_norm.weight", tensor_type=SimpleNamespace(name="F32")))
        reader = SimpleNamespace(GGUFReader=lambda path: SimpleNamespace(tensors=tensors))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"gguf": reader}):
            path = Path(directory) / "unit-fixture.gguf"
            path.write_bytes(b"unit-test fixture only")
            self.assertEqual(inspect(path, True, [3, 5, 11])["tensor_counts"]["Q8_0"], 22)
            tensors[0].tensor_type.name = "Q8_0"
            with self.assertRaises(ValueError):
                inspect(path, True, [3, 5, 11])
            tensors[0].tensor_type.name = "Q4_0"
            tensors.append(SimpleNamespace(name="output.weight", tensor_type=SimpleNamespace(name="F32")))
            with self.assertRaises(ValueError):
                inspect(path, True, [3, 5, 11])

    def test_complete_split_id_contract(self):
        """분할을 포함한 키로 대응시켜 중복 ID를 허용하되 예측 누락과 중복은 거부한다."""
        expected, rows = self.fixtures()
        self.assertEqual(checked_rows(rows, expected), rows)
        for invalid in (rows[:1], rows + rows[:1], []):
            with self.assertRaises(ValueError):
                checked_rows(invalid, expected)

    def test_invalid_numeric_or_source_results(self):
        """본문·정답·토큰·점수 변조를 거부해 잘못된 예측이 성능표에 들어가지 않게 한다."""
        expected, rows = self.fixtures()
        for field, value in (("conditional_p1", float("nan")), ("conditional_p1", 2),
                             ("logit_1", float("inf")), ("tokens", 1), ("tokenizer_match", False),
                             ("label", 0), ("text_sha256", "changed"), ("prediction", 3)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                checked_rows(changed, expected)

    def test_label_choice_and_native_invalid_are_separate(self):
        """무효 첫 토큰을 숨기지 않으면서 논문과 동일한 0/1 선택 지표를 별도 계산한다."""
        _, rows = self.fixtures()
        rows[0]["prediction"] = None
        report = measures(rows)
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["recall"], 1)

    def test_no_result_overwrite(self):
        """이전 결과가 들어 있는 폴더를 새 실험이 덮어쓰지 않도록 확인한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            new_directory(path)
            (path / "existing.json").write_text("{}")
            with self.assertRaises(ValueError):
                new_directory(path)

    def test_train_overlap_exclusion_preserves_originals(self):
        """공백만 다른 학습 전사는 탐색에서 제외하고 남은 전사의 원문·두 클래스를 보존한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "E_train.csv").write_text("id,text,label\nt,같은   통화,1\n", encoding="utf-8")
            (path / "E_validation.csv").write_text(
                "id,text,label\nv0,같은 통화,1\nv1,남은  정상,0\nv2,남은  피싱,1\n", encoding="utf-8")
            kept, removed = validation_rows(path)
            self.assertEqual(removed, ["v0"])
            self.assertEqual([r.text for r in kept], ["남은  정상", "남은  피싱"])

    def test_cli_help_without_model_dependencies(self):
        """의존성 설치 전에도 공개 CLI의 단계와 인자를 확인할 수 있게 한다."""
        script = Path(__file__).resolve().parents[1] / "scripts/public_pipeline.py"
        for command in ([], ["convert"], ["prepare-inputs"], ["quantize"], ["evaluate"], ["sweep"], ["cumulative"]):
            result = subprocess.run([sys.executable, str(script), *command, "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_recorded_cumulative_ranking_and_plan(self):
        """공개한 실제 Validation 수치가 연구 당시 순위와 17개 누적 후보를 그대로 만드는지 확인한다."""
        path = Path(__file__).resolve().parents[1] / "results/single_block_validation.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = [dict(block=int(r["q8_block"]), recall=float(r["recall"]), macro_f1=float(r["macro_f1"]),
                         fpr=float(r["false_positive_rate"]), invalid=int(r["native_invalid_first_token_count"]))
                    for r in csv.DictReader(stream) if r["q8_block"]]
        order = cumulative_order(rows)
        self.assertEqual(order, [11, 3, 5, 10, 6, 0, 14, 17, 13, 15, 1, 8, 16, 12, 4, 2, 7, 9])
        plan = cumulative_plan(order)
        self.assertEqual(len(plan), 17)
        self.assertEqual(plan[0]["blocks"], [3, 11])
        self.assertEqual(plan[1]["blocks"], [3, 5, 11])
        self.assertEqual(plan[-1]["blocks"], list(range(18)))
        with self.assertRaises(ValueError):
            cumulative_order(rows[:-1])

    def test_recorded_cumulative_selection(self):
        """저장된 실제 17개 후보의 Challenge FPR 조건과 Recall로 K03이 유일하게 남는지 확인한다."""
        path = Path(__file__).resolve().parents[1] / "results/cumulative_candidates.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = [dict(k=int(r["q8_block_count"]), challenge=dict(recall=float(r["recall"]),
                    fpr=float(r["false_positive_rate"]), invalid=int(r["native_invalid_first_token_count"])))
                    for r in csv.DictReader(stream) if r["split"] == "challenge"]
        self.assertEqual(len(rows), 17)
        selected = select_cumulative(rows, 0.05)
        self.assertEqual([r["k"] for r in selected["selected_candidates"]], [3])
        self.assertFalse(selected["independent_final_evaluation"])

    def test_selection_gate_retains_ties_and_excludes_invalid(self):
        """FPR 경계는 허용하고 무효 후보는 제외하며 Recall 동률의 추가 순위를 임의로 만들지 않는다."""
        rows = [dict(k=k, challenge=dict(recall=recall, fpr=fpr, invalid=invalid))
                for k, recall, fpr, invalid in ((2, 0.9, 0.05, 0), (3, 0.9, 0.04, 0),
                                               (4, 1.0, 0.06, 0), (5, 1.0, 0.01, 1))]
        self.assertEqual([r["k"] for r in select_cumulative(rows, 0.05)["selected_candidates"]], [2, 3])
        with self.assertRaises(ValueError):
            select_cumulative(rows, 0)
        with self.assertRaises(ValueError):
            select_cumulative(rows, 1.1)


if __name__ == "__main__":
    unittest.main()
