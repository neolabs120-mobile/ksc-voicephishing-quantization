"""추론 라이브러리 없이 분류 지표를 집계한다. 무효 출력은 전체 지표에서 오답이다."""

# rows의 label은 정답, prediction은 0/1 또는 None이다. None을 정상으로 바꾸지 않고 별도 무효 출력으로 센다.
# native(전체 어휘 판정)와 label_choice(0/1 중 선택)는 평가 조건이 다르므로 보고서에서 구분한다.

from collections import Counter


def label_choice(row):
    """전체 어휘 argmax가 0/1이면 동점 판정도 보존하고, 무효 출력만 다시 선택한다."""
    if row["prediction"] in (0, 1):
        return row["prediction"]
    score = row["conditional_p1"]
    if score is None:
        return None
    if score == 0.5:
        raise ValueError("0/1 외 출력의 동점: 원래 토큰 ID/로짓으로 추가 확인 필요")
    return int(score > 0.5)


# 클래스별 TP/FP/FN을 먼저 센 뒤 F1·Recall·FPR을 계산한다. 유효 출력만의 보조 지표도 따로 제공한다.
def classification_report(rows):
    counts = Counter((r["label"], r["prediction"]) for r in rows)
    classes = {}
    for label in (0, 1):
        tp = counts[label, label]
        fp = sum(n for (gold, pred), n in counts.items() if pred == label and gold != label)
        fn = sum(n for (gold, pred), n in counts.items() if gold == label and pred != label)
        classes[f"class_{label}"] = dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / max(1, 2 * tp + fp + fn))
    negatives = sum(n for (gold, _), n in counts.items() if gold == 0)
    positives = sum(n for (gold, _), n in counts.items() if gold == 1)
    valid = [r for r in rows if r["prediction"] in (0, 1)]
    correct = sum(counts[label, label] for label in (0, 1))
    auxiliary = None
    if valid:
        # ECoG/sklearn과 동일하게 유효 출력의 실제·예측 label 합집합에 대해 macro F1을 낸다.
        labels = {r[key] for r in valid for key in ("label", "prediction")}
        f1 = [
            2
            * counts[k, k]
            / max(1, sum(r["label"] == k for r in valid) + sum(r["prediction"] == k for r in valid))
            for k in labels
        ]
        auxiliary = dict(
            accuracy=correct / len(valid),
            positive_precision=counts[1, 1] / max(1, counts[0, 1] + counts[1, 1]),
            positive_recall=counts[1, 1] / max(1, counts[1, 0] + counts[1, 1]),
            macro_f1=sum(f1) / len(f1),
        )
    return dict(
        samples=len(rows), invalid_outputs=sum(r["prediction"] is None for r in rows), **classes,
        macro_f1=sum(c["f1"] for c in classes.values()) / 2, accuracy=correct / max(1, len(rows)),
        phishing_recall=counts[1, 1] / max(1, positives),
        false_positive_rate=counts[0, 1] / max(1, negatives), ecog_valid_only=auxiliary,
        valid_coverage=len(valid) / max(1, len(rows)),
    )
