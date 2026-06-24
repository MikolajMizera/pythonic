from collections.abc import Sequence

from pythonic.data import DECISIONS, Decision


def classification_metrics(
    predictions: Sequence[Decision], labels: Sequence[Decision]
) -> dict[str, float]:
    if not labels or len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have equal, nonzero lengths.")
    pairs = list(zip(predictions, labels, strict=True))
    scores = []
    for label in DECISIONS:
        tp = sum(prediction == label and gold == label for prediction, gold in pairs)
        fp = sum(prediction == label and gold != label for prediction, gold in pairs)
        fn = sum(prediction != label and gold == label for prediction, gold in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "accuracy": sum(prediction == gold for prediction, gold in pairs) / len(labels),
        "macro_f1": sum(scores) / len(scores),
    }
