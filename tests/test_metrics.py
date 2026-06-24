import pytest

from pythonic.metrics import classification_metrics


def test_metrics_respect_minority_classes() -> None:
    metrics = classification_metrics(["yes", "yes", "yes"], ["yes", "no", "maybe"])
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert classification_metrics(["yes", "no", "maybe"], ["yes", "no", "maybe"])["macro_f1"] == 1
    with pytest.raises(ValueError):
        classification_metrics([], [])
