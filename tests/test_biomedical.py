import torch
from transformers import BertConfig, BertModel

from pythonic.biomedical import EvidenceClassifier, evidence_text
from pythonic.data import fixtures


def test_classifier_only_trains_adapters_and_head() -> None:
    encoder = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
        )
    )
    model = EvidenceClassifier(encoder, width=16, rank=2)
    inputs = {"input_ids": torch.randint(0, 32, (2, 8)), "attention_mask": torch.ones(2, 8)}
    model(inputs).square().sum().backward()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(name.startswith("classifier.") or name.endswith((".a", ".b")) for name in trainable)
    assert model.classifier.weight.grad is not None
    assert all(p.grad is None for name, p in model.named_parameters() if name not in trainable)


def test_question_only_has_no_evidence() -> None:
    assert evidence_text(fixtures()[0], "question_only") == ""
