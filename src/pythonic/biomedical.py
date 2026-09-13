from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pythonic.data import DATA_REVISION, DATA_SHA256, DECISIONS, Decision, Paper
from pythonic.experiments import metadata, save_json
from pythonic.lora import inject_lora
from pythonic.metrics import classification_metrics
from pythonic.training import TrainConfig

MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
MODEL_REVISION = "d673b8835373c6fa116d6d8006b33d48734e305d"
Strategy = Literal["original", "question_only", "first", "results_first"]


def evidence_text(paper: Paper, strategy: Strategy = "original") -> str:
    if strategy == "question_only":
        return ""
    sections = list(paper.sections)
    if strategy == "first":
        sections = sections[:1]
    elif strategy == "results_first":
        sections.sort(key=lambda section: "RESULT" not in section.label.upper())
    elif strategy != "original":
        raise ValueError("Unknown evidence strategy.")
    return "\n".join(section.text for section in sections)


class EvidenceClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, width: int, rank: int = 8) -> None:
        super().__init__()
        self.encoder = encoder
        inject_lora(self.encoder, rank)
        self.classifier = nn.Linear(width, len(DECISIONS))

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        hidden = self.encoder(**inputs).last_hidden_state
        return self.classifier(hidden[:, 0])


def load_biomedical(device: str = "cpu", rank: int = 8) -> tuple[EvidenceClassifier, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    encoder = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = EvidenceClassifier(encoder, encoder.config.hidden_size, rank).to(device)
    return model, tokenizer


def encode_evidence(
    tokenizer: Any,
    papers: tuple[Paper, ...],
    budget: int,
    device: torch.device,
    strategy: Strategy = "original",
) -> dict[str, Tensor]:
    if not 8 <= budget <= 512:
        raise ValueError("Token budget must be between 8 and 512.")
    encoded = tokenizer(
        [paper.question for paper in papers],
        [evidence_text(paper, strategy) for paper in papers],
        padding=True,
        truncation=True,
        max_length=budget,
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


@dataclass(frozen=True)
class EvidencePrediction:
    decision: Decision
    confidence: float
    tokens: int


@torch.no_grad()
def predict_evidence(
    model: EvidenceClassifier,
    tokenizer: Any,
    paper: Paper,
    budget: int = 256,
    strategy: Strategy = "original",
) -> EvidencePrediction:
    return predict_text(model, tokenizer, paper.question, evidence_text(paper, strategy), budget)


@torch.no_grad()
def predict_text(
    model: EvidenceClassifier,
    tokenizer: Any,
    question: str,
    context: str,
    budget: int = 256,
) -> EvidencePrediction:
    if not 8 <= budget <= 512:
        raise ValueError("Token budget must be between 8 and 512.")
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(question, context, truncation=True, max_length=budget, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in encoded.items()}
    probabilities = model(inputs).softmax(-1)[0]
    index = int(probabilities.argmax())
    return EvidencePrediction(
        DECISIONS[index], float(probabilities[index]), int(inputs["attention_mask"].sum())
    )


@torch.no_grad()
def evaluate_classifier(
    model: EvidenceClassifier,
    tokenizer: Any,
    papers: tuple[Paper, ...],
    budget: int = 256,
    strategy: Strategy = "original",
) -> dict[str, float]:
    if not papers:
        raise ValueError("Evaluation needs at least one paper.")
    predictions = [
        predict_evidence(model, tokenizer, paper, budget, strategy).decision for paper in papers
    ]
    return classification_metrics(predictions, [paper.decision for paper in papers])


def train_evidence(
    model: EvidenceClassifier,
    tokenizer: Any,
    train: tuple[Paper, ...],
    validation: tuple[Paper, ...],
    config: TrainConfig,
    output: Path,
    budget: int = 256,
    rank: int = 8,
) -> dict[str, Any]:
    if not train or not validation:
        raise ValueError("Training and validation sets must be nonempty.")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed)
    losses = []
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config.accumulation):
            indices = torch.randint(len(train), (config.batch_size,), generator=generator)
            papers = tuple(train[int(index)] for index in indices)
            inputs = encode_evidence(tokenizer, papers, budget, device)
            labels = torch.tensor(
                [DECISIONS.index(paper.decision) for paper in papers], device=device
            )
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(inputs), labels)
            scaler.scale(loss / config.accumulation).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss_sum / config.accumulation)
    metrics = evaluate_classifier(model, tokenizer, validation, budget)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    adapters = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"weights": adapters, "model_id": MODEL_ID, "revision": MODEL_REVISION, "rank": rank},
        output / "adapters.pt",
    )
    report = {
        "metadata": metadata(device, config.seed),
        "train_config": asdict(config),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rank": rank,
        "budget": budget,
        "source_revision": DATA_REVISION,
        "source_sha256": DATA_SHA256,
        "train_pmids": [paper.pmid for paper in train],
        "validation_pmids": [paper.pmid for paper in validation],
        "train_loss": losses,
        "validation": metrics,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    save_json(output / "training.json", report)
    return report


def load_evidence_checkpoint(path: Path, device: str = "cpu") -> tuple[EvidenceClassifier, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["model_id"] != MODEL_ID or state["revision"] != MODEL_REVISION:
        raise ValueError("Adapter checkpoint has a different base model.")
    model, tokenizer = load_biomedical(device, state["rank"])
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state["weights"]) != expected:
        raise ValueError("Adapter checkpoint parameters differ.")
    model.load_state_dict(state["weights"], strict=False)
    return model.eval(), tokenizer
