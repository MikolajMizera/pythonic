from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from pythonic.data import DATA_REVISION, DATA_SHA256, Paper, abstract
from pythonic.experiments import metadata, save_json
from pythonic.model import ByteTokenizer, Decoder, DecoderConfig


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 1
    accumulation: int = 16
    learning_rate: float = 3e-4
    seed: int = 42
    device: str = "cpu"
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.accumulation, self.checkpoint_every) <= 0:
            raise ValueError("Training counts must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")


def token_stream(papers: tuple[Paper, ...]) -> Tensor:
    tokens = [
        token
        for paper in papers
        for token in [*ByteTokenizer.encode(abstract(paper)), ByteTokenizer.eos]
    ]
    return torch.tensor(tokens, dtype=torch.long)


def sample_batch(
    stream: Tensor, length: int, batch: int, generator: torch.Generator, device: torch.device
) -> tuple[Tensor, Tensor]:
    if len(stream) <= length:
        raise ValueError("Corpus must contain more tokens than the training context.")
    starts = torch.randint(len(stream) - length, (batch,), generator=generator)
    inputs = torch.stack([stream[int(start) : int(start) + length] for start in starts])
    targets = torch.stack([stream[int(start) + 1 : int(start) + length + 1] for start in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def validation_loss(model: Decoder, papers: tuple[Paper, ...], batches: int = 8) -> float:
    if batches <= 0:
        raise ValueError("Validation batches must be positive.")
    stream = token_stream(papers)
    generator = torch.Generator().manual_seed(0)
    target = next(model.parameters()).device
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, labels = sample_batch(stream, model.config.context, 1, generator, target)
        losses.append(float(F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten())))
    model.train(was_training)
    return sum(losses) / len(losses)


def load_decoder(path: Path, device: str = "cpu") -> Decoder:
    state = torch.load(path, map_location=device, weights_only=True)
    model = Decoder(DecoderConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def train_decoder(
    train: tuple[Paper, ...],
    validation: tuple[Paper, ...],
    model_config: DecoderConfig,
    config: TrainConfig,
    output: Path,
    resume: Path | None = None,
) -> tuple[Decoder, dict[str, Any]]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = Decoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed)
    stream = token_stream(train)
    start = 0
    losses: list[float] = []
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        if state["model_config"] != asdict(model_config):
            raise ValueError("Checkpoint model configuration differs.")
        previous = state["train_config"]
        for field in ("seed", "batch_size", "accumulation", "learning_rate"):
            if previous[field] != asdict(config)[field]:
                raise ValueError(f"Checkpoint training configuration differs: {field}.")
        if state["train_pmids"] != [paper.pmid for paper in train]:
            raise ValueError("Checkpoint training split differs.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"].cpu())
        start = state["step"]
        losses = state["losses"]
        if start > config.steps:
            raise ValueError("Requested steps precede the checkpoint.")
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint(step: int) -> None:
        temporary = output / "decoder.tmp"
        torch.save(
            {
                "model_config": asdict(model_config),
                "train_config": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "generator": generator.get_state(),
                "step": step,
                "losses": losses,
                "train_pmids": [paper.pmid for paper in train],
            },
            temporary,
        )
        temporary.replace(output / "decoder.pt")

    model.train()
    for step in range(start, config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config.accumulation):
            inputs, labels = sample_batch(
                stream, model_config.context, config.batch_size, generator, device
            )
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten())
            scaler.scale(loss / config.accumulation).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss_sum / config.accumulation)
        if (step + 1) % config.checkpoint_every == 0:
            checkpoint(step + 1)
    checkpoint(config.steps)
    report = {
        "metadata": metadata(device, config.seed),
        "model_config": asdict(model_config),
        "train_config": asdict(config),
        "source_revision": DATA_REVISION,
        "source_sha256": DATA_SHA256,
        "train_loss": losses,
        "validation_loss": validation_loss(model, validation),
        "train_pmids": [paper.pmid for paper in train],
        "validation_pmids": [paper.pmid for paper in validation],
    }
    save_json(output / "training.json", report)
    return model, report
