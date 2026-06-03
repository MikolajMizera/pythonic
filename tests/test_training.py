from pathlib import Path

import torch

from pythonic.data import fixtures
from pythonic.model import DecoderConfig
from pythonic.training import TrainConfig, load_decoder, train_decoder


def test_resumed_training_matches_uninterrupted_training(tmp_path: Path) -> None:
    model_config = DecoderConfig(width=16, layers=1, context=8)
    papers = fixtures()
    full, _ = train_decoder(
        papers, papers, model_config, TrainConfig(steps=4, accumulation=2), tmp_path / "full"
    )
    train_decoder(
        papers, papers, model_config, TrainConfig(steps=2, accumulation=2), tmp_path / "partial"
    )
    resumed, _ = train_decoder(
        papers,
        papers,
        model_config,
        TrainConfig(steps=4, accumulation=2),
        tmp_path / "resumed",
        tmp_path / "partial/decoder.pt",
    )
    loaded = load_decoder(tmp_path / "resumed/decoder.pt")
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)
