# Pythonic

Small experiments in attention, inference, and biomedical evidence.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,notebooks,biomedical,serving]'
pytest
```

For CPU-only PyTorch, install it from https://download.pytorch.org/whl/cpu first.

```sh
pythonic download
pythonic inspect-data
pythonic benchmark --output artifacts/attention.json
```

Data: [PubMedQA](https://github.com/pubmedqa/pubmedqa), Jin et al., EMNLP 2019.
The download is pinned and checked against SHA-256. Splits are local experiments,
not the official leaderboard split. Conclusions and existing predictions are excluded.
Offline fixtures are synthetic.

```sh
pythonic train-decoder --smoke
pythonic train-decoder --device cuda --steps 1000
pythonic train-decoder --device cuda --steps 2000 --resume artifacts/decoder/decoder.pt
HF_HOME=data/hf pythonic train-evidence --device cuda --steps 200 --budget 256
```

BiomedBERT downloads on the first run. Its base stays frozen; only adapters and the
classification head are saved. GPU measurements are pending.
