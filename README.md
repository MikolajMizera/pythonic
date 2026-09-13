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
pythonic generate --device cuda --temperature 0.8
pythonic train-decoder --device cuda --steps 2000 --resume artifacts/decoder/decoder.pt
HF_HOME=data/hf pythonic train-evidence --device cuda --steps 200 --budget 256
HF_HOME=data/hf pythonic evaluate-evidence --device cuda
```

BiomedBERT downloads on the first run. Its base stays frozen; only adapters and the
classification head are saved. GPU measurements are pending.

```sh
pythonic serve --checkpoint artifacts/decoder/decoder.pt
curl -N http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"The trial ","max_new":32}'
HF_HOME=data/hf pythonic ask 'Does treatment reduce fever?' --device cuda
HF_HOME=data/hf pythonic ask --resume --device cuda
```

Agent checkpoints are local JSON files bound to the source index. A crash between
tool completion and checkpoint writing can replay the last read-only operation.
