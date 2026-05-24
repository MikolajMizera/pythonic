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
