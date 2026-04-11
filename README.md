# Pythonic

Small experiments in attention, inference, and biomedical evidence.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,notebooks,biomedical,serving]'
pytest
```

For CPU-only PyTorch, install it from https://download.pytorch.org/whl/cpu first.
