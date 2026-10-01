# Pythonic

Small experiments in attention, inference, and biomedical evidence under a token budget.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,notebooks,biomedical,serving]'
pytest -q
python scripts/check_notebooks.py
```

For GPU runs, install a CUDA build of PyTorch instead of the CPU wheel. Training
defaults target one 6 GB GPU: batch size one, accumulation, short contexts, and FP16.
GPU memory and timing measurements remain pending.

```sh
pythonic download
pythonic inspect-data
pythonic train-decoder --smoke
pythonic benchmark --length 128
```

Real-data runs download the pinned BiomedBERT model on first use:

```sh
export HF_HOME=data/hf
pythonic train-decoder --device cuda --steps 1000
pythonic generate --device cuda --temperature 0.8
pythonic train-decoder --device cuda --steps 2000 --resume artifacts/decoder/decoder.pt
pythonic train-evidence --device cuda --steps 200 --budget 256
pythonic evaluate-evidence --device cuda
pythonic benchmark --device cuda --length 512
pythonic benchmark-generation --device cuda
pythonic evaluate-agent --device cuda
pythonic ask 'Does treatment reduce fever?' --device cuda
pythonic ask --resume --device cuda
```

```sh
pythonic serve --checkpoint artifacts/decoder/decoder.pt
curl -N http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"The trial ","max_new":32}'
```

Omit `--device cuda` for CPU runs. Downloads live in `data/`; checkpoints, traces,
metrics, and executed notebook copies live in `artifacts/`. Source notebooks keep
their outputs cleared. `PYTHONIC_SMOKE=1` selects their offline fixture paths.

| Notebook | Experiment |
|---|---|
| [01](notebooks/01_multihead.ipynb) | Causal multi-head attention |
| [02](notebooks/02_multiquery.ipynb) | Shared KV heads |
| [03](notebooks/03_grouped_query.ipynb) | Grouped-query memory costs |
| [04](notebooks/04_tiled_attention.ipynb) | Online softmax and SDPA |
| [05](notebooks/05_abstract_decoder.ipynb) | Byte-level abstract training |
| [06](notebooks/06_lora_evidence.ipynb) | LoRA and context budgets |
| [07](notebooks/07_cached_decode.ipynb) | KV caching and decode latency |
| [08](notebooks/08_quantization.ipynb) | INT8 storage and error |
| [09](notebooks/09_batching.ipynb) | Static and continuous scheduling |
| [10](notebooks/10_streaming.ipynb) | SSE and cancellation |
| [11](notebooks/11_sparse_retrieval.ipynb) | BM25 source retrieval |
| [12](notebooks/12_hybrid_retrieval.ipynb) | Dense retrieval and rank fusion |
| [13](notebooks/13_evidence_budget.ipynb) | Passage selection and citations |
| [14](notebooks/14_evidence_tools.ipynb) | Validated tool arguments |
| [15](notebooks/15_bounded_agent.ipynb) | Tool loops and checkpoints |
| [16](notebooks/16_evaluation.ipynb) | Task and provenance evaluation |

Data: [PubMedQA](https://github.com/pubmedqa/pubmedqa), Jin et al., EMNLP 2019.
Downloads are pinned and checked against SHA-256. Local stratified splits contain
699/148/153 papers. Conclusions and existing predictions are excluded from inputs;
questions and answer fields are excluded from retrieval indexes. Fixtures are synthetic.

Model: [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract),
Gu et al., 2020. The base stays frozen; only adapters and the classification head are saved.

The tiled reference has no fused kernel or custom backward. Quantized weights are
dequantized for each call. Scheduling uses simulated time. Citation checks establish
provenance; decision support is evaluated separately. Pretraining overlap is unmeasured.
Checkpoint recovery can replay the last read-only tool if a process fails before saving.

[CPU checks](notes/cpu-results.md) record small measured runs. Longer training and
larger evaluations are needed to assess model quality.

```sh
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/pythonic
python scripts/audit_history.py
```
