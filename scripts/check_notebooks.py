import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/notebook-check"
    output.mkdir(parents=True, exist_ok=True)
    passed = []
    with tempfile.TemporaryDirectory(prefix="pythonic-kernel-") as temporary:
        os.environ["IPYTHONDIR"] = temporary
        environment = dict(
            os.environ,
            PYTHONIC_SMOKE="1",
            HF_HUB_OFFLINE="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=temporary,
            IPYTHONDIR=temporary,
            JUPYTER_RUNTIME_DIR=temporary,
        )
        for path in sorted((root / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            manager = KernelManager(kernel_name="python3", transport="ipc")
            manager.kernel_spec.argv[0] = sys.executable
            client = NotebookClient(
                notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}}
            )
            try:
                client.execute(env=environment)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            nbformat.write(notebook, output / path.name)
            passed.append(path.name)
            print(f"Passed {path.name}", flush=True)
    (output / "summary.json").write_text(json.dumps({"passed": passed}, indent=2) + "\n")


if __name__ == "__main__":
    main()
