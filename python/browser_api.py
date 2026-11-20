"""Small JSON/binary interface shared by the browser worker and numerical tests."""

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import time

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp

from simulation import SimulationConfig, simulate


_last = None


def run_simulation(config_json, progress=lambda *_: None):
    global _last
    _last = None
    gc.collect()
    config = SimulationConfig(**json.loads(config_json))
    started = time.monotonic()
    arrays, model, vectors, metadata = simulate(config, progress)
    metadata.update({
        "elapsed_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "Browser Python / Pyodide 0.28.3 (single worker)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "networkx": nx.__version__},
    })
    _last = (arrays, model, vectors, metadata)
    payload = {
        "format": "hatano-nelson-browser/v1", "metadata": metadata,
        "data": {"real": arrays["eigenvalues"].real.tolist(),
                 "imag": arrays["eigenvalues"].imag.tolist(),
                 **{name: arrays[name].tolist() for name in ("l4", "linf", "ipr", "residual")}},
    }
    return json.dumps(payload, allow_nan=False)


def export_file(kind):
    if _last is None:
        raise ValueError("Run a simulation first. Imported results do not include a Hamiltonian or eigenvectors.")
    arrays, model, vectors, _ = _last
    root = Path("/tmp")
    if kind == "metrics":
        path = root / "hatano-nelson-metrics.npz"
        np.savez_compressed(path, **{k: v for k, v in arrays.items() if k != "vector_order"})
    elif kind == "matrix":
        path = root / "hatano-nelson-hamiltonian.npz"
        sp.save_npz(path, model.matrix)
    elif kind == "graph":
        path = root / "hatano-nelson-graph.npz"
        np.savez_compressed(path, edges=model.edges, potential=model.potential, heights=model.heights)
    elif kind == "vectors":
        if vectors is None:
            raise ValueError("Enable 'Keep eigenvectors for download' before running.")
        path = root / "hatano-nelson-eigenvectors.npy"
        # Write a Fortran-order .npy by columns, avoiding an extra N×k array.
        with path.open("wb") as stream:
            np.lib.format.write_array_header_2_0(stream, {
                "descr": np.lib.format.dtype_to_descr(vectors.dtype),
                "fortran_order": True, "shape": vectors.shape,
            })
            order = arrays["vector_order"]
            for start in range(0, len(order), 32):
                stream.write(vectors[:, order[start:start + 32]].tobytes(order="F"))
    else:
        raise ValueError("Unknown export type.")
    return str(path)
