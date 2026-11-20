"""Run a calculation in a separate process, with atomic status/result files."""

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import traceback

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp

from simulation import SimulationConfig, simulate
from file_io import write_json


def run(config_path: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def report(message: str, progress: float) -> None:
        write_json(output / "status.json", {
            "state": "running", "message": message, "progress": progress,
            "elapsed_seconds": time.monotonic() - started,
        }, required=False)

    try:
        parameters = json.loads(config_path.read_text(encoding="utf-8"))
        known = {field.name for field in fields(SimulationConfig)}
        unknown = set(parameters) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config = SimulationConfig(**parameters)
        config.validate()
        write_json(output / "config.json", asdict(config))
        arrays, model, vectors, metadata = simulate(config, report)
        report("Saving results", 0.94)
        sp.save_npz(output / "hamiltonian.npz", model.matrix)
        np.savez_compressed(
            output / "graph_and_disorder.npz", edges=model.edges,
            potential=model.potential, heights=model.heights,
        )
        if vectors is not None:
            # Memory-map the output so reordering never doubles the eigenvector
            # matrix in RAM. Column j matches eigenvalue j in every export.
            saved = np.lib.format.open_memmap(
                output / "eigenvectors.npy", mode="w+", dtype=vectors.dtype, shape=vectors.shape
            )
            order = arrays.pop("vector_order")
            for start in range(0, len(order), 32):
                stop = min(start + 32, len(order))
                saved[:, start:stop] = vectors[:, order[start:stop]]
            saved.flush()
            del saved, vectors
        temporary_result = output / "result.tmp.npz"
        np.savez_compressed(temporary_result, **arrays)
        temporary_result.replace(output / "result.npz")
        table = np.column_stack((
            arrays["eigenvalues"].real, arrays["eigenvalues"].imag,
            arrays["l4"], arrays["linf"], arrays["ipr"], arrays["residual"],
        ))
        np.savetxt(
            output / "eigenvalues.csv", table, delimiter=",", comments="",
            header="real,imag,L4,Linf,IPR,relative_residual",
        )
        metadata.update({
            "schema_version": 1,
            "elapsed_seconds": time.monotonic() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "networkx": nx.__version__,
            },
        })
        write_json(output / "metadata.json", metadata)
        write_json(output / "status.json", {
            "state": "complete", "message": "Calculation complete", "progress": 1.0,
            "elapsed_seconds": metadata["elapsed_seconds"],
        }, required=False)
        return 0
    except Exception as error:
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "status.json", {
            "state": "failed", "message": str(error), "progress": 0.0,
            "elapsed_seconds": time.monotonic() - started,
        }, required=False)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.config, arguments.output))
