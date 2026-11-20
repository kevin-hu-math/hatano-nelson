"""Start and stop local numerical workers without blocking Streamlit reruns."""

import atexit
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from simulation import SimulationConfig
from file_io import write_json


def default_runs_directory() -> Path:
    override = os.environ.get("HATANO_NELSON_RUNS_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "HatanoNelson" / "runs"
    return Path(__file__).parent / "runs"


class Job:
    def __init__(self, config: SimulationConfig):
        root = default_runs_directory()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run-", dir=root)).resolve()
        self.started = time.monotonic()
        (self.directory / "config.json").write_text(json.dumps(asdict(config), indent=2))
        environment = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
            environment[name] = str(config.threads)
        environment["PYTHONUNBUFFERED"] = "1"
        with (self.directory / "worker.log").open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("worker.py")),
                 "--config", str(self.directory / "config.json"),
                 "--output", str(self.directory)],
                stdout=log, stderr=subprocess.STDOUT, env=environment,
            )
        self.cancelled = False
        self.last_status = {"state": "running", "message": "Starting worker", "progress": 0.0}
        atexit.register(self.cancel)

    def status(self) -> dict:
        path = self.directory / "status.json"
        value = self.last_status.copy()
        try:
            value = json.loads(path.read_text())
            self.last_status = value.copy()
        except (OSError, ValueError):
            pass
        if self.cancelled:
            value.update(state="cancelled", message="Calculation cancelled")
        elif self.process.poll() is not None and value["state"] == "running":
            if self.process.returncode == 0 and (self.directory / "metadata.json").exists() and (self.directory / "result.npz").exists():
                value.update(state="complete", message="Calculation complete", progress=1.0)
            else:
                value.update(
                    state="failed",
                    message=(f"The worker exited with code {self.process.returncode}. "
                             "Check the worker log. An operating-system memory limit may have stopped it."),
                )
        value["elapsed_seconds"] = (
            value.get("elapsed_seconds", time.monotonic() - self.started)
            if value["state"] != "running" else time.monotonic() - self.started
        )
        return value

    def cancel(self) -> None:
        if self.process.poll() is not None:
            return
        self.cancelled = True
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        write_json(self.directory / "status.json", {
            "state": "cancelled", "message": "Calculation cancelled", "progress": 0.0,
            "elapsed_seconds": time.monotonic() - self.started,
        }, required=False)
