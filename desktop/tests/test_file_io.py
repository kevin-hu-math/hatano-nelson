import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from file_io import write_json
from jobs import Job, default_runs_directory
from plotting import bin_spectrum, make_figure
import numpy as np


class RegressionTests(unittest.TestCase):
    def test_retry_a_temporary_windows_lock(self):
        real_replace = os.replace
        attempts = []
        def replace(source, destination):
            attempts.append(source)
            if len(attempts) < 3:
                raise PermissionError("simulated Windows file lock")
            return real_replace(source, destination)
        with tempfile.TemporaryDirectory() as root, patch("file_io.os.replace", side_effect=replace), patch("file_io.time.sleep"):
            path = Path(root) / "status.json"
            self.assertTrue(write_json(path, {"state": "running"}))
            self.assertEqual(json.loads(path.read_text())["state"], "running")
            self.assertEqual(len(attempts), 3)
            self.assertEqual(list(Path(root).glob("*.tmp")), [])

    def test_a_locked_progress_file_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as root, patch("file_io.os.replace", side_effect=PermissionError("locked")), patch("file_io.time.sleep"):
            self.assertFalse(write_json(Path(root)/"status.json", {"state":"running"}, required=False))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_windows_default_output_is_outside_source_folder(self):
        with patch("jobs.sys.platform", "win32"), patch.dict(os.environ, {"LOCALAPPDATA":"C:/LocalData"}, clear=True):
            self.assertEqual(default_runs_directory(), Path("C:/LocalData/HatanoNelson/runs"))

    def test_success_detected_even_when_progress_file_is_locked(self):
        with tempfile.TemporaryDirectory() as root:
            job = Job.__new__(Job)
            job.directory, job.started, job.cancelled = Path(root), time.monotonic(), False
            job.process = Mock(returncode=0)
            job.process.poll.return_value = 0
            job.last_status = {"state":"running","message":"Saving results","progress":.94}
            (job.directory / "metadata.json").write_text("{}")
            (job.directory / "result.npz").write_bytes(b"")
            with patch("jobs.Path.read_text", side_effect=PermissionError("locked")):
                self.assertEqual(job.status()["state"], "complete")

    def test_complex_plane_and_fixed_color_scale_for_real_eigenvalues(self):
        values = np.array([-1, 0, 1], dtype=complex)
        metrics = np.array([.1, .2, .3])
        b = bin_spectrum(values, metrics, bins=10, force_complex=True)
        self.assertFalse(b.real_spectrum)
        self.assertEqual(b.counts.sum(), 3)
        for key in ("ipr", "l4", "linf"):
            for view in ("Binned mean", "Individual eigenvalues"):
                fig = make_figure(values, metrics, b, key, True, view)
                trace = fig.data[1]
                if trace.type == "heatmap":
                    self.assertEqual((trace.zmin, trace.zmax), (0, 1))
                else:
                    self.assertEqual((trace.marker.cmin, trace.marker.cmax), (0, 1))
