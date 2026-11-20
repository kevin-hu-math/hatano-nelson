"""Exercise a real background worker through Streamlit's headless test API."""

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest
from jobs import Job
from simulation import SimulationConfig


class AppTests(unittest.TestCase):
    def test_calculate_plot_export_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HATANO_NELSON_RUNS_DIR": directory}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

            def widget(kind, label):
                return next(item for item in getattr(app, kind) if item.label == label)

            self.assertEqual(len(app.exception), 0)
            widget("number_input", "Graph size N").set_value(40)
            widget("button", "Run simulation").click().run()
            job = app.session_state["job"]
            deadline = time.monotonic() + 30
            while job.status()["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(job.status()["state"], "complete", job.status())
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            completed = app.session_state["completed_directory"]
            widget("selectbox", "Eigenvector statistic").select("L∞ norm").run()
            widget("button", "Prepare PNG of current plot").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.session_state["png_export"][1].startswith(b"\x89PNG"))
            # A new browser session can plot completed runs without another solve.
            reopened = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            next(x for x in reopened.text_input if x.label == "Run directory").set_value(completed)
            next(x for x in reopened.button if x.label == "Open saved run").click().run()
            self.assertEqual(len(reopened.exception), 0)
            self.assertEqual(len(reopened.get("plotly_chart")), 1)
            self.assertEqual(reopened.session_state["completed_directory"], completed)

    def test_worker_cancellation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HATANO_NELSON_RUNS_DIR": directory}):
            job = Job(SimulationConfig(n=10000, degree=3))
            job.cancel()
            self.assertIsNotNone(job.process.poll())
            self.assertEqual(job.status()["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
