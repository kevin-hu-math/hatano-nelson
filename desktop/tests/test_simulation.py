from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation import SimulationConfig, make_model, simulate
from plotting import bin_spectrum, make_figure, png_bytes
from worker import run


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(n=60, degree=3, threads=1, g=0.6, save_eigenvectors=True)

    def test_validation(self):
        for config in (replace(self.config, n=11), replace(self.config, degree=60),
                       replace(self.config, n=60.0), replace(self.config, disorder_size=-1),
                       replace(self.config, solver="Near target (sparse)", eigenpairs=59)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_graph_and_reciprocal_edge_products(self):
        model = make_model(self.config)
        u, v = model.edges.T
        degrees = np.bincount(model.edges.ravel(), minlength=self.config.n)
        np.testing.assert_array_equal(degrees, 3)
        self.assertTrue(np.all(u < v))
        self.assertEqual(len(np.unique(model.edges, axis=0)), 90)
        forward, backward = np.asarray(model.matrix[u, v]).ravel(), np.asarray(model.matrix[v, u]).ravel()
        np.testing.assert_allclose(forward * backward, 1)
        np.testing.assert_allclose(np.abs(np.log(forward)), self.config.g)

    def test_independent_seed_streams(self):
        first = make_model(self.config)
        second = make_model(replace(self.config, g=1.2, disorder_size=6))
        np.testing.assert_array_equal(first.edges, second.edges)
        np.testing.assert_allclose(first.potential * 3, second.potential)
        u, v = first.edges.T
        np.testing.assert_allclose(np.asarray(first.matrix[u, v]) ** 2, np.asarray(second.matrix[u, v]))
        third = make_model(replace(self.config, orientation="Root-distance gauge"))
        np.testing.assert_array_equal(first.potential, third.potential)

    def test_two_vertex_exact_spectrum_and_norms(self):
        config = replace(self.config, n=2, degree=1, disorder_size=0)
        arrays, model, vectors, _ = simulate(config)
        np.testing.assert_allclose(arrays["eigenvalues"], [-1, 1])
        p = np.exp(2 * config.g) / (1 + np.exp(2 * config.g))
        np.testing.assert_allclose(arrays["ipr"], p**2 + (1-p)**2)
        np.testing.assert_allclose(arrays["linf"], np.sqrt(p))
        self.assertLess(arrays["residual"].max(), 1e-14)

    def test_gauge_similarity_and_saved_metrics(self):
        config = replace(self.config, orientation="Root-distance gauge", g=1.2)
        arrays, model, vectors, _ = simulate(config)
        reference, _, _, _ = simulate(replace(config, g=0))
        np.testing.assert_allclose(arrays["eigenvalues"], reference["eigenvalues"], atol=1e-12)
        vectors = vectors[:, arrays["vector_order"]]
        scale = np.exp(config.g * model.heights)
        np.testing.assert_allclose(model.matrix.toarray(), scale[:, None] * model.reciprocal.toarray() / scale[None, :])
        np.testing.assert_allclose(np.sum(np.abs(vectors)**2, axis=0), 1, atol=1e-13)
        np.testing.assert_allclose(np.sum(np.abs(vectors)**4, axis=0), arrays["ipr"])
        np.testing.assert_allclose(arrays["l4"]**4, arrays["ipr"])
        np.testing.assert_allclose(np.max(np.abs(vectors), axis=0), arrays["linf"])
        self.assertTrue(np.all(arrays["ipr"] >= 1/config.n - 1e-14))
        self.assertTrue(np.all(arrays["ipr"] <= 1 + 1e-14))
        self.assertLess(arrays["residual"].max(), 1e-13)

    def test_symmetric_and_empty_graphs(self):
        arrays, _, _, _ = simulate(replace(self.config, g=0))
        np.testing.assert_array_equal(arrays["eigenvalues"].imag, 0)
        one, _, _, _ = simulate(replace(self.config, n=1, degree=0, disorder_size=0))
        np.testing.assert_array_equal(one["eigenvalues"], [0])
        np.testing.assert_array_equal(one["ipr"], [1])
        np.testing.assert_array_equal(one["residual"], [0])

    def test_all_disorders(self):
        for law in ("Uniform", "Gaussian", "Cauchy", "Rademacher"):
            config = replace(self.config, disorder=law)
            arrays, model, _, _ = simulate(config)
            self.assertTrue(np.isfinite(model.potential).all())
            self.assertLess(arrays["residual"].max(), 1e-12)
            if law == "Rademacher":
                np.testing.assert_array_equal(np.unique(model.potential), [-2, 2])
            if law == "Uniform":
                self.assertTrue(np.all(np.abs(model.potential) <= 1))
        model = make_model(replace(self.config, disorder="Cauchy", n=1000))
        self.assertGreater(np.max(np.abs(model.potential)), 20 * self.config.disorder_size)

    def test_complex_target_against_dense(self):
        config = replace(self.config, n=100, g=0.9, tolerance=1e-11)
        dense, _, _, _ = simulate(config)
        target = 0.35 + 0.65j
        sparse, _, _, meta = simulate(replace(
            config, solver="Near target (sparse)", eigenpairs=8,
            target_real=target.real, target_imag=target.imag,
        ))
        expected = dense["eigenvalues"][np.argsort(np.abs(dense["eigenvalues"] - target))[:8]]
        distances = np.abs(sparse["eigenvalues"][:, None] - expected[None, :])
        self.assertLess(np.max(np.min(distances, axis=1)), 1e-8)
        self.assertLess(np.max(np.min(distances, axis=0)), 1e-8)
        self.assertTrue(meta["converged"])
        self.assertLess(sparse["residual"].max(), 1e-10)

    def test_real_sparse_and_largest_magnitude(self):
        config = replace(self.config, n=100, g=0, tolerance=1e-11)
        dense, _, _, _ = simulate(config)
        for mode in ("Near target (sparse)", "Largest magnitude (sparse)"):
            arrays, _, _, _ = simulate(replace(config, solver=mode, eigenpairs=6, target_imag=0.3))
            distances = np.abs(arrays["eigenvalues"][:, None] - dense["eigenvalues"][None, :])
            self.assertLess(np.min(distances, axis=1).max(), 1e-9)
            if mode == "Largest magnitude (sparse)":
                np.testing.assert_allclose(np.sort(np.abs(arrays["eigenvalues"])),
                                           np.sort(np.abs(dense["eigenvalues"]))[-6:], atol=1e-9)

    def test_singular_shift_retry(self):
        config = replace(self.config, n=20, degree=0, g=0, disorder_size=0,
                         solver="Near target (sparse)", eigenpairs=3)
        arrays, _, _, meta = simulate(config)
        self.assertTrue(any("singular" in note for note in meta["notes"]))
        np.testing.assert_allclose(arrays["eigenvalues"], 0, atol=1e-18)
        self.assertLess(arrays["residual"].max(), 1e-12)

    def test_worker_exports_and_column_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "input.json"
            config_path.write_text(json.dumps(asdict(self.config)))
            output = root / "output"
            self.assertEqual(run(config_path, output), 0)
            H = sp.load_npz(output / "hamiltonian.npz")
            vectors = np.load(output / "eigenvectors.npy", mmap_mode="r")
            with np.load(output / "result.npz") as data:
                np.testing.assert_allclose(H @ vectors, vectors * data["eigenvalues"][None, :], atol=1e-10)
                np.testing.assert_allclose(np.sum(np.abs(vectors)**4, axis=0), data["ipr"])
                csv = np.loadtxt(output / "eigenvalues.csv", delimiter=",", skiprows=1)
                np.testing.assert_allclose(csv[:, 0] + 1j * csv[:, 1], data["eigenvalues"])
                np.testing.assert_allclose(csv[:, 4], data["ipr"])
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertTrue((output / "config.json").exists())


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([-1-1j, -0.9-0.9j, 1+1j, 0+0j])
        self.metric = np.array([0.1, 0.3, 0.7, 0.5])

    def test_density_integral_and_bin_means(self):
        data = bin_spectrum(self.values, self.metric, bins=2)
        area = np.diff(data.x_edges)[:, None] * np.diff(data.y_edges)[None, :]
        self.assertAlmostEqual(float(np.sum(data.density * area)), 1)
        self.assertAlmostEqual(data.means[0, 0], 0.2)
        self.assertEqual(int(data.counts.sum()), 4)
        clipped = bin_spectrum(self.values, self.metric, bins=2, central_percent=60)
        area = np.diff(clipped.x_edges)[:, None] * np.diff(clipped.y_edges)[None, :]
        self.assertAlmostEqual(float(np.sum(clipped.density * area)), clipped.visible.sum()/4)

    def test_real_density_and_degenerate_range(self):
        values = np.array([0, 0, 1, 2], dtype=complex)
        data = bin_spectrum(values, self.metric, bins=10)
        self.assertTrue(data.real_spectrum)
        self.assertAlmostEqual(float(np.sum(data.density * np.diff(data.x_edges))), 1)
        degenerate = bin_spectrum(np.zeros(4, dtype=complex), self.metric, bins=10)
        self.assertEqual(degenerate.visible.sum(), 4)

    def test_plots_and_png_including_empty_localization(self):
        for values in (self.values, self.values.real.astype(complex)):
            for view in ("Binned mean", "Individual eigenvalues"):
                for log in (False, True):
                    data = bin_spectrum(values, self.metric, bins=10, minimum_count=100)
                    fig = make_figure(values, self.metric, data, "ipr", True, view, log)
                    self.assertEqual(len(fig.data), 2)
                    png = png_bytes(values, self.metric, data, "ipr", True, view, log)
                    self.assertTrue(png.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
