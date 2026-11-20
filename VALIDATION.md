# Validation

The numerical implementation is the desktop Python solver, with its native BLAS thread-pool wrapper replaced by a no-op for the single-worker WebAssembly runtime.

## Actual WebAssembly calculations

The Pyodide 0.28.3 runtime was executed in Node 24 with its compiled numerical packages: Python 3.13.2, NumPy 2.2.5, SciPy 1.14.1, and NetworkX 3.4.2. This exercises the WebAssembly LAPACK, sparse LU, ARPACK, eigenvector statistics, JSON serialization and binary exports.

All runs used degree 3, g = 0.6, seed 42, W = 2, and random edge directions unless noted. Every requested pair converged.

| N | Solver / case | Eigenpairs | Maximum relative residual |
|---:|---|---:|---:|
| 80 | Dense, Uniform | 80 | 9.25 × 10⁻¹⁶ |
| 80 | Dense, Gaussian | 80 | 8.45 × 10⁻¹⁶ |
| 80 | Dense, Cauchy | 80 | 7.54 × 10⁻¹⁶ |
| 80 | Dense, Rademacher | 80 | 7.28 × 10⁻¹⁶ |
| 80 | Dense, root-distance gauge | 80 | 5.15 × 10⁻¹⁶ |
| 80 | Sparse, near 0.3 + 0.4i | 8 | 2.89 × 10⁻¹² |
| 500 | Dense, Uniform | 500 | 6.88 × 10⁻¹⁶ |
| 2,000 | Sparse, near 0.3 + 0.4i | 20 | 6.36 × 10⁻¹³ |

The exported eigenvector NPY was loaded back within Pyodide and checked against the exported Hamiltonian, eigenvalues and IPRs. Metrics, matrix and graph NPZ exports also completed. These checks establish operation of the compiled numerical core; they do not establish a maximum feasible graph size for every browser or device.

The production worker body was also executed in a Node harness with real Pyodide, adapting its CDN import to the installed package and local Python-file fetches to disk reads. Python-to-JavaScript progress callbacks, result messages, and a transferable binary NPY download passed.

## Plot and interface tests

Seven JavaScript tests cover density integration and clipped mass, arithmetic bin means, blank empty bins, CSV round trips and unknown full-spectrum status, graph validation, complex-plane defaults for real eigenvalues, fixed raw [0,1] metric colors for all views, and DOM control/worker-message/cancellation/import wiring. The DOM test uses jsdom with a mocked worker and Plotly renderer; the separate numerical test uses the real WebAssembly solver.

The remote browser could not access the workspace's local HTTP preview, so a rendered browser end-to-end test was unavailable. The static files are served without a build step and all asset/worker/Python paths are relative for GitHub project-page hosting.

## Desktop regression checks

All 21 desktop tests passed. New tests simulate temporary and persistent Windows permission locks, verify retry/cleanup and nonfatal progress reporting, verify successful-job detection with an unavailable status file, check the Windows output-directory choice, and check complex-plane/fixed-color rendering specifications. File locks were simulated on Linux; native Windows execution was not available. The earlier desktop full-spectrum N = 1,000 and sparse N = 5,000 / 10,000 checks are documented in `desktop/VALIDATION.md`.
