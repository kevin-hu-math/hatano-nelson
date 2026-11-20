# Validation on 2026-11-19

All **21 automated tests passed**, including a background calculation through Streamlit's testing API, reopening the completed run in a fresh UI session, PNG generation, worker cancellation, simulated Windows file-lock retries, nonfatal progress-file failures, and completion detection with a locked status file. Windows file locking was simulated; this environment is Linux.

The updated plots default to the complex plane even for real eigenvalues. Regression tests verify fixed color limits [0, 1] for IPR, L⁴ and L∞ in both heat maps and individual-eigenvalue plots.

Additional UI checks exercised a full non-Hermitian run, the real-spectrum display at g = 0, Cauchy disorder with a complex sparse target, all plotting controls, and rejection of an impossible odd Nκ. The static spectral-density/IPR export was visually inspected.

## Numerical runs

All three runs used κ = 3, g = 0.6, Uniform[−1,1] onsite disorder (W = 2), independent random edge directions, seed 42, and four BLAS threads. Sparse tolerance was 10⁻⁹. Every requested eigenpair converged.

| N | Solver | Computed eigenpairs | Worker time | Maximum relative residual |
|---:|---|---:|---:|---:|
| 1,000 | Full dense | 1,000 | 0.960 s | 5.11 × 10⁻¹⁶ |
| 5,000 | Near 0 + 0.3i, sparse | 40 | 4.231 s | 5.38 × 10⁻¹³ |
| 10,000 | Largest magnitude, sparse | 32 | 6.140 s | 1.30 × 10⁻¹¹ |

The worker times include graph generation, diagonalization, statistics and file output, and exclude Python process startup. These are measurements in the development environment, not performance guarantees for another computer. Full diagonalization was checked at N = 1,000; the N = 5,000 and N = 10,000 checks computed selected subsets. A separate N = 10,000 dense job was cancelled successfully.

The small-reference tests compare sparse complex-target eigenvalues with a full dense calculation. Other tests verify finite gauge similarity, the exact two-vertex formula, all four disorder laws, eigenvector export ordering, and probability-density normalization including clipped views and empty bins.

## Environment

Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0; NetworkX 3.7; Streamlit 1.67.0; Plotly 6.9.0; Matplotlib 3.10.8. Each run records its actual numerical-library versions in `metadata.json`.
