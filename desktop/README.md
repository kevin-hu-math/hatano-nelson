# Hatano–Nelson on random regular graphs

A local Python app for the complex spectrum and right-eigenvector localization statistics. Controls include graph size **N**, degree **κ**, nonreciprocity **g**, disorder size **W**, disorder law, seed, and solver. The interface uses Streamlit; computation runs in a separate process, with cancellation and stage updates. Changing the plot does not repeat the calculation.

## Install and run

Use Python 3.10 or newer, preferably 64-bit Python. Extract this folder and open a terminal inside it:

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually http://localhost:8501. Start with the default N = 500 full-spectrum run. Computation happens on your computer; the browser is only the interface. Stop the server with Ctrl+C.

A completed N = 1,000 example is included in `examples/full_1000`. Use its absolute folder path with **Open a saved run** to inspect the plots immediately. The corresponding static plot is `examples/spectrum.png`.

## Matrix and graph conventions

κ is the actual graph degree, so the local tree has forward branching number κ−1. The graph has N labeled vertices, with no self-loops or parallel edges. Nκ must be even and 0 ≤ κ < N. Sampling uses NetworkX's `random_regular_graph`; the app does not enforce connectedness. The number of connected components is recorded.

The hopping scale is 1, without division by √κ. In **Random edge directions**, each undirected edge independently receives a preferred direction u → v, with

```math
H_{vu}=e^g,\qquad H_{uv}=e^{-g},\qquad H_{xx}=V_x.
```

The row index is the arrival vertex. Equivalently, for u < v draw ηᵤᵥ ∈ {−1,+1} independently, then Hᵥᵤ = exp(gηᵤᵥ), Hᵤᵥ = exp(−gηᵤᵥ). In particular, HᵤᵥHᵥᵤ = 1. This convention generally has nontrivial circulation around cycles. It is a specific non-Hermitian extension to a graph, whose directions are part of the model.

**Root-distance gauge** is an optional comparison: set hₓ to graph distance from vertex 0, or the smallest vertex in each other component, and

```math
H=D(A+\operatorname{diag}V)D^{-1},\qquad D_{xx}=e^{gh_x}.
```

An edge between equal-distance shells has hopping 1. This is a finite diagonal similarity, so every eigenvalue is real and independent of g. Right eigenvectors change. The app solves the symmetric matrix and applies D to its eigenvectors using a rescaled exponential. This finite-matrix identity does not imply a bounded similarity or identical spectrum on an infinite tree.

### Disorder scale

All site potentials are real and independent. W ≥ 0 and W = 0 gives zero potential.

| Law | Distribution of Vₓ | Meaning of W |
|---|---|---|
| Uniform | Uniform[−W/2, W/2] | Full box width |
| Gaussian | Normal(0, W²) | Standard deviation |
| Cauchy | W × standard Cauchy | Scale parameter |
| Rademacher | −W or +W, each with probability 1/2 | Amplitude |

The Cauchy sample is **not truncated**. Its extreme diagonal entries can create distant eigenvalues and compress the visible bulk. Use the plotting percentile range to inspect the bulk; this does not remove eigenvalues from saved results. W is not a common standard deviation across these four laws; Cauchy disorder has no finite variance.

Independent seeded streams generate the graph, directions, standardized potential sample, and sparse solver starting vector. Keeping N, κ, and the seed fixed keeps the graph and preferred directions fixed when g or W changes. Keeping the law fixed also keeps the standardized disorder sample fixed. Exact reproducibility is expected within the same dependency versions; versions are saved.

## Solvers and large N

| Solver | Output | Main cost |
|---|---|---|
| Full spectrum (dense) | All N eigenvalues and their eigenvector statistics | O(N³) time and O(N²) RAM |
| Near target (sparse) | k eigenpairs nearest a chosen complex z, subject to convergence | Sparse LU of H−zI, then ARPACK; eigenvector storage O(Nk) |
| Largest magnitude (sparse) | k eigenpairs of largest absolute eigenvalue, subject to convergence | ARPACK matrix-vector iteration; eigenvector storage O(Nk) |

The default full run has N = 500. For non-Hermitian full diagonalization, the displayed RAM planning estimate is 64N² bytes, about 1.49 GiB at N = 5,000 and 5.96 GiB at N = 10,000. Actual peak allocations depend on LAPACK and eigenvector types; allow headroom. The symmetric solver uses a smaller estimate. The app imposes no size cap. Large full-spectrum runs can take a long time even when their matrix fits in RAM.

The sparse matrix itself has O(Nκ) entries. **Sparse LU fill-in on a random regular graph can be substantial**, so sparse shift-invert also has a memory limit in practice. Largest-magnitude mode avoids LU, but does not map out the bulk. A sparse selected subset is neither a uniform sample of the spectrum nor an estimate of the full spectral density. The plots label such output as a selected-eigenvalue histogram. Obtaining all eigenvector statistics for very large N remains an expensive task; the app does not replace those statistics by estimates.

Non-Hermitian sparse runs use complex arithmetic so a complex target really selects eigenvalues near that target. When the requested shift makes factorization exactly singular, the worker tries one small deterministic displacement and records it in the metadata. If ARPACK returns some converged pairs after reaching its limit, they are retained with an explicit note. A run with no converged pairs fails with a useful message. At a tied selection boundary, any basis for the selected eigenspace can be returned. A real-spectrum solver uses the real part of a complex target, which gives the same nearest-eigenvalue ordering.

BLAS thread count is adjustable; more threads are not always faster. A running eigensolve has no reliable percent-complete estimate. The progress bar records stages, and elapsed time continues updating. Cancellation terminates the worker. Completed results stay available while a new job runs.

## Eigenvectors and plots

Every right eigenvector v satisfies Hv = λv and is normalized to ∥v∥₂ = 1. The app measures

```math
\operatorname{IPR}(v)=\sum_x |v_x|^4,\qquad
\|v\|_4=\operatorname{IPR}(v)^{1/4},\qquad
\|v\|_\infty=\max_x|v_x|.
```

Thus L⁴ and IPR carry the same information for individual vectors under this normalization. A flat vector has IPR = 1/N, L⁴ = N⁻¹/⁴, L∞ = N⁻¹/². A vector supported on one vertex has all three equal to 1. These are right-eigenvector statistics, without any biorthogonal weighting. Repeated eigenvalues permit different eigenbases, and their individual-vector statistics can depend on the basis chosen by the solver. They do not by themselves establish an infinite-volume localization transition or a mobility edge.

The two panels use matching spectral coordinates, with the complex plane as the default: Re λ horizontally and Im λ vertically. Real eigenvalues lie along Im λ = 0. In **Plot range and binning**, choose **Automatic (real line for real spectra)** to use a one-dimensional histogram when appropriate. The first panel gives a binned density. The second gives either the arithmetic mean of the chosen statistic in each occupied bin or individual eigenvalues colored by that statistic. IPR, L⁴ and L∞ always use the same linear color scale from 0 to 1, displaying raw statistics. Empty bins are blank. A minimum bin count can suppress poorly populated localization bins. The mean L⁴ of a bin need not equal the fourth root of its mean IPR.

The density divides by the total number of computed eigenvalues and the bin area (bin width for a real spectrum). A clipped view keeps this normalization; it integrates to the displayed fraction of the computed spectrum. A sparse run is normalized by its computed subset size, not by N. The real-spectrum view uses a one-dimensional density and a statistic-versus-energy panel. Real-spectrum detection has relative imaginary tolerance 10⁻¹⁰.

Each run checks and exports

```math
r_j=\frac{\|Hv_j-\lambda_j v_j\|_2}{(\|H\|_F+|\lambda_j|)\|v_j\|_2}.
```

A small residual verifies the computed eigenpair equation; nonnormal eigenvalues can still be sensitive to perturbations. Dense routines do not guarantee an accurate eigenbasis near defective matrices or extreme nonreciprocity. The app uses double precision.

Plots support zoom and hover. Select **Prepare PNG of current plot** for a static export including the model parameters. The Plotly toolbar also provides an image download.

## Saved runs and exports

On Windows, runs are saved under `%LOCALAPPDATA%\HatanoNelson\runs\run-…`, outside a typical OneDrive source folder. On other systems, they are saved in `runs/run-…` beside the app. They remain after closing the app. To choose another root directory, set the environment variable `HATANO_NELSON_RUNS_DIR` before launching. Use **Open a saved run** and its full directory path to resume plotting after a restart. You can delete completed run folders when you no longer need them. Partial files from cancelled/failed runs are kept for inspection and are not loaded as completed results.

Progress-file replacements retry temporary Windows file locks. An unavailable progress file does not terminate the numerical calculation, and the interface can detect successful completion from the worker exit and saved result files. If an old run-directory override points inside OneDrive, remove it or choose a writable local folder. For PowerShell:

```powershell
$env:HATANO_NELSON_RUNS_DIR = "$env:LOCALAPPDATA\HatanoNelson\runs"
python -m streamlit run app.py
```

| File | Contents |
|---|---|
| `config.json` | All run parameters |
| `metadata.json` | Parameters, versions, elapsed time, residual definition, solver notes, graph statistics |
| `eigenvalues.csv` | Columns real, imag, L4, Linf, IPR, relative_residual |
| `result.npz` | Arrays eigenvalues (complex), l4, linf, ipr, residual |
| `hamiltonian.npz` | Exact simulated matrix, readable with `scipy.sparse.load_npz` |
| `graph_and_disorder.npz` | Undirected edges, potential vector, root heights (zero outside gauge mode) |
| `eigenvectors.npy` | Optional N×k array, column j corresponding to eigenvalue j |
| `status.json`, `worker.log` | Worker status and diagnostic output |

Eigenvalues are sorted by real part, then imaginary part. Every statistics column and saved eigenvector column uses that order. Saving vectors is optional because the full N×N array can be large. Read it without a full RAM allocation using `np.load(path, mmap_mode='r')`. The interface offers direct downloads of CSV, NPZ statistics and parameters; the complete matrix and optional vectors are available in the run directory.

Example analysis:

```python
from pathlib import Path
import numpy as np
import scipy.sparse as sp

run = Path("runs/run-REPLACE_WITH_YOUR_RUN")
data = np.load(run / "result.npz")
eigenvalues, ipr = data["eigenvalues"], data["ipr"]
H = sp.load_npz(run / "hamiltonian.npz")
# If vector saving was enabled:
# vectors = np.load(run / "eigenvectors.npy", mmap_mode="r")
```

## Running without the interface

The supplied example selects 80 eigenpairs near 0 + 0.2i for N = 5,000. Edit it and run:

```bash
python worker.py --config example_config.json --output runs/batch-example
```

Use a fresh output directory for each batch run. Read `status.json` for the stage and `error.txt` if the worker fails. To compute the complete spectrum set `"solver": "Full spectrum (dense)"`; eigenpairs/target fields are then ignored. For an unattended batch job, this command runs independently of Streamlit.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests check edge hopping, seeded comparisons, the exact two-vertex spectrum, the finite gauge identity, eigenvector normalization/statistics/residuals, complex-target sparse selection against a dense reference, singular-shift recovery, saved-vector ordering, and histogram normalization/means. See `VALIDATION.md` for the measured runs used to check this release.

Numerical and UI documentation: [SciPy eig](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html), [SciPy eigs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html), [NetworkX regular graphs](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.random_regular_graph.html), [Streamlit](https://docs.streamlit.io/).
