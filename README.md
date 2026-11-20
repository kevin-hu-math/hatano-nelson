# Hatano–Nelson for GitHub Pages

This is a static website. It computes random-regular-graph Hatano–Nelson spectra in your browser using the same Python numerical core as the desktop app, compiled Python/SciPy through Pyodide, and a Web Worker. There is no Python server to install or run on GitHub Pages. The package also contains the updated desktop app in `desktop/`.

## Publish on GitHub Pages

1. Extract the ZIP. Upload the **contents** of the extracted website folder to a GitHub repository. `index.html` must be at the repository root, beside `app.mjs`, `worker.mjs`, `plots.mjs`, `styles.css`, `example.json`, `assets/`, and `python/`.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**. Select **main** and **/(root)**, then **Save**. If your branch has a different name, select that branch.
4. Wait for GitHub's deployment to complete, then open the site URL shown on that page.

No npm build or Python installation is required for hosting. `.nojekyll` is included. All application paths are relative, so a project site such as `https://YOURNAME.github.io/YOURREPOSITORY/` works. Upload the files themselves rather than uploading the ZIP as one repository file. For an existing repository that hosts another page, you can instead place the website files in `docs/` and select **/docs** as the publishing folder.

GitHub's [Pages publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) describe the current controls and repository requirements.

## Use it

Choose N, degree κ, g, disorder size W, disorder law, seed, and solver. Press **Run simulation**. The first calculation downloads the pinned Pyodide 0.28.3 runtime and NumPy/SciPy/NetworkX libraries from jsDelivr; subsequent calculations reuse the initialized runtime. Network speed affects the first startup. The worker sends stage updates while the page remains responsive. **Cancel** terminates the worker immediately; the next calculation initializes a fresh one.

The default plot is the **complex plane**, with Re λ horizontally and Im λ vertically. Real eigenvalues appear on Im λ = 0. Select **Automatic (real line for real spectra)** for the one-dimensional view when the spectrum is numerically real.

The localization panel can show a binned mean or individual eigenvalues colored by IPR, L⁴, or L∞. **All these statistics always use a linear color scale from 0 to 1.** Values are the raw statistics of ℓ²-normalized right eigenvectors. The empirical-density panel has its own density scale. Plot controls do not repeat diagonalization. Empty bins are blank; setting a minimum count masks only the localization bins. Percentile trimming changes only the view, with normalization retained by the total number of computed eigenvalues.

**Load N = 1,000 example** displays an included completed desktop run immediately, without downloading Python. **Import saved JSON or desktop CSV** loads a browser JSON export or the desktop app's `eigenvalues.csv`. CSV lacks graph parameters and a full/partial-spectrum label, so the app explicitly labels its histogram as describing the imported eigenvalues. This also lets you compute large cases with native Python and inspect them on the website.

## Model

κ is the actual graph degree; Nκ must be even and 0 ≤ κ < N. The graph is simple and sampled with NetworkX. Hopping is unnormalized. In the default model, every edge independently gets a preferred direction u → v, and Hᵥᵤ = exp(g), Hᵤᵥ = exp(−g). The diagonal is real independent disorder:

| Law | Potential | Meaning of W |
|---|---|---|
| Uniform | Uniform[−W/2,W/2] | Full box width |
| Gaussian | Normal(0,W²) | Standard deviation |
| Cauchy | W × standard Cauchy | Scale, without truncation |
| Rademacher | ±W with equal probabilities | Amplitude |

The optional **Root-distance gauge** uses H = D(A + diag V)D⁻¹ with Dₓₓ = exp(ghₓ). Its finite spectrum stays real and independent of g; same-shell edges have hopping 1. The app solves the reciprocal symmetric operator and transforms its right eigenvectors stably. This finite similarity does not assert identical spectra of infinite-tree operators.

Independent seeded streams generate the graph, directions and potential. Keeping N, κ and the seed fixed keeps the graph and directions fixed as g or W changes. Reproducibility across different NetworkX/NumPy versions is not assumed; actual versions are exported. See `desktop/README.md` for further model and solver details.

## Computation and browser limits

| Solver | Eigenvalues returned | Main resource cost |
|---|---|---|
| Full spectrum (dense) | All N | O(N³) time, O(N²) RAM |
| Near target (sparse) | k near a complex target | Sparse LU fill-in plus O(Nk) eigenvector storage |
| Largest magnitude (sparse) | k with largest absolute values | ARPACK iteration and O(Nk) storage |

Selected sparse eigenvalues do not estimate the full spectral distribution. Their histograms are labelled accordingly. Sparse complex-target calculations use complex arithmetic; a singular shift gets one recorded deterministic displacement. Partially converged ARPACK results are retained with an explicit note.

Start with N = 400. The runtime uses wasm32, with at most 4 GiB of WebAssembly linear memory and possibly lower practical browser limits; JavaScript plots, package initialization and downloads also consume memory. Full non-Hermitian diagonalization has a planning estimate of 64N² bytes, before this extra overhead. Sparse LU can also be large on random regular graphs. The app has no fixed N cap, but cannot provide native Python's memory capacity or guarantee a particular maximum N. For very large jobs, run the supplied desktop app and import its CSV.

Current desktop Chrome, Edge, Firefox, or Safari with WebAssembly and module-worker support is recommended. Computation uses one numerical worker; a hidden/background tab may run more slowly. Keep the tab open while it computes. No SharedArrayBuffer, cross-origin-isolation headers, server database, API key, or special GitHub Actions build is required.

## Downloads and persistence

Plots and computed data live in the tab's memory. Download before closing or reloading. No simulation data are uploaded to a computation server.

- **Saved run (JSON):** Eigenvalues, statistics, parameters, versions and notes. Re-import to restore plots.
- **Eigenvalues + metrics (CSV):** real, imag, L4, Linf, IPR, relative_residual.
- **Current plot (PNG):** Both panels, with the fixed statistic color range.
- **Metrics (NPZ):** Complex eigenvalues and the statistics arrays, readable in NumPy.
- **Hamiltonian / graph (NPZ):** Exact sparse matrix (`scipy.sparse.load_npz`) or graph edges, potential and heights (`numpy.load`).
- **Eigenvectors (NPY):** Optional normalized right eigenvectors in exported eigenvalue order. Enable **Keep eigenvectors for download** before running. The file may be large.

Binary exports require the corresponding simulation to remain in the worker. Imported JSON/CSV contains no Hamiltonian or eigenvectors. **Release numerical runtime** frees worker memory while keeping the plotted data and JSON/CSV downloads. Starting another simulation replaces retained numerical data. Cancelling a simulation also releases that worker. Export files one at a time to avoid excess browser memory use.

Each right eigenvector has ∥v∥₂ = 1. IPR = ∑|vₓ|⁴, L⁴ = IPR¹/⁴, and L∞ = max|vₓ|. The L⁴ and IPR statistics are related, though the mean L⁴ in a bin need not equal the fourth root of its mean IPR. Degenerate eigenspaces permit basis-dependent individual-vector statistics. The exported residual is ∥Hv−λv∥₂/((∥H∥F+|λ|)∥v∥₂); a small residual does not bound nonnormal eigenvalue sensitivity.

## Updated desktop app and the Windows fix

The `desktop/` folder contains the Streamlit app, including complex-plane default plots and fixed 0–1 statistic colors. From that folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, run outputs now go to `%LOCALAPPDATA%\HatanoNelson\runs`, outside the source folder and typical OneDrive synchronization. Atomic JSON writes retry temporary permission locks, use unique temporary files, and allow the numerical calculation to continue if a progress update remains locked. Successful completion is detected even if the final status update could not be written. The website uses browser memory and does not write a Windows progress file.

## Local preview and developer tests

To serve this folder locally:

```bash
python -m http.server 8000
```

Open http://localhost:8000. Opening `index.html` directly using `file://` cannot load a module worker. Node and npm are needed only to run the JavaScript tests:

```bash
npm install
npm test
npm run test:wasm
npm run test:worker
```

The WebAssembly test downloads its numerical packages on first use and executes the actual Pyodide SciPy solver in Node. Desktop tests: `cd desktop` then `python -m unittest discover -s tests -v`. See `VALIDATION.md` for what was checked. Sources are unbundled and editable: `app.mjs` is the UI, `plots.mjs` is binning and plot specification, `worker.mjs` loads Python, and `python/` is the numerical implementation.

Runtime documentation: [Pyodide 0.28.3 worker usage](https://pyodide.org/en/0.28.3/usage/webworker.html). Plotly.js is vendored locally; Pyodide and the numerical wheels are served from jsDelivr. Third-party notices are in `THIRD_PARTY.md` and `assets/`.
