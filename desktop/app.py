"""Local Streamlit interface. Run: python -m streamlit run app.py"""

import json
import os
from pathlib import Path

import numpy as np
import psutil
import streamlit as st

from jobs import Job
from plotting import METRICS, bin_spectrum, make_figure, png_bytes
from simulation import DISORDERS, ORIENTATIONS, SOLVERS, SimulationConfig, dense_memory_estimate


st.set_page_config(page_title="Hatano–Nelson on regular graphs", layout="wide")
st.title("Hatano–Nelson on random regular graphs")
st.caption("Unnormalized hopping, real onsite disorder, and ℓ²-normalized right eigenvectors.")


def size_label(size: int) -> str:
    return f"{size / 2**30:.2f} GiB" if size >= 2**30 else f"{size / 2**20:.1f} MiB"


@st.cache_data(show_spinner=False, max_entries=3)
def load_result(directory: str) -> tuple[dict, dict]:
    path = Path(directory)
    with np.load(path / "result.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    metadata = json.loads((path / "metadata.json").read_text())
    return arrays, metadata


job = st.session_state.get("job")
running = job is not None and job.status()["state"] == "running"
with st.sidebar:
    st.subheader("Model")
    n = int(st.number_input("Graph size N", min_value=1, value=500, step=100))
    degree = int(st.number_input("Degree κ", min_value=0, value=3, step=1))
    g = float(st.number_input("Nonreciprocity g", value=0.4, step=0.1, format="%.4f"))
    w = float(st.number_input("Disorder size W", min_value=0.0, value=2.0, step=0.5, format="%.4f"))
    disorder = st.selectbox("Disorder law", list(DISORDERS))
    st.caption(DISORDERS[disorder])
    orientation = st.selectbox("Hopping convention", list(ORIENTATIONS))
    st.caption(ORIENTATIONS[orientation])
    seed = int(st.number_input("Random seed", min_value=0, value=42, step=1))
    st.subheader("Solver")
    solver = st.selectbox("Eigenvalue selection", SOLVERS)
    eigenpairs, target_real, target_imag = 80, 0.0, 0.0
    if solver != SOLVERS[0]:
        eigenpairs = int(st.number_input("Number of eigenpairs", min_value=1, value=80, step=10))
        if solver == SOLVERS[1]:
            target_real = float(st.number_input("Target: real part", value=0.0, step=0.1, format="%.5f"))
            target_imag = float(st.number_input("Target: imaginary part", value=0.0, step=0.1, format="%.5f"))
        st.caption("Computes a selected subset. Its histogram does not estimate the full spectral distribution.")
    with st.expander("Numerical options"):
        tolerance = float(st.number_input("Sparse tolerance", min_value=1e-14, max_value=0.1,
                                          value=1e-9, format="%.1e"))
        iterations = int(st.number_input("Maximum sparse iterations", min_value=1, value=10000, step=1000))
        threads = int(st.number_input("BLAS threads", min_value=1, value=min(4, os.cpu_count() or 1), step=1))
        save_vectors = st.checkbox("Save eigenvectors to disk", value=False,
                                   help="Can create a very large file. Column j matches eigenvalue j in the exports.")
    config = SimulationConfig(
        n=n, degree=degree, g=g, disorder_size=w, disorder=disorder, orientation=orientation,
        seed=seed, solver=solver, eigenpairs=eigenpairs, target_real=target_real,
        target_imag=target_imag, tolerance=tolerance, max_iterations=iterations,
        threads=threads, save_eigenvectors=save_vectors,
    )
    if solver == SOLVERS[0]:
        st.caption(f"Dense planning estimate: about {size_label(dense_memory_estimate(config))} RAM. "
                   f"Available now: {size_label(psutil.virtual_memory().available)}. "
                   "Actual peak use varies; runtime scales roughly as N³.")
    else:
        st.caption("Sparse eigenvectors need O(Nk) storage. Shift-invert also needs an LU factorization, "
                   "which can use substantial memory on a regular graph.")
    if st.button("Run simulation", type="primary", disabled=running, width="stretch"):
        try:
            config.validate()
            st.session_state["job"] = Job(config)
            st.session_state["job_handled"] = False
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))
    with st.expander("Open a saved run"):
        saved_path = st.text_input("Run directory", placeholder="/path/to/hatano_nelson_app/runs/run-…")
        if st.button("Open saved run"):
            try:
                previous = str(Path(saved_path).expanduser().resolve())
                loaded_arrays, loaded_metadata = load_result(previous)
                if not {"eigenvalues", "l4", "linf", "ipr", "residual"}.issubset(loaded_arrays):
                    raise ValueError("This folder is missing eigenvalue statistics.")
                if "config" not in loaded_metadata:
                    raise ValueError("This folder is missing simulation parameters.")
                st.session_state["completed_directory"] = previous
                st.session_state.pop("png_export", None)
                st.rerun()
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Could not open that run: {error}")


@st.fragment(run_every="1s" if running else None)
def monitor() -> None:
    active = st.session_state.get("job")
    if active is None:
        return
    status = active.status()
    state = status["state"]
    if state == "running":
        st.write(f"{status['message']} · elapsed {status['elapsed_seconds']:.0f} s")
        st.progress(float(status["progress"]))
        st.caption("The bar marks calculation stages; it does not predict the remaining eigensolver time.")
        if st.button("Cancel calculation"):
            active.cancel()
            st.rerun()
    else:
        if not st.session_state.get("job_handled", False):
            st.session_state["job_handled"] = True
            if state == "complete":
                st.session_state["completed_directory"] = str(active.directory)
                st.session_state.pop("png_export", None)
            st.rerun()
        if state == "failed":
            st.error(status["message"])
            with st.expander("Worker details"):
                for name in ("error.txt", "worker.log"):
                    path = active.directory / name
                    if path.exists():
                        st.code(path.read_text()[-15000:], language="text")
                st.code(str(active.directory), language="text")
        elif state == "cancelled":
            st.info("Calculation cancelled. Any earlier completed result is still displayed below.")


monitor()
directory = st.session_state.get("completed_directory")
if directory is None:
    st.write("Choose the parameters and run a simulation. The default N = 500 gives a small full-spectrum example.")
else:
    arrays, metadata = load_result(directory)
    done = metadata["config"]
    count = metadata["computed_eigenvalues"]
    st.write(
        f"**Displayed run:** N = {done['n']:,}, κ = {done['degree']}, g = {done['g']:g}, "
        f"W = {done['disorder_size']:g}, {done['disorder']} disorder; {done['orientation']}. "
        f"Seed {done['seed']}; {count:,} eigenvalues; {metadata['elapsed_seconds']:.1f} s."
    )
    if not metadata["full_spectrum"]:
        st.info(f"{done['solver']}: {count:,} of {done['n']:,} eigenvalues. "
                "Density and localization plots describe this solver-selected subset.")
    for note in metadata["notes"]:
        st.warning(note)
    controls = st.columns([2, 2, 1])
    metric_name = controls[0].selectbox("Eigenvector statistic", list(METRICS))
    view = controls[1].selectbox("Localization view", ["Binned mean", "Individual eigenvalues"])
    bins = int(controls[2].number_input("Bins per axis", min_value=10, max_value=400, value=70, step=10))
    logarithmic = False
    with st.expander("Plot range and binning"):
        coordinates = st.selectbox("Spectral coordinates", ["Complex plane", "Automatic (real line for real spectra)"])
        central = float(st.number_input("Central percentile range on each axis (%)", min_value=1.0,
                                         max_value=100.0, value=100.0, step=0.5))
        minimum_count = int(st.number_input("Minimum count for a localization bin", min_value=1, value=1, step=1))
        st.caption("Percentile trimming changes only the view. All computed eigenvalues remain in the exports. "
                   "The density stays normalized by the original number computed.")
    metric_key = METRICS[metric_name]
    metric = arrays[metric_key]
    binned = bin_spectrum(arrays["eigenvalues"], metric, bins, central, minimum_count,
                          force_complex=coordinates == "Complex plane")
    figure = make_figure(arrays["eigenvalues"], metric, binned, metric_key,
                         metadata["full_spectrum"], view, logarithmic)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})
    visible = int(binned.visible.sum())
    st.caption(f"View includes {visible:,}/{count:,} eigenvalues. Empty bins are blank; localization bins show arithmetic means. "
               "Statistic colors use a fixed linear scale from 0 to 1. "
               f"Maximum relative eigenpair residual: {metadata['largest_residual']:.2e}. "
               + ("The spectrum is real to numerical precision, so a one-dimensional density is displayed."
                  if binned.real_spectrum else "The density uses area in the complex plane."))
    if not metadata["full_spectrum"]:
        st.caption("Selected-eigenvalue density integrates to one over the complete selected set, not to its fraction of N.")
    subtitle = (f"N={done['n']}, κ={done['degree']}, g={done['g']:g}, W={done['disorder_size']:g}, "
                f"{done['disorder']}; seed={done['seed']}")
    export_key = (directory, metric_key, view, bins, logarithmic, central, minimum_count, coordinates)
    if st.button("Prepare PNG of current plot"):
        with st.spinner("Rendering PNG"):
            content = png_bytes(arrays["eigenvalues"], metric, binned, metric_key,
                                metadata["full_spectrum"], view, logarithmic, subtitle)
            st.session_state["png_export"] = (export_key, content)
    prepared = st.session_state.get("png_export")
    if prepared is not None and prepared[0] == export_key:
        st.download_button("Download PNG", prepared[1], "spectrum.png", "image/png")
    with st.expander("Data and definitions"):
        st.latex(r"\|v\|_2=1,\qquad \mathrm{IPR}(v)=\sum_x |v_x|^4,\quad "
                 r"\|v\|_4=\mathrm{IPR}(v)^{1/4},\quad \|v\|_\infty=\max_x|v_x|.")
        st.write("The L⁴ norm and IPR encode the same statistic under this normalization. "
                 "A flat vector has IPR = 1/N, L⁴ = N⁻¹/⁴, L∞ = N⁻¹/²; a vector on one vertex has all three equal to 1. "
                 "Statistics within a degenerate eigenspace depend on the eigenbasis chosen by the solver.")
        st.caption("These are right-eigenvector statistics. A small residual checks the computed equation Hv = λv; "
                   "it does not bound eigenvalue sensitivity for a nonnormal matrix.")
        columns = st.columns(3)
        for column, name, label, mime in zip(
            columns, ("eigenvalues.csv", "result.npz", "config.json"),
            ("Eigenvalues + metrics (CSV)", "Eigenvalues + metrics (NPZ)", "Parameters (JSON)"),
            ("text/csv", "application/octet-stream", "application/json"),
        ):
            column.download_button(label, (Path(directory) / name).read_bytes(), name, mime=mime)
        st.write("All run files, including the sparse Hamiltonian, graph, potential and metadata, are saved here:")
        st.code(directory, language="text")
        if done["save_eigenvectors"]:
            st.caption("eigenvectors.npy contains the normalized right eigenvectors in exported eigenvalue order. "
                       "Use numpy.load(path, mmap_mode='r') to read it without loading the entire file.")
        st.json({name: metadata[name] for name in (
            "graph_edges", "graph_components", "solver_reference", "residual_definition", "versions"
        )})

with st.expander("Model convention"):
    st.write("κ is the actual graph degree. The graph is simple and sampled with NetworkX's random_regular_graph. "
             "Edges, directions and disorder use separate seeded random streams.")
    st.latex(r"H_{vu}=e^g,\quad H_{uv}=e^{-g}\quad\text{for a preferred edge }u\to v,"
             r"\qquad H_{xx}=V_x.")
    st.write("In the default convention each edge direction is independent. Products around cycles can be nonreciprocal. "
             "Root-distance gauge instead uses H = D(A + diag(V))D⁻¹, Dₓₓ = exp(ghₓ). "
             "Its eigenvalues remain real and independent of g; same-shell hopping is 1. "
             "The app solves the symmetric matrix first and transforms its eigenvectors for this gauge.")
