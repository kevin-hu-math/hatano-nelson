"""Numerics for a real Hatano--Nelson matrix on a simple regular graph."""

from dataclasses import asdict, dataclass
from typing import Callable

import networkx as nx
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits


DISORDERS = {
    "Uniform": "V ~ Uniform[-W/2, W/2]; W is the full box width.",
    "Gaussian": "V ~ Normal(0, W²); W is the standard deviation.",
    "Cauchy": "V = W × standard Cauchy; W is the scale parameter. No truncation.",
    "Rademacher": "V = -W or +W with equal probability; W is the amplitude.",
}
ORIENTATIONS = {
    "Random edge directions": (
        "Each undirected edge independently gets a preferred direction. "
        "H[target, source] is exp(g) in that direction and exp(-g) in reverse."
    ),
    "Root-distance gauge": (
        "H[x, y] = A[x, y] exp(g(h[x] - h[y])) + V[x] δ[x,y], "
        "with h the distance from vertex 0, or the smallest vertex in each other "
        "component. Same-shell edges have hopping 1. The finite spectrum is "
        "independent of g."
    ),
}
SOLVERS = (
    "Full spectrum (dense)",
    "Near target (sparse)",
    "Largest magnitude (sparse)",
)
Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class SimulationConfig:
    n: int = 500
    degree: int = 3
    g: float = 0.4
    disorder_size: float = 2.0
    disorder: str = "Uniform"
    orientation: str = "Random edge directions"
    seed: int = 42
    solver: str = SOLVERS[0]
    eigenpairs: int = 80
    target_real: float = 0.0
    target_imag: float = 0.0
    tolerance: float = 1e-9
    max_iterations: int = 10000
    threads: int = 4
    save_eigenvectors: bool = False

    def validate(self) -> None:
        for name in ("n", "degree", "seed", "eigenpairs", "max_iterations", "threads"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        if self.n < 1 or not 0 <= self.degree < self.n:
            raise ValueError("Use N ≥ 1 and an integer degree 0 ≤ κ < N.")
        if self.n * self.degree % 2:
            raise ValueError("N × κ must be even for a regular graph to exist.")
        if not np.isfinite(self.g) or abs(self.g) > 700:
            raise ValueError("g must be finite and satisfy |g| ≤ 700 in float64 arithmetic.")
        if not np.isfinite(self.disorder_size) or self.disorder_size < 0:
            raise ValueError("W must be finite and nonnegative.")
        if self.disorder not in DISORDERS or self.orientation not in ORIENTATIONS:
            raise ValueError("Unknown disorder or hopping convention.")
        if self.solver not in SOLVERS:
            raise ValueError("Unknown solver.")
        if self.seed < 0:
            raise ValueError("The seed must be a nonnegative integer.")
        if self.solver != SOLVERS[0] and not 1 <= self.eigenpairs < self.n - 1:
            raise ValueError("A sparse run needs 1 ≤ requested eigenpairs ≤ N - 2.")
        if not np.isfinite(self.target_real) or not np.isfinite(self.target_imag):
            raise ValueError("The target must be finite.")
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 1:
            raise ValueError("The sparse tolerance must lie strictly between 0 and 1.")
        if self.max_iterations < 1 or self.threads < 1:
            raise ValueError("Iterations and BLAS threads must be positive integers.")
        if not isinstance(self.save_eigenvectors, bool):
            raise ValueError("save_eigenvectors must be true or false.")


@dataclass
class Model:
    matrix: sp.csr_matrix
    reciprocal: sp.csr_matrix
    edges: np.ndarray
    potential: np.ndarray
    heights: np.ndarray
    components: int


def sample_potential(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """W has the convention recorded in DISORDERS, including for Cauchy tails."""
    if config.disorder_size == 0:
        return np.zeros(config.n)
    if config.disorder == "Uniform":
        values = rng.uniform(-0.5, 0.5, config.n)
    elif config.disorder == "Gaussian":
        values = rng.standard_normal(config.n)
    elif config.disorder == "Cauchy":
        values = rng.standard_cauchy(config.n)
    else:
        values = 2.0 * rng.integers(0, 2, config.n) - 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        potential = config.disorder_size * values
    if not np.isfinite(potential).all():
        raise ValueError("This disorder sample overflowed float64; reduce W or change the seed.")
    return potential


def make_model(config: SimulationConfig, report: Progress = lambda *_: None) -> Model:
    config.validate()
    graph_seed, direction_seed, potential_seed, _ = np.random.SeedSequence(config.seed).spawn(4)
    graph_rng = np.random.default_rng(graph_seed)
    report("Generating a simple random regular graph", 0.04)
    graph = nx.random_regular_graph(
        config.degree, config.n, seed=int(graph_rng.integers(0, 2**32 - 1))
    )
    edge_count = graph.number_of_edges()
    edges = np.fromiter(
        (vertex for edge in graph.edges() for vertex in edge),
        dtype=np.int64,
        count=2 * edge_count,
    ).reshape(-1, 2)
    edges.sort(axis=1)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    u, v = edges.T
    component_sets = list(nx.connected_components(graph))
    heights = np.zeros(config.n, dtype=np.int64)
    if config.orientation == "Root-distance gauge":
        for vertices in component_sets:
            for vertex, distance in nx.single_source_shortest_path_length(graph, min(vertices)).items():
                heights[vertex] = distance
        exponent = config.g * (heights[u] - heights[v])
    else:
        direction_rng = np.random.default_rng(direction_seed)
        directions = 2 * direction_rng.integers(0, 2, edge_count) - 1
        exponent = -config.g * directions

    report("Drawing onsite potentials and constructing the sparse matrix", 0.12)
    potential = sample_potential(config, np.random.default_rng(potential_seed))
    diagonal = np.arange(config.n)
    rows = np.concatenate((u, v, diagonal))
    columns = np.concatenate((v, u, diagonal))
    values = np.concatenate((np.exp(exponent), np.exp(-exponent), potential))
    reciprocal_values = np.concatenate((np.ones(2 * edge_count), potential))
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(config.n, config.n))
    reciprocal = sp.csr_matrix(
        (reciprocal_values, (rows, columns)), shape=matrix.shape
    )
    matrix.eliminate_zeros()
    reciprocal.eliminate_zeros()
    return Model(matrix, reciprocal, edges, potential, heights, len(component_sets))


def uses_reciprocal_solver(config: SimulationConfig) -> bool:
    return config.g == 0 or config.orientation == "Root-distance gauge"


def dense_memory_estimate(config: SimulationConfig) -> int:
    """Planning estimate, not a hard memory limit or a measured LAPACK allocation."""
    bytes_per_entry = 40 if uses_reciprocal_solver(config) else 64
    return bytes_per_entry * config.n**2


def sparse_factorization(
    matrix: sp.csc_matrix, target: complex, real_symmetric: bool, report: Progress
) -> tuple[sla.LinearOperator, complex, list[str]]:
    """Return an explicit shift-invert solve, recording any singular-shift retry."""
    notes = []
    effective = target
    step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(target))
    for attempt in range(2):
        report("Factoring H - zI for shift-invert", 0.23)
        shifted = matrix - effective * sp.eye(matrix.shape[0], format="csc")
        try:
            factor = sla.splu(shifted)
            inverse = sla.LinearOperator(matrix.shape, matvec=factor.solve, dtype=matrix.dtype)
            return inverse, effective, notes
        except RuntimeError as error:
            if attempt or "singular" not in str(error).lower():
                raise
            effective = target + (step if real_symmetric else step * (1 + 0.37j))
            notes.append(
                f"The requested shift {target} was singular. Retried at {effective}."
            )
    raise RuntimeError("Shift-invert factorization failed.")


def sparse_eigensystem(
    model: Model, config: SimulationConfig, report: Progress
) -> tuple[np.ndarray, np.ndarray, dict, list[str]]:
    symmetric = uses_reciprocal_solver(config)
    source = model.reciprocal if symmetric else model.matrix
    # Complex arithmetic is essential for actual distance-to-z selection with a
    # complex shift. Real ARPACK's default OPpart would select a different set.
    matrix = source.tocsc() if symmetric else source.astype(np.complex128).tocsc()
    _, _, _, solver_seed = np.random.SeedSequence(config.seed).spawn(4)
    rng = np.random.default_rng(solver_seed)
    initial = rng.standard_normal(config.n)
    if not symmetric:
        initial = initial + 1j * rng.standard_normal(config.n)
    parameters = dict(
        k=config.eigenpairs,
        v0=initial,
        tol=config.tolerance,
        maxiter=config.max_iterations,
        ncv=min(config.n, max(20, 2 * config.eigenpairs + 1)),
    )
    notes = []
    target_info = {}
    if config.solver == "Near target (sparse)":
        requested = complex(config.target_real, config.target_imag)
        target = float(requested.real) if symmetric else requested
        inverse, effective, factor_notes = sparse_factorization(matrix, target, symmetric, report)
        notes.extend(factor_notes)
        parameters.update(sigma=effective, OPinv=inverse, which="LM")
        target_info = {
            "requested_target": [requested.real, requested.imag],
            "effective_shift": [float(np.real(effective)), float(np.imag(effective))],
        }
        if symmetric and requested.imag != 0:
            notes.append(
                "The spectrum is real in this convention; nearest-to-complex-target "
                "selection is equivalent to nearest to its real part."
            )
    else:
        parameters["which"] = "LM"
    report("Iterating the sparse eigensolver; this stage has no iteration progress callback", 0.35)
    solver = sla.eigsh if symmetric else sla.eigs
    converged = True
    try:
        eigenvalues, eigenvectors = solver(matrix, **parameters)
    except sla.ArpackNoConvergence as error:
        eigenvalues, eigenvectors = error.eigenvalues, error.eigenvectors
        if eigenvalues is None or len(eigenvalues) == 0 or eigenvectors is None:
            raise RuntimeError(
                "ARPACK returned no converged eigenpairs. Try a different target, "
                "fewer eigenpairs, more iterations, or a larger tolerance."
            ) from error
        converged = False
        notes.append(
            f"ARPACK reached its iteration limit. Kept {len(eigenvalues)} converged "
            f"eigenpairs out of {config.eigenpairs} requested."
        )
    return eigenvalues, eigenvectors, {**target_info, "converged": converged}, notes


def normalize_and_measure(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    model: Model,
    config: SimulationConfig,
    report: Progress = lambda *_: None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Work in small column blocks to avoid several extra dense N-by-N arrays."""
    count = len(eigenvalues)
    metrics = {name: np.empty(count) for name in ("l4", "linf", "ipr", "residual")}
    matrix_norm = sla.norm(model.matrix, "fro")
    if not np.isfinite(matrix_norm):
        raise RuntimeError("The matrix norm overflowed double precision. Reduce |g| or W.")
    gauge = config.orientation == "Root-distance gauge" and config.g != 0
    for start in range(0, count, 32):
        stop = min(start + 32, count)
        block = eigenvectors[:, start:stop]
        if gauge:
            magnitude = np.abs(block)
            phase = np.divide(block, magnitude, out=np.zeros_like(block), where=magnitude > 0)
            with np.errstate(divide="ignore"):
                logarithm = np.log(magnitude) + config.g * model.heights[:, None]
            logarithm -= np.max(logarithm, axis=0)
            block[:] = phase * np.exp(logarithm)
        norms = la.norm(block, axis=0)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise RuntimeError("The eigensolver produced a zero or nonfinite eigenvector.")
        block /= norms
        probabilities = np.abs(block) ** 2
        ipr = np.sum(probabilities**2, axis=0)
        metrics["ipr"][start:stop] = ipr
        metrics["l4"][start:stop] = ipr**0.25
        metrics["linf"][start:stop] = np.sqrt(np.max(probabilities, axis=0))
        residual = model.matrix @ block - block * eigenvalues[None, start:stop]
        scale = matrix_norm + np.abs(eigenvalues[start:stop])
        metrics["residual"][start:stop] = np.divide(
            la.norm(residual, axis=0), scale,
            out=np.zeros(stop - start), where=scale > 0,
        )
        report(f"Measuring right eigenvectors ({stop:,}/{count:,})", 0.78 + 0.14 * stop / count)
    return metrics, eigenvectors


def simulate(
    config: SimulationConfig, report: Progress = lambda *_: None
) -> tuple[dict[str, np.ndarray], Model, np.ndarray | None, dict]:
    model = make_model(config, report)
    notes = []
    information = {}
    with threadpool_limits(limits=config.threads):
        if config.solver == SOLVERS[0]:
            report("Diagonalizing the full matrix; this stage can take a long time", 0.25)
            source = model.reciprocal if uses_reciprocal_solver(config) else model.matrix
            dense = source.toarray(order="F")
            if uses_reciprocal_solver(config):
                eigenvalues, eigenvectors = la.eigh(dense, overwrite_a=True, check_finite=True)
            else:
                eigenvalues, eigenvectors = la.eig(dense, overwrite_a=True, check_finite=True)
            del dense
            information["converged"] = True
        else:
            eigenvalues, eigenvectors, information, notes = sparse_eigensystem(model, config, report)
        if not np.isfinite(eigenvalues).all():
            raise RuntimeError("The eigensolver returned nonfinite eigenvalues; reduce |g| or W.")
        order = np.lexsort((np.imag(eigenvalues), np.real(eigenvalues)))
        eigenvalues = eigenvalues[order].astype(np.complex128)
        # Advanced indexing allocates a second copy. Sort only the small result
        # arrays; vectors retain solver order until metrics have been measured.
        inverse_order = np.argsort(order)
        metrics, eigenvectors = normalize_and_measure(
            eigenvalues[inverse_order], eigenvectors, model, config, report
        )
        arrays = {"eigenvalues": eigenvalues}
        arrays.update({name: values[order] for name, values in metrics.items()})
        if config.save_eigenvectors:
            # This final column reordering is also performed in blocks on disk
            # by the worker, so keep the permutation instead of copying N² here.
            arrays["vector_order"] = order
        else:
            eigenvectors = None
    if arrays["residual"].max(initial=0) > max(1e-7, 20 * config.tolerance):
        notes.append("Some eigenpair residuals are large; inspect the residual column before interpretation.")
    metadata = {
        "config": asdict(config),
        "computed_eigenvalues": len(eigenvalues),
        "full_spectrum": config.solver == SOLVERS[0],
        "graph_edges": len(model.edges),
        "graph_components": model.components,
        "largest_residual": float(arrays["residual"].max(initial=0)),
        "residual_definition": "||Hv-λv||₂ / ((||H||_F+|λ|)||v||₂)",
        "solver_reference": "reciprocal real symmetric matrix" if uses_reciprocal_solver(config) else "non-Hermitian matrix",
        "notes": notes,
        **information,
    }
    return arrays, model, eigenvectors, metadata
