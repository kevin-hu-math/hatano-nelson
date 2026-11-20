"""Plot the empirical spectrum and conditional means of eigenvector statistics."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = {"Inverse participation ratio": "ipr", "L⁴ norm": "l4", "L∞ norm": "linf"}
LABELS = {"ipr": "IPR", "l4": "L⁴", "linf": "L∞"}


@dataclass
class SpectralBins:
    x_edges: np.ndarray
    y_edges: np.ndarray
    counts: np.ndarray
    density: np.ndarray
    means: np.ndarray
    visible: np.ndarray
    real_spectrum: bool
    total: int


def expanded_interval(values: np.ndarray, central_percent: float) -> tuple[float, float]:
    tail = (100 - central_percent) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    if low == high:
        pad = max(0.05, 0.025 * abs(float(low)))
        return float(low - pad), float(high + pad)
    # Include endpoints without changing the requested percentile range materially.
    return float(np.nextafter(low, -np.inf)), float(np.nextafter(high, np.inf))


def bin_spectrum(
    eigenvalues: np.ndarray,
    metric: np.ndarray,
    bins: int = 70,
    central_percent: float = 100,
    minimum_count: int = 1,
    force_complex: bool = False,
) -> SpectralBins:
    """Empty bins are NaN; means are arithmetic means of the raw statistics.

    A clipped view keeps the original normalization by the total number of
    computed eigenvalues, so its density integrates to the displayed fraction.
    """
    eigenvalues, metric = np.asarray(eigenvalues), np.asarray(metric)
    if eigenvalues.ndim != 1 or metric.shape != eigenvalues.shape or not len(metric):
        raise ValueError("Supply one metric value per eigenvalue and a nonempty spectrum.")
    if not np.isfinite(eigenvalues).all() or not np.isfinite(metric).all():
        raise ValueError("The plot contains nonfinite eigenvalues or statistics.")
    if bins < 2 or not 0 < central_percent <= 100 or minimum_count < 1:
        raise ValueError("Invalid binning parameters.")
    x, y = eigenvalues.real, eigenvalues.imag
    real_spectrum = not force_complex and bool(np.all(np.abs(y) <= 1e-10 * (1 + np.abs(x))))
    x_edges = np.linspace(*expanded_interval(x, central_percent), bins + 1)
    y_edges = np.linspace(*expanded_interval(y, central_percent), bins + 1)
    visible = (x >= x_edges[0]) & (x <= x_edges[-1])
    if real_spectrum:
        counts, _ = np.histogram(x, bins=x_edges)
        sums, _ = np.histogram(x, bins=x_edges, weights=metric)
        density = counts / (len(x) * np.diff(x_edges))
    else:
        visible &= (y >= y_edges[0]) & (y <= y_edges[-1])
        counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
        sums, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=metric)
        density = counts / (len(x) * np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :])
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    means[counts < minimum_count] = np.nan
    return SpectralBins(x_edges, y_edges, counts, density, means, visible, real_spectrum, len(x))


def color_values(values: np.ndarray, logarithmic: bool) -> np.ndarray:
    # Keep the old argument for callers of the first release; colors now always
    # display raw statistics on the same linear [0, 1] scale.
    return values


def make_figure(
    eigenvalues: np.ndarray,
    metric: np.ndarray,
    data: SpectralBins,
    metric_key: str,
    full_spectrum: bool,
    view: str = "Binned mean",
    logarithmic: bool = False,
) -> go.Figure:
    label = LABELS[metric_key]
    distribution = "Empirical spectral density" if full_spectrum else "Selected-eigenvalue histogram"
    localization_title = f"Mean {label} in each occupied bin" if view == "Binned mean" else f"Right-eigenvector {label}"
    figure = make_subplots(rows=1, cols=2, subplot_titles=(distribution, localization_title), horizontal_spacing=0.17)
    x_centers = (data.x_edges[1:] + data.x_edges[:-1]) / 2
    color_label = label
    if data.real_spectrum:
        figure.add_trace(go.Bar(
            x=x_centers, y=data.density, width=np.diff(data.x_edges), marker_color="#4b718b",
            customdata=data.counts, hovertemplate="E=%{x:.5g}<br>density=%{y:.5g}<br>count=%{customdata}<extra></extra>",
        ), row=1, col=1)
        if view == "Binned mean":
            x, values, count = x_centers, data.means, data.counts
            hover = "E=%{x:.5g}<br>mean=%{y:.5g}<br>count=%{customdata}<extra></extra>"
        else:
            x, values = eigenvalues.real[data.visible], metric[data.visible]
            count = np.ones(len(x), dtype=int)
            hover = "E=%{x:.5g}<br>value=%{y:.5g}<extra></extra>"
        figure.add_trace(go.Scattergl(
            x=x, y=values, mode="markers", customdata=count,
            marker=dict(size=5, color=color_values(values, logarithmic), cmin=0, cmax=1, cauto=False, colorscale="Viridis", showscale=True,
                        colorbar=dict(title=color_label, x=1.02, thickness=12)), hovertemplate=hover,
        ), row=1, col=2)
        figure.update_yaxes(title_text="Density on ℝ", row=1, col=1)
        figure.update_yaxes(title_text=label, type="linear", row=1, col=2)
        figure.update_xaxes(title_text="E", range=[data.x_edges[0], data.x_edges[-1]])
    else:
        y_centers = (data.y_edges[1:] + data.y_edges[:-1]) / 2
        figure.add_trace(go.Heatmap(
            x=x_centers, y=y_centers, z=np.where(data.counts.T > 0, data.density.T, np.nan),
            colorscale="Cividis", colorbar=dict(title="Density", x=0.435, thickness=12),
            customdata=data.counts.T,
            hovertemplate="Re=%{x:.5g}<br>Im=%{y:.5g}<br>density=%{z:.5g}<br>count=%{customdata}<extra></extra>",
        ), row=1, col=1)
        if view == "Binned mean":
            figure.add_trace(go.Heatmap(
                x=x_centers, y=y_centers, z=color_values(data.means.T, logarithmic), colorscale="Viridis", zmin=0, zmax=1, zauto=False,
                colorbar=dict(title=color_label, x=1.02, thickness=12),
                customdata=np.stack((data.means.T, data.counts.T), axis=-1),
                hovertemplate="Re=%{x:.5g}<br>Im=%{y:.5g}<br>mean=%{customdata[0]:.5g}<br>count=%{customdata[1]}<extra></extra>",
            ), row=1, col=2)
        else:
            mask = data.visible
            figure.add_trace(go.Scattergl(
                x=eigenvalues.real[mask], y=eigenvalues.imag[mask], mode="markers", customdata=metric[mask],
                marker=dict(size=5, color=color_values(metric[mask], logarithmic), cmin=0, cmax=1, cauto=False, colorscale="Viridis", showscale=True,
                            colorbar=dict(title=color_label, x=1.02, thickness=12)),
                hovertemplate="Re=%{x:.5g}<br>Im=%{y:.5g}<br>value=%{customdata:.5g}<extra></extra>",
            ), row=1, col=2)
        figure.update_xaxes(title_text="Re λ", range=[data.x_edges[0], data.x_edges[-1]])
        figure.update_yaxes(title_text="Im λ", range=[data.y_edges[0], data.y_edges[-1]])
    figure.update_layout(template="plotly_white", height=510, margin=dict(l=55, r=85, t=65, b=50),
                         showlegend=False, font=dict(size=12), bargap=0)
    for annotation in figure.layout.annotations:
        annotation.font.size = 13
    return figure


def png_bytes(
    eigenvalues: np.ndarray, metric: np.ndarray, data: SpectralBins,
    metric_key: str, full_spectrum: bool, view: str, logarithmic: bool,
    subtitle: str = "",
) -> bytes:
    """A static export without requiring a Plotly browser or Kaleido."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize

    fig = Figure(figsize=(12, 4.8), layout="constrained")
    left, right = fig.subplots(1, 2)
    label = LABELS[metric_key]
    left.set_title("Empirical spectral density" if full_spectrum else "Selected-eigenvalue histogram", fontsize=11)
    right.set_title(f"Mean {label} per occupied bin" if view == "Binned mean" else f"Right-eigenvector {label}", fontsize=11)
    x = (data.x_edges[1:] + data.x_edges[:-1]) / 2
    norm = Normalize(vmin=0, vmax=1)
    if data.real_spectrum:
        left.stairs(data.density, data.x_edges, fill=True, color="#4b718b")
        if view == "Binned mean":
            xx, values = x, data.means
        else:
            xx, values = eigenvalues.real[data.visible], metric[data.visible]
        artist = right.scatter(xx, values, c=values, s=9, cmap="viridis", norm=norm)
        left.set_ylabel("Density on ℝ")
        right.set_ylabel(label)
        for ax in (left, right):
            ax.set_xlabel("E")
            ax.set_xlim(data.x_edges[0], data.x_edges[-1])
    else:
        density = np.ma.masked_where(data.counts.T == 0, data.density.T)
        artist_left = left.pcolormesh(data.x_edges, data.y_edges, density, cmap="cividis", shading="flat")
        fig.colorbar(artist_left, ax=left, label="Density", shrink=0.85)
        if view == "Binned mean":
            artist = right.pcolormesh(data.x_edges, data.y_edges, np.ma.masked_invalid(data.means.T),
                                      cmap="viridis", norm=norm, shading="flat")
        else:
            mask = data.visible
            artist = right.scatter(eigenvalues.real[mask], eigenvalues.imag[mask], c=metric[mask], s=9,
                                   cmap="viridis", norm=norm)
        for ax in (left, right):
            ax.set_xlabel("Re λ")
            ax.set_ylabel("Im λ")
            ax.set_xlim(data.x_edges[0], data.x_edges[-1])
            ax.set_ylim(data.y_edges[0], data.y_edges[-1])
    fig.colorbar(artist, ax=right, label=label, shrink=0.85)
    if subtitle:
        fig.suptitle(subtitle, fontsize=11)
    output = BytesIO()
    fig.savefig(output, format="png", dpi=180)
    return output.getvalue()
