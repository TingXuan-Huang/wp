"""Figures for the experiments."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def heatmap(values, rows, columns, path, title, centered=False):
    """Save a labelled heatmap. With centered, the color scale is symmetric around zero (for differences)."""
    data = np.array(values, dtype=float)
    limit = np.nanmax(np.abs(data)) if centered and np.isfinite(data).any() else None
    fig, ax = plt.subplots(figsize=(2 + 0.9 * len(columns), 1.2 + 0.4 * len(rows)))
    image = ax.imshow(data, aspect="auto", cmap="coolwarm" if centered else "viridis",
                      vmin=-limit if limit else None, vmax=limit)
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for (i, j), value in np.ndenumerate(data):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2g}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curve(xs, ys, path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
