"""Five distinct charts for the adapter zoo.

- inter-adapter cosine heatmap (n x n)
- per-task accuracy heatmap (n x n with diagonal annotation)
- swap-time CDF + histogram (twin axes)
- diagonal vs off-diagonal accuracy violin
- hierarchical-clustering dendrogram of adapter similarity
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    data: dict[str, Any] = json.loads(p.read_text())
    return data


def plot_cosine_heatmap(sweep_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _load(sweep_path)
    if not data:
        out.write_bytes(b"")
        return out
    mat = np.array(data["pairwise_cosine"], dtype=np.float64)
    names = data["names"]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(names) + 2), max(4, 0.4 * len(names) + 2)))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(mat[i, j]) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="cosine(effective delta)")
    ax.set_title("Inter-adapter cosine similarity")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_accuracy_heatmap(sweep_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _load(sweep_path)
    if not data:
        out.write_bytes(b"")
        return out
    acc = np.array(data["accuracy_matrix"], dtype=np.float64)
    names = data["names"]
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * n + 2), max(4, 0.4 * n + 2)))
    im = ax.imshow(acc, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{acc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if acc[i, j] < 0.5 else "black",
            )
    ax.set_xlabel("adapter")
    ax.set_ylabel("task")
    ax.set_title("Per-(task, adapter) accuracy")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_swap_times(sweep_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _load(sweep_path)
    if not data:
        out.write_bytes(b"")
        return out
    samples = np.array(data["swap_ms_samples"], dtype=np.float64)
    fig, ax_hist = plt.subplots(figsize=(7.5, 4.5))
    ax_hist.hist(samples, bins=20, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax_hist.set_xlabel("swap time (ms)")
    ax_hist.set_ylabel("count", color="#1f77b4")
    ax_hist.tick_params(axis="y", labelcolor="#1f77b4")
    ax_hist.set_title(f"Adapter swap-time distribution (n={len(samples)})")

    # twin axis: CDF
    ax_cdf = ax_hist.twinx()
    sv = np.sort(samples)
    cdf = np.arange(1, len(sv) + 1) / len(sv)
    ax_cdf.plot(sv, cdf, color="#d62728", linewidth=1.5, label="CDF")
    ax_cdf.set_ylabel("CDF", color="#d62728")
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.tick_params(axis="y", labelcolor="#d62728")
    p50 = float(np.median(samples))
    p99 = float(np.quantile(samples, 0.99))
    ax_cdf.axvline(p50, color="gray", linestyle=":", linewidth=1)
    ax_cdf.axvline(p99, color="gray", linestyle=":", linewidth=1)
    ax_cdf.text(p50, 1.0, f"p50={p50:.2f}ms", fontsize=8, va="top")
    ax_cdf.text(p99, 0.5, f"p99={p99:.2f}ms", fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_accuracy_split_violin(sweep_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _load(sweep_path)
    if not data:
        out.write_bytes(b"")
        return out
    acc = np.array(data["accuracy_matrix"], dtype=np.float64)
    n = acc.shape[0]
    diag = [float(acc[i, i]) for i in range(n)]
    off = [float(acc[i, j]) for i in range(n) for j in range(n) if i != j]
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot([diag, off], showmeans=True, showmedians=False)
    bodies: Any = parts.get("bodies") or []
    for pc, color in zip(bodies, ["#2ca02c", "#d62728"], strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"diagonal (n={len(diag)})", f"off-diagonal (n={len(off)})"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Adapter-on-own-task vs adapter-on-other-task")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_dendrogram(sweep_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = _load(sweep_path)
    if not data:
        out.write_bytes(b"")
        return out
    mat = np.array(data["pairwise_cosine"], dtype=np.float64)
    names = data["names"]
    # distance = 1 - cosine
    dist = 1 - mat
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    # squareform expects zero diagonal; clamp to avoid tiny negatives
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names) + 2), 4.5))
    dendrogram(link, labels=names, ax=ax, leaf_rotation=30)
    ax.set_ylabel("1 - cosine")
    ax.set_title("Hierarchical clustering of adapters (average-linkage)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
