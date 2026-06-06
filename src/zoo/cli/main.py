from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from tabulate import tabulate

from ..runner import sweep
from ..viz.charts import (
    plot_accuracy_heatmap,
    plot_accuracy_split_violin,
    plot_cosine_heatmap,
    plot_dendrogram,
    plot_swap_times,
)

app = typer.Typer(add_completion=False, help="zoo: LoRA adapter zoo")


@app.command("sweep")
def cmd_sweep(
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    n: Annotated[int, typer.Option(help="number of synthetic adapters")] = 6,
    rank: Annotated[int, typer.Option(help="LoRA rank")] = 8,
) -> None:
    artifact = sweep(out_dir, n=n, rank=rank)
    print()
    print(
        tabulate(
            [
                ("n_adapters", artifact["n_adapters"]),
                ("rank", artifact["rank"]),
                ("diagonal_accuracy_mean", f"{artifact['diagonal_accuracy_mean']:.4f}"),
                ("off_diagonal_accuracy_mean", f"{artifact['off_diagonal_accuracy_mean']:.4f}"),
                ("swap_ms_p50", f"{artifact['swap_ms_p50']:.3f}"),
                ("swap_ms_p99", f"{artifact['swap_ms_p99']:.3f}"),
            ],
            headers=["metric", "value"],
            tablefmt="github",
        )
    )


@app.command("plots")
def cmd_plots(
    sweep_path: Annotated[Path, typer.Option(help="sweep json")] = Path("results/sweep.json"),
    out_dir: Annotated[Path, typer.Option(help="figures dir")] = Path("results/figures"),
) -> None:
    plot_cosine_heatmap(sweep_path, out_dir / "cosine_heatmap.png")
    plot_accuracy_heatmap(sweep_path, out_dir / "accuracy_heatmap.png")
    plot_swap_times(sweep_path, out_dir / "swap_times.png")
    plot_accuracy_split_violin(sweep_path, out_dir / "accuracy_split.png")
    plot_dendrogram(sweep_path, out_dir / "dendrogram.png")
    typer.echo(f"wrote 5 figures to {out_dir}")


if __name__ == "__main__":
    app()
