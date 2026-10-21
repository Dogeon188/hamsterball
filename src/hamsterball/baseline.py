"""hamsterball-baseline (spec §13): human baselines per level and task, from Minari datasets' episode metadata."""

import statistics
from collections import defaultdict
from typing import NamedTuple

import click

from .core.model import DT


class Stats(NamedTuple):
    episodes: int
    success_rate: float
    median_time: float | None  # sim seconds to success, None without a success
    best_time: float | None
    mean_return: float


def group(options):
    """(task, level id) for a file level, (task, difficulty rounded to 0.1) for a generated one."""
    if "difficulty" in options:
        return options["task"], round(options["difficulty"], 1)
    return options["task"], options["level_id"]


def baselines(dataset_ids):
    """{group: Stats} over every episode of the datasets, read from episode metadata only (no steps loaded).

    Tier-agnostic: datasets of any tier or action space pool into the same groups. Each dataset's env step
    length (its `frame_skip` physics steps) turns episode lengths into sim seconds.
    """
    import minari

    episodes = defaultdict(list)  # group -> [(success, sim seconds, return)]
    for dataset_id in dataset_ids:
        ds = minari.load_dataset(dataset_id)
        dt = ds.env_spec.kwargs["frame_skip"] * DT
        for m in ds.storage.get_episode_metadata(ds.episode_indices):
            episodes[group(m["options"])].append((m["success"], m["length"] * dt, m["return"]))
    out = {}
    for g, eps in episodes.items():
        times = [t for success, t, _ in eps if success]
        out[g] = Stats(
            len(eps),
            len(times) / len(eps),
            statistics.median(times) if times else None,
            min(times) if times else None,
            statistics.fmean(r for _, _, r in eps),
        )
    return out


def _sort_key(item):
    (task, level), _ = item
    return isinstance(level, float), task, level  # file levels first


@click.command()
@click.argument("datasets", nargs=-1, required=True)
def main(datasets):
    """Print human baselines for Minari DATASETS (ids such as hamsterball/human-ana-v0), per level and task."""
    try:
        import minari  # noqa: F401
    except ImportError:
        raise click.ClickException("hamsterball-baseline reads Minari datasets: install hamsterball[data]")
    fmt = "{:<14} {:<14} {:>8} {:>8} {:>9} {:>9} {:>11}"
    click.echo(fmt.format("task", "level", "episodes", "success", "median s", "best s", "mean return"))
    for (task, level), s in sorted(baselines(datasets).items(), key=_sort_key):
        level = f"d={level:.1f}" if isinstance(level, float) else level[:12]
        median, best = (f"{t:.2f}" if t is not None else "-" for t in (s.median_time, s.best_time))
        rate = f"{s.success_rate:.0%}"
        click.echo(fmt.format(task, level, s.episodes, rate, median, best, f"{s.mean_return:.2f}"))
