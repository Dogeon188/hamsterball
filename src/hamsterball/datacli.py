"""hamsterball-data (later efforts §7): replay, re-render, video, and LeRobot export of Minari datasets."""

import click

from .core.views import VIEWS


def _size(ctx, param, value):
    """Parse HxW into (H, W); None passes through."""
    if value is None:
        return None
    try:
        h, w = (int(n) for n in value.lower().split("x"))
    except ValueError:
        raise click.BadParameter(f"{value!r} is not HxW, e.g. 84x84")
    return h, w


def _load(dataset_id):
    import minari

    return minari.load_dataset(dataset_id)


@click.group()
def main():
    """Check, re-render, and export Minari datasets that the game or the Collector recorded."""
    try:
        import minari  # noqa: F401
    except ImportError:
        raise click.ClickException("hamsterball-data reads Minari datasets: install hamsterball[data]")


@main.command()
@click.argument("dataset")
@click.option("--episode", "episodes", type=int, multiple=True, help="An episode id; repeat it. [default: all]")
def replay(dataset, episodes):
    """Re-step DATASET's episodes on this build and check every core state bit for bit."""
    from .data import ReplayMismatch, replay

    ds = _load(dataset)
    try:
        replay(ds, list(episodes) or None)
    except ReplayMismatch as e:
        raise click.ClickException(str(e))
    click.echo(f"{len(episodes) or ds.total_episodes} episodes replay bit for bit")


@main.command()
@click.argument("dataset")
@click.argument("out_id")
@click.option("--view", "views", type=click.Choice(VIEWS), multiple=True, required=True, help="A view; repeat it.")
@click.option("--size", callback=_size, default="84x84", show_default=True, help="The views' HxW.")
def rerender(dataset, out_id, views, size):
    """Write DATASET with VIEWs added to its observations as a new dataset OUT_ID."""
    from .data import rerender

    out = rerender(_load(dataset), views, size, out_id)
    click.echo(f"wrote {out_id}: {out.total_episodes} episodes, {out.total_steps} steps")


@main.command()
@click.argument("dataset")
@click.argument("episode", type=int)
@click.argument("view", type=click.Choice(VIEWS))
@click.argument("path", type=click.Path(dir_okay=False))
@click.option("--size", callback=_size, help="HxW, even numbers. [default: the dataset's view size]")
def video(dataset, episode, view, path, size):
    """Write EPISODE of DATASET, re-rendered from VIEW, as an mp4 at PATH."""
    from .data import to_video

    to_video(_load(dataset), episode, view, path, size)
    click.echo(f"wrote {path}")


@main.command()
@click.argument("dataset")
@click.argument("repo_id")
@click.argument("root", type=click.Path(file_okay=False))
@click.option("--images", is_flag=True, help="Store lossless PNG frames instead of lossy video.")
def lerobot(dataset, repo_id, root, images):
    """Export DATASET, a re-rendered dataset, as LeRobot v3 dataset REPO_ID at ROOT."""
    from .data import to_lerobot

    try:
        to_lerobot(_load(dataset), repo_id, root, use_videos=not images)
    except ImportError:
        raise click.ClickException("LeRobot export needs hamsterball[lerobot] (Python 3.12 or 3.13)")
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"wrote {repo_id} at {root}")
