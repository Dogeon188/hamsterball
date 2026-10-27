"""The game menu's headless logic (spec §13): the player name check, the level list, and the remembered choices.

window.py draws the menu with Dear ImGui; everything it decides runs here, without a display.
"""

import json
import re
from pathlib import Path

import yaml

from .. import ENV_VERSION
from ..core.level import Level

CONFIG = Path.home() / ".config" / "hamsterball" / "game.json"
GENERATED = "generated"
TIERS = ("abstract", "articulated")
SAVED = ("player", "task", "level", "difficulty")  # the choices the config file remembers; the seed and the tier are per run
DEFAULTS = {"player": "", "task": "reach_goal", "level": None, "difficulty": 0.5, "seed": None,
            "tier": "abstract"}
NAME = re.compile(r"[a-z0-9_]+")


def name_ok(name):
    """Whether `name` can name a player's dataset: `[a-z0-9_]+`."""
    return NAME.fullmatch(name) is not None


def dataset_id(player, tier="abstract"):
    """The Minari id a player's episodes of `tier` record to."""
    return f"hamsterball/human-{player}-{'articulated-' * (tier == 'articulated')}v{ENV_VERSION}"


def level_choices(levels_dir, task):
    """The menu's level list: the stems of the `*.yaml` files in `levels_dir` that have a `task` block (so a spawn for
    it), sorted, then "generated". A file that fails to load is left out."""
    names = []
    for path in sorted(Path(levels_dir).glob("*.yaml")):
        try:
            if task in Level.load(path).tasks:
                names.append(path.stem)
        except (ValueError, OSError, yaml.YAMLError):
            pass
    return names + [GENERATED]


def resolve(level, levels_dir):
    """The level to play from a menu or `--level` choice: "generated", an existing file, or a stem in `levels_dir`."""
    if level == GENERATED or Path(level).is_file():
        return level
    return str(Path(levels_dir) / f"{level}.yaml")


def load_config(path=CONFIG):
    """The remembered choices, or {} if the file is missing or not a JSON object."""
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def save_config(settings, path=CONFIG):
    """Remember the player, task, level, and difficulty of `settings`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: settings[k] for k in SAVED}, indent=2))


def prefill(flags, config):
    """The menu's starting choices: each flag that's set, else the remembered choice, else the default."""
    return {k: next((v for v in (flags.get(k), config.get(k)) if v is not None), d) for k, d in DEFAULTS.items()}


def parse_seed(text):
    """The menu's seed field: blank is None (a random seed), else a non-negative integer; raises ValueError."""
    if not text.strip():
        return None
    seed = int(text)
    if seed < 0:
        raise ValueError(f"seed {seed} is negative")
    return seed
