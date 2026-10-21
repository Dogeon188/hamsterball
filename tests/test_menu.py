"""The game menu's headless logic (spec §13): the config file, name checks, and the level list."""

from hamsterball.game import menu


def test_player_names_are_lowercase_letters_digits_and_underscores():
    assert menu.name_ok("eric_2")
    for bad in ("", "Eric", "eric mao", "eric-2", "é", "a/b"):
        assert not menu.name_ok(bad)


def test_the_level_list_holds_the_levels_supporting_the_task_then_generated(tmp_path):
    from hamsterball.env import LEVELS_DIR

    assert menu.level_choices(LEVELS_DIR, "reach_goal") == ["first_goal", "hill", "generated"]
    assert menu.level_choices(LEVELS_DIR, "survive") == [
        "first_survive", "survive_collapse", "survive_drift", "survive_pushers", "survive_sweepers", "generated"
    ]
    (tmp_path / "broken.yaml").write_text("tasks: [")
    (tmp_path / "goal.yaml").write_text((LEVELS_DIR / "first_goal.yaml").read_text())
    assert menu.level_choices(tmp_path, "reach_goal") == ["goal", "generated"]  # a malformed file is left out
    assert menu.level_choices(tmp_path, "survive") == ["generated"]


def test_the_menu_prefills_from_flags_over_the_saved_choices_and_saves_them_back(tmp_path):
    path = tmp_path / "hamsterball" / "game.json"
    assert menu.load_config(path) == {}  # first run: no file
    menu.save_config({"player": "eric", "task": "survive", "level": "generated", "difficulty": 0.7, "seed": 3}, path)
    saved = menu.load_config(path)
    assert saved == {"player": "eric", "task": "survive", "level": "generated", "difficulty": 0.7}  # no seed
    flags = {"player": None, "task": "reach_goal", "level": None, "difficulty": None, "seed": 5}
    assert menu.prefill(flags, saved) == {
        "player": "eric", "task": "reach_goal", "level": "generated", "difficulty": 0.7, "seed": 5,
    }
    assert menu.prefill({}, {}) == {"player": "", "task": "reach_goal", "level": None, "difficulty": 0.5, "seed": None}


def test_a_corrupt_config_file_reads_as_empty(tmp_path):
    path = tmp_path / "game.json"
    path.write_text("{not json")
    assert menu.load_config(path) == {}
    path.write_text("[1, 2]")
    assert menu.load_config(path) == {}


def test_a_menu_level_resolves_to_a_file_in_the_levels_dir(tmp_path):
    level = tmp_path / "mine.yaml"
    level.touch()
    assert menu.resolve("mine", tmp_path) == str(level)
    assert menu.resolve(str(level), "elsewhere") == str(level)  # --level FILE
    assert menu.resolve("generated", tmp_path) == "generated"


def test_a_blank_seed_is_random_and_a_seed_is_a_non_negative_integer():
    import pytest

    assert menu.parse_seed(" ") is None
    assert menu.parse_seed("12") == 12
    for bad in ("-1", "x", "1.5"):
        with pytest.raises(ValueError):
            menu.parse_seed(bad)
