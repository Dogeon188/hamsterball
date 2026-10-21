"""Hamsterball: a hamster running inside a rolling shell, as a Gymnasium env and a game on one MuJoCo core."""

from gymnasium.envs.registration import register

register(id="Hamsterball-v0", entry_point="hamsterball.env:HamsterballEnv")
