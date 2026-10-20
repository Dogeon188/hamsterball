# MuJoCo core, and the game wraps the env

The physics core is MuJoCo, run in-process from Python. The game is the Gymnasium env driven by a human at real-time pace and drawn with MuJoCo's renderer, rather than a Godot, Unity, or Unreal front end. Engines that host the physics give up determinism: Godot documents its physics as non-deterministic, and Unity documents determinism only within one session. Those engines also cannot render camera pixels headless. Exact trajectory replay and multi-view pixel observations need both, and a game that wraps the env gives the game and the env one code path and one renderer. The hamsterball model uses only features that MJX and MuJoCo Warp support (a compound convex shell wall, no SDF plugins, no custom collider), so a later GPU backend can load the same model file.

## Considered Options

- **Game engine hosts the physics (godot_rl_agents, ML-Agents, Schola).** Rejected: no determinism, no headless pixels, and Unreal's Gymnasium bridge has no Mac build.
- **MuJoCo core with Godot as renderer and input only.** Deferred: it needs new glue code, and its views would differ from the env's pixels. A later effort can add it through a pose stream out of the core, along with the in-game editor.
- **Bowl SDF or a custom collider for the shell wall.** Rejected: tighter contact (about 4 mm of penetration vs about 10 mm), but MJX cannot load it.

## Consequences

- Exact replay requires the MuJoCo version and CPU architecture to be pinned.
- v1 has no in-game level editor. The level format must still be writable by one later.
