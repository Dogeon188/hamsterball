# Exact MuJoCo pin, and a MuJoCo upgrade is a new env version

hamsterball pins `mujoco` (and `mujoco-warp` in the `gpu` extra) to one exact version in its package metadata. A change that can alter a trajectory, a MuJoCo upgrade included, bumps the shared `-vN` of every env id and is a major release of hamsterball. Each release registers only its current N. Exact re-step replay is a v1 guarantee, a MuJoCo release can change contact results in their last bits, and the pin plus the bump make one env id name one simulator that replays its own datasets bit for bit. The game's human datasets carry the same N in their id (`hamsterball/human-<player>-v<N>`), so episodes from two simulators never share a dataset.

## Considered Options

- **A compatible range (`>=3.16,<3.17`) with a committed lock file.** Rejected: replay would depend on whatever lock each user installs from, while the package would still claim the same env id.
- **Bump `-vN` for semantic changes only, and treat a MuJoCo upgrade as a minor release with a replay warning.** Rejected: the same id would name builds that replay differently, and re-step replay would fail on datasets that carry a valid id.
- **A version N for each id.** Rejected: a MuJoCo upgrade touches every id at once, and a shared N keeps the ids aligned with one release.

## Consequences

- hamsterball can't share a virtual environment with a package that needs a different MuJoCo (dm_control, robosuite).
- Every MuJoCo upgrade is a major release, retires the previous env ids, and starts new human datasets. An older dataset re-steps only under the hamsterball release that recorded it, though re-render still works on any version.
