# honest_drive.py

Throwaway prototype that answers `.scratch/hamsterball/issues/13-honest-physics-for-the-feel.md`: how model C keeps its feel with a force pair that adds no net wrench, what caps its top speed, and how the hamster stays inside the shell on hard hits. It is not production code. Model C and the shell contacts are copied from `shell_contact.py` with the issue 12 verdict fixed: an outer `sphere+roll` collider, 320 tiles, solref `0.04 2`, and a 2 mm margin.

## Interface

```
uv run prototypes/honest_drive.py           # interactive window, starts on the chosen version (PICK)
uv run prototypes/honest_drive.py --sweep   # headless tables, about 3 min
```

Keys:
- **Variants** (each reloads): `P` pair, `H` masses, `L` lean ceiling, `F` feed-forward, `C` speed cap, `V` top speed, `B` containment.
- **Driving:** `WASD` run (camera-relative), `Shift` full effort, `Q`/`E` orbit the camera. A gamepad's left stick sets direction and effort.
- **Run control:** `G` spawns at the top of the drop lane, `R` resets, `Esc` quits.

## How it works

A variant is a `Var(pair, masses, lean, ff, cap, top, contain)`. `PICK` is the play-tested choice.

- **Pair**:
  - `offset`: the 2307efc pair, with +F at the hamster centre and −F at the foot. It adds a net couple of |F|·R_HAM.
  - `centre`: −F acts on the shell along +F's line through the hamster centre.
  - `foot+grip`: both forces act at the foot, and a grip couple `K_GRIP` damps hamster spin relative to the shell. The hamster–wall contact is frictionless (`condim 1`, `priority 1`), so the foot force is the only traction. With friction this pair blows up.
- **Masses** (hamster / shell) and **lean ceiling** are the knobs that win back the drive the offset couple supplied. The foot-force PD gains and cap scale with hamster mass, so every mass ratio leans alike.
- **ff**: the gravity feed-forward from 73c1914.
- **Cap**:
  - `fade` scales effort by `1 − v_along / V_RUN`.
  - `drag` applies `−k |v| v` to the shell whether or not it touches the ground. At speed the outer sphere touches the ground on only 8–20% of steps, so a contact-gated drag barely acted.
  - `calibrate()` sets `V_RUN` or `k` from the uncapped 1–3 s acceleration so the full-effort top speed lands near `top`.
- **Containment**, which runs in `control()` before the airborne cut:
  - `field`:
    - From 5 mm past the resting radius `L_ARM`, an inelastic impulse along the line of centres removes a share of the hamster's outward speed relative to the shell. The share ramps to all of it at 20 mm.
    - Past 20 mm, a position projection moves the hamster back to 20 mm and the shell the other way by mass ratio.
    - Momentum, angular momentum, and the pair's centre of mass are preserved.
  - `solimp` sets solimp `0.9 0.9999 0.02 0.5 2` on the tiles and hamster.

`measure(var)` reports:
- the 1–3 s acceleration
- speed at 3 s at half and full effort
- the mean lean over 2–3 s
- top speed after 10 s (uncapped) or 20 s (capped)
- the coast loss 1–3 s after release from top speed
- the speed change over 40 s of zero input on a frictionless outer sphere
- the shell's yaw rate 8 s after a 90° turn
- distance climbed up a 20° tilt in 3 s

`contain_row(var)` throws the hamster from the shell centre at 0, 4, 8 and 12 m/s (down, up, sideways) and reports the worst sink past `L_ARM`. It then records rattle and wall penetration over 2–6 s at full effort.

The course adds a drop lane at y = −20 to the `shell_contact.py` course:
- a 30° ramp with a 20 m drop
- a curved runout (radius 6 m, 8 segments) onto a 0.3 m deck
- 8 cm bumps, then 15 cm bumps, then a 0.3 m step down at x = −4

`G` places the ball 38 m up the straight. The overlay shows the worst sink since the last `G` or `R`.

## Known quirks

- **Drift isn't zero for the honest pairs.** Both lose speed at zero input on a frictionless roll, because the hamster bumping over the tiles dissipates energy. The chosen version loses 4.5 m/s over 40 s from 4.2 m/s, and that figure includes drag.
- **The feed-forward is mixed.** It raises half-effort speed by 1.5–3 m/s, but it lowers full-effort acceleration in most variants and hurts the climb.
- **The lean sags.** The PD alone holds about 50–55° against a 70° target.
- **Containment is a step-level state edit**, not a MuJoCo constraint. It writes `qvel` and `qpos` once per physics step, so a single step can still overshoot 20 mm by one step's travel (39 mm worst sink on a 12 m/s throw).
- **The calibration is approximate.** Measured top speeds land 3–9% under target, e.g. 14.5 m/s for the chosen drag-16.

## Decisions

- [ADR-0001](../docs/adr/0001-mujoco-core-game-wraps-env.md), MuJoCo core: colliders stay MuJoCo primitives. The drive, cap and containment are core-step code that writes `xfrc_applied`, `qvel` and `qpos`, with no custom collider, so MJX and Warp can run the same model.
- Issue 13 verdict, from play-testing on 2026-10-20: **`foot+grip`, 2.4 / 0.8 kg, 70° lean ceiling, no feed-forward, drag to 16 m/s (k 0.059), `field` containment.**
  - The force pair is honest: +F on the hamster and −F on the shell, both at the foot, with a grip couple `K_GRIP` 2 N·m·s/rad per 2 kg of hamster (2.4 at 2.4 kg). The hamster–wall contact is frictionless.
  - The PD gains and the 30 N force cap scale with hamster mass (1.2× at 2.4 kg).
  - Top speed comes from a k·v² drag on the shell. It isn't gated on ground contact.
  - Containment: the field described in How it works, with a 5 mm start and a 20 mm hard stop.

Measured (flat ground, 2 ms step, M2 CPU):

| | 2307efc feel (offset, 2/1 kg, uncapped) | Chosen |
|---|---|---|
| 1–3 s acceleration | 5.05 m/s² | 4.01 m/s² |
| Speed at 3 s, half / full effort | 8.1 / 14.6 m/s | 5.6 / 9.8 m/s |
| Top speed | none (48 m/s at 10 s) | 14.5 m/s |
| Coast loss from top speed | 0.08 m/s² (the couple feeds energy) | 2.5 m/s² (drag; about 0.55 from rolling friction at low speed) |
| Zero input, frictionless roll, 40 s | +1.09 m/s | −4.5 m/s |
| Yaw 8 s after a turn | 1.67 rad/s | 0.003 rad/s |
| 20° climb in 3 s | 9.9 m | 5.1 m |
| Worst sink, 0 / 4 / 8 / 12 m/s throw | 42 / 78 / escaped / escaped mm (no containment) | 17 / 23 / 31 / 39 mm |
| Drop lane, full effort | hamster escapes at the deck | 22 mm worst sink, 22 m/s peak |

## Tried and rejected

Numbers are from `--sweep` unless noted.

- **`offset` pair (2307efc)**: feeds energy at zero input (+0.7 to +1.7 m/s over 40 s on a frictionless roll) and leaves more yaw after turns.
- **`centre` pair**: honest, but at 2/1 kg it accelerates at 2.98 m/s² and climbs 2.3 m. At 2.7/0.6 kg it reaches 4.6 m/s², slightly behind `foot+grip`. Not chosen in play.
- **`foot+grip` with hamster–wall friction**: blew up (NaN or escape) in 8 of 9 drive variants, with hamster spin climbing to 14 rad/s.
- **2.7 / 0.6 kg**: the closest match to 2307efc's numbers (5.1 m/s², 14.7 m/s at 3 s) but not chosen in play.
- **Lean ceiling 78° or 85°**: rarely helps, because the force cap binds. 85° often accelerates worse than 70°.
- **`fade` cap**: keeps the coast, but its linear fade bites early. Fade 16 cut full-effort speed at 3 s from 8.7 to 7.1 m/s on `centre`.
- **Drag gated on ground contact**: the outer sphere touches the ground on only 8–20% of steps at speed, so drag 8 overshot to 17.8 m/s.
- **Containment by solimp**: barely changes the sink, because the 40 ms solref time constant sets it. On a 12 m/s throw and on the drop lane at full effort (16 m/s into the deck) the hamster escapes. Settings `0.5 0.9999 0.02 0.5 6` and `0.95 0.9999 0.005 0.5 2` were no better. The user confirmed the escape in play.
- **Field as a spring-damper force** (K 20000 N/m, C 1000 N·s/m): blew up on a 2 m/s throw. The relevant mass is the pair's reduced mass (0.6 kg), which puts that damping 3.3× over the explicit stability limit at 2 ms.
- **Field damping capped at the deadbeat value μ/dt**: held to 4 m/s, but on a downward throw the shell is grounded, so it pushes against the hamster's full 2.4 kg and doesn't stop it.
- **Velocity clamp without a position stop**: holds throws to 12 m/s. But a 22 → 15 m/s hit at the deck pushes the hamster outward at about 30 g, it creeps about 1.2 mm per step, and it escapes after about 40 steps.
- **Straight 30° ramp into the deck**: the kink bounced the ball over the bumps. Replaced by the curved runout.

## History

- 2026-10-20, design questions (AskUserQuestion), issue 13: the user chose both honest pairs toggled against the offset baseline, a sweep over masses and lean ceiling, fade and drag caps, and top speeds of 8, 12 and 16 m/s. They approved the grid.
- 2026-10-20, agent findings while building: `accel()` passed the capped variant with cal 0, which gave a negative `V_RUN`. The contact-gated drag overshot its targets. `foot+grip` needs the frictionless contact from 199be5c. The lean sags about 20°, so the feed-forward was added as a sweep axis.
- 2026-10-20, play-test verdict: the user chose `foot+grip`, 2.4 / 0.8 kg, 70°, drag to 16. They reported the hamster escaping the shell on fast vertical bumps and proposed a high-velocity field outside the ball that acts only on the hamster.
- 2026-10-20, agent finding: the soft hamster–tile contact sinks about 1.5 cm per m/s of impact speed, and a plain drop from the shell centre sinks 47 mm, more than the 3 cm wall. The user chose to compare the field with solimp, fully hard by 2 cm.
- 2026-10-20, agent findings: solimp barely helps. The spring-damper field blew up. A momentum-conserving velocity clamp held throws to 12 m/s.
- 2026-10-20, user feedback: the bumps can't be tested because flat ground never gets fast enough, so the ball should start on a high slope and roll onto the bumps. The user chose a 20 m drop onto mixed bumps and a step.
- 2026-10-20, agent finding on the drop lane: the velocity clamp leaks under a sustained 30 g load at the deck. The user chose a position projection that preserves the centre of mass.
- 2026-10-20, user feedback: the straight ramp's kink at the ground bounced the ball over the bumps. The user chose a curved runout.
- 2026-10-20, user verdict: solimp first, then, after re-testing in play and seeing the escape at full effort, **`field` containment**.
