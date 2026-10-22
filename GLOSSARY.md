# Hamsterball

A 3D simulator of a hamster running inside a rolling ball, playable by humans as a game and trainable by agents as a reinforcement-learning environment.

## Language

### Bodies

**Hamster**:
The controlled body inside the shell; it moves the ball only by pushing against the shell's inner wall.
_Avoid_: player, agent (for the body)

**Shell**:
The hollow sphere the hamster runs inside; it rolls through the level by reaction to the hamster.
_Avoid_: ball (alone), sphere

**Hamsterball**:
The hamster and its shell together, as one coupled system.

### Control

**Control tier**:
How much of the hamster the controller drives: the **abstract tier** (hamster as a simple body commanded by run direction and effort) or the **articulated tier** (hamster with jointed legs commanded by joint torques).
_Avoid_: difficulty, mode

**Run direction**:
The heading the hamster runs toward in the abstract tier, measured relative to the controlled third-person view.
_Avoid_: steering, heading (alone)

**Facing**:
The hamster's heading, which sets the frame of actions and observations and the yaw of the chase and first-person views. Only the controller turns it.
_Avoid_: camera yaw, heading (alone), view angle

**Tail**:
A small cone behind the hamster pointing away from its facing, so the chase and follow views show which way the hamster faces. It is a look only: no mass, no contacts.
_Avoid_: arrow, heading marker

**Effort**:
How hard the hamster runs in the abstract tier: together with a run direction, it sets the hamster's target lean.
_Avoid_: throttle, speed command

**Lean**:
The hamster's offset from the bottom of the shell, whose weight rolls the shell.
_Avoid_: tilt, pendulum angle

**Foot force**:
The force the hamster exerts tangentially on the shell's inner wall at its contact point.
_Avoid_: thrust, drive force

**Controller**:
Whoever issues actions to the hamster: a human player or a policy.
_Avoid_: agent (ambiguous with the hamster)

**Low-level policy**:
A policy that turns a run direction and effort into the articulated tier's joint torques, so a human can play the articulated tier.
_Avoid_: gait controller, motor policy

**Physics step**:
One integration step of the core: the smallest unit of simulated time.
_Avoid_: tick, substep (alone)

**Env step**:
One action from the controller, held over a fixed number of physics steps, followed by one observation; the game shows one env step per frame.
_Avoid_: frame, timestep

### World

**Level**:
The static 3D world a hamsterball moves through: terrain, ramps, edges, and task-relevant markers.
_Avoid_: map, stage, track

**Level pool**:
The set of levels an env picks from when an episode starts: hand-authored levels, or a procedural generator.
_Avoid_: level set, curriculum

**Difficulty**:
How hard a generated level is: a number from 0 (easiest) to 1 (hardest) that a level generator maps to its terrain and markers.
_Avoid_: level (as a difficulty grade), tier

**Marker**:
A volume in a level that a task reacts to when the hamsterball's centre enters it: a goal, waypoint, item, key, survive region, or hazard.
_Avoid_: trigger, checkpoint

**Motion**:
A scripted movement of a part of a level over sim time: a closed **path** walked at constant speed or a sine **oscillation** along an axis, either with a constant spin about the vertical, and for a kinematic body optionally a **tilt** (a rocking about a horizontal axis). A mover's pose is a function of its motion and the sim time alone.
_Avoid_: animation, trajectory (a trajectory is a recorded episode)

**Mover**:
A part of a level that has a motion, such as a drifting survive region.
_Avoid_: dynamic object, moving marker (alone)

**Kinematic body**:
A solid part of a level, a group of geoms that collides with the shell and moves by a motion that nothing pushes back on, such as a sweeping bar. It is a mover. A survive region can **attach** to one and ride it, as on a raft.
_Avoid_: mocap, moving platform (alone), obstacle (for any solid)

**Seesaw**:
A solid platform of a level on a passive hinge or ball joint at a pivot, which the hamsterball's weight tips against a spring back to level, up to a tilt limit. It is not a mover: nothing scripts it. A survive region can attach to it as to a kinematic body.
_Avoid_: tilting platform (for a kinematic body's scripted tilt), passive body

**Collapsing tile**:
A floor tile of a level that starts to fall a fixed delay after the shell first touches it, and drops under the kill plane. It is a kinematic body.
_Avoid_: trap tile, crumbling floor

**Kill plane**:
A height below which the hamsterball counts as fallen off the level.
_Avoid_: floor, death height

**Spawn**:
Where a task in a level starts the hamsterball, together with the hamster's initial facing.
_Avoid_: start point, respawn

**Hazard**:
A marker that ends the episode in failure when the hamsterball enters it. It may move by a **Motion**.
_Avoid_: trap, death zone

**Cliff**:
A vertical drop in a level that the hamsterball can go down but not up.
_Avoid_: ledge, step

**Task**:
The objective of an episode within a level, which defines success, failure, and reward. v1 has five: **reach goal**, **waypoint race** (ordered waypoints), **collect items**, **unlock goal** (collect every key, then reach the goal), and **balance/survive** (stay inside the survive region until time runs out).
_Avoid_: mission, challenge

**Current target**:
The marker a task is steering the hamsterball toward right now, such as the next waypoint or the nearest remaining item.
_Avoid_: objective, goal (for non-goal markers)

### Front ends

**Game**:
The human-playable front end: the env driven by a human at real-time pace, with an on-screen view.

**Env**:
The headless, faster-than-real-time front end that policies train against.
_Avoid_: gym, simulator (for the front end)

**Backend**:
The physics implementation an env runs on: the **CPU core**, which is the reference that judges policies and records trajectories, or the **GPU-batched backend**, which runs many worlds at once for training and may drift from the CPU core.
_Avoid_: engine, device

**View**:
A camera from which pixel observations or the game's screen are rendered: the **chase view**, the **follow view**, or the **first-person view**.
_Avoid_: camera angle, perspective

**Chase view**:
The controlled third-person view, which trails the hamsterball and turns with the facing; run direction is measured relative to it.
_Avoid_: controlled camera, main camera

**Follow view**:
A third-person view that tracks the hamsterball's position but keeps a yaw fixed by the level.
_Avoid_: fixed camera, fixed third-person

**First-person view**:
The view from the hamster's eye, looking along the facing, with the shell wall hidden.
_Avoid_: fixed first-person, eye camera

**Camera pitch**:
How far below the horizon the chase view looks; it pitches the chase and first-person views but moves no physics and leaves the frame of actions and observations alone.
_Avoid_: tilt (a mover's rocking), elevation, view angle

**Pose stream**:
The sequence, one **pose** per env step, of what a front end needs to draw the scene without running physics: body poses, view poses, marker progress, and the outcome.
_Avoid_: frame stream, render stream

**Trajectory**:
A recorded episode of a controller's actions and the resulting states.
_Avoid_: replay, demo, log

**Abandoned run**:
An episode the human leaves before success, failure, or the time limit ends it; the game discards it rather than recording it.
_Avoid_: quit episode, partial trajectory

**Human baseline**:
The success rate and completion times humans reach on a level and task, measured from their recorded trajectories.
_Avoid_: high score, leaderboard
