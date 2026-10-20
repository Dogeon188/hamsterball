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

**Marker**:
A volume in a level that a task reacts to when the hamsterball's centre enters it: a goal, waypoint, item, key, survive region, or hazard.
_Avoid_: trigger, checkpoint

**Kill plane**:
A height below which the hamsterball counts as fallen off the level.
_Avoid_: floor, death height

**Spawn**:
Where a task in a level starts the hamsterball, together with the initial camera yaw.
_Avoid_: start point, respawn

**Hazard**:
A marker that ends the episode in failure when the hamsterball enters it.
_Avoid_: trap, death zone

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

**View**:
A camera from which pixel observations or the game's screen are rendered: the **chase view**, the **follow view**, or the **first-person view**.
_Avoid_: camera angle, perspective

**Chase view**:
The controlled third-person view, which trails the hamsterball and turns with the camera yaw; run direction is measured relative to it.
_Avoid_: controlled camera, main camera

**Follow view**:
A third-person view that tracks the hamsterball's position but keeps a yaw fixed by the level.
_Avoid_: fixed camera, fixed third-person

**First-person view**:
The view from the hamster's eye, facing the camera yaw, with the shell wall hidden.
_Avoid_: fixed first-person, eye camera

**Camera yaw**:
The chase view's heading, which defines the frame of actions and observations.
_Avoid_: heading, view angle

**Trajectory**:
A recorded episode of a controller's actions and the resulting states.
_Avoid_: replay, demo, log

**Abandoned run**:
An episode the human leaves before success, failure, or the time limit ends it; the game discards it rather than recording it.
_Avoid_: quit episode, partial trajectory

**Human baseline**:
The success rate and completion times humans reach on a level and task, measured from their recorded trajectories.
_Avoid_: high score, leaderboard
