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

**Controller**:
Whoever issues actions to the hamster: a human player or a policy.
_Avoid_: agent (ambiguous with the hamster)

### World

**Level**:
The static 3D world a hamsterball moves through: terrain, ramps, edges, and task-relevant markers.
_Avoid_: map, stage, track

**Task**:
The objective of an episode within a level (for example reaching a goal), which defines success, failure, and reward.
_Avoid_: mission, challenge

### Front ends

**Game**:
The human-playable, rendered, real-time front end.

**Env**:
The headless, faster-than-real-time front end that policies train against.
_Avoid_: gym, simulator (for the front end)

**View**:
A camera from which pixel observations or the game's screen are rendered (fixed third-person, fixed first-person, or controlled third-person).
_Avoid_: camera angle, perspective

**Trajectory**:
A recorded episode of a controller's actions and the resulting states.
_Avoid_: replay, demo, log
