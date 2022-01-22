# UAV Motion Primitive Trajectory Generation

Implementation of a UAV motion primitive based trajectory generation framework.

The motion primitive generation is based on this paper: 
**Mueller, Hehn, "_A computationally efficient motion primitive for quadrocopter trajectory generation_"**
with the source code available [here](https://github.com/markwmuller/RapidQuadrocopterTrajectories).

## Usage

```shell script
python main.py
```

A global path is first generated using the Phi* path finding algorithm. Then, we compute a local planning solution
using an iterative motion primitive generation algorithm.

