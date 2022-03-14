# teacher-student

This is an environment for learning quadrupedal locomotion depending on [bullet physics engine](https://github.com/bulletphysics/bullet3), containing:

- Interfaces of quadruped states, creating terrain from elevation maps and environmental states
- A PPO implementation and locomotion training with it
- Motor identification, i.e. Actuator Network
- Curriculum learning on disturbance and terrain (not implemented)
- Abundant command, force, torque, terrain and trajectory visualization 
- General Parallelism for accelerating training

Requirements: `python>=3.9`, `pybullet>=3.2`, `torch>=1.10`, `numpy`, `scipy`