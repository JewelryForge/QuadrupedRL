# Quadruped Locomotion Reinforcement Learning

This is an environment for learning quadrupedal locomotion depending on 
either [bullet physics engine](https://github.com/bulletphysics/bullet3)
or [mujoco](https://github.com/deepmind/mujoco) and
[dm_control](https://github.com/deepmind/dm_control) 
(not implemented yet), containing

- Interfaces of quadruped states, creating terrain from elevation maps and environmental states
- Motor identification, i.e. Actuator Network
- Curriculum learning on disturbance and terrain
- Abundant command, force, torque, terrain and trajectory visualization 

[tianshou](https://github.com/thu-ml/tianshou) is used for training in
our examples.

[//]: # (- Imitation Learning from privileged teacher to proprioceptive student)

<div align=center>
<img src="resources/lin.gif" alt="linear" style=" zoom:100%;" />
<img src="resources/rot.gif" alt="rotate" style=" zoom:100%;" />
</div>

Requirements: `python>=3.7`, `pybullet>=3.2`, `torch>=1.10`, `numpy`, `scipy`, `wandb`

For train, run:
```bash
PYTHONPATH=./ python example/loct/train.py --task example/lctv0.yaml \
  --batch-size 8192 --repeat-per-collect 8 --step-per-epoch 160000 \
  --norm-adv 1 --run-name example
```
