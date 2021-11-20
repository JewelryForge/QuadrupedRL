# teacher-student

*Nov20*
- base_twist的测量值从世界坐标系改为基体坐标系
- 对action、TG的基准频率clip
- 修改了原PPO中的noise_std（1.0是否太大了？）
- is_safe的判定更加严格，但仍需要继续考虑（序列的指标）
- 提高速度奖励的权重，训练中可以很快学会站定，但倾向于向前倒