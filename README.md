# teacher-student

_Nov20_
- base_twist的测量值从世界坐标系改为基体坐标系
- 对action、TG的基准频率clip
- 修改了原PPO中的noise_std（1.0是否太大了？）
- is_safe的判定更加严格，但仍需要继续考虑（序列的指标）
- 提高速度奖励的权重，训练中可以很快学会站定，但倾向于向前倒

_Nov21_
- multiprocess的效率太低，如果每步step都创建线程，花的时间是单线程的15~20倍（32环境4线程）
  - 创建线程 ～5ms
  - 循环时间（+start）～40ms
  - 等待时间（join）3e-4～40ms
  - 单线程32环境step ～100ms
- 取消对action的clip，否则会学到一个很诡异的余差（z>0），导致不再迭代；考虑直接在action超限时terminate
- 研究ppo的算法

_Nov22_
- getJointStates的力和力矩的意义不明，接触力改用getContactPoints
- 重构了底层代码，删除legged_gym意义不明的代码


# 目标

_Stage1_
- 输入到神经网络的数据标准化

_Stage2_
- 数据增加噪声
- terrain curriculum