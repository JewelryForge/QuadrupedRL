import sys

sys.path.append('.')
from burl.rl.runner import TgNetTrainer
from burl.utils import g_cfg, log_warn, init_logger, parse_args
import wandb


def update_cfg_from_args():
    for name, value in parse_args().items():
        if not hasattr(g_cfg, name):
            print(f"g_cfg has no attribute named '{name}'")
        setattr(g_cfg, name, value)
        value = getattr(g_cfg, name)
        log_warn(f'{name}: {type(value).__name__} -> {value}')


reward_profile = (('ImitationReward', 0.08),
                  ('YawRateReward', 0.04),
                  ('BodyHeightReward', 0.08),
                  # ('HipAnglePenalty', 0.04),
                  ('RedundantLinearPenalty', 0.04),
                  ('RollPitchRatePenalty', 0.04),
                  ('BodyPosturePenalty', 0.04),
                  ('FootSlipPenalty', 0.04),
                  # ('TorqueGradientPenalty', 0.04),
                  ('AliveReward', 0.1))


def main():
    init_logger()
    g_cfg.rewards_weights = reward_profile
    g_cfg.action_frequency = 500
    g_cfg.init_noise_std = 0.03
    g_cfg.add_disturbance = False
    if len(sys.argv) > 1:
        update_cfg_from_args()
    else:
        g_cfg.learning_rate = 1e-5
        g_cfg.max_sim_iterations = 2000
        g_cfg.num_envs = 1
        g_cfg.trn_type = 'plain'
        g_cfg.rendering = True
        g_cfg.use_mp = False
        g_cfg.use_wandb = False
        g_cfg.sleeping_enabled = False
        g_cfg.schedule = 'fixed'
    wandb.init(project='whole-body-tg-train', config=g_cfg.__dict__, name=g_cfg.run_name, save_code=True,
               mode=None if g_cfg.use_wandb else 'disabled')
    log_warn(f'Training on {g_cfg.device}')
    runner = TgNetTrainer()
    runner.learn()


if __name__ == '__main__':
    main()
