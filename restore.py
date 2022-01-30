import sys

sys.path.append('.')
from burl.rl.runner import OnPolicyRunner
from burl.utils import g_cfg, log_warn, init_logger, parse_args, find_log
import wandb

resume_params = {'run_id': None, 'time': None, 'epoch': None}

raise NotImplementedError


def update_cfg_from_args():
    for name, value in parse_args().items():
        if name in resume_params:
            resume_params[name] = value
        else:
            if not hasattr(g_cfg, name):
                print(f"g_cfg has no attribute named '{name}'")
            setattr(g_cfg, name, value)
            value = getattr(g_cfg, name)
        log_warn(f'{name}: {type(value).__name__} -> {value}')


def main():
    init_logger()
    if len(sys.argv) > 1:
        update_cfg_from_args()
    wandb.init(project='teacher-student', config=g_cfg.__dict__, name=g_cfg.run_name,
               resume="allow", id=resume_params['run_id'])
    log_warn(f'Training on {g_cfg.device}')
    runner = OnPolicyRunner()
    runner.load(find_log(time=resume_params['time'], epoch=resume_params['epoch']))
    runner.learn()


if __name__ == '__main__':
    main()
