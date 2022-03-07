import sys

sys.path.append('.')
from burl.rl.runner import PolicyTrainer
from burl.utils import g_cfg, log_warn, init_logger, parse_args, find_log
import wandb

resume_params = {'run_id': None, 'time': None, 'epoch': None}


def update_cfg_from_wandb_and_args():
    abbrs = {'num_iters': 'num_iterations',
             'rand_dyn': 'random_dynamics'}
    resume_args = dict(parse_args())
    try:
        resume_params['run_id'] = resume_args.pop('run_id')
        resume_params['time'] = resume_args.pop('time')
        resume_params['epoch'] = resume_args.pop('epoch')
    except KeyError:
        raise RuntimeError('Args must contain `run_id`, `time` and `epoch` to resume')

    wandb.init(project='teacher-student', resume="allow", id=resume_params['run_id'])
    g_cfg.update(wandb.config)
    for name, value in resume_args.items():
        if name == 'mp_train':
            g_cfg.use_wandb = True
            g_cfg.use_mp = True
            g_cfg.rendering = False
            log_warn(f'wandb: on')
            log_warn(f'multi-process: on')
        elif name == 'cuda' or name == 'cpu':
            g_cfg.device = name
        elif name == 'on_plain':
            g_cfg.trn_type = 'plain'
            log_warn(f'terrain type: plain')
        else:
            if name in abbrs:
                name = abbrs[name]
            if not hasattr(g_cfg, name):
                raise RuntimeError(f"g_cfg has no attribute named '{name}'")
            setattr(g_cfg, name, value)
            value = getattr(g_cfg, name)
            if isinstance(value, bool):
                log_warn(f"{name}: {'on' if value else 'off'}")
            else:
                log_warn(f'{name}: {type(value).__name__} -> {value}')


def main():
    init_logger()
    update_cfg_from_wandb_and_args()
    log_warn(f'Training on {g_cfg.device}')
    runner = PolicyTrainer(g_cfg.task_type)
    runner.load(find_log(time=resume_params['time'], epoch=resume_params['epoch']))
    runner.learn()


if __name__ == '__main__':
    main()
