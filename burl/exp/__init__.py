from burl.utils import g_cfg, log_warn, parse_args


def update_cfg_from_args():
    abbrs = {'num_iters': 'num_iterations',
             'rand_dyn': 'random_dynamics',
             'centralized': 'use_centralized_curriculum'}
    for name, value in parse_args():
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
