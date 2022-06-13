import torch


class PolicyWrapper(object):
    def __init__(self, net, device):
        self.net = net
        self.device = torch.device(device)

    def __call__(self, obs):
        with torch.inference_mode():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return self.net(obs).detach().cpu().numpy()
