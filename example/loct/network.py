from typing import Type, Union, Any, Dict, Tuple

import numpy as np
import torch
from torch import nn


class ActorNetMLP(nn.Module):
    """
    extero_obs  ->  extero_layers ->|
                                    |-> concat -> action_layers -> action
    proprio_obs -> proprio_layers ->|
    """

    def __init__(
            self,
            extero_obs_dim,
            real_world_obs_dim,
            extero_layer_dims=(72, 64),
            locomotion_layer_dims=(),
            action_layer_dims=(256, 160, 128),
            activation: Type[nn.Module] = nn.Tanh,
            device='cpu'
    ):
        super().__init__()
        self.input_dim = extero_obs_dim + real_world_obs_dim
        self.extero_obs_dim = extero_obs_dim
        self.real_world_obs_dim = real_world_obs_dim
        extero_layers, locomotion_layers, action_layers = [], [], []
        self.extero_obs_dim = extero_feature_dim = extero_obs_dim
        self.device = torch.device(device)
        if extero_layer_dims:
            for dim in extero_layer_dims:
                extero_layers.append(nn.Linear(extero_feature_dim, dim))
                extero_layers.append(activation())
                extero_feature_dim = dim

        self.real_world_obs_dim = locomotion_feature_dim = real_world_obs_dim
        if locomotion_layer_dims:
            for dim in locomotion_layer_dims:
                locomotion_layers.append(nn.Linear(locomotion_feature_dim, dim))
                locomotion_layers.append(activation())
                locomotion_feature_dim = dim

        action_feature_dim = extero_feature_dim + locomotion_feature_dim
        for dim in action_layer_dims:
            action_layers.append(nn.Linear(action_feature_dim, dim))
            action_layers.append(activation())
            action_feature_dim = dim
        self.output_dim = action_feature_dim

        self.extero_layers = nn.Sequential(*extero_layers)
        self.locomotion_layers = nn.Sequential(*locomotion_layers)
        self.action_layers = nn.Sequential(*action_layers)

    def forward(self, x):
        if self.device is not None:
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        extero_obs, real_world_obs = x[..., :self.extero_obs_dim], x[..., self.extero_obs_dim:]
        extero_feature, locomotion_feature = self.extero_layers(extero_obs), self.locomotion_layers(real_world_obs)
        return self.action_layers(torch.cat((extero_feature, locomotion_feature), dim=-1))


class ActorNet(nn.Module):
    def __init__(
            self,
            extero_obs_dim,
            real_world_obs_dim,
            extero_layer_dims=(72, 64),
            locomotion_layer_dims=(),
            action_layer_dims=(256, 160, 128),
            activation: Type[nn.Module] = nn.Tanh,
            device='cpu'
    ) -> None:
        super().__init__()
        self.device = device
        self.model = ActorNetMLP(
            extero_obs_dim,
            real_world_obs_dim,
            extero_layer_dims,
            locomotion_layer_dims,
            action_layer_dims,
            activation,
            device,
        )
        self.output_dim = self.model.output_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        logits = self.model(obs)
        return logits, state
