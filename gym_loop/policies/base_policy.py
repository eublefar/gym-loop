import torch
from torch.nn import Module
from typing import Union, Dict
import numpy as np


class BasePolicy(Module):
    def __init__(self, inp_dim: int, outp_dim: int):
        raise NotImplementedError()

    def act(self, state: Union[np.ndarray, torch.Tensor, Dict]) -> np.ndarray:
        """Act on state"""
        raise NotImplementedError()

    def forward(
        self, state_batch: Union[np.ndarray, torch.Tensor, Dict]
    ) -> Dict[str, torch.Tensor]:
        """Batch prediction"""
        raise NotImplementedError()
