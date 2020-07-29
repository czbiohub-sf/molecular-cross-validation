# implementation of Aggregated Momentum Gradient Descent, adapted from
# the code at https://github.com/AtheMathmo/AggMo

from typing import Sequence

import torch
from torch.optim.optimizer import Optimizer, required


class AggMo(Optimizer):
    """Implements Aggregated Momentum Gradient Descent"""

    def __init__(
        self,
        params: dict,
        lr: float = required,
        betas: Sequence[float] = (0.0, 0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr / len(betas), betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @classmethod
    def from_exp_form(cls, params, lr=required, a=0.1, k=3, weight_decay=0):
        betas = [1 - a ** i for i in range(k)]
        return cls(params, lr, betas, weight_decay)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            betas = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = {}
                    for beta in betas:
                        param_state["momentum_buffer"][beta] = torch.zeros_like(p)

                for beta in betas:
                    buf = param_state["momentum_buffer"][beta]
                    buf.mul_(beta).add_(d_p)
                    p.add_(buf, alpha=-group["lr"])
        return loss
