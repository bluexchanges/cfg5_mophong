"""Single-process, two-device ACT with padding-aware differentiable reduction."""

import torch
from torch import nn


class LossShard(nn.Module):
    """Return vectors for gather, without gathering the large predictions."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, images, actions, is_pad):
        result = self.policy(state, images, actions, is_pad)
        elements = result["valid_action_elements"]
        return {"l1": result["l1"].reshape(1), "kl": result["kl"].reshape(1),
                "elements": elements.reshape(1),
                "samples": elements.new_tensor([len(state)])}


def reduce_shards(result, kl_weight):
    """L1 is weighted by real target elements; KL by samples, including tails."""
    elements = result["elements"].sum()
    samples = result["samples"].sum()
    l1 = (result["l1"] * result["elements"]).sum() / elements
    kl = (result["kl"] * result["samples"]).sum() / samples
    return {"loss": l1 + kl_weight * kl, "l1": l1, "kl": kl,
            "valid_action_elements": elements, "valid_action_steps": elements // 8}


class ParallelACT(nn.Module):
    """Keep plain model checkpoint keys and optimizer parameter identities."""

    def __init__(self, policy, gpu_count):
        super().__init__()
        if gpu_count < 2 or torch.cuda.device_count() < gpu_count:
            raise RuntimeError(f"Cần {gpu_count} GPU CUDA được Slurm cấp")
        self.config = policy.config
        self.parallel = nn.DataParallel(LossShard(policy), device_ids=list(range(gpu_count)))

    @property
    def base_policy(self):
        return self.parallel.module.policy

    def forward(self, state, images, actions, is_pad):
        return reduce_shards(self.parallel(state, images, actions, is_pad),
                             self.config["kl_weight"])

    def state_dict(self, *args, **kwargs):
        return self.base_policy.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.base_policy.load_state_dict(*args, **kwargs)

    def configure_optimizer(self):
        return self.base_policy.configure_optimizer()
