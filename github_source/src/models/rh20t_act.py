"""Adapter RH20T DETRVAE: 8 views, 8D, explicit device, masked loss.

Core DETRVAE/Transformer/encoder and FrozenBatchNorm stay upstream unchanged.
Images supplied here are already normalized by ACTDataset/panda_observation.
"""
from copy import deepcopy
import hashlib
from types import SimpleNamespace
import torch
from torch import nn
from torchvision.models import resnet18
from src.data.common import ROOT, load_config, read_json

COMMIT = "107ce1287a46be3a1a7ea954f8bbdec3bb103288"


def verify_upstream():
    manifest = read_json(ROOT / "third_party/rh20t_act_manifest.json")
    if manifest["commit"] != COMMIT:
        raise ValueError("ACT source commit không khớp")
    for entry in manifest["files"]:
        path = ROOT / "third_party/rh20t_act" / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Upstream source đã đổi: {path}")


verify_upstream()
from third_party.rh20t_act.models.detr.models.detr_vae import DETRVAE, build_encoder
from third_party.rh20t_act.models.detr.models.transformer import Transformer
from third_party.rh20t_act.models.detr.models.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
from third_party.rh20t_act.models.detr.models.position_encoding import PositionEmbeddingSine


class CameraTransformer(Transformer):
    """Add camera identity to repeated per-view 2D positions, then upstream attention."""
    def __init__(self, cameras, identity, **kwargs):
        super().__init__(**kwargs)
        self.camera_count = cameras
        self.camera_embedding = nn.Embedding(cameras, self.d_model) if identity else None

    def camera_positions(self, pos):
        if pos.shape[-1] % self.camera_count:
            raise ValueError("Folded feature width không chia hết camera count")
        if self.camera_embedding is None:
            return pos
        width = pos.shape[-1] // self.camera_count
        identity = self.camera_embedding.weight.T.repeat_interleave(width, dim=1)[None, :, None, :]
        return pos + identity

    def forward(self, src, mask, query_embed, pos_embed, *args):
        return super().forward(src, mask, query_embed, self.camera_positions(pos_embed), *args)


def masked_l1(prediction, target, is_pad):
    if prediction.shape != target.shape or is_pad.shape != target.shape[:2] or is_pad.dtype != torch.bool:
        raise ValueError("Loss shape/mask sai")
    valid = ~is_pad
    if not valid.any(dim=1).all():
        raise ValueError("Mỗi sample cần ít nhất một action thật")
    # Index before reduction: padding values never enter the denominator or sum.
    absolute = (prediction[valid] - target[valid]).abs()
    return absolute.sum() / absolute.numel()


class RH20TACTPolicy(nn.Module):
    def __init__(self, config, data_config):
        super().__init__()
        self.config, self.data_config = deepcopy(config), deepcopy(data_config)
        if config["upstream_commit"] != COMMIT or config["schema_version"] != 1:
            raise ValueError("ACT schema/commit không được hỗ trợ")
        if config["state_dim"] != 8 or config["action_dim"] != 8 or config["chunk_size"] != data_config["chunk_size"]:
            raise ValueError("ACT phải khớp representation8D và chunk_size của data")
        if config["input_normalization"] != "already_imagenet_and_train_state_stats":
            raise ValueError("Adapter chỉ nhận input đã normalize đúng một lần")
        for key in ("hidden_dim", "dim_feedforward", "enc_layers", "dec_layers", "nheads", "chunk_size"):
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"{key} phải là số nguyên dương")
        if config["hidden_dim"] % 4 or config["hidden_dim"] % config["nheads"]:
            raise ValueError("hidden_dim phải chia hết4 và nheads")
        if not 0 <= config["dropout"] < 1 or config["kl_weight"] < 0:
            raise ValueError("Dropout/KL weight không hợp lệ")
        self.camera_names = ["cam_" + serial for serial in data_config["camera_serials"]]
        if len(self.camera_names) != 8 or len(set(self.camera_names)) != 8:
            raise ValueError("Cần8 camera khác nhau")
        if config["backbone"] != "resnet18":
            raise ValueError("Adapter đã kiểm ResNet18")
        # No implicit network access, CUDA, or global argument parsing.
        backbone = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        if config["backbone_weights_path"]:
            path = ROOT / config["backbone_weights_path"]
            backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        feature_net = BackboneBase(backbone, config["lr_backbone"] > 0, 512, False)
        if config["lr_backbone"] == 0:
            feature_net.requires_grad_(False)
        joiner = Joiner(feature_net, PositionEmbeddingSine(config["hidden_dim"] // 2, normalize=True))
        joiner.num_channels = 512
        transformer = CameraTransformer(8, config["camera_identity_embedding"],
            d_model=config["hidden_dim"], nhead=config["nheads"],
            num_encoder_layers=config["enc_layers"], num_decoder_layers=config["dec_layers"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
            normalize_before=config["pre_norm"], return_intermediate_dec=True)
        self.model = DETRVAE([joiner], transformer, build_encoder(SimpleNamespace(**config)),
                             state_dim=8, num_queries=config["chunk_size"], camera_names=self.camera_names)

    def forward(self, state, images, actions=None, is_pad=None):
        expected = (len(state), 8, 3, self.data_config["image_height"], self.data_config["image_width"])
        if state.ndim != 2 or state.shape[1] != 8 or tuple(images.shape) != expected:
            raise ValueError("Input phải state(B,8), images(B,8,3,H,W) theo data config")
        if not torch.isfinite(state).all() or not torch.isfinite(images).all():
            raise ValueError("Input chứa NaN/Inf")
        if actions is None:
            if is_pad is not None:
                raise ValueError("Inference không nhận padding mask")
            prediction, _, _ = self.model(state, images, None)
            return prediction
        if actions.shape != (len(state), self.config["chunk_size"], 8) or is_pad is None:
            raise ValueError("Targets phải (B,K,8), kèm mask")
        if is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
            raise ValueError("is_pad phải bool(B,K), True=padding")
        if not (~is_pad).any(dim=1).all() or not torch.isfinite(actions[~is_pad]).all():
            raise ValueError("Sample không có target thật hoặc target thật không hữu hạn")
        clean_actions = actions.masked_fill(is_pad.unsqueeze(-1), 0)
        prediction, _, (mu, logvar) = self.model(state, images, None, clean_actions, is_pad)
        l1 = masked_l1(prediction, clean_actions, is_pad)
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=1).mean()
        loss = l1 + self.config["kl_weight"] * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("ACT loss không hữu hạn")
        return {"loss": loss, "l1": l1, "kl": kl, "valid_action_steps": (~is_pad).sum(),
                "valid_action_elements": (~is_pad).sum() * 8, "prediction": prediction}

    def configure_optimizer(self):
        cfg = self.config
        groups = [{"params": [p for name, p in self.named_parameters() if p.requires_grad and "backbones" not in name]},
                  {"params": [p for name, p in self.named_parameters() if p.requires_grad and "backbones" in name],
                   "lr": cfg["lr_backbone"]}]
        return torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def load_policy(config_path="configs/act/rh20t_base8.json", device="cpu"):
    cfg = read_json(ROOT / config_path)
    data = load_config(ROOT / cfg["data_config"])
    return RH20TACTPolicy(cfg, data).to(device)
