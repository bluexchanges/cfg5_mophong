"""MCR ResNet50 + existing ACT core, with an explicit cached spatial-feature path.

Separate architecture/checkpoints from the old ResNet18 policy. No upstream edits.
MCR features: eight views, 2048x7x7, INT8 with FP16 scales on disk.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.data.sim_demos import file_hash
from src.models.rh20t_act import RH20TACTPolicy, masked_l1
from third_party.rh20t_act.models.detr.models.backbone import BackboneBase, Joiner
from third_party.rh20t_act.models.detr.models.position_encoding import PositionEmbeddingSine
from third_party.rh20t_act.models.detr.models.detr_vae import reparametrize

MCR_SHA = "21560db9e795384a7b73b0626d33d6525f9d4c5de11fca6ded52089aab14f97c"
PREPROCESS = "resize_short256_center224_imagenet_v1"


def preprocess_mcr(rgb):
    """NCHW uint8 -> author-compatible deterministic resize/crop/normalization."""
    if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("MCR input phải uint8 (N,3,H,W)")
    # Resize float tensors, with antialias=True, identically in cache and rollout.
    image = TF.resize(rgb.float().div(255), 256, InterpolationMode.BILINEAR, antialias=True)
    image = TF.center_crop(image, [224, 224])
    return TF.normalize(image, [.485, .456, .406], [.229, .224, .225])


def make_mcr_backbone(path):
    if file_hash(path) != MCR_SHA:
        raise ValueError("MCR checkpoint sai SHA256 chính thức")
    net = resnet50(weights=None)
    net.fc = nn.Identity()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return net


class MCRACTPolicy(RH20TACTPolicy):
    def __init__(self, config, data_cfg, *, mcr_checkpoint=None):
        cfg = deepcopy(config)
        if cfg.get("backbone") != "mcr_resnet50":
            raise ValueError("Cần config riêng backbone=mcr_resnet50")
        if cfg.get("feature_preprocess") != PREPROCESS:
            raise ValueError("MCR preprocessing chưa được hỗ trợ")
        parent_cfg = {**cfg, "backbone": "resnet18", "backbone_weights_path": None}
        super().__init__(parent_cfg, data_cfg)
        net = make_mcr_backbone(mcr_checkpoint) if mcr_checkpoint else resnet50(weights=None)
        net.fc = nn.Identity()
        trunk = BackboneBase(net, True, 2048, False)
        self.model.backbones = nn.ModuleList([Joiner(trunk, PositionEmbeddingSine(cfg["hidden_dim"] // 2, normalize=True))])
        self.model.input_proj = nn.Conv2d(2048, cfg["hidden_dim"], 1)
        self.config = cfg
        self.frozen = cfg["lr_backbone"] == 0
        self.model.backbones.requires_grad_(not self.frozen)
        if self.frozen:
            self.model.backbones.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.backbones.eval()  # BatchNorm running statistics must stay frozen too.
        return self

    def extract_features(self, rgb, camera_valid=None):
        if rgb.ndim != 5 or rgb.shape[1:3] != (8, 3):
            raise ValueError("RGB phải (B,8,3,H,W)")
        flattened = rgb.flatten(0, 1)
        if camera_valid is not None:
            if camera_valid.shape != rgb.shape[:2] or camera_valid.dtype != torch.bool or not camera_valid.any(1).all():
                raise ValueError("camera_valid must be bool(B,8), at least one real view")
            flattened = flattened[camera_valid.flatten()]
        if rgb.dtype == torch.uint8:
            flat = preprocess_mcr(flattened)
        elif rgb.is_floating_point() and rgb.shape[-2:] == (224, 224):
            flat = flattened  # JPEG loader already applies the same preprocessing.
        else:
            raise ValueError("Float image phải normalized (B,8,3,224,224)")
        features = self.model.backbones[0][0](flat)["0"]
        if camera_valid is not None:
            output = features.new_zeros((len(rgb) * 8, 2048, 7, 7))
            output[camera_valid.flatten()] = features
            return output.reshape(len(rgb), 8, 2048, 7, 7)
        return features.reshape(len(rgb), 8, 2048, 7, 7)

    def forward_features(self, state, features, actions=None, is_pad=None, camera_valid=None):
        if state.shape != (len(features), 8) or features.shape[1:] != (8, 2048, 7, 7):
            raise ValueError("Features phải (B,8,2048,7,7), state(B,8)")
        if not torch.isfinite(state).all() or not torch.isfinite(features).all():
            raise ValueError("Features/state NaN/Inf")
        model = self.model
        view_mask = None
        if camera_valid is not None:
            if camera_valid.shape != (len(state), 8) or camera_valid.dtype != torch.bool or not camera_valid.any(1).all():
                raise ValueError("camera_valid must be bool(B,8), at least one real view")
            # Folded maps are (B,C,7,8*7); prefix latent/proprio tokens stay visible.
            spatial_mask = (~camera_valid)[:, None, :, None].expand(-1, 7, -1, 7).reshape(len(state), -1)
            view_mask = torch.cat([torch.zeros((len(state), 2), dtype=torch.bool, device=state.device), spatial_mask], 1)
        mu = logvar = None
        if actions is not None:
            if actions.shape != (len(state), self.config["chunk_size"], 8) or is_pad is None or is_pad.dtype != torch.bool or is_pad.shape != actions.shape[:2]:
                raise ValueError("Target/mask ACT không hợp lệ")
            if not (~is_pad).any(dim=1).all() or not torch.isfinite(actions[~is_pad]).all():
                raise ValueError("Target không có action thật hoặc NaN")
            actions = actions.masked_fill(is_pad.unsqueeze(-1), 0)
            cls = model.cls_embed.weight.unsqueeze(0).expand(len(state), -1, -1)
            sequence = torch.cat([cls, model.encoder_joint_proj(state).unsqueeze(1), model.encoder_action_proj(actions)], 1).permute(1, 0, 2)
            mask = torch.cat([torch.zeros((len(state), 2), dtype=torch.bool, device=state.device), is_pad], 1)
            encoded = model.encoder(sequence, pos=model.pos_table.permute(1, 0, 2), src_key_padding_mask=mask)[0]
            mu, logvar = model.latent_proj(encoded).chunk(2, dim=1)
            latent = model.latent_out_proj(reparametrize(mu, logvar))
        else:
            if is_pad is not None:
                raise ValueError("Inference không nhận is_pad")
            latent = model.latent_out_proj(state.new_zeros((len(state), model.latent_dim)))
        maps = features.to(state.dtype)
        projected, positions = [], []
        for cam in range(8):
            projected.append(model.input_proj(maps[:, cam]))
            positions.append(model.backbones[0][1](maps[:, cam]))
        hidden = model.transformer(torch.cat(projected, 3), view_mask, model.query_embed.weight,
                                   torch.cat(positions, 3), latent, model.input_proj_robot_state(state),
                                   model.additional_pos_embed.weight)[0]
        prediction = model.action_head(hidden)
        if actions is None:
            return prediction
        l1 = masked_l1(prediction, actions, is_pad)
        kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()
        loss = l1 + self.config["kl_weight"] * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("MCR ACT loss NaN/Inf")
        return {"loss": loss, "l1": l1, "kl": kl, "valid_action_steps": (~is_pad).sum(),
                "valid_action_elements": (~is_pad).sum() * 8, "prediction": prediction}

    def forward(self, state, images, actions=None, is_pad=None, camera_valid=None):
        if self.frozen:
            with torch.no_grad():
                features = self.extract_features(images, camera_valid=camera_valid)
                if self.config.get("cache_quantization"):
                    scale = (features.abs().amax((-2, -1), keepdim=True) / 127).clamp_min(1e-6).half().float()
                    features = (features / scale).round().clamp(-127, 127) * scale
        else:
            features = self.extract_features(images, camera_valid=camera_valid)
        return self.forward_features(state, features, actions, is_pad, camera_valid=camera_valid)

    def initialize_act(self, saved):
        """Explicit transfer of compatible ACT weights; new visual projection is learned."""
        expected = self.state_dict()
        kept, skipped = {}, []
        for key, value in saved["model"].items():
            if key.startswith("model.backbones.") or key.startswith("model.input_proj."):
                skipped.append(key)
                continue
            if key not in expected or expected[key].shape != value.shape:
                raise ValueError(f"ACT architecture khác checkpoint: {key}")
            kept[key] = value
        result = self.load_state_dict(kept, strict=False)
        if result.unexpected_keys or any(not (k.startswith("model.backbones.") or k.startswith("model.input_proj.")) for k in result.missing_keys):
            raise ValueError("Thiếu trọng số ACT ngoài backbone/projection")
        return {"loaded_act_tensors": len(kept), "replaced_visual_tensors": skipped,
                "method": "ACT transfer; MCR backbone and 2048->hidden projection replace old visual path"}
