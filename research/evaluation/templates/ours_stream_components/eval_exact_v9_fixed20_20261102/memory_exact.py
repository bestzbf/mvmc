"""Inference-only interpolation storage change, identical chunk/operator shapes."""
import torch
from torch import Tensor
from typing import Sequence
from types import MethodType
from torch.nn import functional as F

INT_MAX = 1610612736


def custom_interpolate(x, size=None, scale_factor=None, mode='bilinear', align_corners=True,
                       *, _limit=INT_MAX):
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    count = size[0] * size[1] * x.shape[0] * x.shape[1]
    if count <= _limit or torch.is_grad_enabled():
        if count <= _limit:
            return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        return torch.cat([F.interpolate(c, size=size, mode=mode, align_corners=align_corners)
                          for c in torch.chunk(x, count // _limit + 1, dim=0)], dim=0).contiguous()
    # Original routine concatenates these SAME chunks and returns contiguous().
    # Store each completed chunk into its final slice instead of retaining every
    # chunk plus a second full concatenation output. No interpolation is split
    # further and no arithmetic/accumulation order changes.
    result = x.new_empty((x.shape[0], x.shape[1], *size))
    start = 0
    for chunk in torch.chunk(x, count // _limit + 1, dim=0):
        piece = F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners)
        stop = start + len(piece)
        result[start:stop].copy_(piece)
        start = stop
        del piece
    return result


def scratch_forward(self, features):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('DPT lifetime optimization is inference only')
    # The caller supplies a disposable pyramid list. Drop each raw grid after
    # its exact original convolution; retain only the projected skip feature.
    layers = []
    for name in ('layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn'):
        value = features.pop(0)
        layers.append(getattr(self.scratch, name)(value))
        del value
    stack = [fusion_forward(self.scratch.refinenet4, layers.pop(), size=layers[-1].shape[2:])]
    stack.append(fusion_forward(self.scratch.refinenet3, stack.pop(), layers.pop(), size=layers[-1].shape[2:]))
    stack.append(fusion_forward(self.scratch.refinenet2, stack.pop(), layers.pop(), size=layers[-1].shape[2:]))
    stack.append(fusion_forward(self.scratch.refinenet1, stack.pop(), layers.pop()))
    return self.scratch.output_conv1(stack.pop())


def fusion_forward(self, *xs, size=None):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('DPT lifetime optimization is inference only')
    output = xs[0]
    if self.has_residual:
        res = self.resConfUnit1(xs[1])
        output = self.skip_add.add(output, res)
        del res
    del xs
    output = self.resConfUnit2(output)
    if size is None and self.size is None:
        modifier = {'scale_factor': 2}
    else:
        modifier = {'size': self.size if size is None else size}
    output = custom_interpolate(output, **modifier, mode='bilinear', align_corners=self.align_corners)
    return self.out_conv(output)


def install(model):
    model.track_head.forward = MethodType(motion_forward, model.track_head)
    model.track_head.scratch_forward = MethodType(scratch_forward, model.track_head)
    model.track_head._apply_pos_embed = MethodType(position_inplace, model.track_head)
    for i in range(1, 5):
        block = getattr(model.track_head.scratch, 'refinenet'+str(i))
        block.forward = MethodType(fusion_forward, block)


def position_inplace(self, x, W, H, ratio=.1):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Private DPT feature in-place addition is inference only')
    from multicamera4rc.vendor.heads.utils import create_uv_grid, position_grid_to_embed
    pos = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W/H, dtype=x.dtype, device=x.device)
    pos = position_grid_to_embed(pos, x.shape[1]) * ratio
    pos = pos.permute(2,0,1)[None].expand(x.shape[0],-1,-1,-1)
    return x.add_(pos)


def motion_forward(
    self,
    decoded_taps: Sequence[Tensor],
    output_hw: tuple[int, int],
) -> dict[str, Tensor]:
    """Return displacement g, confidence and pre-output-conv2 features.

    decoded_taps consists of four [B,U,L+1,D] tensors. Source-time
    anchoring (g(target)-g(source)) is performed by the enclosing model.
    """
    if self.training or torch.is_grad_enabled():
        raise RuntimeError("Consuming DPT is inference only")
    if len(decoded_taps) != 4 or decoded_taps[0].ndim != 4:
        raise ValueError("decoded_taps must contain four [B,U,L+1,D] tensors")
    height, width = output_hw
    if height < 1 or width < 1:
        raise ValueError("output_hw must be positive")
    hp, wp = height // self.patch_size, width // self.patch_size
    if hp < 1 or wp < 1:
        raise ValueError("output_hw must be at least one patch in each dimension")
    batch, targets, tokens, dim = decoded_taps[0].shape
    if tokens != hp * wp + 1 or dim != self.norm.normalized_shape[0]:
        raise ValueError("decoded tokens do not match output_hw's patch grid/embedding width")
    if any(tap.shape != decoded_taps[0].shape for tap in decoded_taps):
        raise ValueError("all four decoded taps must have the same shape")

    pyramid = []
    for projection, resize in zip(self.projects, self.resize_layers):
        tap = decoded_taps.pop(0)
        patches = self.norm(tap[:, :, 1:].reshape(batch * targets, hp * wp, dim))
        del tap
        grid = patches.transpose(1, 2).reshape(batch * targets, dim, hp, wp)
        grid = projection(grid)
        if self.pos_embed:
            grid = self._apply_pos_embed(grid, width, height)
        pyramid.append(resize(grid))
        del patches, grid

    # scratch_forward includes output_conv1 (features -> features/2)
    # before the final interpolation, preserving 4RC's DPT ordering.
    features = self.scratch_forward(pyramid)
    features = custom_interpolate(
        features, (height, width), mode="bilinear", align_corners=True
    )
    if self.pos_embed:
        features = self._apply_pos_embed(features, width, height)
    raw = self.scratch.output_conv2(features).float()
    xyz = raw[:, :3].permute(0, 2, 3, 1)

    # Compute exponentials in float32 and bound only overflow-scale
    # logits. The where form also keeps the derivative at zero equal to
    # one, unlike sign(x)*expm1(abs(x)).
    displacement = torch.where(
        xyz >= 0,
        torch.expm1(xyz.clamp(max=80.0)),
        -torch.expm1((-xyz).clamp(max=80.0)),
    )
    confidence = 1.0 + torch.exp(raw[:, 3].clamp(max=80.0))
    return {
        "displacement": displacement.reshape(batch, targets, height, width, 3),
        "confidence": confidence.reshape(batch, targets, height, width),
        "features": features.reshape(batch, targets, features.shape[1], height, width),
    }
