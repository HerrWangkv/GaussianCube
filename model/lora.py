# -*- coding: utf-8 -*-
from sklearn import base
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union


def _out_dim_1d(L_in: int, pad: int, dil: int, k: int, stride: int) -> int:
    return (L_in + 2 * pad - dil * (k - 1) - 1) // stride + 1


def _output_padding(input_size, output_size, stride, padding, kernel_size, dilation):
    # Computes required output_padding for conv_transpose
    return [
        output_size[i]
        - (
            (input_size[i] - 1) * stride[i]
            - 2 * padding[i]
            + dilation[i] * (kernel_size[i] - 1)
            + 1
        )
        for i in range(len(input_size))
    ]


def unfold3d(
    x: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Returns patches as a strided view like nn.Unfold (3D):
      Input:  [B, C, D, H, W]
      Output: [B, C * kD * kH * kW, L],  L = D_out * H_out * W_out
    """
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    # Pad symmetrically (matches Conv3d semantics for tuple padding)
    x_pad = F.pad(x, (pW, pW, pH, pH, pD, pD))

    B, C, Dp, Hp, Wp = x_pad.shape
    # Output grid sizes after conv
    D_out = _out_dim_1d(Dp, 0, dD, kD, sD)
    H_out = _out_dim_1d(Hp, 0, dH, kH, sH)
    W_out = _out_dim_1d(Wp, 0, dW, kW, sW)

    # Compute strided view over sliding windows
    s_b, s_c, s_d, s_h, s_w = x_pad.stride()
    # Shape: [B, C, kD, kH, kW, D_out, H_out, W_out]
    shape = (B, C, kD, kH, kW, D_out, H_out, W_out)
    strides = (
        s_b,
        s_c,
        s_d * dD,
        s_h * dH,
        s_w * dW,
        s_d * sD,
        s_h * sH,
        s_w * sW,
    )
    patches = torch.as_strided(x_pad, size=shape, stride=strides)

    # Reorder to [B, C*kD*kH*kW, L]
    patches = patches.reshape(B, C * kD * kH * kW, D_out * H_out * W_out)
    return patches


class LoRALayer(nn.Module):
    """
    Base LoRA layer: y = dropout(x) @ A^T @ B^T * scaling
    Expects last dim of x to be in_features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 32,
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if rank > 0:
            self.lora_A = nn.Parameter(
                torch.randn(rank, in_features, dtype=dtype)
                * (1 / math.sqrt(in_features))
            )
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
            self.reset_parameters()
        else:
            # Placeholders to keep typing simple
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_parameters(self):
        if self.lora_A is not None:
            nn.init.normal_(self.lora_A, std=1 / math.sqrt(self.lora_A.size(1)))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            # Return zeros with correct batch-like leading dims and out_features at the end
            *lead, _ = x.shape
            out_features = self.lora_B.size(0) if self.lora_B is not None else 0
            return torch.zeros(*lead, out_features, device=x.device, dtype=x.dtype)
        return self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


class LoRALinear(nn.Module):
    """
    LoRA adaptation for linear layers.
    """
    def __init__(self, original_layer: nn.Linear, rank=4, alpha=32, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=original_layer.weight.dtype,
        )

        # Freeze original weights
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + self.lora(x)


class LoRAConv1d(nn.Module):
    """
    LoRA adaptation for 1D convolutional layers.
    """
    def __init__(self, original_layer: nn.Conv1d, rank=4, alpha=32, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        kernel_size = original_layer.kernel_size[0]
        in_features = original_layer.in_channels * kernel_size
        out_features = original_layer.out_channels

        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=original_layer.weight.dtype,
        )

        self.kernel_size = kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation

        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)
        if self.lora.rank == 0:
            return original_output

        # unfold along 1D -> simulate 2D unfold
        x_unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride[0], 1),
            padding=(self.padding[0], 0),
            dilation=(self.dilation[0], 1),
        )  # [B, Cin*k, num_patches]
        x_unfolded = x_unfolded.transpose(1, 2)  # [B, num_patches, Cin*k]

        lora_output = self.lora(x_unfolded)  # [B, num_patches, Cout]
        lora_output = lora_output.transpose(1, 2)  # [B, Cout, num_patches]

        # trim in case unfold produced extra
        output_length = original_output.size(2)
        lora_output = lora_output[:, :, :output_length]

        return original_output + lora_output


class LoRAConv2d(nn.Module):
    """
    LoRA adaptation for 2D convolutional layers.
    """
    def __init__(self, original_layer: nn.Conv2d, rank=4, alpha=32, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        kernel_size = original_layer.kernel_size[0] * original_layer.kernel_size[1]
        in_features = original_layer.in_channels * kernel_size
        out_features = original_layer.out_channels

        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=original_layer.weight.dtype,
        )

        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation

        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)
        if self.lora.rank == 0:
            return original_output

        x_unfolded = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )  # [B, Cin*k_h*k_w, num_patches]
        x_unfolded = x_unfolded.transpose(1, 2)  # [B, num_patches, Cin*k_h*k_w]

        lora_output = self.lora(x_unfolded)  # [B, num_patches, Cout]
        lora_output = lora_output.transpose(1, 2)  # [B, Cout, num_patches]

        output_h, output_w = original_output.shape[2:]
        lora_output = F.fold(
            lora_output,
            output_size=(output_h, output_w),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        return original_output + lora_output


class LoRAConv3d(nn.Module):
    """
    TRUE 3D LoRA for Conv3d:
      - Unfold 3D patches (Cin * kD * kH * kW)
      - Apply LoRA to patches -> per-patch Cout
      - Fold back via output_padding (overlap-add) to (B, Cout, D_out, H_out, W_out)
    This mirrors your 1D/2D LoRA and is faithful to the conv receptive field.
    """

    def __init__(
        self,
        original_layer: nn.Conv3d,
        rank: int = 4,
        alpha: float = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer

        kD, kH, kW = original_layer.kernel_size
        in_features = original_layer.in_channels * kD * kH * kW
        out_features = original_layer.out_channels

        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            dtype=original_layer.weight.dtype,
        )

        # Cache conv params
        self.kernel_size = (kD, kH, kW)
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = (
            original_layer.groups
        )  # (LoRA path is independent of groups; base conv uses this)

        # Freeze original conv
        for p in self.original_layer.parameters():
            p.requires_grad = False

        # A grouped transposed-conv kernel of ones does correct overlap-add during "fold"
        # shape: (Cout, 1, kD, kH, kW), groups=Cout
        self.register_buffer(
            "_fold_kernel",
            torch.ones(out_features, 1, kD, kH, kW, dtype=original_layer.weight.dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base conv output
        base = self.original_layer(x)
        if self.lora.rank == 0:
            return base

        B, Cin, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        dD, dH, dW = self.dilation
        Cout = self.original_layer.out_channels

        # Unfold to patches: [B, Cin*kD*kH*kW, L]
        cols = unfold3d(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )  # [B, KKCin, L]
        L = cols.size(-1)

        # [B, L, Cin*kD*kH*kW] for LoRA, then -> [B, L, Cout]
        cols_t = cols.transpose(1, 2).contiguous()
        lora_per_patch = self.lora(cols_t)  # [B, L, Cout]

        # Reshape to grid: [B, Cout, D_out, H_out, W_out]
        D_out = _out_dim_1d(D, pD, dD, kD, sD)
        H_out = _out_dim_1d(H, pH, dH, kH, sH)
        W_out = _out_dim_1d(W, pW, dW, kW, sW)
        assert L == D_out * H_out * W_out, "Unfold size mismatch."

        lora_grid = lora_per_patch.transpose(1, 2).reshape(B, Cout, D_out, H_out, W_out)

        # Compute output_padding needed to match base conv output shape
        output_padding = _output_padding(
            lora_grid.shape[2:],  # (D_out, H_out, W_out)
            base.shape[2:],  # target (D_out, H_out, W_out)
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )

        lora_spatial = F.conv_transpose3d(
            lora_grid,
            self._fold_kernel.to(x.dtype),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=Cout,
            output_padding=tuple(output_padding),
        )

        return base + lora_spatial


def apply_lora_to_module(
    module: nn.Module,
    target_modules: List[str] = None,
    rank: int = 4,
    alpha: float = 32,
    dropout: float = 0.0,
    exclude_modules: List[str] = None,
) -> nn.Module:
    """
    Apply LoRA adaptation to specified modules in a neural network.
    
    Args:
        module: The module to apply LoRA to
        target_modules: List of module names/types to target (e.g., ['Linear', 'Conv1d', 'Conv2d'])
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout rate for LoRA layers
        exclude_modules: List of module names to exclude
    
    Returns:
        Modified module with LoRA adaptations
    """
    if target_modules is None:
        target_modules = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']
    
    if exclude_modules is None:
        exclude_modules = []
    
    # Dictionary to track replaced modules
    replaced_modules = {}
    
    for name, child in module.named_children():
        # Skip if this module should be excluded
        if name in exclude_modules:
            continue
            
        # Check if this module should be replaced with LoRA
        module_type = type(child).__name__
        if module_type in target_modules:
            if isinstance(child, nn.Linear):
                replaced_modules[name] = LoRALinear(child, rank, alpha, dropout)
            elif isinstance(child, nn.Conv1d):
                replaced_modules[name] = LoRAConv1d(child, rank, alpha, dropout)
            elif isinstance(child, nn.Conv2d):
                replaced_modules[name] = LoRAConv2d(child, rank, alpha, dropout)
            elif isinstance(child, nn.Conv3d):
                replaced_modules[name] = LoRAConv3d(child, rank, alpha, dropout)
        else:
            # Recursively apply to child modules
            apply_lora_to_module(child, target_modules, rank, alpha, dropout, exclude_modules)
    
    # Replace modules
    for name, lora_module in replaced_modules.items():
        setattr(module, name, lora_module)
    
    return module


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.
    """
    lora_params = []
    for module in model.modules():
        if hasattr(module, 'lora') and hasattr(module.lora, 'lora_A'):
            lora_params.extend([module.lora.lora_A, module.lora.lora_B])
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only the LoRA weights from a model.
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and hasattr(module.lora, 'lora_A'):
            lora_state_dict[f"{name}.lora.lora_A"] = module.lora.lora_A
            lora_state_dict[f"{name}.lora.lora_B"] = module.lora.lora_B
    
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights into a model.
    """
    lora_state_dict = torch.load(path)
    
    # Create a mapping from LoRA parameter names to modules
    lora_modules = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_modules[name] = module.lora
    
    # Load parameters
    for param_name, param_value in lora_state_dict.items():
        # Extract module name and parameter type
        parts = param_name.split('.')
        if len(parts) >= 3 and parts[-2] == 'lora':
            module_name = '.'.join(parts[:-2])
            param_type = parts[-1]  # 'lora_A' or 'lora_B'
            
            if module_name in lora_modules:
                if hasattr(lora_modules[module_name], param_type):
                    getattr(lora_modules[module_name], param_type).data.copy_(param_value)


def merge_lora_weights(model: nn.Module, scaling: float = 1.0):
    """
    Merge LoRA weights into the original model weights.
    This is useful for inference to avoid the overhead of LoRA computation.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and hasattr(module, 'original_layer'):
            if module.lora.rank > 0:
                # Compute LoRA weight delta
                lora_weight = (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling * scaling
                
                # Add to original weights
                if isinstance(module.original_layer, nn.Linear):
                    module.original_layer.weight.data += lora_weight
                elif isinstance(module.original_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    # Reshape LoRA weight to match conv weight shape
                    original_shape = module.original_layer.weight.shape
                    lora_weight_reshaped = lora_weight.reshape(original_shape)
                    module.original_layer.weight.data += lora_weight_reshaped
                
                # Zero out LoRA parameters after merging
                module.lora.lora_A.data.zero_()
                module.lora.lora_B.data.zero_()


def unmerge_lora_weights(model: nn.Module, scaling: float = 1.0):
    """
    Unmerge LoRA weights from the original model weights.
    This reverses the merge operation.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and hasattr(module, 'original_layer'):
            if module.lora.rank > 0:
                # Compute LoRA weight delta
                lora_weight = (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling * scaling
                
                # Subtract from original weights
                if isinstance(module.original_layer, nn.Linear):
                    module.original_layer.weight.data -= lora_weight
                elif isinstance(module.original_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    # Reshape LoRA weight to match conv weight shape
                    original_shape = module.original_layer.weight.shape
                    lora_weight_reshaped = lora_weight.reshape(original_shape)
                    module.original_layer.weight.data -= lora_weight_reshaped


def print_lora_info(model: nn.Module):
    """
    Print information about LoRA adaptations in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in get_lora_parameters(model))

    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"LoRA parameters percentage: {100 * lora_params / total_params:.2f}%")

    lora_modules = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_modules += 1
            # print(f"LoRA module: {name} (rank={module.lora.rank}, alpha={module.lora.alpha})")

    print(f"Total LoRA modules: {lora_modules}")
