# -*- coding: utf-8 -*-
"""
LoRA-enabled UNet models for fine-tuning with Low-Rank Adaptation.
"""

import torch
import torch.nn as nn
from .unet import UNetModel, SuperResUNetModel
from .lora import apply_lora_to_module, get_lora_parameters, save_lora_weights, load_lora_weights, print_lora_info


class LoRAUNetModel(UNetModel):
    """
    UNet model with LoRA adaptation applied to specified layers.
    """
    
    def __init__(
        self,
        lora_rank: int = 4,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: list = None,
        lora_exclude_modules: list = None,
        *args,
        **kwargs
    ):
        """
        Initialize LoRA UNet model.
        
        Args:
            lora_rank: Rank of LoRA adaptation (lower = fewer parameters)
            lora_alpha: LoRA scaling parameter (higher = stronger adaptation)
            lora_dropout: Dropout rate for LoRA layers
            lora_target_modules: List of module types to apply LoRA to
            lora_exclude_modules: List of module names to exclude from LoRA
            *args, **kwargs: Arguments passed to parent UNetModel
        """
        # Initialize the parent UNet model first
        super().__init__(*args, **kwargs)
        
        # Store LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Default target modules for UNet (focus on key attention and conv layers)
        if lora_target_modules is None:
            lora_target_modules = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']
        
        # Modules to exclude from LoRA (typically output layers and embeddings)
        if lora_exclude_modules is None:
            lora_exclude_modules = ['time_embed', 'label_emb', 'out']
        
        # Apply LoRA to the model
        if lora_rank > 0:
            self._apply_lora(lora_target_modules, lora_exclude_modules)
            print(f"Applied LoRA with rank={lora_rank}, alpha={lora_alpha}")
            self.print_lora_info()
    
    def _apply_lora(self, target_modules, exclude_modules):
        """Apply LoRA adaptations to the model."""
        apply_lora_to_module(
            self,
            target_modules=target_modules,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            exclude_modules=exclude_modules
        )
    
    def get_lora_parameters(self):
        """Get all LoRA parameters."""
        return get_lora_parameters(self)
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights."""
        save_lora_weights(self, path)
        print(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        load_lora_weights(self, path)
        print(f"LoRA weights loaded from {path}")
    
    def print_lora_info(self):
        """Print information about LoRA adaptations."""
        print_lora_info(self)
    
    def enable_lora(self):
        """Enable LoRA layers for training."""
        for module in self.modules():
            if hasattr(module, 'lora'):
                for param in module.lora.parameters():
                    param.requires_grad = True
                # Ensure original layer parameters are frozen
                if hasattr(module, 'original_layer'):
                    for param in module.original_layer.parameters():
                        param.requires_grad = False
    
    def disable_lora(self):
        """Disable LoRA layers."""
        for module in self.modules():
            if hasattr(module, 'lora'):
                for param in module.lora.parameters():
                    param.requires_grad = False
    
    def merge_lora_weights(self, scaling: float = 1.0):
        """Merge LoRA weights into original weights for inference."""
        from .lora import merge_lora_weights
        merge_lora_weights(self, scaling)
        print("LoRA weights merged into original model")
    
    def unmerge_lora_weights(self, scaling: float = 1.0):
        """Unmerge LoRA weights from original weights."""
        from .lora import unmerge_lora_weights
        unmerge_lora_weights(self, scaling)
        print("LoRA weights unmerged from original model")


def create_lora_unet_model(
    lora_rank: int = 4,
    lora_alpha: float = 32,
    lora_dropout: float = 0.0,
    lora_target_modules: list = None,
    lora_exclude_modules: list = None,
    **unet_kwargs
):
    """
    Factory function to create LoRA-enabled UNet models.
    
    Args:
        lora_rank: Rank of LoRA adaptation
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout rate for LoRA layers
        lora_target_modules: List of module types to apply LoRA to
        lora_exclude_modules: List of module names to exclude from LoRA
        **unet_kwargs: Additional arguments for UNet model
    
    Returns:
        LoRA-enabled UNet model
    """
    return LoRAUNetModel(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_exclude_modules=lora_exclude_modules,
        **unet_kwargs
    )


# Utility function to convert existing UNet to LoRA UNet
def convert_unet_to_lora(
    unet_model,
    lora_rank: int = 4,
    lora_alpha: float = 32,
    lora_dropout: float = 0.0,
    lora_target_modules: list = None,
    lora_exclude_modules: list = None,
    **kwargs,
) -> LoRAUNetModel:
    """
    Convert an existing UNet model to a LoRA-enabled version.

    Args:
        unet_model: Existing UNet model to convert
        lora_rank: Rank of LoRA adaptation
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout rate for LoRA layers
        lora_target_modules: List of module types to apply LoRA to
        lora_exclude_modules: List of module names to exclude from LoRA
        **kwargs: Model configuration (already contains correct configs)

    Returns:
        LoRA-enabled UNet model
    """
    # Create a new LoRA UNet model with the provided configuration
    lora_unet = LoRAUNetModel(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_exclude_modules=lora_exclude_modules,
        **kwargs,  # Use provided configuration
    )

    # Copy the original weights to the new LoRA model
    original_state_dict = unet_model.state_dict()
    lora_state_dict = lora_unet.state_dict()

    # Create mapping for weight transfer
    transferred_keys = []
    missing_keys = []

    for name, param in original_state_dict.items():
        if name in lora_state_dict:
            # Direct match - copy the parameter
            lora_state_dict[name].copy_(param)
            transferred_keys.append(name)
        else:
            # Try to find in wrapped LoRA layer
            # LoRA wrapping changes 'layer.weight' to 'layer.original_layer.weight'
            potential_lora_names = [
                f"{name.rsplit('.', 1)[0]}.original_layer.{name.rsplit('.', 1)[1]}",
                f"{name.replace('.weight', '.original_layer.weight')}",
                f"{name.replace('.bias', '.original_layer.bias')}",
            ]

            found = False
            for lora_name in potential_lora_names:
                if lora_name in lora_state_dict:
                    lora_state_dict[lora_name].copy_(param)
                    transferred_keys.append(f"{name} -> {lora_name}")
                    found = True
                    break

            if not found:
                missing_keys.append(name)

    print(f"✓ Transferred {len(transferred_keys)} parameter groups to LoRA model")
    if missing_keys:
        print(
            f"⚠ Could not transfer {len(missing_keys)} keys: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}"
        )

    # Verify the conversion worked by comparing a few key parameters
    print("Verifying LoRA weight transfer...")
    verification_passed = True
    test_keys = list(original_state_dict.keys())[:3]  # Test first 3 parameters

    for name in test_keys:
        original_param = original_state_dict[name]

        if name in lora_state_dict:
            lora_param = lora_state_dict[name]
        else:
            # Check in original_layer
            potential_names = [
                f"{name.rsplit('.', 1)[0]}.original_layer.{name.rsplit('.', 1)[1]}",
                f"{name.replace('.weight', '.original_layer.weight')}",
                f"{name.replace('.bias', '.original_layer.bias')}",
            ]
            lora_param = None
            for lora_name in potential_names:
                if lora_name in lora_state_dict:
                    lora_param = lora_state_dict[lora_name]
                    break

        if lora_param is not None:
            # Ensure both tensors are on the same device for comparison
            original_param_cpu = original_param.cpu() if original_param.is_cuda else original_param
            lora_param_cpu = lora_param.cpu() if lora_param.is_cuda else lora_param
            
            if not torch.allclose(original_param_cpu, lora_param_cpu, rtol=1e-6, atol=1e-6):
                print(f"⚠ Parameter {name} differs after transfer!")
                verification_passed = False
        else:
            print(f"⚠ Could not find transferred parameter for {name}")
            verification_passed = False

    if verification_passed:
        print("✓ LoRA weight transfer verification passed")
    else:
        print("⚠ LoRA weight transfer verification failed - check parameter mapping")

    return lora_unet
