import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.1
    ) -> torch.Tensor:
        """
        x: Tensor of shape (N,3,H,W) or (B,N,3,H,W), values in [-1,1], requires_grad=True
        y: Tensor of shape (N,3,H,W), values in [-1,1], no grad
        """
        # Handle both 4D and 5D input tensors
        original_shape = x.shape
        if len(original_shape) == 5:  # (B,N,3,H,W)
            B, N = original_shape[:2]
            x = x.view(B * N, *original_shape[2:])  # (B*N,3,H,W)
            # Expand y to match the flattened batch size if needed
            if len(y.shape) == 4:  # y is still (N,3,H,W)
                y = (
                    y.unsqueeze(0)
                    .expand(B, -1, -1, -1, -1)
                    .contiguous()
                    .view(B * N, *y.shape[1:])
                )

        feat_x, feat_y = self.net(x), self.net(y)
        # feature difference between generated and reference images
        diff_xy = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res_xy = [l(d).mean((2, 3), True) for d, l in zip(diff_xy, self.lin)]
        loss_xy = torch.sum(torch.cat(res_xy, 0)) / x.shape[0]
        if len(original_shape) == 4:
            return loss_xy
        # feature difference between generated images
        diff_xx = [(fx - torch.roll(fx, shifts=1, dims=0)) ** 2 for fx in feat_x]
        res_xx = [l(d).mean((2, 3), True) for d, l in zip(diff_xx, self.lin)]
        loss_xx = torch.sum(torch.cat(res_xx, 0)) / x.shape[0]
        return loss_xy - alpha * loss_xx
