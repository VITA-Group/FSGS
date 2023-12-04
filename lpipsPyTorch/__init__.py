import torch
# import lpips
from .modules.lpips import LPIPS

# lpips_vgg = lpips.LPIPS(net="vgg").cuda()


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          mask = None,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    if mask is not None:
        x = x * mask + (1 - mask)
        y = y * mask + (1 - mask)
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    loss = criterion(x, y).mean().double()
    return loss


# def lpips(x: torch.Tensor,
#           y: torch.Tensor,
#           net_type: str = 'alex',
#           version: str = '0.1'):
#     score = lpips_vgg(x, y)
#     return float(score.item())