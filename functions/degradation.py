"""Registry of forward operators (super-resolution, motion deblur, inpainting, ...)."""
import numpy as np
import torch
from munch import Munch

import functions.svd_operators as svd_op
from functions import measurements
from utils.inpaint_util import MaskGenerator

__DEGRADATION__ = {}

def register_degradation(name: str):
    def wrapper(fn):
        if __DEGRADATION__.get(name) is not None:
            raise NameError(f'DEGRADATION {name} is already registered')
        __DEGRADATION__[name]=fn
        return fn
    return wrapper

def get_degradation(name: str,
                    deg_config: Munch,
                    device:torch.device):
    if __DEGRADATION__.get(name) is None:
        raise NameError(f'DEGRADATION {name} does not exist.')
    return __DEGRADATION__[name](deg_config, device)

@register_degradation(name='cs_walshhadamard')
def deg_cs_walshhadamard(deg_config, device):
    compressed_size = round(1/deg_config.deg_scale)
    A_funcs = svd_op.WalshHadamardCS(deg_config.channels,
                                     deg_config.image_size,
                                     compressed_size,
                                     torch.randperm(deg_config.image_size**2),
                                     device)
    return A_funcs

@register_degradation(name='cs_blockbased')
def deg_cs_blockbased(deg_config, device):
    cs_ratio = deg_config.deg_scale
    A_funcs = svd_op.CS(deg_config.channels,
                        deg_config.image_size,
                        cs_ratio,
                        device)
    return A_funcs

@register_degradation(name='inpainting')
def deg_inpainting(deg_config, device):
    loaded = np.load("exp/inp_masks/mask_768_half.npy")
    mask = torch.from_numpy(loaded).to(device).reshape(-1)
    missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    A_funcs = svd_op.Inpainting(deg_config.channels,
                                deg_config.image_size,
                                missing,
                                device)
    return A_funcs

@register_degradation(name='denoising')
def deg_denoise(deg_config, device):
    A_funcs = svd_op.Denoising(deg_config.channels,
                               deg_config.image_size,
                               device)
    return A_funcs

@register_degradation(name='colorization')
def deg_colorization(deg_config, device):
    A_funcs = svd_op.Colorization(deg_config.image_size,
                                  device)
    return A_funcs


@register_degradation(name='sr_avgpool')
def deg_sr_avgpool(deg_config, device):
    blur_by = int(deg_config.deg_scale)
    A_funcs = svd_op.SuperResolution(deg_config.channels,
                                     deg_config.image_size,
                                     blur_by,
                                     device)
    return A_funcs

@register_degradation(name='sr_bicubic')
def deg_sr_bicubic(deg_config, device):
    def bicubic_kernel(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
        elif 1 < abs(x) and abs(x) < 2:
            return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
        else:
            return 0

    factor = int(deg_config.deg_scale)
    k = np.zeros((factor * 4))
    for i in range(factor * 4):
        x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    kernel = torch.from_numpy(k).float().to(device)
    A_funcs = svd_op.SRConv(kernel / kernel.sum(),
                            deg_config.channels,
                            deg_config.image_size,
                            device,
                            stride=factor)
    return A_funcs

@register_degradation(name='deblur_uni')
def deg_deblur_uni(deg_config, device):
    A_funcs = svd_op.Deblurring(torch.tensor([1/deg_config.deg_scale]*deg_config.deg_scale).to(device),
                                deg_config.channels,
                                deg_config.image_size,
                                device)
    return A_funcs

@register_degradation(name='deblur_gauss')
def deg_deblur_gauss(deg_config, device):
    sigma = 3.0
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    size = deg_config.deg_scale
    ker = []
    for k in range(-size//2, size//2):
        ker.append(pdf(k))
    kernel = torch.Tensor(ker).to(device)
    A_funcs = svd_op.Deblurring(kernel / kernel.sum(),
                                deg_config.channels,
                                deg_config.image_size,
                                device)
    return A_funcs

@register_degradation(name='deblur_aniso')
def deg_deblur_aniso(deg_config, device):
    sigma = 20
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

    sigma = 1
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

    A_funcs = svd_op.Deblurring2D(kernel1 / kernel1.sum(),
                                  kernel2 / kernel2.sum(),
                                  deg_config.channels,
                                  deg_config.image_size,
                                  device)
    return A_funcs

@register_degradation(name='deblur_motion')
def deg_deblur_motion(deg_config, device):
    A_funcs = measurements.MotionBlurOperator(
        kernel_size=deg_config.deg_scale,
        intensity=0.5,
        device=device
    )
    return A_funcs

@register_degradation(name='deblur_nonuniform')
def deg_deblur_motion(deg_config, device, kernels=None, masks=None):
    A_funcs = measurements.NonuniformBlurOperator(
        deg_config.image_size,
        deg_config.deg_scale,
        device,
        kernels=kernels,
        masks=masks,
    )
    return A_funcs


# ======= FOR arbitraty image size =======
@register_degradation(name='sr_avgpool_gen')
def deg_sr_avgpool_general(deg_config, device):
    blur_by = int(deg_config.deg_scale)
    A_funcs = svd_op.SuperResolutionGeneral(deg_config.channels,
                                            deg_config.imgH,
                                            deg_config.imgW,
                                            blur_by,
                                            device)
    return A_funcs

@register_degradation(name='deblur_gauss_gen')
def deg_deblur_guass_general(deg_config, device):
    A_funcs = measurements.GaussialBlurOperator(
        kernel_size=deg_config.deg_scale,
        intensity=3.0,
        device=device
    )
    return A_funcs


class BoxInpaintingOperator:
    """Box inpainting operator matching FlowChef implementation.

    Masks out a square box region in the center of the image.
    """
    def __init__(self, image_size, box_size, device):
        """Initialize box inpainting operator.

        Args:
            image_size (int): Size of the image (assumes square)
            box_size (int): Size of square box to mask out
            device: torch device
        """
        self.image_size = image_size
        self.box_size = box_size
        self.device = device
        self.name = "inpainting"

        # Precompute mask (1 = keep, 0 = mask out)
        self._mask = None

    def _get_mask(self, shape):
        """Get or create the binary mask."""
        B, C, H, W = shape

        # Center of image
        center_h, center_w = H // 2, W // 2

        # Box boundaries
        h_min = center_h - self.box_size // 2
        h_max = h_min + self.box_size
        w_min = center_w - self.box_size // 2
        w_max = w_min + self.box_size

        # Create mask (1 everywhere except box region)
        mask = torch.ones(shape, device=self.device)
        mask[:, :, h_min:h_max, w_min:w_max] = 0

        return mask

    def A(self, x):
        """Apply box mask to image (forward operator).

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Masked image with box region set to 0
        """
        mask = self._get_mask(x.shape)
        return x * mask

    def At(self, x):
        """Transpose operator (same as forward for inpainting)."""
        return self.A(x)

    def A_pinv(self, x):
        """Pseudo-inverse (same as forward for inpainting)."""
        return self.A(x)

    def get_mask(self, shape):
        """Get the binary mask.

        Args:
            shape (tuple): Shape of input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Binary mask (1 = keep, 0 = mask out)
        """
        return self._get_mask(shape)


@register_degradation(name='box_inpainting')
def deg_box_inpainting(deg_config, device):
    """Create box inpainting operator.

    Config should have:
        - image_size: size of image
        - deg_scale: size of box to mask out
    """
    A_funcs = BoxInpaintingOperator(
        image_size=deg_config.image_size,
        box_size=deg_config.deg_scale,
        device=device
    )
    return A_funcs
