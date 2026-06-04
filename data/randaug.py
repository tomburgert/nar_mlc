import cv2
import torch
import numpy as np

import albumentations as A

from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
from typing import List, Tuple, Optional, Dict


class RandAugment(torch.nn.Module):
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        op_names: List[str] = ['all'],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.op_names = op_names
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        augment_space = {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if self.op_names == ['all']:
            pass
        else:
            augment_space = {key: augment_space[key] for key in self.op_names}
        return augment_space

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            # img has shape (H, W, C)
            op_meta = self._augmentation_space(self.num_magnitude_bins, img.shape[1:])
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


def _apply_op(
    img: Tensor, op_name: str, magnitude: float,
    interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        transform = A.Affine(shear={'x': magnitude}, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "ShearY":
        transform = A.Affine(shear={'y': magnitude}, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "TranslateX":
        transform = A.Affine(translate_px={'x': int(magnitude)}, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "TranslateY":
        transform = A.Affine(translate_px={'y': int(magnitude)}, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Rotate":
        transform = A.Affine(rotate=magnitude, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Brightness":
        transform = A.RandomBrightness(limit=magnitude, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Contrast":
        transform = A.RandomContrast(limit=magnitude, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Sharpness":
        if np.sign(magnitude) == np.sign(-1):
            transform = A.UnsharpMask(alpha=(-1 * magnitude), p=1.0)
        else:
            transform = A.Sharpen(alpha=magnitude, lightness=0.75, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Posterize":
        transform = A.Posterize(num_bits=int(magnitude), p=1.0) 
        img = transform(image=img)['image']
    elif op_name == "Solarize":
        transform = A.Solarize(threshold=magnitude, p=1.0) 
        img = transform(image=img)['image']
    elif op_name == "AutoContrast":
        transform = A.Lambda(image=custom_autocontrast, p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Equalize":
        transform = A.Equalize(p=1.0)
        img = transform(image=img)['image']
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


def custom_autocontrast(img, **kwargs):
    h = cv2.calcHist([img], [0], None, [256], (0, 256)).ravel()

    for lo in range(256):
        if h[lo]:
            break
    for hi in range(255, -1, -1):
        if h[hi]:
            break

    if hi > lo:
        lut = np.zeros(256, dtype=np.uint8)
        scale_coef = 255.0 / (hi - lo)
        offset = -lo * scale_coef
        for ix in range(256):
            lut[ix] = int(np.clip(ix * scale_coef + offset, 0, 255))

        img = cv2.LUT(img, lut)

    return img
