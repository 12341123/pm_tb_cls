from typing import Dict, Any, Union, Tuple, NewType
import logging
import sys

import numpy as np
import cv2

from fvcore.transforms.transform import Transform, NoOpTransform
from detectron2.data.transforms import Augmentation
from detectron2.data.transforms import RandomRotation as _RandomRotation
from detectron2.data.transforms import RotationTransform as _RotationTransform

logger = logging.getLogger(__name__)


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:

        class _TransformToAug(Augmentation):
            def __init__(self, tfm: Transform):
                self.tfm = tfm

            def get_transform(self, *args):
                return self.tfm

            def __repr__(self):
                return repr(self.tfm)

            __str__ = __repr__

        return _TransformToAug(tfm_or_aug)


class RescaleTransform(Transform):
    def __init__(self, scale, rescale_mask=False):
        super(RescaleTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img * float(self.scale)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if self.rescale_mask:
            return self.apply_image(segmentation)
        else:
            return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class Rescale(Augmentation):
    def __init__(self, scale: float, rescale_mask: bool = False):
        self._init(locals())

    def get_transform(self, image) -> Transform:
        return RescaleTransform(self.scale, self.rescale_mask)


class GammaTransform(Transform):
    def __init__(self, gamma: float):
        super(GammaTransform, self).__init__()
        self._set_attributes(locals())

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            lookup_table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** self.gamma) * 255
            lookup_table = lookup_table.astype(np.uint8)
            image = cv2.LUT(image, lookup_table)
        else:
            raise NotImplementedError

        return image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class RandomGamma(Augmentation):
    def __init__(self, gamma_min, gamma_max):
        super(RandomGamma, self).__init__()
        self._init(locals())

    def get_transform(self, image) -> Transform:
        gamma = np.random.uniform(self.gamma_min, self.gamma_max)
        return GammaTransform(gamma)


class RandomApplyOneOf(Augmentation):
    def __init__(self, tfms_or_augs, prob=0.5):
        super(RandomApplyOneOf, self).__init__()
        self.augs = [_transform_to_aug(tfm_or_aug) for tfm_or_aug in tfms_or_augs]
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args) -> Transform:
        do = self._rand_range() <= self.prob
        if do:
            aug = np.random.choice(self.augs)
            return aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input) -> Transform:
        do = self._rand_range() <= self.prob
        if do:
            aug = np.random.choice(self.augs)
            return aug(aug_input)
        else:
            return NoOpTransform()


class ResizeTransform(Transform):
    def __init__(self, h, w, new_h, new_w, interp=None):
        super(ResizeTransform, self).__init__()
        if interp is None:
            interp = cv2.INTER_LINEAR
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp=None):
        """
        Args:
            img (np.ndarray): of size HxW or HxWxC
            interp: cv2 interpolation methods.
        """
        # assert img.shape[:2] == (self.h, self.w), f"img shape {img.shape[:2]} not match {(self.h, self.w)}"
        assert len(img.shape) <= 3

        interpolation = interp if interp is not None else self.interp
        return cv2.resize(img, (self.new_w, self.new_h), interpolation=interpolation)

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class Resize(Augmentation):
    """ Resize image to a fixed target size"""

    def __init__(self, shape, interp=cv2.INTER_LINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
            self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=cv2.INTER_LINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomRotation(_RandomRotation):
    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None, rotate_mask=True):
        super(RandomRotation, self).__init__(angle, expand, center, sample_style, interp)
        self.rotate_mask = rotate_mask

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp, rotate_mask=self.rotate_mask)


class RotationTransform(_RotationTransform):
    def __init__(self, h, w, angle, expand=True, center=None, interp=None, rotate_mask=True):
        super(RotationTransform, self).__init__(h, w, angle, expand, center, interp)
        self.rotate_mask = rotate_mask

    def apply_segmentation(self, segmentation):
        if self.rotate_mask:
            segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation
