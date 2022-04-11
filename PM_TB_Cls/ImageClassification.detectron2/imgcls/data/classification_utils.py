import detectron2.data.transforms as T
import logging
from .transform import Rescale


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    logger = logging.getLogger("detectron2.data.classification_utils")
    tfm_gens = []
    rescale = Rescale(scale=1 / 255.0, )
    # tfm_gens.append(T.Resize((input_size, input_size)))
    if is_train:
        #torchvision model need rescale
        # tfm_gens.append(rescale)
        # tfm_gens.append(T.RandomFlip())
        tfm_gens.append(T.RandomContrast(0.5, 1.5))
        tfm_gens.append(T.RandomBrightness(0.5, 1.5))
        tfm_gens.append(T.RandomSaturation(0.5, 1.5))
        logger.info(
            "TransformGens used in training[Updated]: " + str(tfm_gens))
    else:
        tfm_gens.append(T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST))
        logger.info(
            "TransformGens used in validation[Updated]: " + str(tfm_gens))
    return tfm_gens
