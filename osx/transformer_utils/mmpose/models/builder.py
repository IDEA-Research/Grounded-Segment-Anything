# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn import build_model_from_cfg
from mmcv.utils import Registry, build_from_cfg

MODELS = Registry(
    'models', build_func=build_model_from_cfg, parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
POSENETS = MODELS
MESH_MODELS = MODELS
TRANSFORMER = Registry('Transformer')


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_posenet(cfg):
    """Build posenet."""
    return POSENETS.build(cfg)


def build_mesh_model(cfg):
    """Build mesh model."""
    return MESH_MODELS.build(cfg)

def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)