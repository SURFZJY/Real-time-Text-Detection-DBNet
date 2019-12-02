# -*- coding: utf-8 -*-

from .model import Model
from .loss import DBLoss


def get_model(config):
    model_config = config['arch']['args']
    return Model(model_config)

def get_loss(config):
    alpha = config['loss']['args']['alpha']
    beta = config['loss']['args']['beta']
    ohem_ratio = config['loss']['args']['ohem_ratio']
    return DBLoss(alpha=alpha, beta=beta, ohem_ratio=ohem_ratio)
