import torch
import torch.nn.functional as F
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger
from .focal_loss import get_loss_focal
from .example_ParticleTransformer import ParticleTransformerWrapper
from .example_ParticleTransformer import get_model as get_model_ParT

def get_model(data_config, **kwargs):
    return get_model_ParT(data_config, **kwargs)

def get_loss(data_config, **kwargs): 
    return get_loss_focal(data_config, **kwargs)  # torch.nn.CrossEntropyLoss()
