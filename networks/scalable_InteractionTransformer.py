import torch
import numpy as np
from weaver.nn.model.ParticleTransformer import InteractionTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''

def process_scaling_args(kwargs):
    total_num_layers_default = kwargs['num_layers'] + kwargs['num_cls_layers']

    total_num_layers = kwargs.pop('total_num_layers', total_num_layers_default)
    num_cls_layers_mult = kwargs.pop('num_cls_layers_mult', kwargs['num_cls_layers'] / total_num_layers_default)
    embedding_scale_mult = kwargs.pop('embedding_scale_mult', 1)
    pair_embedding_scale_mult = kwargs.pop('pair_embedding_scale_mult', 1)
    num_neurons_per_head = kwargs.pop('num_neurons_per_head', kwargs['embed_dims'][-1] // kwargs['num_heads'])

    kwargs['num_cls_layers'] = int(np.ceil(num_cls_layers_mult * total_num_layers - 1e-18))
    
    kwargs['num_layers'] = total_num_layers - kwargs['num_cls_layers']

    print(f"{'Neurons per head:':30} {num_neurons_per_head}")

    kwargs['embed_dims'] = list(map(
        lambda x: int(np.round(x * embedding_scale_mult)),
        kwargs['embed_dims']
    ))

    # neurons per head
    if kwargs['embed_dims'][-1] < num_neurons_per_head:
        kwargs['num_heads'] = 1
    else:
        kwargs['num_heads'] = int(np.round(kwargs['embed_dims'][-1] / num_neurons_per_head))

    kwargs['pair_embed_dims'] = list(map(
        lambda x: int(np.round(x * pair_embedding_scale_mult)),
        kwargs['pair_embed_dims']
    )) if kwargs['pair_embed_dims'] is not None else None

    print("Scaled parameters:")
    for k in [
        'num_cls_layers', 'num_layers', 'embed_dims', 'num_heads', 'pair_embed_dims'
    ]:
        print(f"{k:30}: {kwargs[k]}")

class ScalableInteractionTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        process_scaling_args(kwargs)

        self.mod = InteractionTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_seq_len=128,
        interactions_dim=4,
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128], # 8 = 1/16 = 0.0625
        pair_embed_dims=[64, 64, 64], # 8 = 1/8 = 0.125
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        attention='linformer',
        lin_proj_dim=4,
        # misc
        trim=True,
        for_inference=False
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ScalableInteractionTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
