import torch
from weaver.nn.model.ParticleTransformer import InteractionTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class InteractionTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
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
        embed_dims=[8, 16, 8],
        pair_embed_dims=[8, 8, 8],
        num_heads=2,
        num_layers=6,
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

    model = InteractionTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
