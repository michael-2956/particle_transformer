import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer, ParticleTransformerMultipleRuns
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, use_multiple=False, **kwargs) -> None:
        super().__init__()
        self.use_multiple = use_multiple
        if use_multiple:
            self.mod = ParticleTransformerMultipleRuns(**kwargs)
        else:
            self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def load_state_dict(self, state_dict, strict=True):
        if self.use_multiple:
            if not any(k.startswith("mod.pt.") for k in state_dict):
                state_dict = {f"mod.pt.{k[4:]}": v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_runs=128,
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        use_pre_activation_pair=False,
        embed_dims=[8, 16, 8],  # [128, 512, 128]
        pair_embed_dims=None,  # [8, 8, 8],  # [64, 64, 64],
        num_heads=2,
        num_layers=3,
        num_cls_layers=1,
        block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        multiple_pair_embed=False,
        activation='gelu',
        trim=True,
        trim_mode="fixed_shuffle_always",
        trim_mode_fixed_length=8,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(
        use_multiple=True,
        **cfg
    )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
