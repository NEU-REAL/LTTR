from .baseline_template import Baseline3DTemplate
from .baseline import Baseline
__all__ = {
    'Baseline3DTemplate': Baseline3DTemplate,
    'Baseline': Baseline,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
