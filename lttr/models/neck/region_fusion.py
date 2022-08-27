from .region_template import RegionTemplate

class RegionFusion(RegionTemplate):
    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)

    def forward(self, batch_dict):
        sf = batch_dict['search_feature_2d']
        tf = batch_dict['template_feature_2d']

        sf, tf = self.encoder_decoder_forward(sf, tf)
    
        fusion_feature = self.simi_func(sf, tf)
        batch_dict['fusion_feature'] = fusion_feature

        return batch_dict

        