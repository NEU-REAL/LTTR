import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        template_voxel_features, template_voxel_num_points = batch_dict['template_voxels'], batch_dict['template_voxel_num_points']
        template_points_mean = template_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        template_normalizer = torch.clamp_min(template_voxel_num_points.view(-1, 1), min=1.0).type_as(template_voxel_features)

        template_points_mean = template_points_mean / template_normalizer
        batch_dict['template_voxel_features'] = template_points_mean.contiguous()

        search_voxel_features, search_voxel_num_points = batch_dict['search_voxels'], batch_dict['search_voxel_num_points']
        search_points_mean = search_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        search_normalizer = torch.clamp_min(search_voxel_num_points.view(-1, 1), min=1.0).type_as(search_voxel_features)
        search_points_mean = search_points_mean / search_normalizer
        batch_dict['search_voxel_features'] = search_points_mean.contiguous()

        return batch_dict
