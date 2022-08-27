import torch
import torch.nn as nn
from ....utils import common_utils

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        template_pillar_features, template_voxel_coords = batch_dict['template_voxel_features'], batch_dict['template_voxel_coords']
        search_pillar_features, search_voxel_coords = batch_dict['search_voxel_features'], batch_dict['search_voxel_coords']
        tfeature = self.forward_feature(template_pillar_features, template_voxel_coords)
        batch_dict['t_spatial_features'] = tfeature
        
        sfeature = self.forward_feature(search_pillar_features, search_voxel_coords)
        batch_dict['x_spatial_features'] = sfeature
        batch_dict['spatial_features_stride'] = 1
        return batch_dict

    def forward_feature(self, pillar_features, coords):
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        # point_coords = common_utils.get_voxel_centers(
        #     coords[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # point_coords = torch.cat((coords[:, 0:1].float(), point_coords), dim=1)

        return batch_spatial_features
