import torch.nn as nn
import numpy as np
from .center_template import CenterHeadTemplate

class CenterHead(CenterHeadTemplate):
    def __init__(self, model_cfg, input_channels, grid_size, point_cloud_range, voxel_size):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels,
                        grid_size=grid_size, point_cloud_range=point_cloud_range, voxel_size=voxel_size)

        self.cls_layers = self.make_final_layers(input_channels, self.model_cfg.CLS_FC)
        self.xy_layers = self.make_final_layers(input_channels, self.model_cfg.XY_FC)
        self.z_layers = self.make_final_layers(input_channels, self.model_cfg.Z_FC)
        self.ry_layers = self.make_final_layers(input_channels, self.model_cfg.RY_FC)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.cls_layers[-1].bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.xy_layers[-1].weight, mean=0, std=0.001)
        nn.init.normal_(self.z_layers[-1].weight, mean=0, std=0.001)
        nn.init.normal_(self.ry_layers[-1].weight, mean=0, std=0.001)

    def assign_targets(self, input_dict):
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)

        batch_size = gt_boxes.shape[0]
        local_gt = gt_boxes.clone().squeeze(1)
        real_height = local_gt.clone()[:,2]
        center_offset = input_dict['center_offset']
        local_gt[:,:3] -= center_offset
        local_gt = local_gt.unsqueeze(1)

        targets_dict = self.assign_stack_targets(local_gt, real_height)

        return targets_dict

    def forward(self, batch_dict):
        self.feature_map_stride = batch_dict['spatial_features_stride']

        fusion_feature = batch_dict['fusion_feature']
    
        cls_preds = self.cls_layers(fusion_feature)
        xy_preds = self.xy_layers(fusion_feature)
        z_preds = self.z_layers(fusion_feature)
        ry_preds = self.ry_layers(fusion_feature)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        xy_preds = xy_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        z_preds = z_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        ry_preds = ry_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        
        batch_dict['xy_preds'] = xy_preds
        batch_dict['z_preds'] = z_preds
        batch_dict['ry_preds'] = ry_preds
        batch_dict['cls_preds'] = cls_preds
        
        ret_dict = {'cls_preds': cls_preds,
                    'xy_preds': xy_preds,
                    'z_preds': z_preds,
                    'ry_preds': ry_preds,
                    }

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['cls_labels'] = targets_dict['cls_labels']
            ret_dict['reg_labels'] = targets_dict['reg_labels']
            ret_dict['reg_mask'] = targets_dict['reg_mask']
            ret_dict['ind_labels'] = targets_dict['ind_labels']
            ret_dict['object_dim'] = batch_dict['object_dim']
            self.forward_ret_dict = ret_dict
        else:
            batch_dict = self.generate_val_box(batch_dict)

        return batch_dict

        