from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder

class SOTDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, eval_flag=False, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.eval_flag = eval_flag
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.offset_xy = self.dataset_cfg.get('OFFSET_XY', 4)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        self.data_augmentor = None
        if self.dataset_cfg.get('DATA_AUGMENTOR', None) is not None:
            self.data_augmentor = DataAugmentor(
                self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
            ) if self.training else None
            
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, 
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        if not self.eval_flag:
            return 'train' if self.training else 'val'
        else:
            return 'test'


    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        if self.training and self.data_augmentor is not None:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            center_offset = data_dict['center_offset']
            data_dict['gt_boxes'][:3] -= center_offset.reshape(-1)

            data_dict['gt_boxes'] = data_dict['gt_boxes'].reshape(1,-1)
            data_dict = self.data_augmentor.forward(data_dict)
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            data_dict['gt_boxes'][:,:3] += data_dict['center_offset']
            data_dict['gt_boxes'] = data_dict['gt_boxes'].reshape(-1)
        
        if data_dict.get('gt_boxes', None) is not None:
            gt_classes = np.array([self.class_names.index(data_dict['gt_names']) + 1], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'].reshape(1,7), gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['search_voxels', 'object_dim', 'search_voxel_num_points', 'template_voxels', 'template_voxel_num_points', 'center_offset', 'first_template_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['search_points', 'template_points', 'search_voxel_coords', 'template_voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes','local_gt']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['search_range']:
                    ret[key] = np.stack(val[0], axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
