from functools import partial

import numpy as np

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):        
        from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
        self.spconv_ver = 2

        self._voxel_generator = VoxelGenerator(
            vsize_xyz=vsize_xyz,
            coors_range_xyz=coors_range_xyz,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels
        )

    def generate(self, points):
        assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
        voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

# pre-process
class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['search_points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['search_points'] = points

        return data_dict

    '''
    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        search_points = data_dict['search_points']
        search_voxel_output = voxel_generator.generate(search_points)
        if isinstance(search_voxel_output, dict):
            search_voxels, search_coordinates, search_num_points = \
                search_voxel_output['voxels'], search_voxel_output['coordinates'], search_voxel_output['num_points_per_voxel']
        else:
            search_voxels, search_coordinates, search_num_points = search_voxel_output

        if not data_dict['use_lead_xyz']:
            search_voxels = search_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # print('data_process_coordinates: ',coordinates.shape)
        data_dict['search_voxels'] = search_voxels
        data_dict['search_voxel_coords'] = search_coordinates
        data_dict['search_voxel_num_points'] = search_num_points

        template_points = data_dict['template_points']
        template_voxel_output = voxel_generator.generate(template_points)
        if isinstance(template_voxel_output, dict):
            template_voxels, template_voxel_coords, template_voxel_num_points = \
                template_voxel_output['voxels'], template_voxel_output['coordinates'], template_voxel_output['num_points_per_voxel']
        else:
            template_voxels, template_voxel_coords, template_voxel_num_points = template_voxel_output

        if not data_dict['use_lead_xyz']:
            template_voxels = template_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['template_voxels'] = template_voxels # N, 5, 4
        data_dict['template_voxel_coords'] = template_voxel_coords  # N, 3
        data_dict['template_voxel_num_points'] = template_voxel_num_points  # N,
        return data_dict
    '''
    
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels,config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        search_points = data_dict['search_points']
        search_voxel_output = self.voxel_generator.generate(search_points)
        if isinstance(search_voxel_output, dict):
            search_voxels, search_coordinates, search_num_points = \
                search_voxel_output['voxels'], search_voxel_output['coordinates'], search_voxel_output['num_points_per_voxel']
        else:
            search_voxels, search_coordinates, search_num_points = search_voxel_output

        if not data_dict['use_lead_xyz']:
            search_voxels = search_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        # print('data_process_coordinates: ',coordinates.shape)
        data_dict['search_voxels'] = search_voxels
        data_dict['search_voxel_coords'] = search_coordinates
        data_dict['search_voxel_num_points'] = search_num_points

        template_points = data_dict['template_points']
        template_voxel_output = self.voxel_generator.generate(template_points)
        if isinstance(template_voxel_output, dict):
            template_voxels, template_voxel_coords, template_voxel_num_points = \
                template_voxel_output['voxels'], template_voxel_output['coordinates'], template_voxel_output['num_points_per_voxel']
        else:
            template_voxels, template_voxel_coords, template_voxel_num_points = template_voxel_output

        if not data_dict['use_lead_xyz']:
            template_voxels = template_voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['template_voxels'] = template_voxels # N, 5, 4
        data_dict['template_voxel_coords'] = template_voxel_coords  # N, 3
        data_dict['template_voxel_num_points'] = template_voxel_num_points  # N,
        return data_dict


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
