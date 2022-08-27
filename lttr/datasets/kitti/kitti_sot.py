import os
import numpy as np
import copy
import pandas as pd
from pathlib import Path
from ..sotdataset import SOTDatasetTemplate

from ...utils import box_utils, calibration_kitti, tracklet3d_kitti

class KittiSOT(SOTDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, eval_flag=False, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, eval_flag=eval_flag, root_path=root_path, logger=logger
        )

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.velodyne_path = os.path.join(self.root_path, 'velodyne')
        self.label_path = os.path.join(self.root_path, 'label_02')
        self.calib_path = os.path.join(self.root_path, 'calib')

        self.search_dim = [self.point_cloud_range[x::3][1] - self.point_cloud_range[x::3][0] for x in range(3)]

        self.generate_split_list(self.split)

        self.refer_box = None
        self.first_points = None
        self.sequence_points = None

        if self.mode == 'train' or self.mode == 'val':
            self.search_func = self.train_val_search_func
        else:
            self.search_func = self.test_search_func

    def generate_split_list(self, split):
        print('mode: ',self.mode)

        if split == 'train':
            self.sequence = list(range(0,17))
        elif split == 'val':
            self.sequence = list(range(17, 19))
        elif split == 'test':
            self.sequence = list(range(19, 21))

        else:
            self.sequence = list(range(21))

        list_of_sequences = [
            path for path in os.listdir(self.velodyne_path)
            if os.path.isdir(os.path.join(self.velodyne_path, path)) and int(path) in self.sequence
        ]

        list_of_tracklet_anno = []
        self.first_frame_index = [0]
        number = 0
        for sequence in list_of_sequences:
            sequence_label_name = sequence + '.txt'
            label_file = os.path.join(self.label_path, sequence_label_name)

            seq_label = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occlusion",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "ry"
                ])
            seq_label = seq_label[seq_label["type"] == self.class_names[0]]
            ########################
            # KITTI tracking dataset BUG
            if sequence == '0001':
                seq_label = seq_label[(~seq_label['frame'].isin([177,178,179,180]))]
            ########################

            seq_label.insert(loc=0, column="sequence", value=sequence)
            for track_id in seq_label.track_id.unique():
                seq_tracklet = seq_label[seq_label["track_id"] == track_id]
                seq_tracklet = seq_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in seq_tracklet.iterrows()]   
                list_of_tracklet_anno.append(tracklet_anno)
                number += len(tracklet_anno)
                self.first_frame_index.append(number)

        # every tracklet
        self.one_track_infos = self.get_whole_relative_frame(list_of_tracklet_anno)
        self.first_frame_index[-1] -= 1

    def get_whole_relative_frame(self, tracklets_infos):
        all_infos = []
        for one_infos in tracklets_infos:
            track_length = len(one_infos)
            relative_frame = 1
            for frame_info in one_infos:
                frame_info["relative_frame"] = relative_frame
                frame_info["track_length"] = track_length
                relative_frame += 1
                all_infos.append(frame_info)

        return all_infos

    def get_lidar(self, sequence, frame):
        lidar_file = os.path.join(self.velodyne_path, sequence, '{:06}.bin'.format(frame))
        assert Path(lidar_file).exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx):
        return self.one_track_infos[idx]

    def get_calib(self, sequence):
        calib_file = os.path.join(self.calib_path,'{}.txt'.format(sequence))
        assert Path(calib_file).exists()
        return calibration_kitti.Calibration(calib_file)

    def rotat_point(self, points, ry):
        R_M = np.array([[np.cos(ry), -np.sin(ry), 0],
                                [np.sin(ry), np.cos(ry), 0],
                                [0, 0,  1]])  # (3, 3)
        rotated_point = np.matmul(points[:,:3], R_M)  # (N, 3)
        return rotated_point

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return (self.first_frame_index[-1]+1) * self.total_epochs

        return self.first_frame_index[-1]+1

    def __getitem__(self, index):
        return self.get_tracking_item(index)
  
    def find_template_idx(self, index, intervel=5):
        if self.mode == 'train' or self.mode == 'val':
            search_anno = self.one_track_infos[index]
            search_relative_frame = search_anno['relative_frame']
            search_whole_length = search_anno['track_length']
            search_min_index = max(0, search_relative_frame-intervel)
            search_max_index = min(search_relative_frame+intervel, search_whole_length)
            template_relative_frame = np.random.randint(search_min_index, search_max_index)
            template_index = index + template_relative_frame - search_relative_frame
        else:
            template_index = index - 1

        return template_index

    def template_preprocess_lidar(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        cx, cy, cz, dx, dy, dz, ry = label
        label_center = np.stack((cx, cy, cz))

        corner = box_utils.boxes_to_corners_3d(label.reshape(1, 7))[0]

        flag = box_utils.in_hull(point[:,:3], corner)
        crop_points = point[flag]
        new_label = copy.deepcopy(label)
        crop_points[:,:3] -= label_center
        new_label[:3] -=label_center

        return crop_points, new_label

    def search_preprocess_lidar(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        cx, cy, cz, dx, dy, dz, ry = label
        corner = box_utils.boxes_to_corners_3d(label.reshape(1, 7))[0]
        object_flag = box_utils.in_hull(point[:,:3], corner)
        object_points = point[object_flag]
        if object_points.shape[0] != 0:
            offset_xy = np.random.uniform(low=-sample_dx/self.offset_xy, high=sample_dx/self.offset_xy, size=2)
            offset_x = cx + offset_xy[0]
            offset_y = cy + offset_xy[1]
            sample_center = np.stack((offset_x, offset_y, 0))

            search_area = np.stack((offset_x, offset_y, 0, sample_dx, sample_dy, sample_dz, 0))

            # get points in search area
            search_corner = box_utils.boxes_to_corners_3d(search_area.reshape(1, 7))[0]

            flag = box_utils.in_hull(point[:,:3], search_corner)
            crop_points = point[flag]
            crop_points[:,:3] -= sample_center
            return crop_points, sample_center
        else:
            offset_xy = np.random.uniform(low=-sample_dx/2, high=sample_dx/2, size=2)
            offset_x = cx + offset_xy[0]
            offset_y = cy + offset_xy[1]

            sample_center = np.stack((offset_x, offset_y, 0))
            search_area = np.stack((offset_x, offset_y, 0, sample_dx, sample_dy, sample_dz, 0))
            search_corner = box_utils.boxes_to_corners_3d(search_area.reshape(1, 7))[0]
            
            flag = box_utils.in_hull(point[:,:3], search_corner)
            crop_points = point[flag]
            crop_points[:,:3] -= sample_center
            return crop_points, sample_center

    def test_get_search_area(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        cx, cy, cz, dx, dy, dz, ry = label
        sample_center = np.stack((cx, cy, 0))

        area = np.stack((cx, cy, 0, sample_dx, sample_dy, sample_dz, 0)).reshape(1, 7)
        corner = box_utils.boxes_to_corners_3d(area)[0]
        object_flag = box_utils.in_hull(point[:,:3], corner)
        area_points = point[object_flag]      
        area_points[:,:3] -= sample_center
        return area_points, sample_center

    def get_tracking_item(self, index):
        search_info = self.get_label(index)
        gt_name = search_info['type']
        sequence = search_info['sequence']
        frame = search_info['frame']
        calib = self.get_calib(sequence)

        search_points = self.get_lidar(sequence, frame)
        gt_boxes_camera = tracklet3d_kitti.Tracklet3d(search_info).get_box3d().astype(np.float32)  
        gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera.reshape(-1,7), calib)[0]

        template_index = self.find_template_idx(index)
        template_info = self.get_label(template_index)
        sequence = template_info['sequence']
        frame = template_info['frame']
        template_points = self.get_lidar(sequence, frame)
        calib = self.get_calib(sequence)
        template_gt_box_camera = tracklet3d_kitti.Tracklet3d(template_info).get_box3d().astype(np.float32) 
        template_gt_box_lidar = box_utils.boxes3d_kitti_camera_to_lidar(template_gt_box_camera.reshape(-1,7), calib)[0]
        
        local_search_points, center_offset, local_template_points, local_template_label = self.search_func(gt_name, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points)

        norm_tpoints = self.rotat_point(local_template_points, local_template_label[6])
        point_r = local_template_points[:,3].reshape(-1,1)
        norm_tpoints = np.hstack((norm_tpoints, point_r))

        if self.mode == 'train' or self.mode == 'val':
            if norm_tpoints.shape[0] <= 20 or local_search_points.shape[0] <= 20:
                return self.get_tracking_item(np.random.randint(0, self.__len__()))

        norm_tlabel = copy.deepcopy(local_template_label)
        norm_tlabel_xyz = self.rotat_point(local_template_label.reshape(-1,7), local_template_label[6]).reshape(-1)
        norm_tlabel[:3] = norm_tlabel_xyz
        norm_tlabel[6] = 0

        input_dict = {
            'gt_names': gt_name,
            'gt_boxes': gt_boxes_lidar,
            'search_points': local_search_points,
            'center_offset': center_offset.reshape(1,-1),
            'object_dim': norm_tlabel[3:6].reshape(1, 3),
            'template_box': norm_tlabel.reshape(1, 7),
            'template_gt_box': template_gt_box_lidar.reshape(1, 7),
        }

        if self.mode == 'test':
            input_dict.update({
                'first_template_points': norm_tpoints,
            })

        if (self.first_points is not None) and (self.mode == 'test'):
            # first & previous
            norm_tpoints = np.vstack((self.first_points, norm_tpoints))

        input_dict.update({
            'template_points': norm_tpoints,
        })
        
        data_dict = self.prepare_data(data_dict=input_dict)
        if self.mode == 'train' or self.mode == 'val':
            tv = data_dict['template_voxels']
            sv = data_dict['search_voxels']
            if sv.shape[0] <= 20 or tv.shape[0] <= 20:
                return self.get_tracking_item(np.random.randint(0, self.__len__()))
        
        return data_dict

    def train_val_search_func(self, gt_name, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points):
        local_search_points, center_offset = self.search_preprocess_lidar(search_points, gt_boxes_lidar, self.search_dim[0], self.search_dim[1], self.search_dim[2])
        local_template_points, local_template_label = self.template_preprocess_lidar(template_points, template_gt_box_lidar, self.search_dim[0], self.search_dim[1], self.search_dim[2])

        return local_search_points, center_offset, local_template_points, local_template_label

    def test_search_func(self, gt_name, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points):
        if self.refer_box is not None:
            search_box = self.refer_box
            template_box = self.refer_box
        else:
            search_box = template_gt_box_lidar
            template_box = template_gt_box_lidar
        local_search_points, center_offset = self.test_get_search_area(search_points, search_box, self.search_dim[0], self.search_dim[1], self.search_dim[2])
        local_template_points, local_template_label = self.template_preprocess_lidar(template_points, template_box, self.search_dim[0], self.search_dim[1], self.search_dim[2])

        return local_search_points, center_offset, local_template_points, local_template_label

    def set_refer_box(self, refer_box):
        self.refer_box = refer_box

    def set_first_points(self, points):
        self.first_points = points
    
    def reset_all(self):
        self.refer_box = None
        self.first_points = None
        self.sequence_points = None