import pickle
import time

import torch
import tqdm
from lttr.models import load_data_to_gpu
from lttr.ops.iou3d_nms import iou3d_nms_utils
from .track_eval_metrics import Success_torch, Precision_torch

def eval_track_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    dataset = dataloader.dataset
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    first_index = dataset.first_frame_index
    model.eval()

    Success_main = Success_torch()
    Precision_main = Precision_torch()

    try_false_time = 0
    fps = []
    pbar = tqdm.tqdm(total=len(first_index)-1, leave=False, desc='tracklets', dynamic_ncols=True)
    for f_index in range(len(first_index)-1):
        st = first_index[f_index]
        if f_index == len(first_index)-2:
            ov = first_index[f_index+1]+1
        else:
            ov = first_index[f_index+1]
        
        first_point = dataset[st+1]['template_voxels']
        
        length = ov - st - 1
        
        if length > 0:
            for index in range(st+1, ov):
                data = dataset[index]
                if index == st+1:
                    previou_box = data['template_gt_box'].reshape(7)
                    first_point = data['first_template_points']
                    Success_main.add_overlap(torch.ones(1).cuda())
                    Precision_main.add_accuracy(torch.zeros(1).cuda())

                batch_dict = dataset.collate_batch([data])

                load_data_to_gpu(batch_dict)
                gt_box = batch_dict['gt_boxes'].view(-1)[:7]
                center_offset = batch_dict['center_offset'][0]
                # the forward of spconv backbone will fail when the input points are very few, thus we use try-except to avoid error
                try:
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        start = time.time()      
                        pred_box = model(batch_dict).view(-1)
                        torch.cuda.synchronize()
                        end = time.time()
                        fps.append(end-start)   
                    pred_box[:2]+=center_offset[:2]
                except BaseException:
                    try_false_time += 1
                    pred_box = torch.from_numpy(previou_box).float().cuda()

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_box.view(1,-1), gt_box.view(1,-1)).squeeze()
                accuracy = torch.norm(pred_box[:3] - gt_box[:3])
                Success_main.add_overlap(iou3d)
                Precision_main.add_accuracy(accuracy)
                
                dataset.set_first_points(first_point)
                dataset.set_refer_box(pred_box.cpu().numpy())
                previou_box = pred_box.cpu().numpy()
                    
            dataset.reset_all()
        pbar.update()
    pbar.close()
    avs = Success_main.average.item()
    avp = Precision_main.average.item()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    ret_dict = {}

    logger.info('Success: %f' % (avs))
    logger.info('Precision: %f' % (avp))
    logger.info('Try False Times: %d' % (try_false_time))
    logger.info('Mean FPS: %f' % (len(fps)/sum(fps)))

    ret_dict['val/Success'] = avs
    ret_dict['val/Precision'] = avp
    ret_dict['val/Try_false_times'] = try_false_time
    ret_dict['val/FPS'] = len(fps)/sum(fps)

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
