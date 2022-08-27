import torch

# Problem: x, y, z and w, h, l do not match
def gather_feature(feature, index):
    '''
    Args:
        feature: (B, N, C)
        index: (B, K)
    Return:
        feature: (B, K, C)
    '''

    batch, point_num, channel = feature.shape
    index = index.unsqueeze(2).expand(index.shape[0], index.shape[1], channel)

    ind_feature = feature.gather(1, index)
    return ind_feature

def topk_point(scores, K=50):
    batch, channels, point_num = scores.shape

    topk_scores, topk_inds = torch.topk(scores, K)
    topk_inds = topk_inds % point_num
        
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses

def get_region(reg_feature, corner_coors):
    corner_x = corner_coors[:,:,0]
    corner_y = corner_coors[:,:,1]
    corner_z = corner_coors[:,:,2]
    dx = reg_feature[:,:,1]
    dy = reg_feature[:,:,0]
    dz = reg_feature[:,:,2].unsqueeze(2)
    dz = dz / 2
    zeros = corner_coors.new_zeros([corner_x.shape[0],corner_x.shape[1]])

    xmax0 = torch.where(corner_coors[:,:,0]>=0,dx/2,zeros)
    xmin0 = torch.where(corner_coors[:,:,0]<0,-dx/2,zeros)
    ymax0 = torch.where(corner_coors[:,:,1]>=0,dy/2,zeros)
    ymin0 = torch.where(corner_coors[:,:,1]<0,-dy/2,zeros)

    x = (xmax0 + xmin0).unsqueeze(2)
    y = (ymax0 + ymin0).unsqueeze(2)

    center_point = torch.cat((x,y,dz),2)

    return center_point

def rotate_along_corner(center, ry):
    ry = ry.squeeze(2)
    center_x = center[:,:,0]
    center_y = center[:,:,1]
    center_z = center[:,:,2].unsqueeze(2)
    R = torch.sqrt(center_x**2 + center_y**2)
    tana = center_x / center_y
    a = torch.arctan2(tana)
    b = ry + a

    cosb = -torch.cos(b)
    sinb = -torch.sin(b)
    new_x = (R * cosb).unsqueeze(2)
    new_y = (R * sinb).unsqueeze(2)

    rotate_center = torch.cat([new_x, new_y, center_z],dim=-1)
    return rotate_center

def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets

def gather_feature2d(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index  = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def topk_score2d(scores, K=40):
    """
    get top K point in score map
    """
    batch, channel, height, width = scores.shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    # div by K because index is grouped by K(C x K shape)
    topk_clses = (index / K).int()
    topk_inds = gather_feature2d(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature2d(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature2d(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs