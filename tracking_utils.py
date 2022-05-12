import math

import torch
from torch.nn import functional

from libs.test_utils import transform_topk
from libs.track_utils import squeeze_all, match_ref_tar, bbox_in_tar_scale
from test_with_track import forward, model, args


def adjust_bbox(bbox_now, bbox_pre, a, h, w):
    """
    Adjust a bounding box w.r.t previous frame,
    assuming objects don't go under abrupt changes.
    """
    for cnt in bbox_pre.keys():
        if cnt == 0:
            continue
        if cnt in bbox_now and bbox_pre[cnt] is not None and bbox_now[cnt] is not None:
            bbox_now_h = (bbox_now[cnt].top + bbox_now[cnt].bottom) / 2.0
            bbox_now_w = (bbox_now[cnt].left + bbox_now[cnt].right) / 2.0

            bbox_now_height_ = bbox_now[cnt].bottom - bbox_now[cnt].top
            bbox_now_width_ = bbox_now[cnt].right - bbox_now[cnt].left

            bbox_pre_height = bbox_pre[cnt].bottom - bbox_pre[cnt].top
            bbox_pre_width = bbox_pre[cnt].right - bbox_pre[cnt].left

            bbox_now_height = a * bbox_now_height_ + (1 - a) * bbox_pre_height
            bbox_now_width = a * bbox_now_width_ + (1 - a) * bbox_pre_width

            bbox_now[cnt].left = math.floor(bbox_now_w - bbox_now_width / 2.0)
            bbox_now[cnt].right = math.ceil(bbox_now_w + bbox_now_width / 2.0)
            bbox_now[cnt].top = math.floor(bbox_now_h - bbox_now_height / 2.0)
            bbox_now[cnt].bottom = math.ceil(bbox_now_h + bbox_now_height / 2.0)

            bbox_now[cnt].left = max(0, bbox_now[cnt].left)
            bbox_now[cnt].right = min(w, bbox_now[cnt].right)
            bbox_now[cnt].top = max(0, bbox_now[cnt].top)
            bbox_now[cnt].bottom = min(h, bbox_now[cnt].bottom)

    return bbox_now


def bbox_next_frame(img_ref, seg_ref, img_tar, bbox_ref):
    """
    Match bbox from the reference frame to the target frame
    """
    F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
    seg_ref = seg_ref.squeeze(0)
    F_ref, F_tar = squeeze_all(F_ref, F_tar)
    c, h, w = F_ref.size()

    # get coordinates of each point in the target frame
    coords_ref_tar = match_ref_tar(F_ref, F_tar, seg_ref, args.temp)
    # coordinates -> bbox
    bbox_tar = bbox_in_tar_scale(coords_ref_tar, bbox_ref, h, w)
    # adjust bbox
    bbox_tar = adjust_bbox(bbox_tar, bbox_ref, 0.1, h, w)
    return bbox_tar, coords_ref_tar


def recognition(img_ref, img_tar, bbox_ref, bbox_tar, seg_ref, model):
    """
    propagate from bbox in the reference frame to bbox in the target frame
    """
    F_ref, F_tar = forward(img_ref, img_tar, model, seg_ref, return_feature=True)
    seg_ref = seg_ref.squeeze()
    _, c, h, w = F_tar.size()
    seg_pred = torch.zeros(seg_ref.size())

    # calculate affinity only once to save time
    aff_whole = torch.mm(F_ref.view(c, -1).permute(1, 0), F_tar.view(c, -1))
    aff_whole = FUNC.softmax(aff_whole * args.temp, dim=0)

    for cnt, br in bbox_ref.items():
        if not (cnt in bbox_tar):
            continue
        bt = bbox_tar[cnt]
        if br is None or bt is None:
            continue
        seg_cnt = seg_ref[cnt]

        # affinity between two patches
        seg_ref_box = seg_cnt[br.top:br.bottom, br.left:br.right]
        seg_ref_box = seg_ref_box.unsqueeze(0).unsqueeze(0)

        h, w = F_ref.size(2), F_ref.size(3)
        mask = torch.zeros(h, w)
        mask[br.top:br.bottom, br.left:br.right] = 1
        mask = mask.view(-1)
        aff_row = aff_whole[mask.nonzero().squeeze(), :]

        h, w = F_tar.size(2), F_tar.size(3)
        mask = torch.zeros(h, w)
        mask[bt.top:bt.bottom, bt.left:bt.right] = 1
        mask = mask.view(-1)
        aff = aff_row[:, mask.nonzero().squeeze()]

        aff = aff.unsqueeze(0)

        seg_tar_box = transform_topk(aff, seg_ref_box.cuda(), k=args.topk,
                                     h2=bt.bottom - bt.top, w2=bt.right - bt.left)
        seg_pred[cnt, bt.top:bt.bottom, bt.left:bt.right] = seg_tar_box

    return seg_pred


def disappear(seg, bbox_ref, bbox_tar=None):
    """
    Check if bbox disappear in the target frame.
    """
    b, c, h, w = seg.size()
    for cnt in range(c):
        if torch.sum(seg[:, cnt, :, :]) < 3 or (not (cnt in bbox_ref)):
            return True
        if bbox_ref[cnt] is None:
            return True
        if bbox_ref[cnt].right - bbox_ref[cnt].left < 3 or bbox_ref[cnt].bottom - bbox_ref[cnt].top < 3:
            return True

        if bbox_tar is not None:
            if cnt not in bbox_tar.keys():
                return True
            if bbox_tar[cnt] is None:
                return True
            if bbox_tar[cnt].right - bbox_tar[cnt].left < 3 or bbox_tar[cnt].bottom - bbox_tar[cnt].top < 3:
                return True
    return False