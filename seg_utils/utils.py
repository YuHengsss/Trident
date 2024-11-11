import torch
import numpy as np
from skimage import measure
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F
import cv2
import time

def preprocess_image(image, stride=16, slide_crop=336):
    longer_side = max(image.shape[2:])
    h, w = image.shape[2:]
    if h%stride == 0 and w%stride == 0:
        return image
    if longer_side % stride != 0:
        dst_longer = (longer_side // stride + 1) * stride
    else:
        dst_longer = longer_side
    new_h = int(h * dst_longer / longer_side)
    new_w = int(w * dst_longer / longer_side)
    if new_h % stride != 0: new_h = (new_h // stride + 1) * stride
    if new_w % stride != 0: new_w = (new_w // stride + 1) * stride
    new_h, new_w = max(new_h, slide_crop), max(new_w, slide_crop)
    image = torch.nn.functional.interpolate(image, (new_h, new_w), mode='bilinear', align_corners=False)

    return image

def split_connected_regions(segmentation: np.ndarray, seg_logits, split_last=False, minimal_area=0,
                            coarse_thresh=0.0,) -> dict:
    """
    Splits the segmentation mask into separate connected regions for each class.

    Parameters:
        segmentation (np.ndarray): Semantic segmentation mask of shape [C, H, W].
        seg_logits (np.ndarray): Segmentation logits of shape [C, H, W].
        split_last (bool): If True, the last class is not split into connected regions and regarded as the bg.
        minimal_area (int): Minimal area of a connected region to be considered.
    Returns:
        dict: A dictionary where keys are class indices and values are lists of masks.
    """
    C, H, W = segmentation.shape
    regions, boxes, clip_box_score, points = {}, {}, {}, {}
    for c in range(C):
        if not split_last and c == C - 1: continue
        regions[c], points[c], clip_box_score[c],boxes[c] = [], [], [], []
        class_mask = segmentation[c]
        class_logit = seg_logits[c]
        if class_logit.max() < coarse_thresh: continue
        # Label connected regions
        labeled_mask = measure.label(class_mask, connectivity=2)  # 8-connectivity
        num_regions = labeled_mask.max()

        for region_label in range(1, num_regions + 1):
            region_mask = (labeled_mask == region_label).astype(np.uint8)
            if region_mask.sum() < minimal_area: continue
            # Calculate the mean logit score of the region
            region_score = class_logit[region_mask.astype(bool)].mean()
            #find the max logit position of the class_logit(H,W) and add it to points[c]
            region_logit = class_logit * torch.tensor(region_mask).to(class_logit.device)
            max_pos = torch.argmax(region_logit)
            max_h, max_w = max_pos // class_logit.size(1), max_pos % class_logit.size(1)
            points[c].append(torch.tensor([max_w, max_h]))
            regions[c].append(region_mask)
            clip_box_score[c].append(region_score)

        if len(regions[c]):
            regions[c] = torch.from_numpy(np.stack(regions[c]))
        if regions[c] == []: boxes[c] = regions[c]
        else: boxes[c] = masks_to_boxes(regions[c])
    return regions, boxes, clip_box_score, points

def map_refinement_coarse(refined_masks, refined_logits, coarse_boxes, seg_logits, img_shape):
    refined_whole_logits = torch.zeros([len(coarse_boxes),img_shape[0],img_shape[1]],device=refined_masks.device) # C, H, W
    cumsum = 0
    for i in range(len(coarse_boxes)):
        if coarse_boxes[i] == []: continue
        tmp_cls_num = coarse_boxes[i].shape[0]
        tmp_cls_mask = refined_masks[cumsum:cumsum+tmp_cls_num,0].sum(dim=0).bool() #H, W
        tmp_cls_logit = (refined_logits[cumsum:cumsum+tmp_cls_num,0] * seg_logits[i]).max(dim=0)  #H, W
        refined_whole_logits[i] = tmp_cls_logit.values * tmp_cls_mask
        cumsum += tmp_cls_num

    refined_whole_masks = refined_whole_logits.argmax(0, keepdim=True)
    return refined_whole_masks, refined_whole_logits

def map_failed_regions(refined_masks, refined_logits, failed_regions, seg_logits, segmentation):
    failed_area = failed_regions.sum(dim=0, keepdim=True)!=0 #1, H, W
    failed_logit = failed_area * seg_logits #C, H, W
    refined_logits = torch.where(failed_logit>refined_logits, failed_logit, refined_logits)
    refined_masks = torch.where(failed_logit.max(dim=0, keepdim=True)[0]>refined_logits.max(dim=0, keepdim=True)[0], segmentation, refined_masks)
    return refined_masks, refined_logits

def sam_refinement(img, segmentations, seg_logits, num_classes, predictor, coarse_thresh=0.0, minimal_area=0,
                   sam_mask_coff=0.005,sam_iou_thresh=0.9):
    '''
    Args:
        img: str or np.ndarray
        segmentations: torch.Tensor, shape=[C, H, W]
        seg_logits: torch.Tensor, shape=[C, H, W]
        num_classes: int
        predictor: Predictor
        coarse_thresh: float
        minimal_area: int
        sam_mask_coff: float
        sam_iou_thresh: float
    '''
    #downsample the segmentation and seg_logits to accelerate split_connected_regions
    down_ratio = 2
    dh, dw = seg_logits.shape[-2] // down_ratio, seg_logits.shape[-1] // down_ratio
    if down_ratio != 1:
        segmentations_down = F.interpolate(segmentations.unsqueeze(0).half(), [dh, dw], mode='nearest').squeeze(0).long()
        seg_logits_down = F.interpolate(seg_logits.unsqueeze(0), [dh, dw], mode='bilinear', align_corners=False).squeeze(0)
    else:
        segmentations_down = segmentations
        seg_logits_down = seg_logits

    if type(img) == str:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if coarse_thresh > 0:
        segmentations_down[seg_logits_down.max(0, keepdim=True)[0] < coarse_thresh] = num_classes
        cls_pred = F.one_hot(segmentations_down.squeeze(0).long(), num_classes=num_classes+1).permute(2, 0, 1).float()  # [C, H, W]
        split_last = False
    else:
        cls_pred = F.one_hot(segmentations_down.squeeze(0).long(), num_classes=num_classes).permute(2, 0, 1).float()
        split_last = True

    coarse_regions, coarse_boxes, coarse_boxes_score, coarse_points = split_connected_regions(cls_pred.cpu().numpy(), seg_logits_down,
                                                           split_last=split_last, minimal_area=minimal_area,coarse_thresh=coarse_thresh)
    combined_boxes,combined_points = [], []
    cls_wise_masks, cls_wise_scores, cls_wise_logits = [], [], []
    failed_regions = []
    predictor.features = predictor.features.float()
    for i in range(num_classes):
        tmp_r = coarse_regions.get(i, [])
        tmp_b = coarse_boxes.get(i, [])
        tmp_p = coarse_points.get(i, [])

        if tmp_r == []: continue
        cls_wise_r = seg_logits_down[i].unsqueeze(0)
        #resize to longer side = 256
        new_h, new_w = predictor.transform.get_preprocess_shape(img.shape[0], img.shape[1], 256)
        cls_wise_r = F.interpolate(cls_wise_r.unsqueeze(0), [new_h, new_w], mode="bilinear", align_corners=False).squeeze(0)
        #pad to 256x256
        cls_wise_r = F.pad(cls_wise_r, (0, 256-new_w, 0, 256-new_h), mode='constant', value=0)
        cls_wise_r = torch.where(cls_wise_r > coarse_thresh, sam_mask_coff*cls_wise_r, torch.tensor(0.0).type_as(segmentations)).float()
        cls_wise_b = predictor.transform.apply_boxes_torch(tmp_b.type_as(segmentations), [dh, dw]).float()
        cls_wise_p = torch.stack(tmp_p).type_as(segmentations)
        cls_wise_p = predictor.transform.apply_coords_torch(cls_wise_p, [dh, dw]).unsqueeze(1)
        cls_wise_mask, cls_wise_score, cls_wise_logit = predictor.predict_torch(
            point_coords=cls_wise_p.float(),  # input_point,
            point_labels=torch.ones([cls_wise_p.shape[0], 1]).type_as(segmentations).float(),  # input_label,
            boxes=cls_wise_b,
            mask_input=cls_wise_r,
            multimask_output=False,
        )

        valid_masks = (cls_wise_score > sam_iou_thresh).squeeze(-1)
        if valid_masks.sum() != tmp_r.shape[0]:
            failed_regions.append(tmp_r[~valid_masks.cpu()])
        if valid_masks.sum() == 0:
            coarse_boxes[i] = []
            continue
        cls_wise_masks.append(cls_wise_mask[valid_masks])
        cls_wise_scores.append(cls_wise_score[valid_masks])
        cls_wise_logits.append(cls_wise_logit[valid_masks])

        tmp_p = torch.stack(tmp_p)
        tmp_b = tmp_b[valid_masks.cpu()]
        tmp_p = tmp_p[valid_masks.cpu()]
        coarse_boxes[i] = coarse_boxes[i][valid_masks.cpu()]

        combined_boxes.append(tmp_b)
        combined_points.append(tmp_p)

    if len(combined_boxes) == 0:
        return segmentations, None, seg_logits, combined_boxes

    combined_boxes = torch.cat(combined_boxes, dim=0).to(predictor.device)
    masks, scores, logits = torch.cat(cls_wise_masks, dim=0), torch.cat(cls_wise_scores, dim=0), torch.cat(cls_wise_logits, dim=0)
    logits = predictor.model.postprocess_masks(logits, predictor.input_size, predictor.original_size)

    combined_boxes = combined_boxes * down_ratio
    if logits.shape[-1] != img.shape[1] or logits.shape[-2] != img.shape[0]:
        logits = F.interpolate(logits, img.shape[:2], mode='bilinear', align_corners=False)
        masks = logits>predictor.model.mask_threshold

    refined_masks, refined_logits = map_refinement_coarse(masks, logits.sigmoid(), coarse_boxes, seg_logits, img.shape,)

    refined_masks = torch.where(refined_logits.sum(dim=0,keepdim=True) == 0, segmentations, refined_masks)
    if len(failed_regions):
        failed_regions = torch.cat(failed_regions, dim=0).to(predictor.device)
        if down_ratio != 1:
            failed_regions = F.interpolate(failed_regions.unsqueeze(0).float(), img.shape[:2], mode='nearest').squeeze(0).type_as(segmentations)
        refined_masks, refined_logits = map_failed_regions(refined_masks, refined_logits, failed_regions, seg_logits, segmentations)

    return refined_masks, scores, refined_logits,combined_boxes