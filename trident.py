import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import openai_imagenet_template, sub_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from torchvision import transforms
import torch.nn.functional as F

from open_clip import create_model, tokenizer
from segment_anything import sam_model_registry, SamPredictor
from myutils import UnNormalize
from seg_utils.utils import sam_refinement, preprocess_image

import cv2
from pamr import PAMR
import os

@MODELS.register_module()
class Trident(BaseSegmentor):
    def __init__(self, clip_type, model_type, vfm_model, name_path, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336, debug = False,
                 sam_refinement=False, sam_model_type='vit_b', pamr_steps=0, pamr_stride=(8, 16),
                 sam_ckpt='/home/yuheng/project/trident-pure/exclude/sam_vit_b_01ec64.pth',
                 coarse_thresh=0.10, minimal_area=225,sam_mask_coff=0.005, **kwargs):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize
        self.clip_stride = int(model_type[-2:])

        self.vfm_model = vfm_model
        self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.dataset_type = kwargs.get('dataset_type', 'PascalVOCDataset')

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma
        self.debug = debug
        self.cross_patch_fusion = 'ave'
        self.query_words = query_words

        if sam_model_type != 'vit_h' or not sam_refinement:
            #check https://github.com/facebookresearch/segment-anything/issues/540
            self.sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(device=device).eval().half()
        self.sam.prompt_encoder = self.sam.prompt_encoder.float()
        self.sam.mask_decoder = self.sam.mask_decoder.float()

        self.cos_fac = kwargs.get('cos_fac', 1.2)
        self.refine_neg_cos = kwargs.get('refine_neg_cos', True)
        self.sam_iou_thresh = kwargs.get('sam_iou_thresh', 0.9)
        self.sam_predictor = SamPredictor(self.sam)

        self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device) if pamr_steps > 0 else None
        if sam_model_type == 'vit_b':
            self.sam_heads = 12
        elif sam_model_type == 'vit_l':
            self.sam_heads = 16
        elif sam_model_type == 'vit_h':
            self.sam_heads = 16
        if sam_refinement:
            self.sam_refine = True
            self.coarse_thresh = coarse_thresh
            self.minimal_area = minimal_area
            self.sam_mask_coff = sam_mask_coff
        else:
            self.sam_refine = False
        self.kwargs = kwargs

    def get_sam_feat(self,tmp_img, stride):
        self.sam_predictor.set_image(tmp_img)
        sam_r = 1.0
        sam_valid_h, sam_valid_w = sam_r * self.sam_predictor.input_size[0] // stride, sam_r * \
                                   self.sam_predictor.input_size[1] // stride  # get the feature shape in SAM Encoder
        sam_valid_h, sam_valid_w = int(sam_valid_h), int(sam_valid_w)
        sam_enc_feats = self.sam_predictor.features  # [1, 256, 64, 64]
        sam_enc_feats = sam_enc_feats[:, :, :sam_valid_h, :sam_valid_w]  # [1, 256, new_h, new_w]
        sam_hw = int(sam_r * 64)
        sam_attn = self.sam_predictor.model.image_encoder.last_attn  # [1, 12, 64*64, 64*64]
        sam_attn = sam_attn.view(1, self.sam_heads, sam_hw, sam_hw, sam_hw, sam_hw)[:, :, :sam_valid_h, :sam_valid_w,
                   :sam_valid_h, :sam_valid_w]  # [1, 12, new_h, new_w, new_h, new_w]
        sam_attn = sam_attn.flatten(2, 3).flatten(3, 4)  # [1, 12, new_h*new_w, new_h*new_w]
        sam_v = self.sam_predictor.model.image_encoder.last_v
        sam_v = sam_v[:, :, :sam_valid_h, :sam_valid_w, :]  # [1, head, new_h, new_w, head_dim]

        return sam_enc_feats, sam_attn, sam_v, sam_valid_h, sam_valid_w

    @torch.no_grad()
    def get_trident_seg(self, src_img, img_path, img_metas):
        #TODO： replace the tmp_img with the src_img
        stride = self.clip_stride

        tmp_img = cv2.imread(img_path)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_h, tmp_w = tmp_img.shape[:2]
        if tmp_h % stride != 0: tmp_h = (tmp_h // stride + 1) * stride
        if tmp_w % stride != 0: tmp_w = (tmp_w // stride + 1) * stride
        tmp_img = cv2.resize(tmp_img, (tmp_w, tmp_h))
        if self.dataset_type == 'CityscapesDataset':
            #special cases for cityscapes
            tmp_img_1 = tmp_img[:, :tmp_w // 2, :]
            tmp_img_2 = tmp_img[:, tmp_w // 2:, :]
            tmp_feats_1, sam_attn_1, sam_v_1, sam_valid_h_1, sam_valid_w_1 = self.get_sam_feat(tmp_img_1, 16)
            tmp_feats_2, sam_attn_2, sam_v_2, sam_valid_h_2, sam_valid_w_2 = self.get_sam_feat(tmp_img_2, 16)
            sam_enc_feats = [tmp_feats_1, tmp_feats_2]
            sam_attn = [sam_attn_1, sam_attn_2]
            sam_v = [sam_v_1, sam_v_2]
            sam_valid_w = sam_valid_w_1 * 2
            sam_valid_h = sam_valid_h_1
            if self.sam_refine: self.sam_predictor.set_image(tmp_img)
        else:
            sam_enc_feats, sam_attn, sam_v, sam_valid_h, sam_valid_w = self.get_sam_feat(tmp_img, 16)

        #preprocess the image to make the longer size is stride times and padding the shorter size also to stride times
        processed_img = preprocess_image(src_img, stride, self.slide_crop)
        clip_whole_h, clip_whole_w = processed_img.shape[-2:]
        clip_feat_h, clip_feat_w = clip_whole_h // stride, clip_whole_w // stride
        img_batch, paddings, patch_locs, win_sizes = self.get_windowed_imgs(processed_img, stride)

        imgs_norm = [self.norm(self.unnorm(img_batch[i])) for i in range(len(img_batch))]  # replace norm here
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        if self.vfm_model == 'dino':
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)
        # Forward pass in the model
        patch_size = self.vfm.patch_embed.patch_size
        if type(patch_size) is tuple: patch_size = patch_size[0]
        feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
        nb_im = feat.shape[0]  # Batch size
        vfm_h, vfm_w = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size
        vfm_feats = feat[:, 1:, :].reshape(nb_im, vfm_h, vfm_w, -1).permute(0, 3, 1, 2) #batch, c, h, w

        clip_features = self.clip.encode_image(img_batch.half(),external_feats=vfm_feats, beta=self.beta, gamma=self.gamma,
                                               paddings=paddings,dst_coords=patch_locs,win_sizes=win_sizes,
                                               dst_vh=clip_feat_h, dst_vw=clip_feat_w, sam_attn=sam_attn, sam_v=sam_v,
                                               cos_fac=self.cos_fac, vfm_token_size = (vfm_h, vfm_w),
                                               refine_neg_cos=self.refine_neg_cos)
        clip_features /= clip_features.norm(dim=-1, keepdim=True)
        logits = (clip_features) @ self.query_features.T #BLC
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], sam_valid_h, sam_valid_w) #B, C, H, W
        logits = F.interpolate(logits, size=img_metas[0]['ori_shape'][:2], mode='bilinear')
        return logits.float()

    def get_windowed_imgs(self, img, patch_size=16):
        stride, crop_size = self.slide_stride, self.slide_crop
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        crop_imgs, paddings, patch_locs = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                assert y1 % patch_size == 0 and x1 % patch_size == 0
                assert y2 % patch_size == 0 and x2 % patch_size == 0
                patch_locs.append(torch.tensor([y1//patch_size, x1//patch_size, y2//patch_size, x2//patch_size]))
                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)
                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                crop_imgs.append(crop_img)
                paddings.append(pad)
        batched_imgs = torch.cat(crop_imgs, dim=0) # [n_patches, 3, h, w]
        return batched_imgs, paddings, patch_locs, (h_grids,w_grids)

    def predict(self, inputs, data_samples, debug_img_path=None):
        if data_samples is not None:
            if self.debug: batch_img_metas = data_samples
            else:
                batch_img_metas = [
                    data_sample.metainfo for data_sample in data_samples
                ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        if self.debug and debug_img_path:
            img_path = debug_img_path
        else:
            img_path = data_samples[0].img_path

        seg_logits = self.get_trident_seg(inputs, img_path, batch_img_metas)

        if self.pamr:
            img_size = batch_img_metas[0]['ori_shape']
            img = nn.functional.interpolate(inputs, size=img_size, mode='bilinear', align_corners=False)
            try:
                seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)
            except RuntimeError as e:
                print(f"Error in PAMR: {e}")

        return self.postprocess_result(seg_logits, data_samples, debug_img_path)

    def postprocess_result(self, seg_logits, data_samples, debug_img_path=None):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_fg = seg_logits[-num_cls+1:]
                seg_bg = seg_logits[:-num_cls+1].max(0)[0]
                seg_logits = torch.cat([seg_bg.unsqueeze(0), seg_fg], dim=0)

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            if self.debug and debug_img_path: tmp_img = debug_img_path
            else: tmp_img = data_samples[i].img_path
            if self.sam_refine:
                assert (data_samples is not None or debug_img_path is not None)
                refined_masks,scores,refined_logits,prompt_boxes = sam_refinement(tmp_img, seg_pred, seg_logits, num_cls,
                                                                                  self.sam_predictor, self.coarse_thresh, self.minimal_area,
                                                                                  self.sam_mask_coff, self.sam_iou_thresh,)
                seg_pred = refined_masks
                seg_logits = refined_logits

            if data_samples is None or self.debug:
                return seg_pred, seg_logits
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices