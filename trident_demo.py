import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")

from PIL import Image
from torchvision import transforms
from trident import Trident
import torch.nn.functional as F
from matplotlib.patches import Patch


img_path = 'images/frog.jpg'
name_list = ['frog', 'water', 'turtle', 'snail']

#SAM　settings
sam_checkpoint = "/home/yuheng/project/trident-pure/exclude/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
coarse_thresh = 0.2

#set random seed for numpy
np.random.seed(42)
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

def show_mask(mask, ax, random_color=False, default_color =  [30/255, 144/255, 255/255]):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.concatenate([default_color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color

def show_results(image, seg_pred, seg_logit, name_list, add_bg=False, vis_thresh = 0.01):
    if type(image) == str:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cls_pred = F.one_hot(seg_pred.squeeze(0).long(), num_classes=len(name_list)+int(add_bg)).permute(2, 0, 1).float()  # [C, H, W]
    patches = []  # To store legend patches
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    for i in range(len(name_list)):
        bool_mask = cls_pred[i].bool() & (seg_logit[i] > vis_thresh)
        cls_color = [c / 255 for c in VOC_COLORMAP[1:][i]] # Skip the first color (black)
        color = show_mask(bool_mask.cpu().numpy(), plt.gca(), default_color=cls_color)
        patch = Patch(color=color, label=str(i) + name_list[i])
        patches.append(patch)
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def trident_demo(img_path, name_list):

    with open('./configs/my_name.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list) - 1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
    writers.close()


    img = Image.open(img_path)
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])(img)
    img_tensor = img_tensor.unsqueeze(0).to('cuda')
    model = Trident(clip_type='openai', model_type='ViT-B/16', vfm_model='dino', name_path='./configs/my_name.txt', sam_refinement=True,
                    coarse_thresh=coarse_thresh, minimal_area=225, debug=True)
    seg_pred, seg_logits = model.predict(img_tensor, data_samples=None, debug_img_path=img_path)  # [1, H, W]
    show_results(img_path, seg_pred, seg_logits, name_list, vis_thresh=0.01)
    pass


image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
trident_demo(img_path, name_list)


