# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse

from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.nn as nn
import random
from datetime import datetime
import shutil
import glob
from tqdm import tqdm
import monai



# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4) 

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    # low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg
    # return low_res_pred


# args
# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="./seg4medicine/medsam-data/npy/CT_5",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="./seg4medicine/medsam-ouput/CT_5",
    help="path to the segmentation folder",
)
# parser.add_argument(
#     "--box",
#     type=str,
#     default='[95, 255, 190, 350]',
#     help="bounding box of the segmentation target",
# )
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="./seg4medicine/MedSAM/work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-num_workers", type=int, default=8)
parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
parser.add_argument("--world_size", type=int, help="world size")
args = parser.parse_args()

def main():
    if torch.cuda.device_count()>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    # medsam_model = medsam_model.to(device)
    medsam_model = nn.DataParallel(medsam_model)
    medsam_model.eval()

    # %% test
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    log_save_path = join(args.seg_path, 'MedSAM_inference' + "-" + run_id)
    device = torch.device(args.device)
    test_dataset = NpyDataset(args.data_path)
    print("Number of testing samples: ", len(test_dataset))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    seg_loss = monai.losses.DiceLoss(sigmoid=False, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    for step, (image, gt2D, boxes, img_names) in enumerate(tqdm(test_dataloader)):
        boxes_np = boxes.detach().cpu().numpy() #[B, 4]
        image, gt2D = image.to(device), gt2D.to(device) #[B, 3, H, W]
        B, _, H, W = image.shape
        # box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]]) 
        # transfer box_np t0 1024x1024 scale
        box_1024 = boxes_np/ np.array([W, H, W, H]) * 1024 #[B, 4]
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(image)  # (1, 256, 64, 64)
            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        
        for seg_idx, name in enumerate(img_names):
            io.imsave(
                join(args.seg_path, "seg_" + name),
                medsam_seg[seg_idx],
                check_contrast=False,
            )
        loss = seg_loss(medsam_seg, gt2D)
        for idx, item in enumerate(loss.item()):
            with open(os.join(args.seg_path, 'results.txt'), 'a+') as f:
                result = img_names[idx] + ": " + item + '\n'
                f.write(result)




if __name__ == "__main__":
    main()

