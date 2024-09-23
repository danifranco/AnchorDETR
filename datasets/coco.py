# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from PIL import Image
import datasets.transforms as T

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
from util.box_ops import box_cxcywh_to_xyxy


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.unnorm = T.UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    
    def get_transformed_samples(self, num_examples, out_dir, random_images=False):
        # Generate the examples
        print("Creating samples of data augmentation . . .")
        for i in range(num_examples):
            if random_images:
                pos = random.randint(0, len(self.ids) - 1) if len(self.ids) > 2 else 0
            else:
                pos = i
            img, target = super(CocoDetection, self).__getitem__(pos)
            image_id = self.ids[pos]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)

            self.save_image(img, target, out_dir, f'image_{pos}.png')

            # Apply transformations
            if self._transforms is not None:
                img, target = self._transforms(img, target)

            # Undo normalization
            self.unnorm(img)
            self.save_image(img, target, out_dir, f'image_{pos}_transformed.png')

    def save_image(self, image, target, data_out_dir, filename):
        if isinstance(image, Image.Image):
            image = np.array(image)
            boxes = target['boxes']
        # DA has been applied 
        elif isinstance(image, torch.Tensor):
            image = np.array(image).transpose(1,2,0)
            boxes = target['boxes']
            boxes = box_cxcywh_to_xyxy(boxes)
            boxes *= np.array([image.shape[1],image.shape[0],image.shape[1],image.shape[0]]) 

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax=plt.gca()
        for box in boxes:
            ax.add_patch(plt.Rectangle(
                (int(box[0]), int(box[1])), 
                int(box[2]-box[0]), int(box[3]-box[1]), 
                edgecolor='red', 
                facecolor=(0, 0, 0, 0), 
                lw=2)
            ) 
        plt.axis('off')
        os.makedirs(data_out_dir, exist_ok=True)
        plt.savefig(os.path.join(data_out_dir, filename))
    
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, sam2=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # if not sam2:
    #     scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    #     if image_set == 'train':
    #         return T.Compose([
    #             T.RandomHorizontalFlip(),
    #             T.RandomSelect(
    #                 T.RandomResize(scales, max_size=1333),
    #                 T.Compose([
    #                     T.RandomResize([400, 500, 600]),
    #                     T.RandomSizeCrop(384, 600),
    #                     T.RandomResize(scales, max_size=1333),
    #                 ])
    #             ),
    #             normalize,
    #         ])

    #     if image_set == 'val' or image_set == 'test':
    #         return T.Compose([
    #             T.RandomResize([800], max_size=1333),
    #             normalize,
    #         ])
    # else:
    scales = [1024]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.SquareRandomResize(scales),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.SquareRandomResize(scales),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    if args.dataset_file == 'cell':
        img_folder = root / f"{image_set}"
        ann_file = root / "annotations" / f'coco_{image_set}.json'
    else: # COCO
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json'),
        }

        img_folder, ann_file = PATHS[image_set]

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, sam2=("sam2" == args.backbone)), 
        return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    
    # Generate examples of data augmentation
    print("Creating generator samples . . .")
    dataset.get_transformed_samples(
        10,
        out_dir=os.path.join(args.output_dir, "aug"),
        random_images=True,
    )

    return dataset
