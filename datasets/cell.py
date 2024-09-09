import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from .data_utils import read_img_as_ndarray, normalize_to_range
import datasets.transforms as T

class CellDetection(Dataset):
    """
    Cell detection generator. 

    Parameters
    ----------
    X : str
        Path to the images.

    Y : dtr
        Path to masks. 
    """

    def __init__(self, X, Y, transforms=None):
        self.X_dir = X
        self.Y_dir = Y
        self.X = sorted(next(os.walk(X))[2])
        self.Y = sorted(next(os.walk(Y))[2])
        assert len(self.X) == len(self.Y), f"Different number of X and Y samples found in {X} and {Y} folders respectively"
        
        self.length = len(self.X)
        self.transforms = transforms

    def __len__(self):
        """Defines the number of samples per epoch."""
        return self.length

    def __getitem__(self, idx):
        """
        Generation of one pair of data.

        Parameters
        ----------
        idx : int
            Index counter.

        Returns
        -------
        img : 3D Torch tensor
            Loaded image. E.g. (channels, y, x).

        target : dict
            Target information containing the following keys:
                * 'image_id': integer. Number of the image. 
                * 'boxes': (N, 4) tensor. Each row contains a bbox of the object to find within the image.
                * 'labels': classes of each bbox object. This is expected in AnchorDETR code so we just set 
                  all labels to same value. However it is not used to calculate the loss.  
                * 'orig_size': original size of the image. 
                * 'size': current image size.
        """
        # X data
        img = read_img_as_ndarray(os.path.join(self.X_dir, self.X[idx]))
        img = normalize_to_range(img, img.min(), img.max(), out_min=0, out_max=255, out_type=np.uint8)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        w, h = img.size

        # Y data
        annotation = pd.read_csv(os.path.join(self.Y_dir, self.Y[idx]))
        boxes = [[a,b,c,d] for a,b,c,d in zip(annotation["bbox_min_y"],annotation["bbox_min_x"],annotation["bbox_max_y"],annotation["bbox_max_x"])]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        target =  {
            'image_id': torch.tensor([idx]), 
            'boxes': boxes,
            'labels': torch.ones(len(boxes)), # this is expected so we just set all labels to same value 
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def make_cell_transforms(image_set):
    """
    Prepare the transformation to be performed to the images and bboxes. 
    
    Parameters
    ----------
    image_set : str
        Phase of the workflow to build transformations for. Options: ['train', 'val', 'test']. 

    Returns
    -------
    transformations : list
        List of transformations to be applied to the data. 
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "x", root / "train" / "y"),
        "val": (root / "val"/ "x", root / "val" / "y"),
        "test": (root / "test"/ "x", root / "test" / "y" ),
    }

    img_folder, ann_folder = PATHS[image_set]
    dataset = CellDetection(img_folder, ann_folder, transforms=make_cell_transforms(image_set))
    return dataset
