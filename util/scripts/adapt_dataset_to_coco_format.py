# Used steps in: https://www.kaggle.com/code/impiyush/simply-convert-data-to-coco-format/notebook

import os
import json
import sys
import argparse
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops

# python adapt_dataset_to_bboxes.py --input_dir "/scratch/dfranco/thesis/data2/dfranco/datasets/cellSAM_dataset/dataset/train" \
#     --out_dir "/scratch/dfranco/thesis/data2/dfranco/datasets/cellSAM_dataset/prepared_dataset/train" \
#     --biapy_dir ../../../BiaPy/

parser = argparse.ArgumentParser(description="Adapts dataset into COCO format",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-input_dir", "--input_dir", required=True, help="cellSAM dataset root path")
parser.add_argument("-out_dir", "--out_dir", required=True, help="Output directory to store the new data")
parser.add_argument("-data_type", "--data_type", required=True, help="Type of data", choices=['train', 'val', 'test'])
parser.add_argument("-biapy_dir", "--biapy_dir", required=True, help="BiaPy directory")

args = vars(parser.parse_args())

if not os.path.exists(args["input_dir"]):
    raise FileNotFoundError("{} directory not found".format(args["input_dir"]))

if not os.path.exists(args["biapy_dir"]):
    raise FileNotFoundError("{} directory not found".format(args["biapy_dir"]))

# Import some aux functions of BiaPy
sys.path.insert(0, args["biapy_dir"])
from biapy.data.data_manipulation import read_img_as_ndarray
from biapy.utils.util import save_tif

x_out_folder = os.path.join(args["out_dir"], args['data_type'])
y_out_folder = os.path.join(args["out_dir"], "annotations")

images = []
bboxes = []
heights = []
widths = []
image_filenames = []
image_id = 1
folders = sorted(next(os.walk(args["input_dir"]))[1])
for i, folder in tqdm(enumerate(folders)):
    print(f"Processing folder '{folder}' . . .")
    subdataset_dir = os.path.join(args["input_dir"], folder)

    x_ids = [os.path.basename(x) for x in glob.glob(f'{subdataset_dir}/*.X.npy')]
    y_ids = [os.path.basename(x) for x in glob.glob(f'{subdataset_dir}/*.y.npy')]
    x_ids.sort()
    y_ids.sort()
    assert len(x_ids) == len(y_ids), f"Different number of X and Y samples found in {subdataset_dir}"

    for j, id_ in tqdm(enumerate(x_ids), total=len(x_ids)):
        # X data
        sample_path = os.path.join(subdataset_dir, id_)
        new_sample_name = os.path.basename(subdataset_dir) + "_" + os.path.splitext(id_)[0] + ".tif"
        img = read_img_as_ndarray(sample_path, is_3d=False)
        save_tif(np.expand_dims(img,0), x_out_folder, [new_sample_name], verbose=False)

        # Y data
        sample_path = os.path.join(subdataset_dir, y_ids[j])
        mask = read_img_as_ndarray(sample_path, is_3d=False).squeeze()

        # Extract all bboxes from instances
        regions = regionprops(mask)
        image_bboxes  = []
        for k, props in enumerate(regions):
            miny, minx, maxy, maxx = props.bbox
            image_bboxes.append([miny, minx, abs(maxy-miny), abs(maxx-minx)])

        heights.append(img.shape[0])
        widths.append(img.shape[1])
        image_filenames.append(new_sample_name)
        images.append(image_id)
        bboxes.append(image_bboxes)
        image_id += 1

df = pd.DataFrame(list(zip(image_filenames,images, heights, widths, bboxes)), columns=['filename','image_id','height', 'width', 'bboxes'])

coco_base = { "info": {},
    "licenses": [], 
    "images": [],
    "annotations": [],
    "categories": []
}

coco_base["info"] = {
    "description": "CellSAM dataset",
    "url": "https://github.com/vanvalenlab/cellSAM",
    "version": "1.0",
    "year": 2023,
    "contributor": "https://github.com/vanvalenlab/cellSAM/graphs/contributors",
    "date_created": "2023/11/18"
}

coco_base["licenses"].append(
    {
        "url": "https://opensource.org/license/apache-2-0",
        "id": 1,
        "name": "Apache License"
    }
)

coco_base["categories"].append({"supercategory": "image", "id": 1, "name": "image"})

def set_coco_images(df):
    """author: @impiyush"""
    images = []
    for _, row in tqdm(df.iterrows()):
        img_dict = {
            "license": 1, 
            "height": int(row['height']), 
            "width": int(row['width']),
            "id": int(row['image_id']),
            "date_captured": "2023/11/18",
            "file_name": "{}".format(row['filename']),
            }
        images.append(img_dict)

    return images

coco_base["images"] = set_coco_images(df)
assert len(coco_base["images"])==len(df), "Number of images differ from df"

def set_coco_annotations(df):
    """author: @impiyush"""
    annos = []
    id_cnt = 1
    for _,row in tqdm(df.iterrows(), total=len(df)):
        anno = {
            'segmentation': [],
            'iscrowd': 0,
            'image_id': int(row['image_id']),
            'category_id': 1,
        }
                
        bboxes = row['bboxes']
        for ix, box in enumerate(bboxes):
            anno['bbox'] = box # x,y,w,h
            anno['area'] = box[2] * box[3] 
            anno['id'] = f"{id_cnt:05}"
            annos.append(anno.copy()) 
            id_cnt += 1
    
    return annos

coco_base['annotations'] = set_coco_annotations(df)

os.makedirs(y_out_folder, exist_ok=True)
with open(os.path.join(y_out_folder, 'coco_{}.json'.format(args['data_type'])),'w') as train_coco:
    json.dump(coco_base, train_coco)

print("Finished!")
