import os
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

parser = argparse.ArgumentParser(description="Generate bounding boxes for the cellSAM dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-input_dir", "--input_dir", required=True, help="cellSAM dataset root path")
parser.add_argument("-out_dir", "--out_dir", required=True, help="Output directory to store the new data")
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

folders = sorted(next(os.walk(args["input_dir"]))[1])
for folder in tqdm(folders):
    print(f"Processing folder '{folder}' . . .")
    subdataset_dir = os.path.join(args["input_dir"], folder)

    x_ids = [os.path.basename(x) for x in glob.glob(f'{subdataset_dir}/*.X.npy')]
    y_ids = [os.path.basename(x) for x in glob.glob(f'{subdataset_dir}/*.y.npy')]
    x_ids.sort()
    y_ids.sort()
    assert len(x_ids) == len(y_ids), f"Different number of X and Y samples found in {subdataset_dir}"

    x_out_folder = os.path.join(args["out_dir"], "x")
    y_out_folder = os.path.join(args["out_dir"], "y")
    for j, id_ in tqdm(enumerate(x_ids), total=len(x_ids)):
        # X data
        sample_path = os.path.join(subdataset_dir, id_)
        img = read_img_as_ndarray(sample_path, is_3d=False)
        save_tif(np.expand_dims(img,0), x_out_folder, [id_], verbose=False)

        # Y data
        sample_path = os.path.join(subdataset_dir, y_ids[j])
        mask = read_img_as_ndarray(sample_path, is_3d=False).squeeze()

        # Extract all bboxes from instances
        df = pd.DataFrame(columns=['label','bbox_min_y','bbox_min_y','bbox_min_y','bbox_min_y'])
        regions = regionprops(mask)
        for k, props in enumerate(regions):
            miny, minx, maxy, maxx = props.bbox
            df.loc[k] = [props.label, miny, minx, maxy, maxx]

        os.makedirs(y_out_folder, exist_ok=True)
        df.to_csv(os.path.join(y_out_folder, os.path.splitext(id_)[0]+".csv"), encoding='utf-8', index=False)

print("Finished!")
