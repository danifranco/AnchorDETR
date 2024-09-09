import torch 

import numpy as np
from skimage.io import imread

def read_img_as_ndarray(path):
    """
    Read an image from a given path.

    Parameters
    ----------
    path : str
        Path to the image to read.

    Returns
    -------
    img : Numpy 3D/4D array
        Image read. E.g. ``(z, y, x, num_classes)`` for 3D or ``(y, x, num_classes)`` for 2D.
    """
    # Read image
    if path.endswith(".npy"):
        img = np.load(path)
    else:
        img = imread(path)
    img = np.squeeze(img)

    img = ensure_2d_shape(img, path)

    return img


def ensure_2d_shape(img, path=None):
    """
    Read an image from a given path.

    Parameters
    ----------
    img : ndarray
        Image read.

    path : str
        Path of the image (just use to print possible errors).

    Returns
    -------
    img : Numpy 3D array
        Image read. E.g. ``(y, x, num_classes)``.
    """
    if img.ndim > 3:
        if path is not None:
            m = "Read image seems to be 3D: {}. Path: {}".format(img.shape, path)
        else:
            m = "Read image seems to be 3D: {}".format(img.shape)
        raise ValueError(m)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    else:
        if img.shape[0] <= 3:
            img = img.transpose((1, 2, 0))
    return img


def normalize_zero_mean_one_std(data, means, stds, out_type="float32"):
    numpy_torch_dtype_dict = {
        "bool": [torch.bool, bool],
        "uint8": [torch.uint8, np.uint8],
        "int8": [torch.int8, np.int8],
        "int16": [torch.int16, np.int16],
        "int32": [torch.int32, np.int32],
        "int64": [torch.int64, np.int64],
        "float16": [torch.float16, np.float16],
        "float32": [torch.float32, np.float32],
        "float64": [torch.float64, np.float64],
        "complex64": [torch.complex64, np.complex64],
        "complex128": [torch.complex128, np.complex128],
    }
    if torch.is_tensor(data):
        if stds == 0:
            return data.to(numpy_torch_dtype_dict[out_type][0])
        else:
            return ((data - means) / stds).to(numpy_torch_dtype_dict[out_type][0])
    else:
        if stds == 0:
            return data.astype(numpy_torch_dtype_dict[out_type][1])
        else:
            return ((data - means) / stds).astype(numpy_torch_dtype_dict[out_type][1])

def normalize_to_range(x, x_min, x_max, out_min=0, out_max=1, out_type=np.float32):
    if isinstance(x, np.ndarray):
        return ((np.array((x - x_min) / (x_max - x_min)) * (out_max - out_min)) + out_min).astype(out_type)
    else:  # Tensor considered
        return ((((x - x_min) / (x_max - x_min)) * (out_max - out_min)) + out_min).to(out_type)