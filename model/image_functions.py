import os
import glob
import sys

import math
import numpy as np

import skimage
import skimage.io as skio
from skimage.transform import resize


def imadjust(image, in_bound=(0.001, 0.999), out_bound=(0, 1)):
    """
    See https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python/44529776#44529776
    image : input one-layer image (numpy array)
    in_bound  : src image bounds
    out_bound : dst image bounds
    output : output img
    """
    image_dtype = image.dtype

    if image_dtype == 'uint8':
        range_value = 255
    elif image_dtype == 'uint16':
        range_value = 65535
    else:
        range_value = 1

    # Compute in and out limits
    in_bound = np.percentile(image, np.multiply(in_bound,100))
    out_bound = np.multiply(out_bound, range_value)

    # Stretching
    scale = (out_bound[1] - out_bound[0]) / (in_bound[1] - in_bound[0])

    image = image - in_bound[0]
    image[image < 0] = 0

    output = (image) * scale + out_bound[0]
    output[output > out_bound[1]] = out_bound[1]

    output = output.astype(image_dtype)

    return output


# -------------------------------------------- Folder IO Functions --------------------------------------------------- #
def list_images(image_dir, image_ext='*.tif'):
    """List images in the directory with the given file extension

    Parameters
    ----------
    image_dir : `str`
        Directory to look for image files
    image_ext : `str`, optional
        [Default: '*.tif'] File extension of the image file

    Returns
    ----------
    image_list : `list`
        List of images found in the directory with the given file extension

    Notes
    ----------
    For linux based systems, please ensure that the file extensions are either in all lowercase or all uppercase.
    """
    # TODO: bypass case sensitivity of file extensions in linux and possibly other systems
    if sys.platform in ["win32", ]:
        image_extension = [image_ext]
    else:
        image_extension = [image_ext.lower(), image_ext.upper()]

    image_list = []
    for ext in image_extension:
        image_list.extend(glob.glob(os.path.join(image_dir, ext)))

    return image_list


def generate_file_list(dataset_dir, subfolder=None, image_ext="*.tif"):
    """Finds all images with extension `image_ext` in `dataset_dir`

    Parameters
    ----------
    dataset_dir : str
        Folder to load the dataset from.
    subfolder : str or none, optional
        Subfolder in which to look for the image/ground_truth file
    image_ext : `str`, optional
        [Default: '*.tif'] File extension of the image file
    """
    image_dirs = next(os.walk(dataset_dir))[1]

    image_list = []

    for directory in image_dirs:
        # skip hidden directories or special characters '.' and '..'
        if directory[0] == ".":
            continue

        image_list.extend(list_images(os.path.join(dataset_dir, directory, subfolder), image_ext=image_ext))

    return image_list


def generate_folder_list(dataset_dir, subfolder=None):
    """Finds all directories
    """
    image_dirs = next(os.walk(dataset_dir))[1]

    folder_list = []

    for directory in image_dirs:
        # skip hidden directories or special characters '.' and '..'
        if directory[0] == ".":
            continue
        folder_list.append(os.path.join(dataset_dir, directory, subfolder))

    return folder_list


# -------------------------------------------- Image IO Functions ---------------------------------------------------- #
def load_image(image_path):
    """Loads images found in ``image_path``

    Parameters
    ----------
    image_path : `str`
        Path to look for image files

    Returns
    ----------
    image : `array_like`
        Loaded image
    """
    return skio.imread(image_path)


def load_ground_truth(folder_path, image_ext='*.tif', binary_operations=None):
    """Loads ground truth images found in ``image_path`` and performs erosion/dilation/inversion if needed

    Parameters
    ----------
    folder_path : `str`
        Folder to look for ground truth images
    image_ext : `str`, optional
        [Default: '*.tif'] File extension of ground truth image file
    binary_operations : func, optional
        Function that accepts an image as an input and performs a binary operation on it

    Returns
    ----------
    output_ground_truth : `array_like`
        Stacked array of ground truth images found in the directory with the given file extension [h,w,c]

    class_ids : `list`
        List of class ids of the ground truth images
    """
    image_list = list_images(folder_path, image_ext=image_ext)

    output_ground_truth = []
    class_ids = []

    if len(image_list) == 1:
        # Load image
        output_ground_truth = skio.imread(image_list[0])
        output_ground_truth = np.asarray(output_ground_truth)
        if output_ground_truth.ndim == 2:
            output_ground_truth = output_ground_truth[..., np.newaxis]
        else:
            output_ground_truth = np.moveaxis(output_ground_truth, 0, -1)

    else:
        for ground_truth_path in image_list:
            # Load image
            ground_truth_img = skio.imread(ground_truth_path)

            # takes list of operations and operates on ground_truth image
            if binary_operations is not None:
                assert isinstance(binary_operations, (list, str))
                for operation in binary_operations:
                    ground_truth_img = operation(ground_truth_img)

            output_ground_truth.append(ground_truth_img)

        output_ground_truth = np.asarray(output_ground_truth)
        output_ground_truth = np.moveaxis(output_ground_truth, 0, -1)

        # create background class
        if output_ground_truth.shape[-1] > 1:
            background = (np.sum(output_ground_truth, axis=-1, keepdims=True) == 0)
            output_ground_truth = np.concatenate((output_ground_truth, background), axis=-1)

    return output_ground_truth, class_ids


def save_image(image, image_path, model_dir=None,
               subfolder='Masks', output_subfolder=None, suffix='-preds',
               save_dtype='float32', check_contrast=True):
    """Saves image to image_path

    Final location of image is as follows:
      - image_path
          - subfolder
             - model/weights file name

    Parameters
    ----------
    image : `array_like`
        Image to be saved
    image_path : `str`
        Location to save the image in
    model_dir : `str`
        [Default: None] Path to obtain name of model
    subfolder : `str`
        [Default: 'Masks'] Subfolder in which the image is to be saved in
    output_subfolder  : `str`
        [Default: None] Additional subfolder to facilitate the organization of output files
    suffix : `str`
        [Default: '-preds'] Suffix to append to the filename of the predicted image
    save_dtype : `str`
        [Default: 'float32'] Specify dtype of saved image
    check_contrast : `bool`
        [Default: True] Specify if contract checking should be done for the saved image
    """
    image_dir = os.path.dirname(image_path)

    output_dir = os.path.join(image_dir, subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    basename, _ = os.path.splitext(os.path.basename(model_dir))

    if output_subfolder is not None:
        output_dir = os.path.join(output_dir, basename, output_subfolder)
    else:
        output_dir = os.path.join(output_dir, basename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename, _ = os.path.splitext(os.path.basename(image_path))

    output_path = os.path.join(output_dir, "{}{}.tif".format(filename, suffix))

    if save_dtype == 'uint8':
        image = skimage.util.img_as_ubyte(image)
    if save_dtype == 'uint16':
        image = skimage.util.img_as_uint(image)
    elif save_dtype == 'float32':
        image = skimage.util.img_as_float32(image)

    skimage.io.imsave(output_path, image, check_contrast=check_contrast, compress=6)


# -------------------------------------------- Image  Functions ---------------------------------------------------- #
def binary_erosion_function(disk_size):
    def helper_function(image):
        from skimage.morphology import binary_erosion, disk
        # sets dtype back to unsigned integer in order for some augmentations to work
        image_dtype = image.dtype
        image = binary_erosion(image, disk(disk_size))
        image = image.astype(image_dtype)
        return image

    return helper_function


def binary_dilation(disk_size):
    def helper_function(image):
        from skimage.morphology import binary_dilation, disk
        # sets dtype back to unsigned integer in order for some augmentations to work
        image_dtype = image.dtype
        image = binary_dilation(image, disk(disk_size))
        image = image.astype(image_dtype)
        return image

    return helper_function


def reshape_image(image):
    """Reshapes the image to the correct dimenstions for Unet

    Parameters
    ----------
    image : `array_like`
        Image to be reshaped

    Returns
    ----------
    image : `array_like`
        Reshaped image 
    """
    image_shape = tuple(image.shape)
    image = np.reshape(image, image_shape + (-1,))
    return image


def resize_image(image, resize_factor):
    image_shape = image.shape
    if image.ndim == 3:
        image = skimage.transform.resize(image, (image_shape[0], image_shape[1] // resize_factor,
                                                 image_shape[2] // resize_factor))
    else:
        image = skimage.transform.resize(image, (image_shape[0] // resize_factor, image_shape[1] // resize_factor))
    return image


def adjust_log(image):
    image_min = np.amin(image)
    image_max = np.amax(image)

    if (image_max - image_min) == 0:
        return image
    else:
        image = np.log10(image, where=(image > 0)) * (image_max - image_min) / np.log10((image_max - image_min))
        #     image = skimage.exposure.adjust_log(image)
        return image


# --------------------------------------- Image Normalization Functions ---------------------------------------------- #
def percentile_normalization(image, in_bound=(3, 99.8), output_min_max=False, clip_values=False):
    """Performs percentile normalization on the image

    Parameters
    ----------
    image : `array_like`
        Image to be normalized
    in_bound : `list`
        Upper and lower percentile used to normalize image

    Returns
    ----------
    image : `array_like`
        Normalized image

    image_min : `int`
        Min value of ``image``

    image_max : `int`
        Max value of ``image``
    """
    image_min = np.percentile(image, in_bound[0])
    image_max = np.percentile(image, in_bound[1])
    image = (image - image_min) / (image_max - image_min)

    if clip_values is True:
        image = np.clip(image, 0, 1)

    if output_min_max is True:
        return image, image_min, image_max
    else:
        return image


def min_max_normalization(image):
    image_min = np.amin(image)
    image_max = np.amax(image)
    image = (image - image_min) / (image_max - image_min)

    return image


# -------------------------------------------- Padding Functions ---------------------------------------------------- #
def pad_image(image, image_size, mode='reflect', channels_first=True):
    """Pad image to specified image_size

    Parameters
    ----------
    image : `array_like`
        Image to be padded
    image_size : `list`
        Final size of padded image
    mode : `str`, optional
        [Default: 'constant'] Mode to pad the image

    Returns
    ----------
    image : `array_like`
        Padded image

    padding : `list`
        List containing the number of pixels padded to each direction
    """
    if image.ndim == 3:
        if channels_first==True:
            h, w = image.shape[1:3]
        else:
            h, w = image.shape[:2]
    else:
        h, w = image.shape[0:2]

    top_pad = (image_size[0] - h) // 2
    bottom_pad = image_size[0] - h - top_pad

    left_pad = (image_size[1] - w) // 2
    right_pad = image_size[1] - w - left_pad

    if image.ndim == 3:
        if channels_first==True:
            padding = ((0, 0), (top_pad, bottom_pad), (left_pad, right_pad))
        else:
            padding = ((top_pad, bottom_pad), (left_pad, right_pad), (0,0))
    else:
        padding = ((top_pad, bottom_pad), (left_pad, right_pad))
    image = np.pad(image, padding, mode=mode)

    return image, padding


def remove_pad_image(image, padding):
    """Removes pad from image

    Parameters
    ----------
    image : `array_like`
        Padded image
    padding : `list`
        List containing the number of padded pixels in each direction

    Returns
    ----------
    image : `array_like`
        Image without padding
    """

    if len(padding) == 2:
        h, w = image.shape[:2]
        return image[padding[0][0]:h - padding[0][1], padding[1][0]:w - padding[1][1]]
    if len(padding) == 3:
        h, w = image.shape[1:3]
        return image[:, padding[1][0]:h - padding[1][1], padding[2][0]:w - padding[2][1]]


# --------------------------------------------- Tiling Functions ----------------------------------------------------- #
def tile_image(image, tile_size, tile_overlap_size, channels_first=True):
    """Converts an image into a list of tiled images

    Parameters
    ----------
    image : `array_like`
        Image to be tiled
    tile_size : `list`
        Size of each individual tile
    tile_overlap_size : `list`
        Amount of overlap (in pixels) between each tile

    Returns
    ----------
    image : `array_like`
        Image without padding
    """
    if image.ndim == 3:
        if channels_first == True:
            image_height, image_width = image.shape[1:3]
        else:
            image_height, image_width = image.shape[:2]
    else:
        image_height, image_width = image.shape[:2]
    tile_height_without_overlap = tile_size[0] - tile_overlap_size[0]
    tile_width_without_overlap = tile_size[1] - tile_overlap_size[1]

    if image_height <= tile_size[0] and image_width <= tile_size[1]:
        print("image_height is less than tile_height or image_width is less than tile_width. Returning original image.")
        return image

    num_rows = math.ceil((image_height - tile_overlap_size[0]) / tile_height_without_overlap)
    num_cols = math.ceil((image_width - tile_overlap_size[1]) / tile_width_without_overlap)
    num_tiles = num_rows * num_cols

    # pad image to fit tile size
    padding_height = (num_rows - 1) * tile_height_without_overlap + tile_size[0]
    padding_width = (num_cols - 1) * tile_width_without_overlap + tile_size[1]
    
    if image.ndim == 3:
        image, padding = pad_image(image, [padding_height, padding_width], channels_first=channels_first)
    else:
        image, padding = pad_image(image, [padding_height, padding_width])

    tile_image_list = []

    for tile_no in range(num_tiles):
        tile_x_start = (tile_no % num_cols) * tile_width_without_overlap
        tile_x_end = tile_x_start + tile_size[1]

        tile_y_start = (tile_no // num_cols) * tile_height_without_overlap
        tile_y_end = tile_y_start + tile_size[0]

        if image.ndim == 3:
            if channels_first == True:
                tiled_image = image[:,tile_y_start: tile_y_end, tile_x_start:tile_x_end]
            else:
                tiled_image = image[tile_y_start: tile_y_end, tile_x_start:tile_x_end, :]
        else:
            tiled_image = image[tile_y_start: tile_y_end, tile_x_start:tile_x_end]

        tiled_image = reshape_image_dimensions(tiled_image, channels_first=channels_first)

        tile_image_list.append(tiled_image)

    return tile_image_list, num_rows, num_cols, padding


def reshape_image_dimensions(image, channels_first=True):
    """ Reshape image dimensions in order to the same shape as the expected NN input

    :param image:
    :return:
    """
    if image.ndim == 3:
        if channels_first == True:
            image = np.moveaxis(image, 0, -1)
        image = image[np.newaxis, ...]
    else:
        image = image[np.newaxis, ..., np.newaxis]

    return image


def weights_array(tile_size, tile_overlap_size, exclude_side=''):
    """ Generate weights array for blending of tiles
    
    Notes:
        tile_overlap_size must be less than half of the corresponding tile_size
    """
    # assert tile_overlap_size[0] <= tile_size[0]/2
    # assert tile_overlap_size[1] <= tile_size[1]/2

    if 'l' in exclude_side:
        left_gradient = np.linspace(1, 1, tile_overlap_size[0])
    else:
        left_gradient = np.linspace(0, 1, tile_overlap_size[0])

    if 'r' in exclude_side:
        right_gradient = np.linspace(1, 1, tile_overlap_size[0])
    else:
        right_gradient = np.linspace(1, 0, tile_overlap_size[0])

    if 't' in exclude_side:
        top_gradient = np.linspace(1, 1, tile_overlap_size[0])
    else:
        top_gradient = np.linspace(0, 1, tile_overlap_size[0])

    if 'b' in exclude_side:
        bottom_gradient = np.linspace(1, 1, tile_overlap_size[0])
    else:
        bottom_gradient = np.linspace(1, 0, tile_overlap_size[0])

    x = np.concatenate((top_gradient, np.linspace(1, 1, tile_size[1] - tile_overlap_size[1] * 2), bottom_gradient),
                       axis=0)
    y = np.concatenate((left_gradient, np.linspace(1, 1, tile_size[0] - tile_overlap_size[0] * 2), right_gradient),
                       axis=0)
    meshgrid = np.meshgrid(y, x)
    return np.multiply(meshgrid[0], meshgrid[1], dtype=np.float32)


def no_blend(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding):
    """Stitches a list of tiled images back into a single image

    Parameters
    ----------
    tile_list : `list`
        List of tiled images
    tile_size : `list`
        Size of each individual tile
    tile_overlap_size : `list`
        Amount of overlap (in pixels) between each tile
    num_rows : `int`
        Number of rows of tiles
    num_cols : `int`
        Number of cols of tiles
    padding : `list`
        Amount of padding used during tiling

    Returns
    ----------
    image : `array_like`
        Image without padding
    """
    image_height = (num_rows - 1) * (tile_size[0] - tile_overlap_size[0]) + tile_size[0]
    image_width = (num_cols - 1) * (tile_size[1] - tile_overlap_size[1]) + tile_size[1]
    image = np.zeros((image_height, image_width), dtype=np.float32)

    tile_mask = np.zeros(tile_size)

    start = int(tile_overlap_size[1] // 2)
    end = int(tile_size[0] - start)
    tile_mask[start:end, start:end] = 1

    for row in range(num_rows):
        for col in range(num_cols):
            tile = tile_list[row * num_cols + col]
            tile = np.reshape(tile, tile.shape[:2])
            tile = tile * tile_mask

            top_pad = row * (tile_size[0] - tile_overlap_size[0])
            bottom_pad = image_height - top_pad - tile_size[0]
            left_pad = col * (tile_size[1] - tile_overlap_size[1])
            right_pad = image_width - left_pad - tile_size[1]

            tile_expanded = np.pad(tile, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
            image = np.maximum(image, tile_expanded)

    image = remove_pad_image(image, padding=padding)
    return image


def no_blend_v2(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding):
    image_height = (num_rows - 1) * (tile_size[0] - tile_overlap_size[0]) + tile_size[0]
    image_width = (num_cols - 1) * (tile_size[1] - tile_overlap_size[1]) + tile_size[1]
    image = np.zeros((image_height, image_width), dtype=np.float32)

    for i, tile in enumerate(tile_list):
        if tile.shape != tile_size:
            tile = np.full(tile_size, tile)

            h, w = np.unravel_index(i, (num_rows, num_cols))
            center_h = tile_size[0] - tile_overlap_size[0]
            center_w = tile_size[1] - tile_overlap_size[1]
            half_overlap_h = tile_overlap_size[0] // 2
            half_overlap_w = tile_overlap_size[1] // 2

            image[(0 if h == 0 else h * center_h + half_overlap_h):
                  (image_height if h == (num_rows - 1) else (h + 1) * center_h + half_overlap_h),
                  (0 if w == 0 else w * center_w + half_overlap_w):
                  (image_width if w == (num_cols - 1) else (w + 1) * center_w + half_overlap_w)] = \
                tile[(0 if h == 0 else half_overlap_h): (tile_size[0] if h == ( num_rows - 1) else -half_overlap_h),
                     (0 if w == 0 else half_overlap_w): (tile_size[1] if w == (num_cols - 1) else -half_overlap_w)]

    image = remove_pad_image(image, padding=padding)
    return image


def merge_values(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding=None):
    image = np.zeros((num_rows, num_cols), dtype=np.float32)

    for i, tile in enumerate(tile_list):
        h, w = np.unravel_index(i, (num_rows, num_cols))
        image[h, w] = tile

    return image


def untile_function(method="linear_blend"):
    allowed_methods = ["linear_blend", "max_blend", "no_blend", "merge_values"]

    if method not in allowed_methods:
        raise ValueError("Method not found in allowed_methods")

    if method == "linear_blend":
        return linear_blend_v2
    elif method == "max_blend":
        return max_blend_v2
    elif method == "no_blend":
        return no_blend_v2
    elif method == "merge_values":
        return merge_values


def linear_blend_v2(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding):
    image_height = (num_rows - 1) * (tile_size[0] - tile_overlap_size[0]) + tile_size[0]
    image_width = (num_cols - 1) * (tile_size[1] - tile_overlap_size[1]) + tile_size[1]
    image = np.zeros((image_height, image_width), dtype=np.float32)

    standard_weights = weights_array(tile_size, tile_overlap_size, exclude_side='')

    for row in range(num_rows):
        for col in range(num_cols):
            tile = tile_list[row * num_cols + col]
            tile = np.squeeze(tile)

            exclude_side = ''
            if row == 0:
                exclude_side += 't'
            if row == num_rows - 1:
                exclude_side += 'b'
            if col == 0:
                exclude_side += 'l'
            if col == num_cols - 1:
                exclude_side += 'r'

            if exclude_side == '':
                tile = np.multiply(tile, standard_weights, dtype=np.float32)
            else:
                weights = weights_array(tile_size, tile_overlap_size, exclude_side=exclude_side)
                tile = np.multiply(tile, weights, dtype=np.float32)

            top_pad = row * (tile_size[0] - tile_overlap_size[0])
            left_pad = col * (tile_size[1] - tile_overlap_size[1])

            image[top_pad: top_pad + tile_size[0],
                  left_pad: left_pad + tile_size[1]] = image[top_pad: top_pad + tile_size[0],
                                                             left_pad: left_pad + tile_size[1]] + tile

    image = remove_pad_image(image, padding=padding)
    return image


def linear_blend(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding):
    image_height = (num_rows - 1) * (tile_size[0] - tile_overlap_size[0]) + tile_size[0]
    image_width = (num_cols - 1) * (tile_size[1] - tile_overlap_size[1]) + tile_size[1]
    image = np.zeros((image_height, image_width), dtype=np.float32)

    for row in range(num_rows):
        for col in range(num_cols):
            tile = tile_list[row * num_cols + col]
            tile = np.reshape(tile, tile.shape[:2])

            exclude_side = ''
            if row == 0:
                exclude_side += 't'
            if row == num_rows - 1:
                exclude_side += 'b'
            if col == 0:
                exclude_side += 'l'
            if col == num_cols - 1:
                exclude_side += 'r'

            weights = weights_array(tile_size, tile_overlap_size, exclude_side=exclude_side)
            tile = np.multiply(tile, weights, dtype=np.float32)

            top_pad = row * (tile_size[0] - tile_overlap_size[0])
            bottom_pad = image_height - top_pad - tile_size[0]
            left_pad = col * (tile_size[1] - tile_overlap_size[1])
            right_pad = image_width - left_pad - tile_size[1]

            tile_expanded = np.pad(tile, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
            image = np.add(image, tile_expanded)

    image = remove_pad_image(image, padding=padding)
    return image


def max_blend(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding):
    image_height = (num_rows - 1) * (tile_size[0] - tile_overlap_size[0]) + tile_size[0]
    image_width = (num_cols - 1) * (tile_size[1] - tile_overlap_size[1]) + tile_size[1]
    image = np.zeros((image_height, image_width), dtype=np.float32)
    for row in range(num_rows):
        for col in range(num_cols):
            tile = tile_list[row * num_cols + col]
            tile = np.reshape(tile, tile.shape[:2])

            top_pad = row * (tile_size[0] - tile_overlap_size[0])
            bottom_pad = image_height - top_pad - tile_size[0]
            left_pad = col * (tile_size[1] - tile_overlap_size[1])
            right_pad = image_width - left_pad - tile_size[1]

            tile_expanded = np.pad(tile, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
            image = np.maximum(image, tile_expanded)

    image = remove_pad_image(image, padding=padding)
    return image


# TODO: use numba for faster performance?
def max_blend_v2(tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding):
    image_height = (num_rows - 1) * (tile_size[0] - tile_overlap_size[0]) + tile_size[0]
    image_width = (num_cols - 1) * (tile_size[1] - tile_overlap_size[1]) + tile_size[1]
    image = np.full((image_height, image_width), -np.inf, dtype=np.float32)

    for row in range(num_rows):
        for col in range(num_cols):
            tile = tile_list[row * num_cols + col]
            tile = np.squeeze(tile)

            top_pad = row * (tile_size[0] - tile_overlap_size[0])
            left_pad = col * (tile_size[1] - tile_overlap_size[1])

            image[top_pad:top_pad + tile_size[0], left_pad: left_pad + tile_size[1]] = np.maximum(
                image[top_pad:top_pad + tile_size[0], left_pad: left_pad + tile_size[1]], tile)

    image = remove_pad_image(image, padding=padding)
    return image
