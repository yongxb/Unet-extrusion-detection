import scipy.signal
import scipy.ndimage
import numpy as np
import skimage

from tensorpack.dataflow import MultiThreadMapData, PrefetchDataZMQ, BatchData, MultiProcessRunnerZMQ, RNGDataFlow

from .image_functions import generate_file_list, percentile_normalization, generate_folder_list, \
    load_image, load_ground_truth, resize_image, adjust_log
from .network_config import get_parameter


# SVLS: see https://arxiv.org/pdf/2104.05788.pdf
gauss = 1/(np.sqrt(2*np.pi))*np.exp(-0.5 * np.array([1, 0, 1]) / np.square(1))
kernel = np.outer(gauss, gauss)
kernel[1, 1] = np.sum(kernel)-kernel[1, 1]
kernel = kernel/kernel[1, 1]


class Dataset2D:
    def __init__(self, config):
        self.config = config

    def dataflow(self, dataset_dir=None, num_proc=5, batch_size=32, is_train=True):
        # update dataset_dir if specified. If not, load dataset_dir from config file
        if dataset_dir is None:
            if is_train is True:
                dataset_dir = get_parameter(self.config, "dataset_dir")
            else:
                dataset_dir = get_parameter(self.config, "val_dataset_dir")
        
        if type(dataset_dir) != str:
            image_filelist = []
            ground_truth_folders = []
            for data_dir in dataset_dir:
                image_filelist = image_filelist + generate_file_list(data_dir, subfolder=get_parameter(self.config, "image_subfolder"))
                if get_parameter(self.config, "ground_truth_subfolder") != "None":
                    
                    ground_truth_folders = ground_truth_folders + generate_folder_list(data_dir,
                                                                                       subfolder=get_parameter(self.config, "ground_truth_subfolder"))
                else:
                    ground_truth_folders = "None"
        else:
            image_filelist = generate_file_list(dataset_dir, subfolder=get_parameter(self.config, "image_subfolder"))

            if get_parameter(self.config, "ground_truth_subfolder") != "None":
                ground_truth_folders = generate_folder_list(dataset_dir,
                                                            subfolder=get_parameter(self.config, "ground_truth_subfolder"))
            else:
                ground_truth_folders = "None"

        if is_train is True:
            data = Dataflow2D(image_filelist, ground_truth_folders,
                              use_log_adjust=get_parameter(self.config, "use_log_adjust"),
                              resize_factor=get_parameter(self.config, "resize_factor"),
                              invert_ground_truth=get_parameter(self.config, "invert_ground_truth"),
                              use_percentile_normalization=get_parameter(self.config, "use_percentile_normalization"),
                              percentile=get_parameter(self.config, "percentile"))

            data1 = MultiThreadMapData(
                data, num_thread=20,
                map_func=lambda dp: self.data_processing(dp[0], dp[1]),
                buffer_size=1000)

            data1 = PrefetchDataZMQ(data1, num_proc=num_proc)
            batch_data = BatchData(data1, batch_size, remainder=False)
        else:
            data = Dataflow2D(image_filelist, ground_truth_folders, shuffle=False,
                              use_log_adjust=get_parameter(self.config, "use_log_adjust"),
                              resize_factor=get_parameter(self.config, "resize_factor"),
                              invert_ground_truth=get_parameter(self.config, "invert_ground_truth"),
                              use_percentile_normalization=get_parameter(self.config, "use_percentile_normalization"),
                              percentile=get_parameter(self.config, "percentile"))

            data1 = MultiThreadMapData(
                data, num_thread=50,
                map_func=lambda dp: self.data_processing(dp[0], dp[1]),
                buffer_size=1000)

            data1 = MultiProcessRunnerZMQ(data1, num_proc=num_proc)
            batch_data = BatchData(data1, batch_size, remainder=True)

        return batch_data
    
    def augmentations(self, p=None, additional_targets=None):
        from albumentations import (
            RandomCrop, HorizontalFlip, VerticalFlip, CLAHE, RandomRotate90,
            Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, ElasticTransform,
            MotionBlur, MedianBlur, IAASharpen, RandomBrightnessContrast,
            OneOf, Compose
        )

        augmentation_list = []

        if get_parameter(self.config, "random_rotate") is True:
            augmentation_list.append(RandomRotate90(p=get_parameter(self.config, "random_rotate_p")))  # 0.9

        if get_parameter(self.config, "vertical_flip") is True:
            augmentation_list.append(VerticalFlip())

        if get_parameter(self.config, "horizontal_flip") is True:
            augmentation_list.append(HorizontalFlip())

        if get_parameter(self.config, "transpose") is True:
            augmentation_list.append(Transpose())

        if get_parameter(self.config, "blur_group") is True:
            blur_augmentation = []
            if get_parameter(self.config, "motion_blur") is True:
                blur_augmentation.append(MotionBlur(p=get_parameter(self.config, "motion_blur_p")))
            if get_parameter(self.config, "median_blur") is True:
                blur_augmentation.append(MedianBlur(blur_limit=get_parameter(self.config, "median_blur_limit"),
                                                    p=get_parameter(self.config, "median_blur_p")))
            if get_parameter(self.config, "blur") is True:
                blur_augmentation.append(
                    Blur(blur_limit=get_parameter(self.config, "blur_limit"), p=get_parameter(self.config, "blur_p")))
            augmentation_list.append(OneOf(blur_augmentation, p=get_parameter(self.config, "blur_group_p")))

        if get_parameter(self.config, "shift_scale_rotate") is True:
            augmentation_list.append(ShiftScaleRotate(shift_limit=get_parameter(self.config, "shift_limit"),
                                                      scale_limit=get_parameter(self.config, "scale_limit"),
                                                      rotate_limit=get_parameter(self.config, "rotate_limit"),
                                                      p=get_parameter(self.config, "shift_scale_rotate_p")))

        if get_parameter(self.config, "distortion_group") is True:
            distortion_augmentation = []
            if get_parameter(self.config, "optical_distortion") is True:
                distortion_augmentation.append(OpticalDistortion(p=get_parameter(self.config, "optical_distortion_p")))
            if get_parameter(self.config, "elastic_transform") is True:
                distortion_augmentation.append(ElasticTransform(p=get_parameter(self.config, "elastic_transform_p")))
            if get_parameter(self.config, "grid_distortion") is True:
                distortion_augmentation.append(GridDistortion(p=get_parameter(self.config, "grid_distortion_p")))

            augmentation_list.append(OneOf(distortion_augmentation, p=get_parameter(self.config, "distortion_group_p")))

        if get_parameter(self.config, "brightness_contrast_group") is True:
            contrast_augmentation = []
            if get_parameter(self.config, "clahe") is True:
                contrast_augmentation.append(CLAHE())
            if get_parameter(self.config, "sharpen") is True:
                contrast_augmentation.append(IAASharpen())
            if get_parameter(self.config, "random_brightness_contrast") is True:
                contrast_augmentation.append(RandomBrightnessContrast())

            augmentation_list.append(
                OneOf(contrast_augmentation, p=get_parameter(self.config, "brightness_contrast_group_p")))

        augmentation_list.append(
            RandomCrop(get_parameter(self.config, "patch_size")[0], get_parameter(self.config, "patch_size")[1],
                       always_apply=True))

        return Compose(augmentation_list, p=p, additional_targets=additional_targets)

    def data_processing(self, image, ground_truth=None, deep_supervision=True):

        augmentor = self.augmentations(p=get_parameter(self.config, "augmentations_p"))
        # target must be image and mask in order for albumentations to work
        data = {"image": image}

        if get_parameter(self.config, "ground_truth_subfolder") != "None":
            data["mask"] = ground_truth

        augmented = augmentor(**data)

        augmented_image = augmented["image"]
        inputs = [augmented_image]

        if get_parameter(self.config, "ground_truth_subfolder") != "None":
            ground_truth = np.ndarray.astype(augmented["mask"], np.bool)
            
            if deep_supervision==True:
                logits_4 = skimage.transform.downscale_local_mean(ground_truth, (2, 2, 1)) > 0
                shape = logits_4.shape
                logits_4 = scipy.signal.convolve2d(np.squeeze(logits_4), kernel, mode='same')/2
                logits_4 = np.reshape(logits_4, shape)
                
                logits_3 = skimage.transform.downscale_local_mean(ground_truth, (4, 4, 1)) > 0
                shape = logits_3.shape
                logits_3 = scipy.signal.convolve2d(np.squeeze(logits_3), kernel, mode='same')/2
                logits_3 = np.reshape(logits_3, shape)
                
                logits_2 = skimage.transform.downscale_local_mean(ground_truth, (8, 8, 1)) > 0
                shape = logits_2.shape
                logits_2 = scipy.signal.convolve2d(np.squeeze(logits_2), kernel, mode='same')/2
                logits_2 = np.reshape(logits_2, shape)
                
                logits_1 = skimage.transform.downscale_local_mean(ground_truth, (16, 16, 1)) > 0
                shape = logits_1.shape
                logits_1 = scipy.signal.convolve2d(np.squeeze(logits_1), kernel, mode='same')/2
                logits_1 = np.reshape(logits_1, shape)
                
                shape = ground_truth.shape
                ground_truth = scipy.signal.convolve2d(np.squeeze(ground_truth), kernel, mode='same')/2
                ground_truth = np.reshape(ground_truth, shape)
                
                inputs.extend([ground_truth, logits_1, logits_2, logits_3, logits_4])
            else:
                inputs.extend([ground_truth])

        return inputs


# TODO: validation dataset loading to be different from current setup
class Dataflow2D(RNGDataFlow):
    def __init__(self,
                 image_filelist, ground_truth_folders="None", num_images=64, random_crop=False, shuffle=True,
                 invert_ground_truth=False, use_percentile_normalization=True, percentile=(0, 100), use_log_adjust=True,
                 resize_factor=4):

        self.shuffle = shuffle
        self.num_images = num_images

        self.images = []
        self.ground_truth = []

        for i in range(len(image_filelist)):
            # loading of images
            loaded_image = load_image(image_filelist[i])
            if use_log_adjust is True:
                loaded_image = adjust_log(loaded_image)
            if use_percentile_normalization is True:
                loaded_image = percentile_normalization(loaded_image, in_bound=percentile)
                #  image = image*2-1

            if resize_factor > 1:
                loaded_image = resize_image(loaded_image, resize_factor)

            if loaded_image.ndim == 2:
                loaded_image = loaded_image[np.newaxis, ...]

            loaded_image = np.moveaxis(loaded_image, 0, -1)

            if random_crop is True:
                self.images.extend(self.random_crops(loaded_image))
            else:
                self.images.append(loaded_image)

            # loading of ground truth
            if ground_truth_folders != "None":
                ground_truth, _ = load_ground_truth(ground_truth_folders[i])

                if invert_ground_truth is True:
                    ground_truth = skimage.util.invert(ground_truth)

                if resize_factor > 1:
                    ground_truth = resize_image(ground_truth, resize_factor)

                self.ground_truth.append(ground_truth)

    def __len__(self):
        return self.num_images

    def __iter__(self):
        indexes = np.arange(len(self.images))
        if self.shuffle is True:
            self.rng.shuffle(indexes)

        # assumes a small dataset that can be loaded into memory
        for image_index in indexes:
            output = [self.images[image_index]]
            if self.ground_truth != []:
                output.extend([self.ground_truth[image_index]])
            yield output

    @staticmethod
    def random_crops(image, min_size=(512, 512)):
        h, w = image.shape[:2]

        rows = round(h / min_size[0])
        cols = round(w / min_size[1])
        num_crops = int(rows * cols)

        from albumentations import RandomCrop, Compose
        augmentor = Compose([RandomCrop(min_size[0], min_size[1], always_apply=True)], p=1)

        data = {"image": image}
        crop_list = []

        for _ in range(num_crops):
            augmented = augmentor(**data)
            crop_list.extend([augmented["image"]])

        return crop_list
