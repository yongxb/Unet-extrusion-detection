import tensorflow as tf
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, TowerContext
from tqdm import tqdm

from .NNBase import NNBase
from .image_functions import *


class CNNPredictionBase(NNBase):
    def __init__(self, model_dir=None, config_filepath=None, **kwargs):
        """Creates the base neural network class with basic functions

        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Folder where the model is stored
        config_filepath : `str`, optional
            [Default: None] File path to the config file
        **kwargs
            Parameters that are passed to :class:`network_config.Network_Config`

        Attributes
        ----------
        config : :class:`network_config.Network_Config`
            Network_config object containing the config and necessary functions
        """

        super().__init__(model_dir=model_dir, config_filepath=config_filepath, **kwargs)

    # -------------------------------------------- Prediction Functions ---------------------------------------------- #
    def build_prediction_graph(self):
        images = tf.placeholder(tf.float32, shape=[None, ] + self.get_parameter("input_size"), name='images')
        with TowerContext('', is_training=False):
            self.model.build_pred_graph(images)

    def create_predict_config(self, session_init):
        return PredictConfig(model=self, input_names=['images'], output_names=['prediction'], session_init=session_init)

    def create_prediction_model(self, weights_path):
        self.prediction_model = OfflinePredictor(self.create_predict_config(SmartInit(weights_path)))

    def predict_image(self, image):
        return self.prediction_model(image)

    def preprocess_image(self, image):
        if self.get_parameter("resize_factor") != 1:
            image = resize_image(image, self.get_parameter("resize_factor"))

        if self.get_parameter("use_log_adjust") is True:
            image = adjust_log(image)

        if self.get_parameter("use_percentile_normalization") is True:
            image = percentile_normalization(image, in_bound=self.get_parameter("percentile"))

        return image

    def predict_images_from_directory(self, image_dir, image_ext='*.tif', suffix=None, return_output=True):
        """Perform prediction on images found in ``image_dir``

        Parameters
        ----------
        image_dir : `str`
            Directory containing the images to perform prediction on
        image_ext : 'str'
            Regex to identify image files
        suffix : 'str'
            Suffix used to append to the end of the image file


        Returns
        ----------
        image : `array_like`
            Last image that prediction was performed on

        """
        image_list = list_images(image_dir, image_ext=image_ext)
        
        for image_path in image_list:
            image = load_image(image_path=image_path)
            image = self.preprocess_image(image)

            if self.get_parameter("use_tiling") is True:
                tile_image_list, num_rows, num_cols, padding = tile_image(image, self.get_parameter("patch_size"),
                                                                          self.get_parameter("overlap_size"))

                try:
                    tile_array = np.concatenate(tile_image_list, axis=0)
                    prediction_list = self.predict_image(tile_array)
                except tf.errors.ResourceExhaustedError:
                    prediction_list = []
                    for tile in tile_image_list:
                        prediction_list.append(self.predict_image(tile))

                untile_image_function = untile_function(method=self.get_parameter("untiling_method"))
                output_image = untile_image_function(prediction_list, self.get_parameter("patch_size"),
                                                     self.get_parameter("overlap_size"),
                                                     num_rows, num_cols, padding=padding)

            elif self.get_parameter("pad_image") is True:
                image, padding = pad_image(image, self.get_parameter("patch_size"), mode='reflect')

                input_image = reshape_image_dimensions(image)

                output_image = self.predict_image(input_image)[0]
                output_image = np.squeeze(output_image)

                output_image = remove_pad_image(output_image, padding=padding)

            else:
                input_image = reshape_image_dimensions(image)

                output_image = self.predict_image(input_image)
                output_image = np.squeeze(output_image)

            if self.get_parameter("resize_factor") != 1:
                output_image = resize_image(output_image, 1/self.get_parameter("resize_factor"))
            
            output_image = np.moveaxis(output_image, -1, 0)
            save_image(output_image, image_path, model_dir=self.get_parameter("model_dir"), suffix=suffix,
                       output_subfolder="output", save_dtype='uint8', check_contrast=False)
            
        if return_output is True:
            return output_image

    def predict_batch_from_directory(self, image_dir, batchsize=4, image_ext='*.tif', suffix=None, return_output=True, check_contrast=False):
        """Perform prediction on images found in ``image_dir``

        Parameters
        ----------
        image_dir : `str`
            Directory containing the images to perform prediction on
        image_ext : 'str'
            Regex to identify image files
        suffix : 'str'
            Suffix used to append to the end of the image file

        Returns
        ----------
        image : `array_like`
            Last image that prediction was performed on

        """
        image_list = list_images(image_dir, image_ext=image_ext)
        print(image_list)
        main_image = load_image(image_path=image_list[0])
        main_image = imadjust(main_image)

        channels = self.get_parameter("num_channels")
        total_image_no = main_image.shape[0]-channels+1
        no_batches = total_image_no // batchsize + 1
        output_image = np.zeros((total_image_no, main_image.shape[1], main_image.shape[2]), dtype=np.float32)
        
        image_size = self.get_parameter("patch_size")
        main_image, padding = pad_image(main_image, image_size, mode='reflect')
        main_image = np.moveaxis(main_image, 0, -1)
        
        for i in tqdm(range(no_batches)):
            no_images = np.min((total_image_no, (i+1)*batchsize)) - i*batchsize
            input_image = np.zeros((no_images, image_size[0], image_size[1], channels), dtype=np.float32)
            for j in range(no_images):
                current_index = i*batchsize + j
                input_image[j, ...] = main_image[..., current_index:current_index+channels]
        
            pred_image = self.predict_image(input_image)[0]
            pred_image = np.squeeze(pred_image, axis=3)
            pred_image = remove_pad_image(pred_image, padding=padding)

            output_image[i*batchsize:i*batchsize+no_images, ...] = pred_image

        save_image(output_image, os.path.join(image_dir, f"image.tif"), model_dir=self.get_parameter("model_dir"),
                   suffix=suffix, output_subfolder=None, save_dtype='uint8', check_contrast=check_contrast)

        if return_output is True:
            return output_image
