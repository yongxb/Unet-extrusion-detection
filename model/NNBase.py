import platform
import os
import math
import subprocess
import datetime
import numpy as np

import tensorflow as tf
from tensorpack.train.model_desc import ModelDesc
from tensorpack.tfutils import sessinit
from tensorpack.callbacks import ModelSaver, GPUUtilizationTracker, PeakMemoryTracker, InjectShell, EstimatedTimeLeft, \
                                 StatMonitorParamSetter, PeriodicTrigger, HyperParamSetterWithFunc
from tensorpack.utils import logger

from .network_config import get_parameter, find_key, update_parameter, load_config_from_model_dir, load_config_from_file, write_config
from .Dataset2d import Dataset2D


_datasets = {"2D": Dataset2D}


class NNBase(ModelDesc):
    def __init__(self, model_dir=None, config_filepath=None, **kwargs):
        """Creates the base  neural network class with basic functions
        
        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Folder where the model is stored
        config_filepath : `str`, optional
            [Default: None] Filepath to the config file
        **kwargs
            Parameters that are passed to :class:`network_config.Network_Config`

        Attributes
        ----------
        config : :class:`network_config.Network_Config`
            Network_config object containing the config and necessary functions
        """

        super().__init__()

        self.initialize_config(model_dir=model_dir, config_filepath=config_filepath, **kwargs)
        self.session_init = None

        # save time that model is initialized
        self.update_parameter(["general", "now"], datetime.datetime.now())

        self.server_name = platform.node()
        self.update_parameter(["general", "server_name"], self.server_name)

        if self.get_parameter("use_cpu") is True:
            self.initialize_cpu()
            self.use_cpu = True
        else:
            self.initialize_gpu()
            self.use_cpu = False

        self.dataset = _datasets[self.get_parameter("image_type")](self.config)

    # ---------------------------------------------- Config Functions ------------------------------------------------ #
    def get_parameter(self, *args):
        return get_parameter(self.config, *args)

    def update_parameter(self, *args):
        return update_parameter(self.config, *args)

    def initialize_config(self, model_dir=None, config_filepath=None, **kwargs):
        """Initializes config that contains the network parameters and functions needed to manipulate these parameters.
    
        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Folder where the model is to be saved/read from
        config_filepath : `str`, optional
            [Default: None] Filepath to the config file that will be loaded
        **kwargs
            For network parameters that are to be changed from the loaded config file

        Attributes
        ----------
        yaml : :class:`ruamel.yaml.YAML`
            YAML class with function needed to read/write YAML files 
        config : `dict`
            Dictionary containing the config parameters
        """

        # load config file from model_dir
        if config_filepath is not None:
            self.config = load_config_from_file(config_filepath)
            logger.info(f"Loaded config file from {config_filepath}")
        else:
            try:
                self.config = load_config_from_model_dir(model_dir)
                logger.info(f"Loaded config file from {model_dir}")
            except IndexError:
                logger.error("Please ensure that config_filepath is set or there is a config file in model_dir")
                raise

        if model_dir is not None:
            # update model_dir in config
            logger.info(f"Updating model_dir to {model_dir}")
            self.update_parameter(["general", "model_dir"], model_dir)

        # overwrite network parameters with parameters given during initialization
        for key, value in kwargs.items():
            try:
                self.update_parameter(find_key(self.config, key), value)
            except AssertionError:
                logger.warning("Config parameter not found. Adding as a standalone key.")
                self.update_parameter([key], value)

        if self.get_parameter("pad_image") is True:
            # TODO to move factor to model definition
            factor = 32

            image_size = self.get_parameter("patch_size")

            height = math.ceil(image_size[0] / factor) * factor
            width = math.ceil(image_size[1] / factor) * factor

            self.update_parameter(["images", "patch_size"], [height, width])

        # perform calculations
        self.update_parameter(["model", "image_input_size"],
                              self.get_parameter("patch_size") + [self.get_parameter("num_channels"), ])
        self.update_parameter(["model", "gt_input_size"], self.get_parameter("patch_size") + [
            1 if self.get_parameter("num_classes") == 1 else (self.get_parameter("num_classes") + 1), ])
        self.update_parameter(["model", "batch_size"], self.get_parameter("batch_size_per_GPU"))  # * self.gpu_count

    # ------------------------------------------------- Properties --------------------------------------------------- #
    @property
    def model(self):
        return self

    # ------------------------------------------------ Log Functions ------------------------------------------------- #
    def init_logs(self):
        """Initiates the parameters required for the log file
        """
        # Directory for training logs
        logger.info("Training run: {}-{:%Y%m%dT%H%M}".format(self.get_parameter("name"), self.get_parameter("now")))
        if self.get_parameter("folder_suffix") is None:
            self.log_dir = os.path.join(self.get_parameter("model_dir"),
                                        "{}-{:%Y%m%dT%H%M}".format(self.get_parameter("name"),
                                                                   self.get_parameter("now")))
        else:
            self.log_dir = os.path.join(self.get_parameter("model_dir"),
                                        "{}-{:%Y%m%dT%H%M}{}".format(self.get_parameter("name"),
                                                                     self.get_parameter("now"),
                                                                     self.get_parameter("folder_suffix")))
        logger.set_logger_dir(self.log_dir)

    def write_logs(self):
        """Writes the log file
        """
        # Create log_dir if it does not exist
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)

        # save the parameters used in current run to logs dir
        write_config(self.config, os.path.join(self.log_dir,
                                               "{}-{:%Y%m%dT%H%M}-config.yml".format(self.get_parameter("name"),
                                                                                     self.get_parameter("now"))))

    # ------------------------------------------ Initialization Functions -------------------------------------------- #
    @property
    def summary(self):
        """Summary of the layers in the model
        """
        return self.model.summary()

    @staticmethod
    def initialize_cpu():
        """Sets the session to only use the CPU
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # needs to be a string

    @staticmethod
    def get_free_gpu():
        """Selects the gpu with the most free memory
        """
        output = subprocess.Popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free', stdout=subprocess.PIPE,
                                  shell=True).communicate()[0]
        output = output.decode("ascii")
        mem_gpu = output.strip().split("\n")
        memory_available = [int(x.split()[2]) for x in mem_gpu]
        if not memory_available:
            return

        logger.info("Setting GPU to use to PID {}".format(np.argmax(memory_available)))
        return np.argmax(memory_available)

    def initialize_gpu(self):
        """Sets the seesion to use the gpu specified in config file
        """
        gpu = self.get_parameter("visible_gpu")
        if gpu == "None":
            gpu = self.get_free_gpu()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)  # needs to be a string

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = self.get_parameter("allow_growth")

    def initialize_model(self):
        """Initializes the logs, builds the model, and chooses the correct initialization function
        """
        # write parameters to yaml file
        self.init_logs()

        # build model
        self.model_config = self.get_train_config(self.model)

        logger.info("{} using single {}.".format("Predicting" if self.get_parameter("for_prediction") else "Training",
                                                 "CPU" if self.use_cpu is True else "GPU"))

    def load_session(self, model_path, ignore=()):
        self.session_init = sessinit.SaverRestore(model_path, ignore=ignore)

    def load_dictionary(self, dict_path):
        self.session_init = sessinit.SmartInit(dict_path)

    # --------------------------------------------- Optimizer Functions ---------------------------------------------- #
    def optimizer(self):
        lr = tf.compat.v1.get_variable('learning_rate', initializer=self.get_parameter("learning_rate"), trainable=False)
        tf.compat.v1.summary.scalar('learning_rate-summary', lr)
        opt = self.get_optimizer_function(lr)

        return opt

    def get_optimizer_function(self, lr):
        optimizer_function = self.get_parameter("optimizer_function")
        logger.info("Using {} as optimizer with lr={}".format(optimizer_function, self.get_parameter("learning_rate")))
        optimizer_function = optimizer_function.lower()
        _optimizers = {"rmsprop": tf.compat.v1.train.RMSPropOptimizer(lr, self.get_parameter("momentum")),
                       "adam": tf.compat.v1.train.AdamOptimizer(lr),
                       'proxada': tf.compat.v1.train.ProximalAdagradOptimizer(lr, initial_accumulator_value=0.1, l1_regularization_strength=0.0,
                                                                              l2_regularization_strength=0.0, use_locking=False, name='ProximalAdagrad'
                                                                              ),
                       }

        return _optimizers[optimizer_function]

    # --------------------------------------------- Callbacks Functions ---------------------------------------------- #
    def default_callbacks(self):
        if self.get_parameter("for_prediction") is False:
            self.callbacks = [
                # save the model every epoch
                PeriodicTrigger(ModelSaver(), every_k_steps=5000),  # PeriodicTrigger
                # record GPU utilization during training
                GPUUtilizationTracker(),
                PeakMemoryTracker(),
                # touch a file to pause the training and start a debug shell, to observe what's going on
                InjectShell(shell='ipython'),
                # estimate time until completion
                EstimatedTimeLeft(),
            ]

        if self.get_parameter("cosine_LR") is True:
            self.callbacks.append(HyperParamSetterWithFunc('learning_rate', lambda e, x: max(0.5*(1+math.cos(e*math.pi/self.get_parameter("num_epochs")))*self.get_parameter("learning_rate"), 1e-6)))
            
        if self.get_parameter("reduce_LR_on_plateau") is True:
            min_lr = self.get_parameter("reduce_LR_min_lr")
            self.callbacks.append(
                StatMonitorParamSetter('learning_rate', self.get_parameter("reduce_LR_monitor"),
                                       lambda x: x * self.get_parameter("reduce_LR_factor") if x > min_lr else min_lr,
                                       threshold=0, last_k=self.get_parameter("reduce_LR_patience")))

    # ---------------------------------------------- Training Functions ---------------------------------------------- #
    @staticmethod
    def end_training():
        """Deletes model and releases gpu memory held by tensorflow
        """
        # clear memory
        tf.compat.v1.reset_default_graph()

        # take hold of cuda device to shut it down
        from numba import cuda
        cuda.select_device(0)
        cuda.close()
