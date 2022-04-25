import tensorpack
from tensorpack import QueueInput
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.train import TrainConfig, launch_train_with_config, SimpleTrainer

from .NNBase import NNBase
from .losses import focal_loss, focal_tversky, unified_focal_loss
import tensorflow as tf


class CNNBase(NNBase):
    def __init__(self, model_dir=None, config_filepath=None, **kwargs):
        """Creates the base neural network class with basic functions
        
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

        super().__init__(model_dir=model_dir, config_filepath=config_filepath, **kwargs)

    # ------------------------------------------ Initialization Functions ------------------------------------------- #
    def get_train_config(self, model):
        data = QueueInput(self.dataset.dataflow(batch_size=self.get_parameter("batch_size_per_GPU")))

        self.default_callbacks()

        return TrainConfig(
            model=model,
            data=data,
            callbacks=self.callbacks,
            steps_per_epoch=self.get_parameter("num_train_steps"),
            session_init=self.session_init,
            max_epoch=self.get_parameter("num_epochs"),
        )

    # ----------------------------------------------- Loss Functions ------------------------------------------------ #
    def compute_loss(self, loss_fn, y_true, y_pred, **kwargs):
        return loss_fn(y_true, y_pred, **kwargs)

    def get_loss_function(self, loss=None):
        """Initialize loss function
        
        Parameters
        ----------
        loss : `str`
            Name of the loss function
            
        Returns
        ----------
        loss
            Function to call loss function
        """
        _loss = {"focal_tversky_loss": focal_tversky,
                 "unified_focal_loss": unified_focal_loss,
                 }

        if loss is None:
            loss = self.get_parameter("loss")

        return _loss[loss]

    # -------------------------------------------- Model Functions --------------------------------------------------- #
    def inputs(self):  # to change to multi-class
        inputs = [tf.TensorSpec([None, ] + self.get_parameter("image_input_size"), tf.float32, 'images')]

        if self.get_parameter("ground_truth_subfolder") != "None":
            inputs.extend([tf.TensorSpec([None, ] + self.get_parameter("gt_input_size"), tf.float32, 'ground_truth')])

        return inputs

    def build_graph(self, *args):
        list_inputs = self.input_names
        inputs = {}
        for arg, name in zip(args, list_inputs):
            inputs[name] = arg

        logits = self.backbone(inputs["images"])

        prediction = self.get_final_activation(logits)

        loss_fn = self.get_loss_function(self.get_parameter("loss"))

        loss = self.compute_loss(loss_fn, inputs["ground_truth"],
                                 logits if "logits" in self.get_parameter("loss") else prediction)

        total_cost = tf.identity(loss, name='cost')
        add_moving_summary(total_cost)

        return total_cost

    def get_final_activation(self, logits):
        activation_dict = {"sigmoid": tf.nn.sigmoid,
                           "relu": tf.nn.relu,
                           "softmax": tf.nn.softmax}

        final_activation_fn = activation_dict[self.get_parameter("final_activation")]

        return final_activation_fn(logits, name='prediction')

    # ---------------------------------------------- Train Model ----------------------------------------------------- #
    def train_model(self, verbose=True):
        """Trains model
        
        Parameters
        ----------
        verbose : `int`, optional
            [Default: True] Verbose output
        """
        trainer = SimpleTrainer()
        launch_train_with_config(self.model_config, trainer)

        self.end_training()
