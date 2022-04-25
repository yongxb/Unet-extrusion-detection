import tensorflow.compat.v1 as tf
from tensorpack.models import BNReLU, Conv2D, MaxPooling, Conv2DTranspose, ConcatWith, AvgPooling
from tensorpack.models.regularize import Dropout
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.summary import add_moving_summary

from .CNNBase import CNNBase
from .CNNPredictionBase import CNNPredictionBase
from .layers import batchnorm_swish, batchnorm_function, concat_block


class AttentionResnetBuilder:
    # see https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/resnet_model.py

    @staticmethod
    def reshape_shortcut(shortcut, layer, stride=1):
        # assumes 'NHWC'
        filters = layer.get_shape().as_list()[-1]

        if stride == 2:  # change dimension when channel is not the same
            shortcut = AvgPooling("convavg", shortcut, 2)
            return Conv2D('convshortcut', shortcut, filters, 1, stride=1,
                          activation=batchnorm_function(), use_bias=False)
        elif layer.get_shape().as_list() != shortcut.get_shape().as_list():
            return Conv2D('convshortcut', shortcut, layer.get_shape().as_list()[-1], 1, stride=1,
                          activation=batchnorm_function(), use_bias=False)
        else:
            return shortcut

    # --------------------------------------------------- Blocks ----------------------------------------------------- #
    def bottleneck_block(self, layer, features, layer_no, stride=1, dropout_rate=0.2):
        shortcut_layer = layer
        bn_fn = batchnorm_swish if self.get_parameter("activation_function") == "swish" else BNReLU

        layer = BNReLU('preact', layer)
        layer = Conv2D('conv1', layer, features, 1, activation=bn_fn, use_bias=False)
        layer = Conv2D('conv2', layer, features, 3, stride=stride, activation=bn_fn, use_bias=False)
        layer = Conv2D('conv3', layer, features * 4, 1, activation=None, use_bias=False)

        layer = self.SE_block(layer, features, features * 4, 0.25)

        if dropout_rate > 0:
            layer = Dropout(layer, rate=dropout_rate)

        layer = layer + self.reshape_shortcut(shortcut_layer, layer, stride)
        return layer

    def bottleneck_group(self, name, layer, features, layer_count, stride, **kwarg):
        with tf.variable_scope(name):
            for i in range(0, layer_count):
                with tf.variable_scope(f'block{i}'):
                    layer = self.bottleneck_block(layer, features, i, stride if i == 0 else 1, **kwarg)
        return layer

    def simple_block(self, name, layer, features, stride, dropout_rate=0.2):
        bn_fn = batchnorm_swish if self.get_parameter("activation_function") == "swish" else BNReLU

        with tf.variable_scope(name):
            shortcut_layer = layer

            layer = BNReLU('preact', layer)
            layer = Conv2D('conv1', layer, features, 3, activation=bn_fn, use_bias=False)

            if dropout_rate > 0:
                layer = Dropout(layer, rate=dropout_rate)

            layer = layer + self.reshape_shortcut(shortcut_layer, layer, 1)
        return layer

    @staticmethod
    def attention_gate(input_layer, gate_layer):
        # See https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/grid_attention_layer.py
        # https://arxiv.org/pdf/1804.03999.pdf

        input_layer_shape = tf.shape(input_layer)
        inter_filters = input_layer.get_shape().as_list()[-1] // 2

        theta_x = Conv2D('input_signal', input_layer, inter_filters, (2, 2), strides=(2, 2), use_bias=False)

        phi_g = Conv2D('gating_signal', gate_layer, inter_filters, (1, 1), strides=(1, 1), use_bias=True)

        layer = tf.add(phi_g, theta_x)
        layer = tf.nn.relu(layer)

        layer = Conv2D('psi_layer', layer, 1, (1, 1), strides=(1, 1), use_bias=True)
        layer = tf.nn.sigmoid(layer)

        layer = tf.image.resize_images(layer, [input_layer_shape[1], input_layer_shape[2]], align_corners=True)

        layer = layer * input_layer
        return layer

    def attention_block(self, name, input_layer, gate_layer):
        with tf.variable_scope(name):
            filters = input_layer.get_shape().as_list()[-1]

            layer = self.attention_gate(input_layer, gate_layer)

            if layer.get_shape().as_list()[1:2] != gate_layer.get_shape().as_list()[1:2]:
                gate_layer = Conv2DTranspose('conv_up', gate_layer, filters, 2, stride=2)

            layer_1 = ConcatWith(layer, gate_layer, -1)
            return layer_1

    def SE_block(self, layer, input_filters, output_filters, se_ratio):
        se_filters = max(1, int(input_filters * se_ratio))
        se_tensor = tf.reduce_mean(layer, axis=(1, 2), keepdims=True)
        se_tensor = Conv2D('SE_conv1', se_tensor, se_filters, 1, use_bias=False)
        se_tensor = tf.nn.relu(se_tensor)
        se_tensor = Conv2D('SE_conv2', se_tensor, output_filters, 1, use_bias=False)
        layer = tf.nn.sigmoid(se_tensor) * layer
        return layer

    # ----------------------------------------------- Model Backbone ------------------------------------------------- #
    def attention_unet_backbone(self, image):

        # get parameters from config file
        filters = self.get_parameter("filters")
        dropout_rate = self.get_parameter("dropout_rate")
        layer_count = self.get_parameter("conv_layer")

        bn_fn = batchnorm_swish if self.get_parameter("activation_function") == "swish" else BNReLU

        with argscope(Conv2D, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            layer = Conv2D('down1', image, filters, 3, activation=bn_fn, use_bias=False)
            layer_store = [layer]

            layer = MaxPooling("maxpool", layer, 2)
            layer = self.simple_block('down2', layer, filters, 1, dropout_rate=dropout_rate)
            layer_store.append(layer)

            layer = self.bottleneck_group('down3', layer, filters, layer_count[0], 2, dropout_rate=dropout_rate)
            filters = filters * 2
            layer_store.append(layer)

            layer = self.bottleneck_group('down4', layer, filters, layer_count[1], 2, dropout_rate=dropout_rate)
            filters = filters * 2
            layer_store.append(layer)

            layer = self.bottleneck_group('down5', layer, filters, layer_count[2], 2, dropout_rate=dropout_rate)
            filters = filters * 2
            layer_store.append(layer)

            layer = self.bottleneck_group('down6', layer, filters, layer_count[3], 2, dropout_rate=dropout_rate)

            filters = filters // 2
            layer = self.attention_block('attention1', layer_store[-1], layer)
            layer = self.bottleneck_group('up1', layer, filters, layer_count[2], 1, dropout_rate=dropout_rate)
            last_layer_filter = self.get_parameter("num_classes")
            logits_1 = BNReLU('logits1_preact', layer)
            logits_1 = Conv2D('logits1', logits_1, 1 if last_layer_filter == 1 else (last_layer_filter + 1), 1)

            filters = filters // 2
            layer = self.attention_block('attention2', layer_store[-2], layer)
            layer = self.bottleneck_group('up2', layer, filters, layer_count[1], 1, dropout_rate=dropout_rate)
            logits_2 = BNReLU('logits2_preact', layer)
            logits_2 = Conv2D('logits2', logits_2, 1 if last_layer_filter == 1 else (last_layer_filter + 1), 1)

            filters = filters // 2
            layer = self.attention_block('attention3', layer_store[-3], layer)
            layer = self.bottleneck_group('up3', layer, filters, layer_count[0], 1, dropout_rate=dropout_rate)
            logits_3 = BNReLU('logits_3_preact', layer)
            logits_3 = Conv2D('logits3', logits_3, 1 if last_layer_filter == 1 else (last_layer_filter + 1), 1)

            layer = self.attention_block('attention4', layer_store[-4], layer)
            layer = self.simple_block('up4', layer, filters, 1, dropout_rate=dropout_rate)
            logits_4 = BNReLU('logits_4_preact', layer)
            logits_4 = Conv2D('logits4', logits_4, 1 if last_layer_filter == 1 else (last_layer_filter + 1), 1)

            layer = Conv2DTranspose('conv_up', layer, filters, 2, stride=2, activation=bn_fn)
            layer = concat_block('concat5', layer_store[-5], layer)
            #             layer = self.attention_block('attention5', layer_store[-5], layer)
            layer = Conv2D('up5', layer, filters, 3, activation=bn_fn, use_bias=False)

            logits_final = Conv2D('logits_final', layer, 1 if last_layer_filter == 1 else (last_layer_filter + 1), 1)

        return logits_1, logits_2, logits_3, logits_4, logits_final

    def attention_unet_pred_backbone(self, image):

        # get parameters from config file
        filters = self.get_parameter("filters")
        dropout_rate = self.get_parameter("dropout_rate")
        layer_count = self.get_parameter("conv_layer")

        bn_fn = batchnorm_swish if self.get_parameter("activation_function") == "swish" else BNReLU

        with argscope(Conv2D, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            layer = Conv2D('down1', image, filters, 3, activation=bn_fn, use_bias=False)
            layer_store = [layer]

            layer = MaxPooling("maxpool", layer, 2)
            layer = self.simple_block('down2', layer, filters, 1, dropout_rate=dropout_rate)
            layer_store.append(layer)

            layer = self.bottleneck_group('down3', layer, filters, layer_count[0], 2, dropout_rate=dropout_rate)
            filters = filters * 2
            layer_store.append(layer)

            layer = self.bottleneck_group('down4', layer, filters, layer_count[1], 2, dropout_rate=dropout_rate)
            filters = filters * 2
            layer_store.append(layer)

            layer = self.bottleneck_group('down5', layer, filters, layer_count[2], 2, dropout_rate=dropout_rate)
            filters = filters * 2
            layer_store.append(layer)

            layer = self.bottleneck_group('down6', layer, filters, layer_count[3], 2, dropout_rate=dropout_rate)

            filters = filters // 2
            layer = self.attention_block('attention1', layer_store[-1], layer)
            layer = self.bottleneck_group('up1', layer, filters, layer_count[2], 1, dropout_rate=dropout_rate)

            filters = filters // 2
            layer = self.attention_block('attention2', layer_store[-2], layer)
            layer = self.bottleneck_group('up2', layer, filters, layer_count[1], 1, dropout_rate=dropout_rate)

            filters = filters // 2
            layer = self.attention_block('attention3', layer_store[-3], layer)
            layer = self.bottleneck_group('up3', layer, filters, layer_count[0], 1, dropout_rate=dropout_rate)

            layer = self.attention_block('attention4', layer_store[-4], layer)
            layer = self.simple_block('up4', layer, filters, 1, dropout_rate=dropout_rate)

            layer = Conv2DTranspose('conv_up', layer, filters, 2, stride=2, activation=bn_fn)
            layer = concat_block('concat5', layer_store[-5], layer)
            layer = Conv2D('up5', layer, filters, 3, activation=bn_fn, use_bias=False)

            last_layer_filter = self.get_parameter("num_classes")
            logits_final = Conv2D('logits_final', layer, 1 if last_layer_filter == 1 else (last_layer_filter + 1), 1, )

        return logits_final


class AttentionUnetModel(CNNBase, CNNPredictionBase, AttentionResnetBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.backbone = self.attention_unet_backbone

    def inputs(self):  # to change to multi-class
        inputs = [tf.TensorSpec([None, ] + self.get_parameter("image_input_size"), tf.float32, 'images')]

        if self.get_parameter("ground_truth_subfolder") != "None":
            inputs.extend([
                tf.TensorSpec([None, ] + self.get_parameter("gt_input_size"), tf.float32, 'ground_truth'),
                tf.TensorSpec([None, ] + [max(1, x // 16) for x in self.get_parameter("gt_input_size")], tf.float32,
                              'logits_1'),
                tf.TensorSpec([None, ] + [max(1, x // 8) for x in self.get_parameter("gt_input_size")], tf.float32,
                              'logits_2'),
                tf.TensorSpec([None, ] + [max(1, x // 4) for x in self.get_parameter("gt_input_size")], tf.float32,
                              'logits_3'),
                tf.TensorSpec([None, ] + [max(1, x // 2) for x in self.get_parameter("gt_input_size")], tf.float32,
                              'logits_4'),
            ])

        return inputs

    def build_graph(self, *args):
        list_inputs = self.input_names
        inputs = {}
        for arg, name in zip(args, list_inputs):
            inputs[name] = arg

        logits_1, logits_2, logits_3, logits_4, logits_final = self.backbone(inputs["images"])

        #         prediction = self.get_final_activation(logits_final)

        loss_fn = self.get_loss_function(self.get_parameter("loss"))

        gt_loss = tf.identity(
            self.compute_loss(loss_fn, inputs["ground_truth"], self.get_final_activation(logits_final)), name='gt_loss')
        logits_1_loss = tf.identity(self.compute_loss(loss_fn, inputs["logits_1"], self.get_final_activation(logits_1)),
                                    name='logits_1_loss')
        logits_2_loss = tf.identity(self.compute_loss(loss_fn, inputs["logits_2"], self.get_final_activation(logits_2)),
                                    name='logits_2_loss')
        logits_3_loss = tf.identity(self.compute_loss(loss_fn, inputs["logits_3"], self.get_final_activation(logits_3)),
                                    name='logits_3_loss')
        logits_4_loss = tf.identity(self.compute_loss(loss_fn, inputs["logits_4"], self.get_final_activation(logits_4)),
                                    name='logits_4_loss')

        loss = gt_loss + logits_1_loss + logits_2_loss + logits_3_loss + logits_4_loss
        total_cost = tf.identity(loss, name='cost')

        add_moving_summary(total_cost)
        add_moving_summary(gt_loss)
        add_moving_summary(logits_1_loss)
        add_moving_summary(logits_2_loss)
        add_moving_summary(logits_3_loss)
        add_moving_summary(logits_4_loss)

        return total_cost

    def build_pred_graph(self, *args):

        list_inputs = self.input_names
        inputs = {}
        for arg, name in zip(args, list_inputs):
            inputs[name] = arg

        logits_final = self.attention_unet_pred_backbone(inputs["images"])

        prediction = self.get_final_activation(logits_final)

    def get_final_activation(self, logits):
        activation_dict = {"sigmoid": tf.nn.sigmoid, "relu": tf.nn.relu, "softmax": tf.nn.softmax}

        final_activation_fn = activation_dict[self.get_parameter("final_activation")]

        return final_activation_fn(logits, name='prediction')


class AttentionUnetResnet101(AttentionUnetModel):
    def __init__(self, name='Attention_Unet_Resnet101', **kwargs):
        super().__init__(**kwargs)

        self.update_parameter(["model", "name"], name)
        self.update_parameter(["model", "conv_layer"], (3, 4, 23, 3))

        # store parameters for ease of use (may need to remove in the future)
        self.conv_layer = self.get_parameter("conv_layer")


class AttentionUnetResnet50(AttentionUnetModel):
    def __init__(self, name='Attention_Unet_Resnet50', **kwargs):
        super().__init__(**kwargs)

        self.update_parameter(["model", "name"], name)
        self.update_parameter(["model", "conv_layer"], (3, 4, 6, 3))

        # store parameters for ease of use (may need to remove in the future)
        self.conv_layer = self.get_parameter("conv_layer")
