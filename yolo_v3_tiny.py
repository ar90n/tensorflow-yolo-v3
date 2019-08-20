# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from yolo_v3 import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _detection_layer, _upsample

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

def _reorg(inputs, stride):
    return f.extract_image_patches(inputs,
            [1, stride, stride, 1], [1, stride, stride, 1], [1, 1, 1, 1], 'VALID')

def yolo_v3_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 tiny model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    _ANCHORS = [(10, 14),  (23, 27),  (37, 58), (81, 82),  (135, 169),  (344, 319)]
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with tf.variable_scope('yolo-v3-tiny'):
                    for i in range(6):
                        inputs = _conv2d_fixed_padding(
                            inputs, 16 * pow(2, i), 3)

                        if i == 4:
                            route_1 = inputs

                        if i == 5:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], stride=1, padding="SAME", scope='pool2')
                        else:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], scope='pool2')

                    inputs = _conv2d_fixed_padding(inputs, 1024, 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    route_2 = inputs

                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

                    detect_1 = _detection_layer(
                        inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='detect_1')

                    inputs = _conv2d_fixed_padding(route_2, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = _upsample(inputs, upsample_size, data_format)

                    inputs = tf.concat([inputs, route_1],
                                       axis=1 if data_format == 'NCHW' else 3)

                    inputs = _conv2d_fixed_padding(inputs, 256, 3)
                    # inputs = _conv2d_fixed_padding(inputs, 255, 1)

                    detect_2 = _detection_layer(
                        inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    detections = tf.concat([detect_1, detect_2], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections

def yolo_v3_tiny_pan(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 tiny pan model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    _ANCHORS = [(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with tf.variable_scope('yolo-v3-tiny-pan'):
                    # 0 conv     16       3 x 3/ 1    544 x 544 x   3 ->  544 x 544 x  16 0.256 BF
                    inputs = _conv2d_fixed_padding(inputs, 16, 3)
                    # 1 max               2 x 2/ 2    544 x 544 x  16 ->  272 x 272 x  16 0.005 BF
                    inputs = slim.max_pool2d( inputs, [2, 2], scope='pool2')
                    route_1 = inputs

                    #2 conv     32       3 x 3/ 1    272 x 272 x  16 ->  272 x 272 x  32 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 16, 3)
                    #3 max               2 x 2/ 2    272 x 272 x  32 ->  136 x 136 x  32 0.002 BF
                    inputs = slim.max_pool2d( inputs, [2, 2], scope='pool2')
                    route_3 = inputs

                    #4 conv     64       3 x 3/ 1    136 x 136 x  32 ->  136 x 136 x  64 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 16, 3)
                    #5 max               2 x 2/ 2    136 x 136 x  64 ->   68 x  68 x  64 0.001 BF
                    inputs = slim.max_pool2d( inputs, [2, 2], scope='pool2')
                    route_5 = inputs

                    #6 conv    128       3 x 3/ 1     68 x  68 x  64 ->   68 x  68 x 128 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 16, 3)
                    route_6 = inputs
                    #7 max               2 x 2/ 2     68 x  68 x 128 ->   34 x  34 x 128 0.001 BF
                    inputs = slim.max_pool2d( inputs, [2, 2], scope='pool2')
                    route_7 = inputs

                    #8 conv    256       3 x 3/ 1     34 x  34 x 128 ->   34 x  34 x 256 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 16, 3)
                    route_8 = inputs
                    #9 max               2 x 2/ 2     34 x  34 x 256 ->   17 x  17 x 256 0.000 BF
                    inputs = slim.max_pool2d( inputs, [2, 2], scope='pool2')
                    route_9 = inputs

                    #10 conv    512       3 x 3/ 1     17 x  17 x 256 ->   17 x  17 x 512 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 16, 3)
                    #11 max               2 x 2/ 1     17 x  17 x 512 ->   17 x  17 x 512 0.001 BF
                    inputs = slim.max_pool2d( inputs, [2, 2], stride=1, padding="SAME", scope='pool2')

                    #12 conv   1024       3 x 3/ 1     17 x  17 x 512 ->   17 x  17 x1024 2.727 BF
                    inputs = _conv2d_fixed_padding(inputs, 1024, 3)
                    #13 conv    256       1 x 1/ 1     17 x  17 x1024 ->   17 x  17 x 256 0.152 BF
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    #14 conv    512       3 x 3/ 1     17 x  17 x 256 ->   17 x  17 x 512 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    route_14 = inputs
                    #15 conv    128       1 x 1/ 1     17 x  17 x 512 ->   17 x  17 x 128 0.038 BF
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)


                    #16 upsample                 2x    17 x  17 x 128 ->   34 x  34 x 128
                    inputs = _upsample(inputs, route_8.get_shape().as_list(), data_format)

                    #17 route  16 8
                    inputs = tf.concat([inputs, route_8], axis=1 if data_format == 'NCHW' else 3)

                    #18 conv    128       1 x 1/ 1     34 x  34 x 384 ->   34 x  34 x 128 0.114 BF
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)
                    #19 conv    256       3 x 3/ 1     34 x  34 x 128 ->   34 x  34 x 256 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 256, 3)
                    route_19 = inputs
                    #20 conv    128       1 x 1/ 1     34 x  34 x 256 ->   34 x  34 x 128 0.076 BF
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)


                    #21 upsample                 2x    34 x  34 x 128 ->   68 x  68 x 128
                    inputs = _upsample(inputs, route_6.get_shape().as_list(), data_format)

                    #22 route  21 6
                    inputs = tf.concat([inputs, route_6], axis=1 if data_format == 'NCHW' else 3)

                    #23 conv     64       1 x 1/ 1     68 x  68 x 256 ->   68 x  68 x  64 0.152 BF
                    inputs = _conv2d_fixed_padding(inputs, 64, 1)
                    #24 conv    128       3 x 3/ 1     68 x  68 x  64 ->   68 x  68 x 128 0.682 BF
                    inputs = _conv2d_fixed_padding(inputs, 128, 3)
                    route_24 = inputs

                    #25 route  1
                    inputs = route_1
                    #26 reorg                    / 2  272 x 272 x  16 ->  136 x 136 x  64
                    inputs = _reorg(inputs, 2)
                    #27 route  3 26
                    inputs = tf.concat([route_3, inputs], axis=1 if data_format == 'NCHW' else 3)
                    #28 reorg                    / 2  136 x 136 x  96 ->   68 x  68 x 384
                    inputs = _reorg(inputs, 2)
                    #29 route  5 28
                    inputs = tf.concat([route_5, inputs], axis=1 if data_format == 'NCHW' else 3)
                    #30 reorg                    / 2   68 x  68 x 448 ->   34 x  34 x1792
                    inputs = _reorg(inputs, 2)
                    #31 route  7 30
                    inputs = tf.concat([route_7, inputs], axis=1 if data_format == 'NCHW' else 3)
                    #32 reorg                    / 2   34 x  34 x1920 ->   17 x  17 x7680
                    inputs = _reorg(inputs, 2)
                    #33 route  9 32
                    inputs = tf.concat([route_9, inputs], axis=1 if data_format == 'NCHW' else 3)
                    route_33 = inputs

                    #34 conv     64       1 x 1/ 1     17 x  17 x7936 ->   17 x  17 x  64 0.294 BF
                    inputs = _conv2d_fixed_padding(inputs, 64, 1)
                    #35 upsample                 4x    17 x  17 x  64 ->   68 x  68 x  64
                    inputs = _upsample(inputs, route_24.get_shape().as_list(), data_format)
                    #36 route  35 24
                    inputs = tf.concat([inputs, route_24], axis=1 if data_format == 'NCHW' else 3)
                    #37 conv    128       3 x 3/ 1     68 x  68 x 192 ->   68 x  68 x 128 2.046 BF
                    inputs = _conv2d_fixed_padding(inputs, 128, 3)
                    #38 conv     18       1 x 1/ 1     68 x  68 x 128 ->   68 x  68 x  18 0.021 BF
                    inputs = _conv2d_fixed_padding(inputs, 18, 1)

                    #39 yolo
                    detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='detect_1')

                    #40 route  33
                    inputs = route_33
                    #41 conv    128       1 x 1/ 1     17 x  17 x7936 ->   17 x  17 x 128 0.587 BF
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)
                    #42 upsample                 2x    17 x  17 x 128 ->   34 x  34 x 128
                    inputs = _upsample(inputs, route_19.get_shape().as_list(), data_format)
                    #43 route  42 19
                    inputs = tf.concat([inputs, route_19], axis=1 if data_format == 'NCHW' else 3)
                    #44 conv    256       3 x 3/ 1     34 x  34 x 384 ->   34 x  34 x 256 2.046 BF
                    inputs = _conv2d_fixed_padding(inputs, 256, 3)
                    #45 conv     18       1 x 1/ 1     34 x  34 x 256 ->   34 x  34 x  18 0.011 BF
                    inputs = _conv2d_fixed_padding(inputs, 18, 1)
                    #46 yolo
                    detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    #47 route  33
                    inputs = route_33
                    #48 conv    256       1 x 1/ 1     17 x  17 x7936 ->   17 x  17 x 256 1.174 BF
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    #49 route  48 14
                    inputs = tf.concat([inputs, route_14], axis=1 if data_format == 'NCHW' else 3)
                    #50 conv    512       3 x 3/ 1     17 x  17 x 768 ->   17 x  17 x 512 2.046 BF
                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    #51 conv     18       1 x 1/ 1     17 x  17 x 512 ->   17 x  17 x  18 0.005 BF
                    inputs = _conv2d_fixed_padding(inputs, 18, 1)
                    #52 yolo
                    detect_3 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                    detect_3 = tf.identity(detect_3, name='detect_3')

                    detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections
