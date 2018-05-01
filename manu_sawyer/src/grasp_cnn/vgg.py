# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(reuse = False, weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu, reuse = reuse,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


# def vgg_16(inputs,
#            num_classes=1000,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            scope='vgg_16',
#            update_top_only = False,
#            fc_conv_padding='VALID'):
#   """Oxford Net VGG 16-Layers version D Example.

#   Note: All the fully_connected layers have been transformed to conv2d layers.
#         To use in classification mode, resize input to 224x224.

#   Args:
#     inputs: a tensor of size [batch_size, height, width, channels].
#     num_classes: number of predicted classes.
#     is_training: whether or not the model is being trained.
#     dropout_keep_prob: the probability that activations are kept in the dropout
#       layers during training.
#     spatial_squeeze: whether or not should squeeze the spatial dimensions of the
#       outputs. Useful to remove unnecessary dimensions for classification.
#     scope: Optional scope for the variables.
#     fc_conv_padding: the type of padding to use for the fully connected layer
#       that is implemented as a convolutional layer. Use 'SAME' padding if you
#       are applying the network in a fully convolutional manner and want to
#       get a prediction map downsampled by a factor of 32 as an output. Otherwise,
#       the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

#   Returns:
#     the last op containing the log predictions and end_points dict.
#   """
#   with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
#     end_points_collection = sc.name + '_end_points'
#     # Collect outputs for conv2d, fully_connected and max_pool2d.
#     with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
#                         outputs_collections=end_points_collection):
#       net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#       net = slim.max_pool2d(net, [2, 2], scope='pool1')
#       net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#       net = slim.max_pool2d(net, [2, 2], scope='pool2')
#       net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#       net = slim.max_pool2d(net, [2, 2], scope='pool3')
#       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#       net = slim.max_pool2d(net, [2, 2], scope='pool4')
#       if update_top_only:
#         net = tf.stop_gradient(net)
#       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#       net = slim.max_pool2d(net, [2, 2], scope='pool5')
#       # Use conv2d instead of fully_connected layers.
#       net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout6')
#       net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout7')
#       net = slim.conv2d(net, num_classes, [1, 1],
#                         activation_fn=None,
#                         normalizer_fn=None,
#                         scope='fc8')
#       # Convert end_points_collection into a end_point dict.
#       end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#       if spatial_squeeze:
#         net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#         end_points[sc.name + '/fc8'] = net
#       return net, end_points



def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           update_top_only = False,
           fc_conv_padding='VALID',
           reuse = None):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc, \
           slim.arg_scope(vgg_arg_scope(reuse = reuse)):
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      if update_top_only:
        net = tf.stop_gradient(net)
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points

        
vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
       net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
       end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19


# def vgg_dual_16(inputs1,
#                 inputs2,
#                 num_classes=1000,
#                 is_training=True,
#                 dropout_keep_prob=0.5,
#                 spatial_squeeze=True,
#                 scope='vgg_16',
#                 update_top_only = False,
#                 fc_conv_padding='VALID'):
#   with tf.variable_scope(scope, 'vgg_16', [inputs1]) as sc:
#     end_points_collection = sc.name + '_end_points'
#     # Collect outputs for conv2d, fully_connected and max_pool2d.
#     with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
#                         outputs_collections=end_points_collection):
#       nets = []
#       for i, inputs in enumerate([inputs1, inputs2]):
#         with slim.arg_scope(vgg_arg_scope(reuse = (i > 0))):
#           net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#           net = slim.max_pool2d(net, [2, 2], scope='pool1')
#           net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#           net = slim.max_pool2d(net, [2, 2], scope='pool2')
#           net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#           net = slim.max_pool2d(net, [2, 2], scope='pool3')
#           net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#           net = slim.max_pool2d(net, [2, 2], scope='pool4')
#           if update_top_only:
#             net = tf.stop_gradient(net)
#           net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#           net = slim.max_pool2d(net, [2, 2], scope='pool5')
#           # Use conv2d instead of fully_connected layers.
#           net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#           net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout6')
#           # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#           # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout7')
#           nets.append(net)
#         net = tf.concat(nets, 0)
#         net = slim.conv2d(net, 2048, [1, 1], scope='fc7_')
#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout7_')
#         net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
#                           normalizer_fn = None, scope = 'fc8')
#         # Convert end_points_collection into a end_point dict.
#         end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#         if spatial_squeeze:
#           net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#           end_points[sc.name + '/fc8'] = net
#         return net, end_points



def vgg_dual_16(inputs1,
                inputs2,
                num_classes=1000,
                is_training=True,
                dropout_keep_prob=0.5,
                spatial_squeeze=True,
                scope='vgg_16',
                update_top_only = False,
                fc_conv_padding='VALID',
                reuse = False):
  with tf.variable_scope(scope, 'vgg_16', [inputs1]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      nets = []
      for i, inputs in enumerate([inputs1, inputs2]):
        with slim.arg_scope(vgg_arg_scope(reuse = reuse or (i > 0))):
          net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
          net = slim.max_pool2d(net, [2, 2], scope='pool4')
          if update_top_only:
            net = tf.stop_gradient(net)
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
          nets.append(net)
      with slim.arg_scope(vgg_arg_scope(reuse = reuse)):
        net = tf.concat(nets, 3)
        net = slim.conv2d(net, 512, [1, 1], scope='conv6')
        net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool6')
        #net = slim.max_pool2d(net, [2, 2], scope='pool6')
        # Use conv2d instead of fully_connected layers.
        #net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.conv2d(net, 2048, [7, 7], padding=fc_conv_padding, scope = 'fc6_')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout6')
        # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout7')

        net = slim.conv2d(net, 2048, [1, 1], scope='fc7_')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout7_')
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn = None, scope = 'fc8')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
        return net, end_points

def pair_vgg(inputs1,
             inputs2,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             spatial_squeeze=True,
             scope='vgg_16',
             update_top_only = False,
             fc_conv_padding='VALID',
             reuse = False):
  #with tf.variable_scope(scope, 'vgg_16', [inputs1]), \
  with tf.variable_scope(scope), \
           slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
    nets = []
    for i, inputs in enumerate([inputs1, inputs2]):
      with slim.arg_scope(vgg_arg_scope(reuse = reuse or (i > 0))):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        if update_top_only:
          net = tf.stop_gradient(net)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        nets.append(net)
    with slim.arg_scope(vgg_arg_scope(reuse = reuse)):
      net = tf.concat(nets, 3)
      net = slim.conv2d(net, 128, [1, 1], scope='conv6')
      net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool6')
  return net

# def vgg_gel2(gel0_pre, gel0_post,
#              gel1_pre, gel1_post,
#              num_classes = 2,
#              is_training = True,
#              update_top_only = False,
#              fc_conv_padding='VALID',
#              dropout_keep_prob = 0.5):
#   net0 = pair_vgg(gel0_pre, gel0_post, is_training = is_training, update_top_only = update_top_only)
#   net1 = pair_vgg(gel1_pre, gel1_post, reuse = True, is_training = is_training, update_top_only = update_top_only)
#   with slim.arg_scope(vgg_arg_scope(False)):
#     net = tf.concat([net0, net1], 3)
#     net = slim.conv2d(net, 2048, [7, 7], padding = fc_conv_padding, scope = 'fc6_')
#     net = slim.dropout(net, dropout_keep_prob, is_training = is_training, scope = 'dropout6')
#     net = slim.conv2d(net, 2048, [1, 1], scope='fc7_')
#     net = slim.dropout(net, dropout_keep_prob, is_training = is_training, scope = 'dropout7')
#     net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'fc8')
#     net = net[:, 0, 0, :]
#   return net


def vgg_gel2(gel0_pre, gel0_post,
             gel1_pre, gel1_post,
             num_classes = 2,
             is_training = True,
             update_top_only = False,
             fc_conv_padding='VALID',
             dropout_keep_prob = 0.5,
             diff = True,
             reuse = False,
             scope = 'vgg_16'):
  print('reuse =', reuse)
  if diff:
    nets = []
    r = reuse
    if gel0_pre is not None:
      nets.append(vgg_dual_16(gel0_post - gel0_pre, gel0_post, reuse = r, is_training = is_training, 
                              num_classes = None, update_top_only = update_top_only, scope = scope)[0])
      r = True
    if gel1_pre is not None:    
      nets.append(vgg_dual_16(gel1_post - gel1_pre, gel1_post, reuse = r, is_training = is_training, 
                              num_classes = None, update_top_only = update_top_only, scope = scope)[0])
      r = True
    return tf.concat(nets, 1)
  else:
    net0 = pair_vgg(gel0_post, gel0_pre, is_training = is_training, update_top_only = update_top_only, scope = scope)
    net1 = pair_vgg(gel1_post, gel1_pre, reuse = True, is_training = is_training, update_top_only = update_top_only, scope = scope)

    with tf.variable_scope(scope, scope), \
             slim.arg_scope(vgg_arg_scope(reuse)):
      net = tf.concat([net0, net1], 3)
      net = slim.conv2d(net, 2048, [7, 7], padding = fc_conv_padding, scope = 'fc6_')
      net = slim.dropout(net, dropout_keep_prob, is_training = is_training, scope = 'dropout6')
      net = slim.conv2d(net, 2048, [1, 1], scope =  'fc7_')
      net = slim.dropout(net, dropout_keep_prob, is_training = is_training, scope = 'dropout7')
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'fc8_')
      net = net[:, 0, 0, :]
  return net
