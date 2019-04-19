from __future__ import print_function, division
import os
import dill
import copy
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from utils.dualprint import dualprint
# from app.anom_v2.utils.lrelu import lrelu
from utils.lrelu import lrelu
from utils.generic_utils import make_batches
# from male.utils.disp_utils import tile_raster_images
from utils.func_utils import sigmoid
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops import control_flow_ops
from pix2pix_so_v1_func import deprocess

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
config.allow_soft_placement = True

def conv_out_size_same(size, stride):
    return math.ceil(float(size) / float(stride))

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", with_w = False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        # w = tf.get_variable('w', [k_h, k_w, tf.shape(input_)[-1], output_dim],
        #                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        pre_act = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        # conv = tf.nn.bias_add(conv, biases)
        if with_w:
            return pre_act, w, biases, conv
        else:
            return pre_act


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            output_shape1 = tf.stack(output_shape)
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape1,
                                            strides=[1, d_h, d_w, 1])
            # deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
            #                                 strides=[1, d_h, d_w, 1])

        # Support for versions of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def linear(input, output_dim, scope='linear', stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def batchnorm(inputs, is_training, decay=0.999, scope='batchnorm'):
    with tf.variable_scope(scope):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))

        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, 0.001)



def tile_images(data_display, img_shape, tile_shape=(10, 10),
                                             tile_spacing=(1, 1)):
    num_dim = len(img_shape)

    if num_dim == 2:
        img_sz = img_shape + [1]
    else:
        img_sz = img_shape
    tile_shape = list(tile_shape)+[1]

    img_shape_ex = []
    for i in range(len(img_sz)):
        if i < len(img_sz)-1:
            img_shape_ex.append(img_sz[i]+tile_spacing[i])
        else:
            img_shape_ex.append(img_sz[i])

    disp_img = np.tile(np.zeros(img_shape_ex), tile_shape)
    c = 0
    for i in range(tile_shape[0]):
        y_start = i * img_shape_ex[0]
        y_end = (i+1) * img_shape_ex[0] - 1

        for j in range(tile_shape[1]):
            x_start = j * img_shape_ex[1]
            x_end = (j + 1) * img_shape_ex[1] -1
            # disp_img[y_start:y_end, x_start:x_end, :] = np.reshape(data_display[c], img_shape)
            disp_img[y_start:y_end, x_start:x_end, :] = np.reshape(data_display[c], img_sz)
            c += 1
    disp_img = disp_img[:-1, :-1, :]
    if num_dim == 2:
        disp_img = np.reshape(disp_img, [disp_img.shape[0], disp_img.shape[1]])
    return disp_img


class ConvAEv4():
    def __init__(self, input_width, input_height, input_channels,
                 batch_size=100,
                 lr_rate = 0.01,
                 w_init = 0.1,
                 gamma = 1.0,
                 num_epochs = 100,
                 k_h = 5,
                 k_w = 5,
                 d_h = 2,
                 d_w = 2,
                 use_bn = False,
                 denoising = False,
                 saved_folder = None,
                 encoder_dims = None,
                 encoder_act = None,
                 decoder_dims = None,
                 decoder_act =  None,
                 debug = False,
                 optimizer_name = 'Adagrad',
                 disp_freq = 1,
                 save_freq=10,
                 device = '/device:GPU:0',
                 model_name='ConvAEv4'):
        self.model_name = model_name
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.lr_rate = lr_rate
        self.w_init = w_init
        self.k_h = k_h
        self.k_w = k_w
        self.d_h = d_h
        self.d_w = d_w
        self.use_bn = use_bn
        self.denoising = denoising
        self.num_epochs = num_epochs
        self.saved_folder = saved_folder
        self.encoder_dims = encoder_dims
        self.encoder_act = encoder_act
        self.decoder_dims = decoder_dims
        self.decoder_act = decoder_act
        self.is_training = None
        self.gamma = gamma
        self.debug = debug
        self.device = device
        self.disp_freq = disp_freq
        self.save_freq = save_freq
        self.layer_height = [self.input_height]
        self.layer_width = [self.input_width]
        self.optimizer_name = optimizer_name
        for i in range(len(self.encoder_dims)):
            self.layer_height.append(int( self.layer_height[-1]/self.d_h))
            self.layer_width.append(int(self.layer_width[-1]/self.d_w))


    def get_params(self):
        dict = {'model_name': self.model_name,
                'input_width': self.input_width,
                'input_height': self.input_height,
                'input_channels': self.input_channels,
                'batch_size': self.batch_size,
                'lr_rate': self.lr_rate,
                'w_init': self.w_init,
                'num_epochs': self.num_epochs,
                'saved_folder': self.saved_folder,
                'k_h': self.k_h,
                'k_w': self.k_w,
                'd_h': self.d_h,
                'd_w': self.d_w,
                'use_bn': self.use_bn,
                'optimizer_name': self.optimizer_name,
                'denoising': self.denoising,
                'encoder_dims': self.encoder_dims,
                'encoder_act': self.encoder_act,
                'decoder_dims': self.decoder_dims,
                'decoder_act': self.decoder_act,
                'gamma': self.gamma,
                'debug': self.debug,
                'device': self.device,
                'disp_freq': self.disp_freq,
                'save_freq': self.save_freq}
        return dict

    def create_decoder(self, x, is_training):

        decoder_params = []
        de_layers = []
        layer_i = x
        de_layers.append(layer_i)
        s_h, s_w = [self.input_height], [self.input_width]
        for i in range(len(self.decoder_dims)-1):
            s_h = [conv_out_size_same(s_h[0], self.d_h)] + s_h
            s_w = [conv_out_size_same(s_w[0], self.d_w)] + s_w

        num_decoder_layer = len(self.decoder_dims)
        # print('kernel= %d x %d' % (self.k_h, self.k_w))
        for i in range(0, num_decoder_layer):
            # print(tf.shape(layer_i))
            layer_i, w_i, b_i = deconv2d(layer_i, output_shape=[tf.shape(x)[0], s_h[i], s_w[i], self.decoder_dims[i]],
                                         k_h=self.k_h, k_w=self.k_w,
                                         stddev=self.w_init,
                                         name='decoder-deconv-%d' % (i),
                                         with_w=True)

            # layer_i, w_i, b_i = deconv2d(layer_i, output_shape=np.array(
            #     [shape[0], s_h[i], s_w[i], self.decoder_dims[i]],
            #     dtype=np.int32),
            #                              stddev=self.w_init,
            #                              name='decoder-deconv-%d' % (i),
            #                              with_w=True)
            # # if i < num_layer_cnn - 1:
            # print('i = %d' % i)
            # print(layer_i.get_shape())
            layer_i = tf.reshape(layer_i, tf.stack([-1, s_h[i], s_w[i], self.decoder_dims[i]]))

            if self.use_bn:
                layer_i = batch_norm(layer_i,
                                     decay=0.9,
                                     updates_collections=None,
                                     epsilon=1e-5,
                                     scale=True,
                                     is_training=is_training,
                                     scope="decoder-bn-%d" % i)
            if self.decoder_act[i] == 'relu':
                layer_i = tf.nn.relu(layer_i)
            elif self.decoder_act[i] == 'lrelu':
                    layer_i = lrelu(layer_i)
            elif self.decoder_act[i] == 'sigmoid':
                layer_i = tf.sigmoid(layer_i)
            elif self.decoder_act[i] == 'tanh':
                layer_i = tf.tanh(layer_i)
            else:
                layer_i = None

            decoder_params.append([w_i, b_i])
            de_layers.append(layer_i)

        if len(self.decoder_params) == 0:
            self.decoder_params = decoder_params
        if len(self.de_layers) == 0:
            self.de_layers = de_layers
        return layer_i


    def create_encoder(self, x, is_training, start_layer = 0):
        encoder_params = []
        en_layers = []
        conv_layers = []
        layer_i = x
        en_layers.append(layer_i)
        num_en_layers = len(self.encoder_dims)
        # print('kernel= %d x %d' % (self.k_h, self.k_w))
        for i in range(start_layer, num_en_layers):
            layer_i, w_i, b_i, conv_i = conv2d(layer_i, self.encoder_dims[i],
                                       k_h=self.k_h, k_w=self.k_w, d_h=self.d_h, d_w=self.d_w,
                                       # stddev=0.02,
                                       stddev=self.w_init,
                                       name="encoder-conv-%d" % i,
                                       with_w=True)
            encoder_params.append([w_i, b_i])
            conv_layers.append(conv_i)
            if self.use_bn:
                layer_i = batch_norm(layer_i,
                                     decay=0.9,
                                     updates_collections=None,
                                     epsilon=1e-5,
                                     scale=True,
                                     is_training=is_training,
                                     scope="encoder-bn-%d" % i)

            if self.encoder_act[i] == 'relu':
                layer_i = tf.nn.relu(layer_i)
            elif self.encoder_act[i] == 'lrelu':
                    layer_i = lrelu(layer_i)
            elif self.encoder_act[i] == 'sigmoid':
                layer_i = tf.sigmoid(layer_i)
            elif self.encoder_act[i] == 'tanh':
                layer_i = tf.tanh(layer_i)
            else:
                layer_i = None

            en_layers.append(layer_i)


        if len(self.encoder_params) == 0:
            self.encoder_params = encoder_params

        if len(self.en_layers) == 0:
            self.en_layers = en_layers
        return layer_i, conv_layers
    # def create_encoder_visual(self, x, is_training, start_layer = 0):
    #     encoder_params = []
    #     conv_layers = []
    #     layer_i = x
    #     conv_layers.append(layer_i)
    #     num_en_layers = len(self.encoder_dims)
    #     for i in range(start_layer, num_en_layers):
    #         layer_i, w_i, b_i, conv = conv2d(layer_i, self.encoder_dims[i],
    #                                    k_h=self.k_h, k_w=self.k_w, d_h=2, d_w=2,
    #                                    # stddev=0.02,
    #                                    stddev=self.w_init,
    #                                    name="encoder-conv-%d" % i,
    #                                    with_w=True)
    #         encoder_params.append([w_i, b_i])
    #         layer_i = batch_norm(layer_i,
    #                              decay=0.9,
    #                              updates_collections=None,
    #                              epsilon=1e-5,
    #                              scale=True,
    #                              is_training=is_training,
    #                              scope="encoder-bn-%d" % i)
    #
    #         if self.encoder_act[i] == 'relu':
    #             layer_i = tf.nn.relu(layer_i)
    #         elif self.encoder_act[i] == 'lrelu':
    #                 layer_i = lrelu(layer_i)
    #         elif self.encoder_act[i] == 'sigmoid':
    #             layer_i = tf.sigmoid(layer_i)
    #         elif self.encoder_act[i] == 'tanh':
    #             layer_i = tf.tanh(layer_i)
    #         else:
    #             layer_i = None
    #
    #         conv_layers.append(conv)
    #
    #     return conv_layers, encoder_params

    def build_model(self):
        self.encoder_params = []
        self.decoder_params = []
        self.en_layers = []
        self.de_layers = []

        # with tf.device(self.device):
        # with tf.variable_scope(tf.get_variable_scope()) as cem_scope:
        with tf.device(self.device):
            with tf.variable_scope('ConvAE') as convae_scope:
                self.x = tf.placeholder(tf.float32, shape=[None,
                                                           self.input_height,
                                                           self.input_width,
                                                           self.input_channels])

                self.x_org = tf.placeholder(tf.float32, shape=[None,
                                                               self.input_height,
                                                               self.input_width,
                                                               self.input_channels])



                self.lrate = tf.placeholder(tf.float32, shape=())

                self.mtum = tf.placeholder(tf.float32, shape=())


                self.is_training = tf.placeholder(tf.bool, shape=())

                self.x_encode, self.conv_layers = self.create_encoder(self.x, is_training = self.is_training)
                self.x_decode = self.create_decoder(self.x_encode, is_training = self.is_training)

                self.visual_x = [self.x]
                self.visual_decode = [self.x_decode]


                # self.x_opt = tf.get_variable('x_opt', [100, self.input_height,
                #                                        self.input_width,
                #                                        self.input_channels],
                #                              initializer=tf.random_uniform_initializer(minval=0.0,
                #                                                                        maxval=1.0,
                #                                                                        dtype=tf.float32))

                tf.get_variable_scope().reuse_variables()


                # self.conv_layers, self.en_params_visual = self.create_encoder_visual(self.x_opt, is_training=self.is_training)


                for i in range(1, len(self.encoder_dims)+1):

                    visual_layer_i = tf.placeholder(tf.float32, shape=self.en_layers[i].get_shape())
                    self.visual_x.append(visual_layer_i)

                    # with tf.variable_scope('visual_layer%d' % i) as visual_layer:
                    visual_encode, _ = self.create_encoder(visual_layer_i, is_training=self.is_training,
                                                        start_layer=i)
                    visual_decode = self.create_decoder(visual_encode, is_training=self.is_training)
                    self.visual_decode.append(visual_decode)

                self.reg_term = 0.0
                for i in range(len(self.encoder_params)):
                    # self.reg_term += tf.reduce_mean(tf.square(self.encoder_params[i][0]))
                    self.reg_term += tf.reduce_sum(tf.square(self.encoder_params[i][0]))

                for i in range(len(self.decoder_params)):
                    # self.reg_term += tf.reduce_mean(tf.square(self.decoder_params[i][0]))
                    self.reg_term += tf.reduce_sum(tf.square(self.decoder_params[i][0]))
                # minimize
                # self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.x_decode), axis=(1, 2, 3)))
                self.recon_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(self.x_org - self.x_decode), axis=(1, 2, 3)))
                self.loss = self.recon_loss + self.gamma*self.reg_term


            # self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr_rate ).minimize(
            #     self.loss)
            if self.optimizer_name == 'Adam':
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr_rate)
                self.opt = self.trainer.minimize(self.loss)
            elif self.optimizer_name == 'Adagrad':
                # self.trainer = tf.train.AdagradOptimizer(learning_rate=self.lr_rate)
                self.trainer = tf.train.AdagradOptimizer(learning_rate=self.lrate)
                self.opt = self.trainer.minimize(self.loss)



            self.loss_filter = []
            self.x_opt_grad = []
            for conv_idx in range(len(self.conv_layers)):
                loss_layer = []
                x_opt_grad_layer = []
                x_pos = int(self.layer_height[conv_idx + 1]*0.5)
                y_pos = int(self.layer_width[conv_idx + 1]*0.5)

                for i in range(self.encoder_dims[conv_idx]):

                    # loss_filter_i = tf.reduce_mean(self.conv_layers[conv_idx][0, :, :, i])
                    loss_filter_i = tf.reduce_mean(self.conv_layers[conv_idx][0, y_pos, x_pos, i])
                    loss_layer.append(loss_filter_i)

                    x_opt_grad_i = tf.gradients(loss_filter_i, [self.x])[0]
                    grad_norm_i = tf.sqrt(tf.reduce_mean(tf.square(x_opt_grad_i)))
                    x_opt_grad_norm_i = tf.divide(x_opt_grad_i, grad_norm_i)

                    x_opt_grad_layer.append(x_opt_grad_norm_i)
                # x_opt_norm = tf.sqrt(tf.reduce_sum(tf.square(self.x_opt)))
                # self.x_opt_norm_update = tf.assign(self.x_opt, tf.divide(self.x_opt, x_opt_norm))
                self.loss_filter.append(loss_layer)
                self.x_opt_grad.append(x_opt_grad_layer)


        # self.opt_D = tf.train.AdagradOptimizer(learning_rate=self.lrate_D).minimize(
        #     -self.adver_D_loss)

        # self.opt = tf.train.MomentumOptimizer(learning_rate=self.lrate, momentum=self.mtum).minimize(
        #     -self.loglike)

    def fit(self, data, data_org):

        print('Training: ...')
        print('Denoising: %d' % self.denoising)
        print('Gamma = %f' % self.gamma)
        param_file = '%s/param.dill' % self.saved_folder
        dill.dump(self.get_params(), open(param_file, 'wb'))

        log_file = '%s/log.txt' % self.saved_folder
        flog = open(log_file, 'wt')

        self.data = data
        self.saver = tf.train.Saver()
        if self.debug:
            fig = plt.figure()
            n_subplot = 2
            m_subplot = 2
            ax1 = plt.subplot(n_subplot, m_subplot, 1)
            ax2 = plt.subplot(n_subplot, m_subplot, 2)
            ax3 = plt.subplot(n_subplot, m_subplot, 3)
            ax4 = plt.subplot(n_subplot, m_subplot, 4)
            # ax5 = plt.subplot(n_subplot, m_subplot, 5)
            # ax6 = plt.subplot(n_subplot, m_subplot, 6)
        #     ax7 = plt.subplot(n_subplot, m_subplot, 7)
        #     ax8 = plt.subplot(n_subplot, m_subplot, 8)
        #     ax9 = plt.subplot(n_subplot, m_subplot, 9)
        #
            plt.show(block=False)


        num_batches = int(data.shape[0] / self.batch_size)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            num_data = data.shape[0]
            batches = make_batches(num_data, self.batch_size)
            self.epoch = 0
            self.stop_training = False

            epoch_list = []

            loss_list = []
            recon_loss_list = []
            reg_term_list = []
            # lr_array = [self.learning_rate]
            # for i in range(int(self.num_epochs / 20)):
            #     lr_array.append(lr_array[-1] * 1.0 / 10.0)

            current_learning_rate = self.lr_rate
            while (self.epoch < self.num_epochs) and (not self.stop_training):
                num_last_epoch = 5

                if len(loss_list) > num_last_epoch+1:
                    count = 0
                    for i in range(1, num_last_epoch+1):
                        if loss_list[-i] > loss_list[-6]:
                            count += 1
                    if count == num_last_epoch:
                        current_learning_rate = current_learning_rate *0.3
                print('Learning rate = %f\n' % current_learning_rate)
                if self.optimizer_name == 'Adam':
                    dualprint('Optimizer''s _lr = %f \t_lr_t = %f\n' % (self.trainer._lr, sess.run(self.trainer._lr_t)), flog)
                elif self.optimizer_name == 'Adagrad':
                    dualprint(
                        'Optimizer''s learning rate = %f \t_learning_rate_tensor=%f\n' % (sess.run(self.trainer._learning_rate, feed_dict={self.lrate:current_learning_rate} ),
                                                                                          sess.run(self.trainer._learning_rate_tensor, feed_dict={self.lrate:current_learning_rate})), flog)
                # current_learning_rate = self.learning_rate  - self.epoch*1.0/(self.num_epochs-1)*(self.learning_rate - lwr_learning_rate)
                # current_learning_rate = lr_array[int(self.epoch / 20)]
                # if self.momentum_method == 'none':
                #     current_momentum = 0.0
                # else:
                #     if self.epoch < self.momentum_iteration:
                #         current_momentum = self.initial_momentum
                #     else:
                #         current_momentum = self.final_momentum

                total_loss = 0.0
                total_recon_loss = 0.0
                total_reg_term = 0.0

                for batch_idx, (batch_start, batch_end) in enumerate(batches):

                    x_batch = data[batch_start:batch_end, :, :, :]
                    x_batch_org = data_org[batch_start: batch_end, :, :, :]


                    sess.run(self.opt, feed_dict={self.x: x_batch,
                                                  self.x_org: x_batch_org,
                                                  self.lrate: current_learning_rate,
                                                  # self.mtum: current_momentum,
                                                  self.is_training: 1})


                    [loss_batch, recon_loss_batch, reg_term_batch] = sess.run([self.loss, self.recon_loss, self.reg_term],
                                                                              feed_dict={self.x: x_batch,
                                                                              self.x_org:x_batch_org,
                                                                              self.lrate: current_learning_rate,

                                                                              # self.mtum: current_momentum,
                                                                              self.is_training: 1})


                    x_decode_batch = sess.run(self.x_decode,
                                              feed_dict={   self.x: x_batch,
                                                            self.lrate: current_learning_rate,
                                                            # self.mtum: current_momentum,
                                                            self.is_training: 1})

                    # batch_error = np.sum(np.sqrt(
                    #     np.mean(np.square(x_batch - x_decode_batch), axis=(1, 2, 3),
                    #             keepdims=False)))
                    total_loss += loss_batch
                    total_recon_loss += recon_loss_batch
                    total_reg_term += reg_term_batch

                total_loss = total_loss / num_data
                total_recon_loss = total_recon_loss / num_data
                total_reg_term = total_reg_term / num_data
                epoch_list += [self.epoch]
                loss_list.append(total_loss)
                recon_loss_list.append(total_recon_loss)
                reg_term_list.append(total_reg_term)
                # print('Epoch %d: recon error = %.5f constraint = %.5f loglike = %.5f\n' % (self.epoch, total_error,total_constraint, total_loglike))
                dualprint('Epoch %d: loss = %.5f recon_lost = %f regularizor = %f' % (self.epoch, total_loss, total_recon_loss, total_reg_term), flog)
                #

                if (self.epoch + 1) % self.save_freq == 0:
                    # model_file = '%s/model.dill' % self.saved_folder
                    # dill.dump(self, open(model_file, 'wb'))
                    # print('Model saved to filee %s\n' % model_file)

                    save_path = self.saver.save(sess, "%s/model.ckpt" % self.saved_folder)
                    dualprint("Model saved in path: %s" % save_path, flog)
                    learn_info_file = '%s/lr_curve.dill' % self.saved_folder
                    dill.dump({'loss': loss_list, 'recon_loss': recon_loss_list,
                               'reg_term': reg_term_list,
                               'epochs': epoch_list}, open(learn_info_file, 'wb'))

                # if self.debug and (self.epoch + 1) % self.disp_freq == 0 and batch_idx == len(batches) - 1:
                if self.debug and (self.epoch + 1) % self.disp_freq == 0:
                    ax1.plot(epoch_list, loss_list, '-r')
                    plt.title('loss')
                    ax2.plot(epoch_list, recon_loss_list, '-r')
                    plt.title('recon_loss')
                    ax3.plot(epoch_list, reg_term_list, '-r')
                    plt.title('reg term')
                    plt.pause(0.05)

                #
                #     x_visual = x[np.random.permutation(num_data)[:100], :, :, :]
                #     x_visual_recon = sess.run(self.v_mean,
                #                               feed_dict={self.v: x_visual, self.beta: 1.0,
                #                                          self.is_training: 1})
                    img_row = int(np.sqrt(x_batch.shape[0]))
                    if img_row % 2 == 1:
                        img_row -= 1
                    print('x_batch_size = %d' % x_batch.shape[0])
                    print('img_row %d' % img_row)
                    img_col = img_row
                    num_img = img_row * img_col
                    x_visual =  deprocess(x_batch[:int(num_img*0.5), :, :, :])
                    x_visual_recon = deprocess(x_decode_batch[:int(num_img*0.5), :, :, :])

                    data_display = np.zeros([x_visual.shape[0] * 2, np.prod(x_visual.shape[1:])])
                    for i in range(int(data_display.shape[0] / 2)):
                        data_display[2 * i, :] = x_visual[i, :].flatten()
                        data_display[2 * i + 1, :] = x_visual_recon[i, :].flatten()

                    title = 'Reconstruction data'
                    # self._disp_images(data_display, fig=ax1.figure,
                    #                   img_shape=[self.input_height, self.input_width],
                    #                   title=title)
                    ax4.grid(0)
                    ax4.set_title(title)
                    ax4.set_xticklabels([])
                    ax4.set_yticklabels([])
                    ax4.set_xlabel("epoch #{}".format(self.epoch), fontsize=28)

                    img = tile_images(data_display, img_shape=[self.input_height, self.input_width, self.input_channels],
                                      tile_shape=(img_row, img_col),
                                      tile_spacing=(1, 1))
                    img_norm = (img - img.min())/(img.max() - img.min())
                    ax4.imshow(img_norm, aspect='auto', cmap='Greys_r', interpolation='none',
                               vmin=0.0, vmax=1.0)

                    ax4.axis('off')
                    ax4.grid(False)
                    recon_file = '%s/recon.jpg' % self.saved_folder
                    cv2.imwrite(recon_file, img*255.0)

                    fig_file = '%s/train.pdf' % self.saved_folder
                    fig.savefig(fig_file)

                    plt.pause(0.05)
                #     # plt.tight_layout()
                #
                #     # img = tile_raster_images(data_display, img_shape=disp_dim, tile_shape=tile_shape,
                #     #                          tile_spacing=(1, 1),
                #     #                          scale_rows_to_unit_interval=False,
                #     #                          output_pixel_vals=output_pixel_vals)
                #
                #     ax2.plot(epoch_list, total_error_list, '-r')
                #     ax2.set_title('Recon error')
                #
                #     ax3.plot(epoch_list, total_constraint_list, '-r')
                #     ax3.set_title('Constraint')
                #
                #     ax4.plot(epoch_list, total_loglike_list, '-r')
                #     ax4.set_title('Loglike')
                #
                #     data_display = v_particle.reshape([v_particle.shape[0], -1])
                #
                #     title = 'Gibbs samples'
                #     # self._disp_images(data_display, fig=ax1.figure,
                #     #                   img_shape=[self.input_height, self.input_width],
                #     #                   title=title)
                #     ax5.grid(0)
                #     ax5.set_title(title)
                #     ax5.set_xticklabels([])
                #     ax5.set_yticklabels([])
                #     ax5.set_xlabel("epoch #{}".format(self.epoch), fontsize=28)
                #
                #     # img = tile_raster_images(data_display,
                #     #                          img_shape=[self.input_height, self.input_width],
                #     #                          tile_shape=(10, 10),
                #     #                          tile_spacing=(1, 1),
                #     #                          scale_rows_to_unit_interval=False,
                #     #                          output_pixel_vals=False)
                #
                #     img = tile_images(data_display,
                #                       img_shape=[self.input_height, self.input_width,
                #                                  self.input_channels],
                #                       tile_shape=(10, 10),
                #                       tile_spacing=(1, 1))
                #     ax5.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none',
                #                vmin=0.0, vmax=1.0)
                #
                #     ax5.axis('off')
                #     ax5.grid(False)
                #
                #     data_display = v4_particle.reshape([v4_particle.shape[0], -1])
                #
                #     title = 'v4 particles'
                #     # self._disp_images(data_display, fig=ax1.figure,
                #     #                   img_shape=[self.input_height, self.input_width],
                #     #                   title=title)
                #     ax6.grid(0)
                #     ax6.set_title(title)
                #     ax6.set_xticklabels([])
                #     ax6.set_yticklabels([])
                #     ax6.set_xlabel("epoch #{}".format(self.epoch), fontsize=28)
                #
                #     img = tile_images(data_display,
                #                       img_shape=[self.input_height, self.input_width,
                #                                  self.input_channels],
                #                       tile_shape=(10, 10),
                #                       tile_spacing=(1, 1))
                #
                #     ax6.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none',
                #                vmin=0.0, vmax=1.0)
                #
                #     ax6.axis('off')
                #     ax6.grid(False)
                #
                #     ax7.plot(epoch_list, total_D_loss_list, '-r')
                #     ax7.set_title('D loss')
                #
                #     ax8.plot(epoch_list, total_adver_CEM_loss_list, '-r')
                #     ax8.set_title('G loss')
                #
                #     [v3_Gibbs_next, h3_Gibbs_next] = sess.run([self.v_Gibbs, self.h_Gibbs],
                #                                               feed_dict={
                #                                                   self.v_model: v3_particle,
                #                                                   self.h_model: h3_particle,
                #                                                   self.beta: 1.0,
                #                                                   # self.lrate: current_learning_rate,
                #                                                   self.mtum: current_momentum,
                #                                                   self.is_training: 1})
                #
                #     data_display = v3_Gibbs_next.reshape([v3_Gibbs_next.shape[0], -1])
                #
                #     title = 'v3 particles Gibbs'
                #     # self._disp_images(data_display, fig=ax1.figure,
                #     #                   img_shape=[self.input_height, self.input_width],
                #     #                   title=title)
                #     ax9.grid(0)
                #     ax9.set_title(title)
                #     ax9.set_xticklabels([])
                #     ax9.set_yticklabels([])
                #     ax9.set_xlabel("epoch #{}".format(self.epoch), fontsize=28)
                #
                #     img = tile_images(data_display,
                #                       img_shape=[self.input_height, self.input_width,
                #                                  self.input_channels],
                #                       tile_shape=(10, 10),
                #                       tile_spacing=(1, 1))
                #
                #     ax9.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none',
                #                vmin=0.0, vmax=1.0)
                #
                #     ax9.axis('off')
                #     ax9.grid(False)
                #
                #     fig_file = '%s/train.pdf' % self.saved_folder
                #     fig.savefig(fig_file)


                self.epoch += 1


            save_path = self.saver.save(sess, "%s/model.ckpt" % self.saved_folder)
            dualprint("Model saved in path: %s" % save_path, flog)
            learn_info_file = '%s/lr_curve.dill' % self.saved_folder
            dill.dump({'loss': loss_list, 'recon_loss': recon_loss_list, 'reg_term': reg_term_list,
                       'epochs': epoch_list}, open(learn_info_file, 'wb'))
            dualprint("Learning info saved as: %s" % learn_info_file, flog)

            fig_file = '%s/train.pdf' % self.saved_folder
            fig.savefig(fig_file)
            dualprint("Figure saved as: %s" % fig_file, flog)

            param_file = '%s/param.dill' % self.saved_folder
            dill.dump(self.get_params(), open(param_file, 'wb'))

        dualprint('Training ends.', flog)
        flog.close()
