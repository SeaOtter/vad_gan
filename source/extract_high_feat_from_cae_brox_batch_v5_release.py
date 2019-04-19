from __future__ import print_function, division
import numpy as np
import socket

import sys
import dill

import h5py
import scipy.io as sio
import tensorflow as tf

from pix2pix_so_v1_func import deprocess, preprocess, norm_OF_01
from ConvAEv4 import ConvAEv4

computer_name = socket.gethostname()
# SYSINFO = system_config()
# if SYSINFO['display']==False:
#     import matplotlib
#     matplotlib.use('Agg')
# else:
#     import matplotlib

from matplotlib import pyplot as plt
import cv2
import os

from utils.anom_UCSDholderv1 import anom_UCSDholder
from utils.read_list_from_file import read_list_from_file
from utils.dualprint import dualprint

# def frame_process(data_F, resz = None):
#     # convert to [-1, 1]
#     # resize into [256, 256]
#
#     data_F = convert01tom1p1(data_F)
#
#     # # trim the OF data and convert to [-1.0, 1.0]
#     if resz is not None:
#         F_resz = np.zeros([data_F.shape[0], resz[0], resz[1]])
#
#
#         for i in range(F_resz.shape[0]):
#             F_resz[i, :, :] = cv2.resize(data_F[i, :, :], (resz[1], resz[0]))
#     else:
#         F_resz = data_F.copy()
#
#     print(F_resz.shape)
#
#     print('F min %f max %f' % (F_resz.min(), F_resz.max()))
#
#     data_F3c = np.stack((F_resz, F_resz, F_resz), axis=3)
#     return data_F3c

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # # convert to RGB array
    # x *= 255
    # x = x.transpose((1, 2, 0))
    # x = np.clip(x, 0, 255).astype('uint8')
    return x

if len(sys.argv) > 1:
    # dataset name
    data_str = sys.argv[1]
    # the id of the trained CAE that is used to extract the high-level features
    cae_list = [int(sys.argv[2])]
    # training batch size
    batch_size = int(sys.argv[3])
    # the name of the folder where the trained CAE is stored
    model_folder_name = sys.argv[4]
    # the name of file that contains the raw feature filename list
    all_str = sys.argv[5]
    # which device to run on, e.g., cpu or gpu (in TensorFlow format, e.g., '/device:GPU:0')
    device = sys.argv[6]
else:
    data_str = 'UCSDped2_demo'

    cae_list = [1]
    # cae_list = [2]
    batch_size = 50
    model_folder_name = 'hvad-32-16-8-release'
    all_str = 'all'
    device = '/device:GPU:0'

print('Data set: %s' % data_str)
print('cae_list:', cae_list)
print('batch_size:', batch_size)
print('model_folder_name: %s' % model_folder_name)

bshow = 0
bh5py = 1
OF_scale = 0.3
resz = [256, 256]


dataholder = anom_UCSDholder(data_str, resz)
data_folder = dataholder.data_folder

feat_folder = '%s/feat' % (data_folder)
if not os.path.exists(feat_folder):
    os.mkdir(feat_folder)

model_folder = '%s/model' % (data_folder)
MODEL_DIR = model_folder
res_folder = '%s/result' % (data_folder)

hvad_model_folder = '%s/%s' % (model_folder, model_folder_name)

ext = dataholder.ext
imsz = dataholder.imsz


frame_file_format = '%s/%s_resz240x360_raw_v3.npz'
OF_file_format = '%s/%s_sz240x360_BroxOF.mat'
test_list = read_list_from_file('%s/%s.lst' % (data_folder, all_str))

for i in range(len(cae_list)):
    cae_id = cae_list[i]
    # loading the data
    print('Load the model information')
    cae_folder = '%s/cae%d' % (hvad_model_folder, cae_id)

    # load model parameters
    dill_data = dill.load(open('%s/param.dill' % (cae_folder), 'rb'))

    input_height = dill_data['input_height']
    input_width = dill_data['input_width']
    input_channels = dill_data['input_channels']

    num_epochs = dill_data['num_epochs']
    cae_lr_rate = dill_data['lr_rate']
    w_init = dill_data['w_init']
    encoder_dims = dill_data['encoder_dims']
    encoder_act = dill_data['encoder_act']
    decoder_dims = dill_data['decoder_dims']
    decoder_act = dill_data['decoder_act']
    gamma = dill_data['gamma']
    k_h = dill_data['k_h']
    k_w = dill_data['k_w']
    d_h = dill_data['d_h']
    d_w = dill_data['d_w']

    use_bn = dill_data['use_bn']
    optimizer_name = dill_data['optimizer_name']

    disp_freq = 10
    save_freq = 10

    print('Extracting features from cae %d' % cae_id)
    cae = ConvAEv4(input_height = input_height,
                            input_width = input_width,
                            input_channels = input_channels,
                            batch_size=batch_size,
                            num_epochs = num_epochs,
                            lr_rate = cae_lr_rate,
                            w_init = w_init,
                            gamma = gamma,
                            k_h = k_h,
                            k_w = k_w,
                            use_bn = use_bn,
                            saved_folder = cae_folder,
                            # encode_dims = [256, 128],
                            # decode_dims = [256, data1.shape[3]],
                            encoder_dims=encoder_dims,
                            encoder_act = encoder_act,
                            decoder_dims = decoder_dims,
                            decoder_act = decoder_act,
                            optimizer_name = optimizer_name,
                            debug = True,
                            disp_freq = disp_freq,
                            save_freq = save_freq,
                            device = device)
    cae.build_model()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,'%s/model.ckpt' % cae_folder)
        if bshow:
            plt.figure()
            plt.show(block=False)

        for s in test_list:
            print('Video %s' % s)
            if cae_id==1:

                frame_file = frame_file_format % (feat_folder, s)
                if os.path.isfile(frame_file):
                    F = np.load(frame_file)
                    print('F shape', F.shape)

                else:
                    print('File %s doesn''t exists' % frame_file)

                print('Convert frame and optical flow into [-1.0, 1.0]')
                # convert frame data in [0.0, 1.0] to [-1.0, 1.0]
                F = preprocess(F)
                # resize the frame and optical flow into [h_resz, w_resz]
                data_s = np.zeros([F.shape[0], resz[0], resz[1]])
                # O_resz = np.zeros([data_O.shape[0], h_resz, w_resz, 3])

                for i in range(data_s.shape[0]):
                    data_s[i, :, :] = cv2.resize(F[i, :, :], (resz[1], resz[0]))
                data_s = np.stack((data_s, data_s, data_s), axis=3)
            elif cae_id==2:
                OF_file = OF_file_format % (feat_folder, s)
                if os.path.isfile(OF_file):

                    if bh5py == 1:
                        f_h5py = h5py.File(OF_file, 'r')
                        OF = f_h5py['O']
                        OF = np.array(OF).T
                        print(OF.shape)
                    else:
                        mat_data = sio.loadmat(OF_file)
                        OF = mat_data['O']
                    last_OF = OF[-1, :, :, :]
                    last_OF = np.reshape(last_OF, [1, *last_OF.shape])
                    data_O = np.concatenate([OF, last_OF], axis=0)
                else:
                    dualprint('File %s doesn''t exists' % OF_file, flog)

                dualprint('Convert frame and optical flow into [-1.0, 1.0]')
                # convert frame data in [0.0, 1.0] to [-1.0, 1.0]
                data_O = norm_OF_01(data_O, scale=OF_scale)
                data_O = preprocess(data_O)

                print('O shape', data_O.shape)
                print('O min %f max %f' % (data_O.min(), data_O.max()))
                data_s = np.zeros([data_O.shape[0], resz[0], resz[1], 3])
                for i in range(data_O.shape[0]):
                    data_s[i, :, :, :] = cv2.resize(data_O[i, :, :, :], (resz[1], resz[0]))

            print('data shape', data_s.shape)
            print('data min and max [%f, %f]' % (data_s.min(), data_s.max()))

            # batch_size = 500
            if data_s.shape[0]< batch_size:
                rep = sess.run(cae.en_layers, feed_dict={cae.x: data_s,
                                                              cae.is_training: 0})
            else:
                # using batch size
                rep = None

                for i in range(0, data_s.shape[0], batch_size):

                    i_end = np.minimum(i + batch_size, data_s.shape[0])
                    print('batch: %d --> %d' % (i, i_end))
                    rep_i = sess.run(cae.en_layers, feed_dict={cae.x: data_s[i:i_end, :, :, :],
                                                             cae.is_training: 0})
                    if rep is None:
                        rep = []
                        for j in range(len(rep_i)):
                            rep.append(rep_i[j])
                    else:
                        for j in range(len(rep_i)):
                            rep[j] = np.concatenate((rep[j],rep_i[j]), axis=0)

                for i in range(len(rep)):
                    print(rep[i].shape)

            if bshow==1:
                n_c = 5
                n_r = 4
                count = 1
                idx = 20
                plt.subplot(n_r, n_c, 1)
                plt.imshow(deprocess(data_s[idx, :, :, :]))
                count +=1
                count = 5
                for j in range(rep[-1].shape[3]):
                    plt.subplot(n_r, n_c, count + j)
                    a = rep[-1][idx, :, :, j ]
                    a = (a - a.min())/(a.max() - a.min())
                    plt.imshow(a , cmap='Greys_r')
                    plt.grid(False)
                    plt.title('channel %d'% j)
                plt.show(block=False)


            for l in range(1, len(rep)):
                high_feat_file = '%s/%s_resz%dx%d_cae%d_layer%d' % (hvad_model_folder, s, resz[0], resz[1], cae_id, l)
                np.savez(high_feat_file, feat=rep[l])
                print('Saving %s' % high_feat_file)

print('Finished.')