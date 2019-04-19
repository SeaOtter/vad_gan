
from __future__ import print_function, division
import numpy as np
import dill
import copy
import h5py
import scipy.io as sio
import tensorflow as tf
import pix2pix_so_v1_func_largev1a as pix2pix
from pix2pix_so_v1_func_largev1a import norm_OF_01
from utils.util import convert01tom1p1

from utils.sytem_config import system_config
from train_hvad_GANv5_brox_largev2_reshape_release import reshape_feat, process_feat_reshape_resize
epsilon = 1e-8
SYSINFO = system_config()

if SYSINFO['display']==False:
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib

import cv2
import os
import time
from utils.anom_UCSDholderv1 import anom_UCSDholder
from utils.util import load_feat
import sys

bdebug = False

from utils.read_list_from_file import read_list_from_file

from utils.dualprint import  dualprint
from utils.ParamManager import ParamManager

def build_model(tf_model_folder, h_resz, w_resz, num_channels):
    M = {}
    paths_batch = tf.placeholder(tf.string)
    inputs_batch = tf.placeholder(tf.float32, shape=[1, h_resz, w_resz, num_channels])
    targets_batch = tf.placeholder(tf.float32, shape=[1, h_resz, w_resz, num_channels])

    paths = paths_batch
    inputs = inputs_batch
    targets = targets_batch

    gen_layers1 = [(256, 64),
                   (128, 128),
                   (64, 256),
                   (32, 512),
                   (16, 512),
                   (8, 512),
                   (4, 512),
                   (2, 512)]

    gen_layers2 = [
        (4, 512, 0.5),
        (8, 512, 0.5),
        (16, 512, 0.5),
        (32, 512, 0.0),
        (64, 256, 0.0),
        (128, 128, 0.0),
        (256, 64, 0.0)]

    gen_layers_specs1 = []
    gen_layers_specs2 = []

    for i in range(len(gen_layers1)):
        if gen_layers1[i][0] == h_resz:
            num_gen_feats = gen_layers1[i][1]
            num_dis_feats = num_gen_feats
            gen_layers_specs1 = [out_dim for _, out_dim in gen_layers1[i + 1:]]
            break
    print('num_gen_feats = %d' % num_gen_feats)
    print('num_dis_feats = %d' % num_dis_feats)
    print('gen_layers_specs1', gen_layers_specs1)
    for i in range(len(gen_layers2)):
        if gen_layers2[i][0] == h_resz:
            gen_layers_specs2 = gen_layers2[:i + 1]
            gen_layers_specs2 = [(out_dim, drop) for _, out_dim, drop in gen_layers_specs2]
            break
    # n = int(np.log2(data_F.shape[1])) - 1
    # print('n = %d' % n)
    # layer_specs = layer_specs_full[:n]
    dis_layers = [
        [256, 64, 2],  # gdf, feat_dims, stride
        [128, 128, 2],
        [64, 256, 2],
        [32, 512, 1],
        [31, 1, 1]
    ]
    if h_resz >= 256:
        strides = [2, 2, 2, 1, 1]
    elif h_resz >= 128:
        strides = [2, 2, 1, 1, 1]
    elif h_resz >= 64:
        strides = [2, 1, 1, 1, 1]
    else:
        strides = [1, 1, 1, 1, 1]

    dis_layers_specs = copy.copy(dis_layers)
    for i in range(len(dis_layers)):
        dis_layers_specs[i][2] = strides[i]
    if h_resz <= 32:
        dis_layers_specs = [[32, 512, 1]]
    else:
        for i in range(len(dis_layers)):
            if dis_layers[i][0] == h_resz:
                dis_layers_specs = dis_layers[i:-1]

    pix2pix.a.ngf = num_gen_feats
    pix2pix.a.ndf = num_dis_feats
    pix2pix.a.gen_layers_specs1 = gen_layers_specs1
    pix2pix.a.gen_layers_specs2 = gen_layers_specs2
    pix2pix.a.dis_layers_specs = dis_layers_specs

    # inputs and targets are [batch_size, height, width, channels]
    model = pix2pix.create_model(inputs_batch, targets_batch)

    outputs = model.outputs

    M['paths_batch'] = paths_batch
    M['model'] = model
    M['inputs_batch'] = inputs_batch
    M['targets_batch'] = targets_batch


    M['inputs'] = inputs
    M['targets'] = targets
    M['outputs'] = outputs

    return M


def reconstruct(A_resz, B_resz, M, sess):
    h_resz = A_resz.shape[1]
    w_resz = A_resz.shape[2]
    num_channels = A_resz.shape[3]

    C = np.zeros(B_resz.shape)
    start = time.time()
    for step in range(A_resz.shape[0]):
        results_outputs = sess.run(
            M['outputs'],

            feed_dict={M['paths_batch']: "%s" % step,
                       M['inputs_batch']: np.reshape(A_resz[step, :, :, :],
                                                [1, h_resz, w_resz, num_channels]),
                       M['targets_batch']: np.reshape(B_resz[step, :, :, :],
                                                 [1, h_resz, w_resz, num_channels])})
        C[step, :, :, :] = results_outputs

    print("time (seconds)", (time.time() - start))
    return C


def test_compute_recon(params):
    # experiment params
    mode = params.get_value('mode')
    mode_explain = params.get_value('mode_explain')

    layers = params.get_value('layers')
    layers_with_cluster = params.get_value('layers_with_cluster')
    data_str = params.get_value('data_str')
    test_str = params.get_value('test_str')
    gan0_folder_name = params.get_value('gan0_folder_name')

    bh5py = params.get_value('bh5py')

    resz = params.get_value('resz')
    data_range = params.get_value('data_range')

    OF_scale = 0.3
    alpha = 2.0
    dataholder = anom_UCSDholder(data_str, resz)
    data_folder = dataholder.data_folder


    feat_folder = '%s/feat' % (data_folder)
    if not os.path.exists(feat_folder):
        os.mkdir(feat_folder)

    model_folder = '%s/model' % (data_folder)

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    res_folder = '%s/result' % (data_folder)
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)

    time_recon_start = time.time()
    cae_id = mode + 1
    for l in range(len(layers)):

        if layers[l] == 0:
            gan_model_folder = '%s/%s/layer%d-usecluster%d-%s-large-v2-reshape' % (model_folder, gan0_folder_name, layers[l], layers_with_cluster[l], mode_explain[mode])

        else:
            gan_model_folder = '%s/%s/layer%d-usecluster%d-%s-large-v2-reshape' % (model_folder, cae_folder_name, layers[l], layers_with_cluster[l], mode_explain[mode])

        cae_folder = '%s/%s' % (model_folder, cae_folder_name)
        if layers[l] == 0:
            vis_folder = '%s/%s/recon' % (model_folder, gan0_folder_name)
        else:
            vis_folder = '%s/recon' % (cae_folder)

        if os.path.isdir(vis_folder) == False:
            os.makedirs(vis_folder)
        flog = open('%s/test_compute_recon_log_mode%d_largev2_reshape.txt' % (vis_folder, mode), 'wt')
        dualprint('mode = %d' % mode, flog)

        pix2pix.a.checkpoint = gan_model_folder
        # load a file Test001 to get the num_channel
        if layers[0] == 0:
            num_channel = 3
            feat_sz = resz
        else:

            reshape_info_file = '%s/reshape_info_layer%d_large_v2.dill' % (cae_folder, layers[l])
            shape_info = dill.load(open(reshape_info_file, 'rb'))
            print(shape_info)
            feat_sz = [shape_info['resize'][0], shape_info['resize'][1]]
            num_channel = shape_info['resize'][2]

        gan = build_model(gan_model_folder, feat_sz[0], feat_sz[1], num_channel)

        saver = tf.train.Saver(max_to_keep=1)

        sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            dualprint("loading model from checkpoint %s" % gan_model_folder, flog)
            checkpoint = tf.train.latest_checkpoint(gan_model_folder)

            saver.restore(sess, checkpoint)

            test_list = read_list_from_file('%s/%s.lst' % (data_folder, test_str))
            data_mean_F, data_std_F, data_scale_F = None, None, None
            data_mean_M, data_std_M, data_scale_M = None, None, None

            for i_s in range(len(test_list)):
                s = test_list[i_s]
                dualprint('[%s]' % s, flog)
                if layers[l] == 0:

                    # load raw 3 contiguous frame data
                    feat_file_format = '%s/%s_resz%sx%s_raw_v3.npz' % ('%s', '%s', resz[0], resz[1])
                    frame_file = feat_file_format % (feat_folder, s)
                    if os.path.isfile(frame_file):
                        dualprint('Loading %s' % s, flog)
                        F = np.load(frame_file)
                        print('F shape', F.shape)

                    else:
                        dualprint('File %s doesn''t exists' % frame_file, flog)

                    dualprint('Convert frame and optical flow into [-1.0, 1.0]')
                    # convert frame data in [0.0, 1.0] to [-1.0, 1.0]
                    F = convert01tom1p1(F)
                    F_resz = np.zeros([F.shape[0], resz[0], resz[1]])
                    for i in range(F.shape[0]):
                        F_resz[i, :, :] = cv2.resize(F[i, :, :], (resz[1], resz[0]))

                    data_F_s_resz = np.stack((F_resz, F_resz, F_resz), axis=3)

                    OF_file_format = '%s/%s_sz240x360_BroxOF.mat'
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
                        OF = np.concatenate([OF, last_OF], axis=0)

                    else:
                        dualprint('File %s doesn''t exists' % OF_file, flog)

                    OF = norm_OF_01(OF, scale=OF_scale)
                    OF = convert01tom1p1(OF)

                    print('O shape', OF.shape)
                    print('O min %f max %f' % (OF.min(), OF.max()))
                    OF_resz = np.zeros([OF.shape[0], resz[0], resz[1], 3])
                    for i in range(OF.shape[0]):
                        OF_resz[i, :, :, :] = cv2.resize(OF[i, :, :, :], (resz[1], resz[0]))

                    data_M_s_resz = OF_resz
                else:
                    feat_F_file_format = '%s/%s_resz%sx%s_cae1_layer%d.npz' % (
                                            '%s', '%s', resz[0], resz[1], layers[l])
                    npz_data = load_feat([s], cae_folder, feat_F_file_format)
                    data_F_s = npz_data['feat']

                    feat_M_file_format = '%s/%s_resz%sx%s_cae2_layer%d.npz' % (
                        '%s', '%s', resz[0], resz[1], layers[l])
                    npz_data = load_feat([s], cae_folder, feat_M_file_format)
                    data_M_s = npz_data['feat']

                    if data_mean_F is None:
                        mean_std_file = '%s/mean_std_layer%d_large_v2.dill' %  (cae_folder, layers[l])
                        dill_data = dill.load(open(mean_std_file,'rb'))
                        dualprint('Loading mean-std file: %s' % mean_std_file)
                        data_mean_F = dill_data['cae1_mean']
                        data_std_F = dill_data['cae1_std']
                        data_scale_F = dill_data['cae1_scale']
                        data_mean_M = dill_data['cae2_mean']
                        data_std_M = dill_data['cae2_std']
                        data_scale_M = dill_data['cae2_scale']

                    dualprint('Original F shape: %d x %d' % (data_F_s.shape[1], data_F_s.shape[2]))
                    dualprint('Original M shape: %d x %d' % (data_M_s.shape[1], data_M_s.shape[2]))

                    dualprint('Normalizing to [%d, %d] using trained mean and standard deviation' % (data_range[0], data_range[1]))

                    data_F_s = np.divide(data_F_s - data_mean_F, data_std_F+epsilon)*data_scale_F
                    data_F_s = np.minimum(np.maximum(data_F_s, data_range[0]), data_range[1])

                    data_M_s = np.divide(data_M_s - data_mean_M, data_std_M+epsilon)*data_scale_M
                    data_M_s = np.minimum(np.maximum(data_M_s, data_range[0]), data_range[1])
                    # resize to [256, 256]
                    print('Resizing to [%d, %d]' % (resz[0], resz[1]))
                    data_F_s_resz = data_F_s
                    data_M_s_resz = data_M_s


                print('data F shape', data_F_s_resz.shape)
                print('data F min and max [%f, %f]' % (data_F_s_resz.min(), data_F_s_resz.max()))

                print('data M shape', data_M_s_resz.shape)
                print('data M min and max [%f, %f]' % (data_M_s_resz.min(), data_M_s_resz.max()))

                if layers[l]>0:
                    print('reshaping and resizing')
                    data_F_s_resz = process_feat_reshape_resize(data_F_s_resz, resz)
                    data_M_s_resz = process_feat_reshape_resize(data_M_s_resz, resz)
                    print('After reshaping and resizing')
                    print('data F shape', data_F_s_resz.shape)
                    print(
                        'data F min and max [%f, %f]' % (data_F_s_resz.min(), data_F_s_resz.max()))

                    print('data M shape', data_M_s_resz.shape)
                    print(
                        'data M min and max [%f, %f]' % (data_M_s_resz.min(), data_M_s_resz.max()))

                if mode == 0:
                    M_s_recon = reconstruct(data_F_s_resz, data_M_s_resz, gan, sess)
                    M_s_recon_file = '%s/%s_M_recon_layer%d_usecluster%d_large_v2_reshape.npz' % (vis_folder, s, layers[l], layers_with_cluster[l])
                    np.savez(M_s_recon_file, M_recon=M_s_recon)
                    print('Saving %s' % M_s_recon_file)
                elif mode == 1:
                    F_s_recon = reconstruct(data_M_s_resz, data_F_s_resz, gan, sess)
                    F_s_recon_file = '%s/%s_F_recon_layer%d_usecluster%d_large_v2_reshape.npz' % (vis_folder, s, layers[l], layers_with_cluster[l])
                    np.savez(F_s_recon_file, F_recon=F_s_recon)
                    print('Saving %s' % F_s_recon_file)

    time_recon_end = time.time()
    dualprint('Test time %f (seconds):' % (time_recon_end - time_recon_start), flog)

    print('Finished.')
    flog.close()

if __name__ == "__main__":

    # python3 test_compute_recon 0 0 True hvad-64-32-16-16-16-lrelu-k5-gamma0.00-denoise0.20-bn1-Adagrad1-lr0.10-v4
    if len(sys.argv) > 1:
        # dataset name
        data_str = sys.argv[1]

        # mode=0: use GAN_F->M (generate motion maps from frames)
        # mode=1: use GAN_M->F (generate frames from motion maps)
        mode = int(sys.argv[2])

        # the corresponding layer
        # layers = [0]: generate maps between low-level features
        # layers = [1]: generate maps between high-level features at the first hidden units
        # layers = [2]: generate maps between high-level features at the second hidden units
        # ...
        layers = [int(sys.argv[3])]

        # not used
        layers_with_cluster = [sys.argv[4] in ['True', 'T', 'true', 't', '1']]

        # the folder name of trained CAEs
        if len(sys.argv) >= 6:
            cae_folder_name = sys.argv[5]
        else:
            cae_folder_name = None

        # the folder name of trained low-level GANs
        if len(sys.argv) >= 7:
            gan0_folder_name = sys.argv[6]
        else:
            gan0_folder_name = None

    else:
        raise ValueError("Please provide some arguments")

    mode_explain = ['FtoM', 'MtoF']
    params = ParamManager()

    # experiment parameters
    data_range = [-1.0, 1.0]

    print('Mode: %d' % mode)
    print('layers: ', layers)
    print('layers_with_cluster', layers_with_cluster)
    print('cae_folder_name', cae_folder_name)
    print('gan0_folder_name', gan0_folder_name)

    train_str = 'train'
    test_str = 'test'

    bshow = 1
    bsave = 1
    bh5py = 1

    params.add('mode', mode, 'hvad')
    params.add('mode_explain', mode_explain, 'hvad')
    params.add('layers', layers, 'hvad')
    params.add('layers_with_cluster', layers_with_cluster, 'hvad')
    params.add('cae_folder_name', cae_folder_name, 'hvad')
    params.add('gan0_folder_name', gan0_folder_name, 'hvad')
    params.add('data_str', data_str, 'hvad')
    params.add('test_str', test_str, 'hvad')
    params.add('bshow', bshow, 'hvad')
    params.add('bsave', bsave, 'hvad')
    params.add('bh5py', bh5py, 'hvad')

    params.add('data_range', data_range, 'hvad')


    resz = [256, 256]
    thresh = 0.9

    params.add('resz', resz, 'detector')
    params.add('thresh', thresh, 'detector')

    test_compute_recon(params)


