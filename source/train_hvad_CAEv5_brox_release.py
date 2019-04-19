
from __future__ import print_function, division
import numpy as np
import socket

import sys
import h5py
import scipy.io as sio

from pix2pix_so_v1_func import preprocess, deprocess, norm_OF_01

from ConvAEv4 import ConvAEv4
from utils.generic_utils import make_batches

computer_name = socket.gethostname()

# SYSINFO = system_config()
# if SYSINFO['display']==False:
#     import matplotlib
#     matplotlib.use('Agg')
# else:
#     import matplotlib

import cv2
import os
import time

from utils.anom_UCSDholderv1 import anom_UCSDholder

bdebug = False
from utils.read_list_from_file import read_list_from_file
from utils.dualprint import  dualprint
from utils.ParamManager import ParamManager

def train_hvad(params):
    # experiment params
    mode = params.get_value('mode')
    data_str = params.get_value('data_str')
    train_str = params.get_value('train_str')
    bshow = params.get_value('bshow')
    bh5py = params.get_value('bh5py')
    batch_size = params.get_value('batch_size')
    encoder_dims = params.get_value('encoder_dims')
    imsz = params.get_value('imsz')
    frame_step = params.get_value('frame_step')

    # learner parameters
    # lr_rate = params.get_value('lr_rate')
    w_init = params.get_value('w_init')
    data_range = params.get_value('data_range')
    # a.batch_size = 1

    OF_scale = 0.3
    h_resz, w_resz = 256, 256

    noise_sigma = 0.2
    gamma = 0.0
    denoising = True
    k_h = 5
    k_w = 5
    d_h = 2
    d_w = 2
    use_bn = True
    # optimizer_name = 'Adam'
    optimizer_name = 'Adagrad'
    cae_lr_rate = 0.1

    dataholder = anom_UCSDholder(data_str, imsz)
    data_folder = dataholder.data_folder

    if data_str in ['Avenue','Avenue_sz240x360fr1', 'UCSDped1']:
        skip_frame = 2
    else:
        skip_frame = 1

    feat_folder = '%s/feat' % (data_folder)
    if not os.path.exists(feat_folder):
        os.mkdir(feat_folder)

    model_folder = '%s/model' % (data_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    res_folder = '%s/result' % (data_folder)
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)

    encoder_act = ['lrelu'] * len(encoder_dims)
    decoder_dims = encoder_dims[:(len(encoder_dims) - 1)]
    decoder_dims.reverse()
    # print(decoder_dims)
    decoder_dims = decoder_dims + [None]
    print('decoder_dims', decoder_dims)
    decoder_act = ['lrelu'] * len(decoder_dims)

    layer_str = '-'.join(str(s) for s in encoder_dims)
    # model_folder_name = 'hvad-%s-lrelu-k%d-gamma%0.2f-denoise%0.2f-bn%d-%s-lr%0.2f-v5-brox' % (
    #     layer_str, k_h, gamma, noise_sigma, use_bn, optimizer_name, cae_lr_rate)
    model_folder_name = 'hvad-%s-release' % (layer_str)
    hvad_model_folder = '%s/%s' % (model_folder, model_folder_name)
    if os.path.isdir(hvad_model_folder) == False:
        os.mkdir(hvad_model_folder)
    flog = open('%s/log.txt' % hvad_model_folder, 'wt')

    # print('Mode = %d' % mode)
    # load features
    frame_file_format = '%s/%s_resz240x360_raw_v3.npz'

    OF_file_format = '%s/%s_sz240x360_BroxOF.mat'
    train_list = read_list_from_file('%s/%s.lst' % (data_folder, train_str))

    train_time_start = time.time()
    if mode in [0, 2]:
        print('Training cae 1:')

        # load all data for training
        data_F = None
        for s in train_list:
            frame_file = frame_file_format % (feat_folder, s)
            if os.path.isfile(frame_file):
                dualprint('Loading %s' % s, flog)
                F = np.load(frame_file)
                print('F shape', F.shape)
                if skip_frame > 1:
                    F = F[::skip_frame, :, :]
                    print('Skipping frame:', F.shape)

                if data_F is None:
                    data_F = F
                else:
                    data_F = np.concatenate([data_F, F], axis=0)
            else:
                dualprint('File %s doesn''t exists' % frame_file, flog)

        dualprint('Convert frame and optical flow into [-1.0, 1.0]')
        # convert frame data in [0.0, 1.0] to [-1.0, 1.0]
        data_F = preprocess(data_F)

        # resize the frame and optical flow into [h_resz, w_resz]
        F_resz = np.zeros([data_F.shape[0], h_resz, w_resz])

        for i in range(F_resz.shape[0]):
            F_resz[i, :, :] = cv2.resize(data_F[i, :, :], (w_resz, h_resz))
        print('F shape', F_resz.shape)
        # print(O_resz.shape)
        print('F min %f max %f' % (F_resz.min(), F_resz.max()))
        # print('O min %f max %f' % (O_resz.min(), O_resz.max()))

        data1 = np.stack((F_resz, F_resz, F_resz), axis=3)

        decoder_dims[-1] = data1.shape[3]

        cae_trainer1_folder = '%s/cae1' % hvad_model_folder
        if os.path.isdir(cae_trainer1_folder) == False:
            os.mkdir(cae_trainer1_folder)
        cae1 = ConvAEv4(   input_height = data1.shape[1],
                                input_width = data1.shape[2],
                                input_channels = data1.shape[3],
                                batch_size=batch_size,
                                num_epochs = num_epochs,
                                lr_rate = cae_lr_rate,
                                w_init = w_init,
                                gamma = gamma,
                                saved_folder = cae_trainer1_folder,
                                k_h = k_h,
                                k_w = k_w,
                                d_h = d_h,
                                d_w = d_w,
                                use_bn = use_bn,
                                denoising = denoising,
                                # encode_dims = [256, 128],
                                # decode_dims = [256, data1.shape[3]],
                                encoder_dims = encoder_dims,
                                encoder_act = encoder_act,
                                decoder_dims = decoder_dims,
                                decoder_act = decoder_act,
                                optimizer_name = optimizer_name,
                                debug = True,
                                disp_freq = disp_freq,
                                save_freq = save_freq,
                                device = device)
        cae1.build_model()


        idx = np.random.permutation(data1.shape[0])
        data1 = data1[idx, :, :, :]


        data1_org = data1.copy()
        if denoising==True:
            # corrupt the data with Gaussian noise

            batches = make_batches(data1.shape[0], 100)
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                print('batch_idx, batch_start, batch_end', batch_idx, batch_start, batch_end)
                data1[batch_start:batch_end, :, :, :] = data1_org[batch_start:batch_end, :, :, :] + np.random.normal(0.0, noise_sigma, size=data1_org[batch_start:batch_end, :, :, :].shape)

                data1[batch_start:batch_end, :, :, :] = np.maximum(data1[batch_start:batch_end, :, :, :], data_range[0])
                data1[batch_start:batch_end, :, :, :] = np.minimum(data1[batch_start:batch_end, :, :, :], data_range[1])

        cae1.fit(data1, data1_org)

    if mode in [1, 2]:
        print('Training cae 2')
        data_O = None
        for s in train_list:
            dualprint('Loading %s' % s, flog)
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

                # print(OF.shape)
                if skip_frame > 1:
                    OF = OF[::skip_frame, :, :, :]
                    print('after skipping frames')
                    print(OF.shape)

                if data_O is None:
                    data_O = OF
                else:
                    data_O = np.concatenate([data_O, OF], axis=0)
            else:
                dualprint('File %s doesn''t exists' % OF_file, flog)

        dualprint('Convert frame and optical flow into [-1.0, 1.0]')
        # convert frame data in [0.0, 1.0] to [-1.0, 1.0]
        data_O = norm_OF_01(data_O, scale=OF_scale)
        data_O = preprocess(data_O)

        print('O shape', data_O.shape)
        print('O min %f max %f' % (data_O.min(), data_O.max()))
        data2 = np.zeros([data_O.shape[0], h_resz, w_resz, 3])
        for i in range(data_O.shape[0]):
            data2[i, :, :, :] = cv2.resize(data_O[i, :, :, :], (w_resz, h_resz))

        decoder_dims[-1] = data2.shape[3]

        cae2_folder = '%s/cae2' % hvad_model_folder
        if os.path.isdir(cae2_folder) == False:
            os.mkdir(cae2_folder)
        cae2 = ConvAEv4(input_height=data2.shape[1],
                        input_width=data2.shape[2],
                        input_channels=data2.shape[3],
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        lr_rate=cae_lr_rate,
                        w_init=w_init,
                        gamma=gamma,
                        saved_folder=cae2_folder,
                        k_h=k_h,
                        k_w=k_w,
                        d_h=d_h,
                        d_w=d_w,
                        use_bn=use_bn,
                        denoising=denoising,
                        # encode_dims = [256, 128],
                        # decode_dims = [256, data1.shape[3]],
                        encoder_dims=encoder_dims,
                        encoder_act=encoder_act,
                        decoder_dims=decoder_dims,
                        decoder_act=decoder_act,
                        optimizer_name=optimizer_name,
                        debug=True,
                        disp_freq=disp_freq,
                        save_freq=save_freq,
                        device=device)
        cae2.build_model()

        idx = np.random.permutation(data2.shape[0])
        data2 = data2[idx, :, :, :]


        data2_org = data2.copy()
        if denoising == True:
            batches = make_batches(data2.shape[0], 100)
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                print('batch_idx, batch_start, batch_end', batch_idx, batch_start, batch_end)
                data2[batch_start:batch_end, :, :, :] = data2_org[batch_start:batch_end, :, :, :] + np.random.normal(0.0, noise_sigma, size=data2[batch_start:batch_end, :, :, :].shape)
                data2[batch_start:batch_end, :, :, :] = np.maximum(data2[batch_start:batch_end, :, :, :], data_range[0])
                data2[batch_start:batch_end, :, :, :] = np.minimum(data2[batch_start:batch_end, :, :, :], data_range[1])

        cae2.fit(data2, data2_org)



    train_time_end = time.time()
    dualprint('Training time: %f (seconds)' % (train_time_end - train_time_start), flog)
    flog.close()
    print('Finished.')

if __name__ == "__main__":

    if len(sys.argv) > 1:
        #  mode = 0: cae 1 on data 1
        #  mode=1: cae 2 on data 2
        #  mode=2: cae 1 on data 1 and cae 2 on data 2
        mode = int(sys.argv[1])

        # dataset name
        data_str = sys.argv[2]

        # training batch size
        batch_size = int(sys.argv[3])

        # encoding structure, e.g., "32-16-8" (3 layers whose # filters are 32, 16, 8 respectively
        encoder_dims = [int(s) for s in sys.argv[4].split('-')]

        # update the monitoring figure after disp_freq epochs
        disp_freq = int(sys.argv[5])
        # save the model after save_freq epochs
        save_freq = int(sys.argv[6])
        # the number of training epochs
        num_epochs = int(sys.argv[7])
        # which device to run on, e.g., cpu or gpu (in TensorFlow format, e.g., '/device:GPU:0')
        device = sys.argv[8]
    else:
        mode = 0
        data_str = 'UCSDped2'
        batch_size = 100
        encoder_dims = [32, 16, 8]
        disp_freq = 10
        save_freq = 10
        num_epochs = 500
        device = '/cpu:0'

    print('mode = %d' % mode)
    print('data_str = %s' % data_str)
    print('batch_size = %d' % batch_size)
    print('encoder_dims :', encoder_dims)

    params = ParamManager()

    train_str = 'train'

    bshow = 1
    bh5py = 1

    frame_step = 5
    params.add('mode', mode, 'hvad_cae')
    params.add('data_str', data_str, 'hvad_cae')
    params.add('train_str', train_str, 'hvad_cae')
    params.add('bshow', bshow, 'hvad_cae')
    params.add('bh5py', bh5py, 'hvad_cae')
    params.add('frame_step', frame_step, 'hvad_cae')
    params.add('batch_size', batch_size, 'hvad_cae')
    params.add('encoder_dims', encoder_dims, 'hvad_cae')


    params.add('num_epochs', num_epochs, 'hvad_cae')
    params.add('disp_freq', disp_freq, 'hvad_cae')
    params.add('save_freq', save_freq, 'hvad_cae')
    params.add('device', device, 'hvad_cae')

    imsz = [240, 360]

    params.add('imsz', imsz, 'hvad_cae')
    # learner paramters
    w_init = 0.01
    data_range = [-1.0, 1.0]

    params.add('w_init', w_init, 'hvad_cae')
    params.add('data_range', data_range, 'hvad_cae')

    train_hvad(params)


