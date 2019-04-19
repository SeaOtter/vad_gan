from __future__ import print_function, division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
import math
from utils.read_list_from_file import read_list_from_file
from utils.anom_UCSDholderv1 import anom_UCSDholder

def feat_resize(data_str, imsz, resz, bshow=0):

    # train_str = 'test'
    # test_str = 'test'

    dataholder = anom_UCSDholder(data_str, imsz)

    data_folder = dataholder.data_folder

    feat_folder = '%s/feat' % (data_folder)
    if not os.path.exists(feat_folder):
        os.mkdir(feat_folder)



    feat_file_format = '%s/%s_resz%dx%d_raw_v3.npz'

    if bshow==1:
        fig = plt.figure('Resize')


    test_list = read_list_from_file('%s/all.lst' % (data_folder))


    for s in test_list:
        frm_folder = "%s/%s" % (data_folder, s)

        feat_file = feat_file_format % (
                        feat_folder, s, imsz[0], imsz[1])
        resz_file = feat_file_format % (feat_folder, s, resz[0], resz[1])
        if os.path.isfile(feat_file):
            print('Loading %s' % s)
            data_test = np.load(feat_file)
            data_im = data_test
            num_data, h, w = data_test.shape
            D = np.zeros([num_data, *resz])
            for i in range(0, num_data):
                # im = data_im[i,:,:].astype('uint8')
                im = data_im[i, :, :]
                im_resz = cv2.resize(im, (resz[1], resz[0]))

                # im_resz = im_resz/255.0
                D[i, :, :] = im_resz

                if bshow==1:

                    plt.clf()
                    plt.subplot(1, 2, 1)
                    plt.imshow(im,cmap='Greys_r')

                    plt.title('Original %d' % i)

                    plt.subplot(1, 2, 2)
                    plt.imshow(im_resz, cmap='Greys_r')
                    plt.title('Resize %d' % i)

                    plt.show(block=False)
                    plt.pause(0.05)
            np.save(open(resz_file, 'wb'), D)
            print('saved to %s' % resz_file)
    print('Finished.')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # dataset name list
        data_list = [sys.argv[1]]

        # souce feature size
        imsz = eval(sys.argv[2])

        # the new size of the features
        resz = eval(sys.argv[3])
    else:
        data_list = ['UCSDped2_demo']
        imsz = [240, 360]
        resz = [256, 256]

    bshow = 0
    for data_str in data_list:
        print(data_str)
        feat_resize(data_str, imsz, resz, bshow=bshow)