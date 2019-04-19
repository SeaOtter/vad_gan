# from __future__ import print_function, division
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.read_list_from_file import read_list_from_file

import math
import cv2
import glob
import os
import sys

from utils.anom_UCSDholderv1 import anom_UCSDholder

def preprocess(img_folder, output_file, imsz, resz=None, ext='jpg', feature=0, bshow=1):
    if bshow == 1:
        plt.figure()

    img_files = glob.glob("%s/*.%s" % (img_folder, ext))
    img_files.sort()

    N = len(img_files)
    X = None
    winSize = (resz[1], resz[0])

    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    if feature ==1:
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    for j in range(N):
        im_name = img_files[j]
        if j % 1000 == 0:
            print('%d/%d' % (j, N))
        im = cv2.imread(im_name, 0)
        if resz is None:
            im_resz = im
        else:
            # im_resz = cv2.resize(im, (resz[1],resz[0]), interpolation=cv2.INTER_CUBIC)
            im_resz = cv2.resize(im, (resz[1], resz[0]), interpolation=cv2.INTER_LINEAR)

        if bshow == 1:
            plt.subplot(2, 4, 1)
            plt.imshow(im, cmap='Greys_r')
            plt.title('Original Image')
            plt.subplot(2, 4, 2)
            plt.imshow(im_resz, cmap='Greys_r')
            plt.title('Processed image')

        if feature == 0:
            feat = im_resz / 255.0
        elif feature == 1:
            # patch_scale = cv2.resize(im_resz, winSize, interpolation=cv2.INTER_CUBIC)
            h = hog.compute(im_resz)
            feat = np.squeeze(h)

        if X is None:
            X = np.zeros((N, feat.shape[0], feat.shape[1]))
        X[j, :, :] = feat
        if bshow == 1:
            plt.show()

    print(X.shape)
    print('Saving to %s' % output_file)
    np.save(open(output_file, 'wb'), X)
    print('Saved')

def feat_extract(data_str, resz):
    feature = 0 # 0: raw pixel, 1: HOG
    if data_str == 'UCSDped2_demo':
        dataholder = anom_UCSDholder(data_str, resz)

    data_folder = dataholder.data_folder


    feat_folder = '%s/feat' % (data_folder)
    if not os.path.exists(feat_folder):
        os.mkdir(feat_folder)

    model_folder = '%s/model' % (data_folder)
    MODEL_DIR = model_folder
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    ext = dataholder.ext
    imsz = dataholder.imsz

    if resz is None:
        resz = imsz
    if feature == 1:
        resz = (math.ceil(resz[0]*1.0/16)*16,math.ceil(resz[1]*1.0/16)*16)
        resz = (int(resz[0]), int(resz[1]))

    h, w = 12, 18
    h_step, w_step = 6, 9

    # feat_file_format = '%s/%s_resz%dx%d_cellsz%dx%d_step%dx%d%s_v2.npz'
    feat_file_format = '%s/%s_resz%dx%d_%s_v3.npz'

    if feature == 0:
        strfeature = "raw"
    elif feature == 1:
        strfeature = "HOG"

    print('Extracting %s feature from training videos...' % strfeature)
    video_list = read_list_from_file('%s/all.lst' % data_folder)

    for s in video_list:
        frm_folder = "%s/%s" % (data_folder, s)
        print(frm_folder)
        feat_file = feat_file_format % (
            feat_folder, s, resz[0], resz[1], strfeature)
        if os.path.isfile(feat_file)==False:
            preprocess(frm_folder, feat_file, imsz, resz,  ext, feature, bshow=0)
        else:
            print('File exists.')

    print('Finished.')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # dataset name
        data_str = sys.argv[1]
        # raw feature size
        resz = eval(sys.argv[2])
    else:
        data_str = 'UCSDped2_demo'
        resz = [240, 360]
    feat_extract(data_str, resz)