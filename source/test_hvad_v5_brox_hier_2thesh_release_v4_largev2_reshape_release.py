from __future__ import print_function, division
import numpy as np
import sys
import dill
import copy
import h5py

import scipy.io as sio
from utils.util import empty_struct

from utils.util import convert01tom1p1, convertm1p1to01
from pix2pix_so_v1_func_largev1a import norm_OF_01
from train_hvad_GANv5_brox_largev2_reshape_release import reshape_feat, process_feat_reshape_resize, compute_reshape
epsilon = 1e-8

def find_components3d(D):
    Mask = D.copy()*(-1.0)
    compid = 1


    N, M, L = D.shape
    D_index = np.argwhere(D==1.0)

    for l in range(D_index.shape[0]):
        i, y, x = D_index[l, 0], D_index[l, 1], D_index[l, 2]

        if Mask[i, y, x] < 0:

            nb = get_neighbours3d(Mask, (i, y, x))
            if nb.size == 0:
                Mask[i, y, x] = compid
                compid += 1
            else:
                lbl = nb[:, 3].max()
                if lbl < 0:
                    lbl = compid
                    compid += 1
                if i == 29 and y == 154 and x == 356:
                    print('abc')
                Mask = propagate(Mask, (i, y, x), lbl, bgvalue=0)

    # vcount[i, 0]: component size
    # vcount[i, 1]: longevity
    # vcount[i, 2]: start frame id
    # vcount[i, 3]: end frame id
    vcount = np.zeros((compid - 1, 4))
    for i in range(len(vcount)):
        mask = (Mask == i + 1)
        vcount[i, 0] = np.sum(mask)
        t = np.sum(mask, axis=(1, 2))
        vcount[i, 1] = (t > 0).sum()

        for j in range(len(t)):
            if t[j] > 0:
                vcount[i, 2] = j
                break

        for j in reversed(range(len(t))):
            if t[j] > 0:
                vcount[i, 3] = j
                break
    return Mask, vcount

def propagate(D, pos, lbl, bgvalue=0):
    Q = []
    inQ = np.zeros(D.shape, dtype=np.bool)
    Q.append(pos)
    inQ[pos[0], pos[1], pos[2]] = True
    while len(Q) > 0:
        p = Q.pop()
        z0, y0, x0 = p
        nb = get_neighbours3d(D, (z0, y0, x0), bgvalue=bgvalue)
        D[z0, y0, x0] = lbl
        for i in range(nb.shape[0]):
            z, y, x = nb[i, 0], nb[i, 1], nb[i, 2]
            # if D[z, y, x] < 0:
            if D[z, y, x] < 0 and inQ[z, y, x] == False:
                Q.append((z, y, x))
                inQ[z, y, x] = True

    return D
def get_neighbours3d(D, posi, bgvalue=0, radius_x=1, radius_y=1, radius_z=1):
    num_z, num_y, num_x = D.shape
    zc, yc, xc = posi
    z1 = np.maximum(zc - radius_z, 0)
    z2 = np.minimum(zc + radius_z + 1, num_z)
    y1 = np.maximum(yc - radius_y, 0)
    y2 = np.minimum(yc + radius_y + 1, num_y)
    x1 = np.maximum(xc - radius_x, 0)
    x2 = np.minimum(xc + radius_x + 1, num_x)
    l = []
    for z in range(z1, z2):
        for y in range(y1, y2):
            for x in range(x1, x2):
                if D[z, y, x] != bgvalue:
                    if z != zc or y != yc or x != xc:
                        l.append([z, y, x, D[z, y, x]])
    return np.array(l).astype(int)

def reverse_channel(im):
    im_rev = im.copy()
    if len(im.shape) >= 3:
        im_rev[:, :, 0] = im[:, :, 2]
        im_rev[:, :, 2] = im[:, :, 0]
    # else:
    #     print('No reversion operation is applied.')
    return im_rev


from evalv1 import evalv1

from utils.sytem_config import system_config
SYSINFO = system_config()
if SYSINFO['display']==False:
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib

from matplotlib import pyplot as plt
from pylab import cm
if sys.version_info[0]<3:
    import cPickle as pkl
else:
    import pickle as pkl

import cv2
import glob
import os
import time

from utils.anom_UCSDholderv1 import anom_UCSDholder
from utils.util import load_feat
import sys

bdebug = False

from utils.read_list_from_file import read_list_from_file
from utils.dualprint import  dualprint
from utils.ParamManager import ParamManager

def visualize_save_v2(vis_data, threshold,
                  title, im_format, gt_im_files, vis_folder=None, bshow=True, frame_step=5, pause_time = 1.0):
    F0_diff = vis_data.F0_diff
    F0_diff_min = F0_diff.min()
    F0_diff_max = F0_diff.max()

    if len(vis_data.M0_diff.shape) == 4:
        M0_diff_OF = vis_data.M0_diff[:, :, :, 1]
    else:
        M0_diff_OF = vis_data.M0_diff
    M0_diff_OF_min = M0_diff_OF.min()
    M0_diff_OF_max = M0_diff_OF.max()

    F4_diff = vis_data.F4_diff
    F4_diff_min = F4_diff.min()
    F4_diff_max = F4_diff.max()

    M4_diff = vis_data.M4_diff
    M4_diff_min = M4_diff.min()
    M4_diff_max = M4_diff.max()

    E_map = vis_data.E_map
    E_map_min = E_map.min()
    E_map_max = E_map.max()

    # D_map = vis_data.D_map
    D_map_2 = vis_data.D_map_2

    fontsz = 7
    titlefontsz = fontsz
    matplotlib.rcParams.update({'font.size': fontsz})
    # resz = grid['imsz']
    num_data, height, width, n_channels = vis_data.F0.shape

    nr = 5
    nc = 4

    if bshow==True:

        figsize = matplotlib.rcParams['figure.figsize']
        figsize = [figsize[0], figsize[1]*2.0]
        fig, axes = plt.subplots(nr, nc, figsize = figsize)
        plt.show(block=False)

    for i in range(0, num_data, frame_step):

        if bshow == True:
            c = 1
            plt.clf()
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            # plt.imshow(im_resz, cmap='Greys_r', vmin=0, vmax=255)
            if vis_data.layer_ids[0]==0:
                plt.imshow(convertm1p1to01(vis_data.F0[i, :, :, :]), cmap='Greys_r', vmin=0, vmax=1.0)
            else:
                plt.imshow(convertm1p1to01(vis_data.F0[i, :, :, :]).mean(axis=2), cmap='jet', vmin=0, vmax=1.0)

            ax.set_title('F0', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            if vis_data.layer_ids[0] == 0:
                plt.imshow(convertm1p1to01(vis_data.M0[i, :, :, :]), cmap='jet', vmin=0.0, vmax=1.0)
            else:
                plt.imshow(convertm1p1to01(vis_data.M0[i, :, :, :]).mean(axis=2), cmap='jet', vmin=0.0, vmax=1.0)
            ax.set_title('M0-OF' , fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            plt.imshow(convertm1p1to01(vis_data.F4[i, :, :, :]).mean(axis=2), cmap='jet', vmin = 0.0, vmax = 1.0)
            ax.set_title('F4-mean', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            plt.imshow(convertm1p1to01(vis_data.M4[i, :, :, :]).mean(axis=2), cmap='jet', vmin = 0.0, vmax = 1.0)

            ax.set_title('M4-mean', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.colorbar()

            # reconstruction maps
            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            # plt.imshow(im_resz, cmap='Greys_r', vmin=0, vmax=255)
            if vis_data.layer_ids[0] == 0:
                plt.imshow(convertm1p1to01(vis_data.F0_recon[i, :, :, :]), cmap='Greys_r', vmin=0,
                           vmax=1.0)
            else:
                plt.imshow(convertm1p1to01(vis_data.F0_recon[i, :, :, :]).mean(axis=2), cmap='jet',
                           vmin=0, vmax=1.0)

            ax.set_title('F0_recon', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            if vis_data.layer_ids[0] == 0:
                plt.imshow(convertm1p1to01(vis_data.M0_recon[i, :, :, :]), cmap='jet', vmin=0.0,
                           vmax=1.0)
            else:
                plt.imshow(convertm1p1to01(vis_data.M0_recon[i, :, :, :]).mean(axis=2), cmap='jet',
                           vmin=0.0, vmax=1.0)
            ax.set_title('M0_recon', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar()
            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            plt.imshow(convertm1p1to01(vis_data.Fl_recon[i, :, :, :]).mean(axis=2), cmap='jet',
                       vmin=0.0, vmax=1.0)
            ax.set_title('Fl_recon-mean', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            plt.imshow(convertm1p1to01(vis_data.Ml_recon[i, :, :, :]).mean(axis=2), cmap='jet',
                       vmin=0.0, vmax=1.0)

            ax.set_title('Ml_recon-mean', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.colorbar()
            # difference map
            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            if len(F0_diff.shape) == 4:
                plt.imshow(F0_diff[i, :, :, :], cmap='jet', vmin=F0_diff_min, vmax=F0_diff_max)
            else:
                plt.imshow(F0_diff[i, :, :], cmap='jet', vmin=F0_diff_min, vmax=F0_diff_max)
            # if pause_time < 0:
            #     plt.colorbar()
            # elif i == 0:
            #     plt.colorbar()
            plt.colorbar()
            ax.set_title('F0_diff', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            if len(M0_diff_OF.shape) == 4:
                pass
            else:
                plt.imshow(M0_diff_OF[i, :, :], cmap='jet', vmin=M0_diff_OF_min,
                           vmax=M0_diff_OF_max)

            # if pause_time < 0:
            #     plt.colorbar()
            # elif i == 0:
            #     plt.colorbar()
            plt.colorbar()

            ax.set_title('M0_diff_OF', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            c += 1

            ax = plt.subplot(nr, nc, c)
            plt.cla()
            if len(F4_diff.shape) == 4:
                plt.imshow(F4_diff[i, :, :, :].mean(axis=2), cmap='jet', vmin=F4_diff_min,
                           vmax=F4_diff_max)
            else:
                plt.imshow(F4_diff[i, :, :], cmap='jet', vmin=F4_diff_min, vmax=F4_diff_max)
            ax.set_title('F4_diff-mean', fontsize=titlefontsz)

            # if pause_time < 0:
            #     plt.colorbar()
            # elif i == 0:
            #     plt.colorbar()
            plt.colorbar()
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            if len(M4_diff.shape) == 4:
                plt.imshow(M4_diff[i, :, :, :].mean(axis=2), cmap='jet', vmin=M4_diff_min,
                           vmax=M4_diff_max)
            else:
                plt.imshow(M4_diff[i, :, :], cmap='jet', vmin=M4_diff_min,
                           vmax=M4_diff_max)
            ax.set_title('M4_diff-mean', fontsize=titlefontsz)

            # if pause_time < 0:
            #     plt.colorbar()
            # elif i == 0:
            #     plt.colorbar()
            plt.colorbar()
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            # detection maps
            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            # plt.imshow(vis_data.F0_recon[i, :, :], cmap='jet',
            #            vmin=vis_data.F0_recon.min(), vmax=vis_data.F0_recon.max())

            plt.imshow(vis_data.D_map0[i, :, :], cmap='Greys_r')

            if pause_time < 0:
                plt.colorbar()
            elif i == 0:
                plt.colorbar()
            # plt.imshow(convertm1p1to01(vis_data.F0_recon[i, :, :, :]), cmap='Greys_r', vmin=0.0,
            #            vmax=1.0)
            # ax.set_title('E_map0', fontsize=titlefontsz)
            ax.set_title('D_map0', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            plt.imshow(vis_data.D_map2[i, :, :], cmap='Greys_r',
                       vmin=0.0, vmax=1.0)
            # plt.imshow(convertm1p1to01(vis_data.M0_recon[i, :, :, 1]), cmap='jet', vmin=0.0,
            #            vmax=1.0)
            # ax.set_title('D_map0', fontsize=titlefontsz)
            ax.set_title('D_map2', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()

            plt.imshow(vis_data.D_map_abs0[i, :, :], cmap='Greys_r')


            ax.set_title('D_map_abs0', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()

            plt.imshow(vis_data.D_map_abs2[i, :, :], cmap='Greys_r',
                       vmin=0.0, vmax=1.0)
            ax.set_title('D_map_abs2', fontsize=titlefontsz)

            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # final decision
            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            plt.imshow(E_map[i, :, :], cmap='jet', vmin=E_map_min, vmax=E_map_max)

            ax.set_title('E_map_final', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()

            plt.imshow(convertm1p1to01(vis_data.D_map[i, :, :]).mean(axis=2), cmap='jet', vmin=0.0,
                       vmax=1.0)
            ax.set_title('M4_recon-mean', fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.colorbar()

            c += 1
            ax = plt.subplot(nr, nc, c)
            plt.cla()
            a_i = vis_data.D_map_2[i, :, :]
            a_i = (a_i - a_i.min()) / (a_i.max() - a_i.min())
            plt.imshow(a_i, cmap='Greys_r',  vmin = 0.0, vmax = 1.0)
            ax.set_title('D_map_final' , fontsize=titlefontsz)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            if len(gt_im_files) > 0:
                gt = cv2.imread(gt_im_files[i], 0)
                c += 1
                ax = plt.subplot(nr, nc, c)
                plt.cla()
                plt.imshow(gt, cmap='Greys_r', vmin=0, vmax=255)
                ax.set_title('[%d] Groundtruth' % i, fontsize=titlefontsz)
                ax.xaxis.grid(False)
                ax.yaxis.grid(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.suptitle('%s-frame %d' % (title, i))

            # plt.show(block=False)
            if pause_time > 0:
                plt.pause(pause_time)
            else:
                pass
            # we should save the figure when pause_time > 0 because it makes the
            # figure be resized and not good for visualizaition
            if vis_folder is not None:
                    fig_file = '%s/%08d.jpg' % (vis_folder, i)
                    plt.savefig(fig_file, dpi=1200)

        if vis_folder is not None:

            score_max = E_map.max()
            heat_score =  cm.jet(E_map[i, :, :]/score_max)
            cv2.imwrite('%s/%08d_score.jpg' % (vis_folder, i), reverse_channel(heat_score*255.0))
            # cv2.imwrite('%s/%08d_det.jpg' % (vis_folder, i), D_map[i, :, :]*255.0)
            cv2.imwrite('%s/%08d_det_2.jpg' % (vis_folder, i), D_map_2[i, :, :] * 255.0)

def remove_start_end(D_map, E_map, Mask, vcomp_list, min_size, E_min):
    Mask_2 = Mask.copy()
    D_map_2 = D_map.copy()
    E_map_2 = E_map.copy()
    vcomp_list_2 = copy.copy(vcomp_list)

    for i in range(len(vcomp_list)):
        obj_id = vcomp_list[i][0]
        # id: 0
        # #pixles: 1
        # longevity: 2
        # frame_start: 3
        # frame_end: 4

        frame_start = vcomp_list[i][3]
        frame_end = vcomp_list[i][4]
        mask_i = (Mask==obj_id)
        pix_per_frm_counts = mask_i.sum(axis=(1, 2))

        frame_start_2 = int(frame_start)
        frame_end_2 = int(frame_end)
        for j in range(int(frame_start), int(frame_end) + 1):
            if pix_per_frm_counts[j] < min_size:
                D_map_2_j =D_map_2[j,:, :].copy()
                D_map_2_j[mask_i[j, :, :].astype(bool)] = 0

                D_map_2[j, :, :] = D_map_2_j
                Mask_2_j = Mask_2[j, :, :]
                Mask_2_j[mask_i[j, :, :].astype(bool)] = 0
                Mask_2[j, :, :] = Mask_2_j

                E_map_2_j = E_map_2[j, :, :].copy()
                E_map_2_j[mask_i[j, :, :].astype(bool)] = E_min
                E_map_2[j, :, :] = E_map_2_j

            else:
                frame_start_2 = j
                break

        for j in reversed(range(int(frame_start), int(frame_end) + 1)):
            if pix_per_frm_counts[j] < min_size:
                D_map_2_j = D_map_2[j, :, :].copy()
                D_map_2_j[mask_i[j, :, :].astype(bool)] = 0
                D_map_2[j, :, :] = D_map_2_j

                Mask_2_j = Mask_2[j, :, :]
                Mask_2_j[mask_i[j, :, :].astype(bool)] = 0
                Mask_2[j, :, :] = Mask_2_j

                E_map_2_j = E_map_2[j, :, :].copy()
                E_map_2_j[mask_i[j, :, :].astype(bool)] = E_min
                E_map_2[j, :, :] = E_map_2_j

            else:
                frame_end_2 = j
                break

        vcomp_list_2[i][1] = pix_per_frm_counts[frame_start_2:frame_end_2+1].sum()
        vcomp_list_2[i][2] = frame_end_2 - frame_start_2 + 1
        vcomp_list_2[i][3] = frame_start_2
        vcomp_list_2[i][4] = frame_end_2

    return D_map_2, E_map_2, Mask_2, vcomp_list_2

def dilate(D_map, E_map, Mask, vcomp_list):
    D_map_2 = D_map.copy()
    E_map_2 = E_map.copy()
    Mask_2 = Mask.copy()
    vcomp_list_2 = copy.copy(vcomp_list)

    # plt.figure()
    # plt.show(block=False)
    for i in range(len(vcomp_list)):
        obj_id = vcomp_list[i][0]
        # id: 0
        # #pixles: 1
        # longevity: 2
        # frame_start: 3
        # frame_end: 4

        longevity = int(vcomp_list[i][2])
        frame_start = int(vcomp_list[i][3])
        frame_end = int(vcomp_list[i][4])
        mask_i = (Mask == obj_id)
        # obj_lbl = mask_i.max()
        pix_per_frm_counts = mask_i.sum(axis=(1, 2))
        mean_size = pix_per_frm_counts.sum() * 1.0/longevity
        for j in range(frame_start, frame_end+1):
            if pix_per_frm_counts[j]<mean_size:
            # kernel_width = int(np.floor(np.sqrt(mean_size / pix_per_frm_counts[j]))*2 + 1)
                kernel_width = int(np.floor( 0.5 * (np.sqrt(mean_size) - np.sqrt(pix_per_frm_counts[j]))) * 2 + 1)
            else:
                kernel_width = 0
            kernel_width += 9
            kernel = np.ones((kernel_width, kernel_width), np.float32)
            # print('Frame [%d] kernel width = %d' % (j, kernel_width))
            # do dilation operation
            mask_ij = mask_i[j, :, :]
            D_map_j = D_map_2[j, :, :]
            E_map_j = E_map_2[j, :, :]
            Mask_2_j = Mask_2[j, :, :]
            mask_ij_dilate = cv2.dilate(mask_ij.astype(np.float32), kernel, iterations=1)
            E_map_j_dilate = cv2.dilate(E_map_j, kernel, iterations=1)
            D_map_j[mask_ij_dilate.astype(bool)] = 1
            D_map_2[j, :, :] = D_map_j
            E_map_2[j, :, :] = E_map_j_dilate
            # Mask_2_j[mask_ij_dilate.astype(bool)] = obj_lbl
            Mask_2_j[mask_ij_dilate.astype(bool)] = obj_id
            Mask_2[j, :, :] = Mask_2_j
            pix_per_frm_counts[j] = mask_ij_dilate.sum()

        vcomp_list_2[i][1] = pix_per_frm_counts.sum()
    return D_map_2, E_map_2, Mask_2, vcomp_list_2


def load_and_convert_FM_data(s, layer_id, feat_folder, resz, data_range, bh5py =0):
    if layer_id == 0:

        # load raw frame data
        feat_file_format = '%s/%s_resz%sx%s_raw_v3.npz' % ('%s', '%s', resz[0], resz[1])
        data_F_s = load_feat([s], feat_folder, feat_file_format)

        OF_file_format = '%s/%s_sz240x360_BroxOF.mat'

        data_O = None
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

            if data_O is None:
                data_O = OF
            else:
                data_O = np.concatenate([data_O, OF], axis=0)
        else:
            print('File %s doesn''t exists' % OF_file)

        data_M_s = np.zeros([data_O.shape[0], resz[1], resz[0], 3])

        for i in range(data_O.shape[0]):
            data_M_s[i, :, :, :] = cv2.resize(data_O[i, :, :, :], (resz[1], resz[0]))



        dualprint('Original F shape: %d x %d' % (data_F_s.shape[1], data_F_s.shape[2]))
        dualprint('Original M shape: %d x %d' % (data_M_s.shape[1], data_M_s.shape[2]))

        dualprint('Convert frame and optical flow into [-1.0, 1.0]')
        # data_F_s = frame_process(data_F_s, resz)

        data_F = convert01tom1p1(data_F_s)

        # # trim the OF data and convert to [-1.0, 1.0]
        if resz is not None:
            F_resz = np.zeros([data_F.shape[0], resz[0], resz[1]])

            for i in range(F_resz.shape[0]):
                F_resz[i, :, :] = cv2.resize(data_F[i, :, :], (resz[1], resz[0]))
        else:
            F_resz = data_F.copy()

        print(F_resz.shape)

        print('F min %f max %f' % (F_resz.min(), F_resz.max()))

        data_F_s = np.stack((F_resz, F_resz, F_resz), axis=3)
        scale = 0.3
        data_M_s = norm_OF_01(data_M_s, scale=scale)
        data_M_s = convert01tom1p1(data_M_s)
        print('M min %f max %f' % (data_M_s.min(), data_M_s.max()))
        data_F_s_resz = data_F_s
        data_M_s_resz = data_M_s

    else:
        feat_F_file_format = '%s/%s_resz%sx%s_cae1_layer%d.npz' % (
            '%s', '%s', resz[0], resz[1], layer_id)
        npz_data = load_feat([s], feat_folder, feat_F_file_format)
        data_F_s = npz_data['feat']

        feat_M_file_format = '%s/%s_resz%sx%s_cae2_layer%d.npz' % (
            '%s', '%s', resz[0], resz[1], layer_id)
        npz_data = load_feat([s], feat_folder, feat_M_file_format)
        data_M_s = npz_data['feat']


        mean_std_file = '%s/mean_std_layer%d_large_v2.dill' % (feat_folder, layer_id)
        dill_data = dill.load(open(mean_std_file, 'rb'))
        dualprint('Loading mean-std file: %s' % mean_std_file)
        data_mean_F = dill_data['cae1_mean']
        data_std_F = dill_data['cae1_std']
        data_scale_F = dill_data['cae1_scale']
        data_mean_M = dill_data['cae2_mean']
        data_std_M = dill_data['cae2_std']
        data_scale_M = dill_data['cae2_scale']

        dualprint('Original F shape: %d x %d' % (data_F_s.shape[1], data_F_s.shape[2]))
        dualprint('Original M shape: %d x %d' % (data_M_s.shape[1], data_M_s.shape[2]))

        dualprint('Normalizing to [%d, %d] using trained mean and standard deviation' % (
        data_range[0], data_range[1]))
        data_F_s = np.divide(data_F_s - data_mean_F, data_std_F + epsilon) * data_scale_F
        data_F_s = np.minimum(np.maximum(data_F_s, data_range[0]), data_range[1])

        data_M_s = np.divide(data_M_s - data_mean_M, data_std_M + epsilon) * data_scale_M
        data_M_s = np.minimum(np.maximum(data_M_s, data_range[0]), data_range[1])

        data_F_s_resz = data_F_s
        data_M_s_resz = data_M_s

    return data_F_s_resz, data_M_s_resz

def print_stat(A, A_name):
    print('%s: info' % A_name)
    print(A.shape)
    print('[min,max]=[%f, %f]' % (A.min(), A.max()))


def combine(vcomp_list_2, Mask_2, D_map_2, E_map_enh,
            Mask_abs, Mask_abs_2, D_map_abs_2, E_map_enh_abs, fr_rate_2obj, min_size):
    D_map_combined = D_map_2.copy()
    E_map_enh_combined = E_map_enh.copy()
    for i in range(len(vcomp_list_2)):
        obj_id = vcomp_list_2[i][0]
        D_mask = (Mask_2 == obj_id)
        num_frames = np.sum(D_mask.sum(axis=(1, 2)) > 0)

        shared_mask = np.multiply(D_mask, D_map_abs_2)
        num_shared_frames = np.sum(shared_mask.sum(axis=(1, 2)) > 0)
        if num_shared_frames * 1.0 / num_frames > 0.75:
        # if num_shared_frames * 1.0 / num_frames > 0.6:

            relevant_obj_abs = np.unique(np.multiply(shared_mask, Mask_abs))

            for j in range(len(relevant_obj_abs)):
                obj_abs = relevant_obj_abs[j]
                if obj_abs > 0:
                    mask_j = (Mask_abs == obj_abs)
                    mask_j_2 = (Mask_abs_2 == obj_abs)
                    mask_j_sub_2 = ((mask_j_2 - D_map_2) > 0.5).astype(bool)
                    mask_j_union_2 = ((D_mask + mask_j_2)> 0.0).astype(bool)
                    mask_j_inter = ((mask_j + D_map_2) > 1.5).astype(bool)
                    frame_inter = mask_j_inter.sum(axis=(1, 2)) > 0
                    shared_mean_area = mask_j_inter[frame_inter, :,
                                       :].sum() * 1.0 / frame_inter.sum()
                    if shared_mean_area > min_size:

                        present_frames = mask_j.sum(axis=(1, 2))
                        present_frames_2 = mask_j_2.sum(axis=(1, 2))
                        num_shared_frame = frame_inter.sum()
                        # thresh_sep_objs = 0.5*num_shared_frame
                        thresh_sep_objs = fr_rate_2obj * num_shared_frame
                        thresh_sep_objs = np.maximum(5, thresh_sep_objs)

                        count = 0
                        flag = True
                        for k in range(len(present_frames)):
                            if present_frames[k] > 0:

                                im_contour, contours, hierarchy = cv2.findContours(
                                    mask_j_2[k, :, :].astype(np.uint8),
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
                                if len(contours) > 1:

                                    all_contours = np.concatenate(contours, axis=0)
                                    rotrect = cv2.minAreaRect(all_contours)
                                    bRect_w, bRect_h = rotrect[1]
                                    bRect_w += 1
                                    bRect_h += 1

                                    if present_frames_2[k] * 1.0 / (
                                            bRect_w * bRect_h) < 0.4:  # two seperate objects
                                        count += 1
                                        # if count > 5:
                                        if count > thresh_sep_objs:
                                            flag = False
                                            break

                        if flag:
                            D_map_combined = ((D_map_combined + mask_j_2) > 0).astype(int)

                            # E_map_enh_combined[mask_j_sub_2] = np.maximum(
                            #     E_map_enh[mask_j_sub_2],
                            #     E_map_enh_abs[mask_j_sub_2])
                            E_map_enh_combined[mask_j_union_2] = np.maximum(
                                    E_map_enh[mask_j_union_2],
                                    E_map_enh_abs[mask_j_union_2])



    return D_map_combined, E_map_enh_combined

def test_hvad(params):
    # experiment params

    cae_folder_name = params.get_value('cae_folder_name')
    gan_layer0_folder_name = params.get_value('gan_layer0_folder_name')
    data_str = params.get_value('data_str')
    test_str = params.get_value('test_str')
    bshow = params.get_value('bshow')
    bsave = params.get_value('bsave')
    beval = params.get_value('beval')
    bh5py = params.get_value('bh5py')

    resz = params.get_value('resz')
    thresh = params.get_value('thresh')
    fr_rate_2obj = params.get_value('fr_rate_2obj')
    frame_step = params.get_value('frame_step')
    min_size = params.get_value('min_size')
    data_range = params.get_value('data_range')
    pause_time = params.get_value('pause_time')
    longevity = params.get_value('longevity')
    folder_name = params.get_value('folder_name')
    layer_ids = params.get_value('layer_ids')


    scale = 0.3
    alpha = 2.0
    # alpha = 1.0
    resz = [256, 256]

    # frame_feat = 'conv5' # 'raw'

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


    imsz = dataholder.imsz

    test_list = read_list_from_file('%s/%s.lst' % (data_folder, test_str))


    vis_folder = '%s/%s' % (res_folder, folder_name)

    if os.path.isdir(vis_folder) == False:
        os.mkdir(vis_folder)
    flog = open('%s/test_hvad_v5_brox_hier_2thresh_release_v4_log.txt' % (vis_folder), 'wt')
    cae_folder = '%s/%s' % (model_folder, cae_folder_name)
    gan_layer0_folder = '%s/%s' % (model_folder, gan_layer0_folder_name)
    time_test_start = time.time()
    time_test_novisual_total = 0.0

    eps = 1e-10
    anom_map_final_list = []


    for s in test_list:
        time_start = time.time()

        anom_map_list = []

        F_all = []
        M_all = []
        F_recon_all = []
        M_recon_all = []
        Delta_F_norm_all = []
        Delta_M_norm_all = []
        D_map_all = []

        Mask_all = []
        Mask_all_2 = []
        D_map_all_2 = []
        E_map_enh_all = []
        vcomp_list_all_2 = []
        for l in range(len(layer_ids)):
            layer_id_l = layer_ids[l]
            if layer_id_l == 0:
                Fl, Ml = load_and_convert_FM_data(s, layer_id_l, feat_folder, resz, data_range,
                                              bh5py=bh5py)
            else:
                Fl, Ml = load_and_convert_FM_data(s, layer_id_l, cae_folder, resz, data_range,
                                                  bh5py=bh5py)

            print_stat(Fl, 'Fl')
            print_stat(Ml, 'Ml')

            if Fl.shape[1] % 2 == 1:
                dualprint('This resizing step is just for Alexnet only')
                new_h = int(np.floor(np.log2(Fl.shape[1])) + 1)
                new_h = int(2 ** new_h)
                new_w = int(np.floor(np.log2(Fl.shape[2])) + 1)
                new_w = int(2 ** new_w)

                print('Resizing the data [%d, %d]-->[%d, %d]' % (
                    Fl.shape[1], Fl.shape[2], new_h, new_w))
                Fl_resz = np.zeros([Fl.shape[0], new_h, new_w, Fl.shape[3]])

                Ml_resz = np.zeros([Ml.shape[0], new_h, new_w, Ml.shape[3]])

                for i in range(Fl.shape[0]):
                    Fl_resz[i, :, :, :] = cv2.resize(Fl[i, :, :, :], (new_w, new_h))
                    Ml_resz[i, :, :, :] = cv2.resize(Ml[i, :, :, :],
                                                           (new_w, new_h))
                Fl = Fl_resz
                Ml = Ml_resz

            F_all.append(Fl)
            M_all.append(Ml)
            print_stat(Fl, 'Fl')
            print_stat(Ml, 'Ml')


            if l==0:
                if layer_ids[0] == 0:
                    F0 = Fl
                else:
                    F0, _ = load_and_convert_FM_data(s, 0, feat_folder, resz, data_range,
                                                     bh5py=bh5py)

                pix_diff = np.zeros(F0.shape[:-1])
                for i in range(F0.shape[0] - 1):
                    pix_diff[i, :, :] = F0[i + 1, :, :, 1] - F0[i, :, :, 1]

                OF_Mask = np.abs(pix_diff) > 0.05

                kernel = np.ones((5, 5), np.uint8)
                for i in range(OF_Mask.shape[0]):
                    img = OF_Mask[i, :, :].astype(np.uint8)
                    img = cv2.dilate(img, kernel, iterations=1)
                    OF_Mask[i, :, :] = img

                OF_Mask_reshape = np.reshape(OF_Mask, [*OF_Mask.shape, 1])



            if layer_id_l == 0:
                Fl_recon_file = '%s/recon/%s_F_recon_layer%d_usecluster0_large_v2_reshape.npz' % (
                    gan_layer0_folder, s, layer_id_l)
                Ml_recon_file = '%s/recon/%s_M_recon_layer%d_usecluster0_large_v2_reshape.npz' % (
                    gan_layer0_folder, s, layer_id_l)

            else:
                Fl_recon_file = '%s/recon/%s_F_recon_layer%d_usecluster0_large_v2_reshape.npz' % (
                    cae_folder, s, layer_id_l)
                Ml_recon_file = '%s/recon/%s_M_recon_layer%d_usecluster0_large_v2_reshape.npz' % (
                    cae_folder, s, layer_id_l)

            print('Loading %s' % Fl_recon_file)
            F_recon_data = np.load(Fl_recon_file)
            Fl_recon = F_recon_data['F_recon']

            print('Loading  %s' % Ml_recon_file)
            M_recon_data = np.load(Ml_recon_file)
            Ml_recon = M_recon_data['M_recon']

            if layer_id_l > 0:
                reshape_info_file = '%s/reshape_info_layer%d_large_v2.dill' % (cae_folder, layer_id_l)
                shape_info = dill.load(open(reshape_info_file, 'rb'))

                print(shape_info)
                reshape_h = shape_info['reshape'][0]
                reshape_w = shape_info['reshape'][1]
                reshape_c = shape_info['reshape'][2]
                Fl_recon_reshape = np.zeros([Fl_recon.shape[0], reshape_h, reshape_w, reshape_c])
                Ml_recon_reshape = np.zeros([Fl_recon.shape[0], reshape_h, reshape_w, reshape_c])
                for i in range(Fl_recon.shape[0]):
                    Fl_recon_reshape[i, :, :, :] = cv2.resize(Fl_recon[i, :, :, :], (reshape_w, reshape_h))
                    Ml_recon_reshape[i, :, :, :] = cv2.resize(Ml_recon[i, :, :, :], (reshape_w, reshape_h))

                Fl_recon = reshape_feat(Fl_recon_reshape, Fl.shape[1], Fl.shape[2], Fl.shape[3])
                Ml_recon = reshape_feat(Ml_recon_reshape, Ml.shape[1], Ml.shape[2], Ml.shape[3])

            # # resize the reconstructed images
            if layer_id_l ==0:
                OF_Mask_reshape_resz = OF_Mask_reshape
            else:
                OF_Mask_reshape_resz = np.zeros([Ml.shape[0], Ml.shape[1], Ml.shape[2]])

                for i in range(Ml.shape[0]):
                    OF_Mask_reshape_resz[i, :, :] = cv2.resize(OF_Mask_reshape[i, :, :].astype(np.float32), (Ml.shape[2], Ml.shape[1]))

                OF_Mask_reshape_resz = OF_Mask_reshape_resz > 0.0
                OF_Mask_reshape_resz = OF_Mask_reshape_resz.reshape([*OF_Mask_reshape_resz.shape, 1])


            F_recon_all.append(Fl_recon)
            M_recon_all.append(Ml_recon)

            print_stat(Fl_recon, 'Fl_recon')
            print_stat(Ml_recon, 'Ml_recon')

            Fl_diff = Fl - Fl_recon
            Fl_diff = np.maximum(Fl_diff, 0.0)


            Ml = Ml * OF_Mask_reshape_resz

            Ml_recon = Ml_recon * OF_Mask_reshape_resz

            Ml_diff = np.abs(Ml) - np.abs(Ml_recon)
            Ml_diff = np.maximum(Ml_diff, 0.0)

            Delta_Fl = Fl_diff
            mFl_channel = Delta_Fl.max(axis=(0, 1, 2), keepdims=True)
            Delta_Fl = Delta_Fl / (mFl_channel + eps)
            Delta_Fl = Delta_Fl.mean(axis=3)

            Delta_Ml = Ml_diff
            mMl_channel = Delta_Ml.max(axis=(0, 1, 2), keepdims=True)
            Delta_Ml = Delta_Ml / (mMl_channel + eps)
            Delta_Ml = Delta_Ml.mean(axis=3)


            Delta_Fl_norm = Delta_Fl / (Delta_Fl.max() + eps)
            Delta_Ml_norm = Delta_Ml / (Delta_Ml.max() + eps)

            Delta_F_norm_all.append(Delta_Fl_norm)
            Delta_M_norm_all.append(Delta_Ml_norm)

            E_map_l = Delta_Fl_norm + alpha * Delta_Ml_norm
            anom_map_list.append(E_map_l)

            dualprint('Resizing the maps from [%d,%d] into [%d,%d]' % (
            E_map_l.shape[1], E_map_l.shape[2], imsz[0], imsz[1]), flog)

            E_map_enh_l = np.zeros([Fl.shape[0], imsz[0], imsz[1]])

            for i in range(Fl.shape[0]):
                E_map_enh_l[i, :, :] = cv2.resize(E_map_l[i, :, :], (imsz[1], imsz[0]))

            frame_window = 2
            E_map_enh_l_temp = E_map_enh_l.copy()
            for i in range(frame_window, E_map_enh_l.shape[0] - frame_window):
                if l > 0:
                    E_map_enh_l_temp[i, :, :] = np.mean(E_map_enh_l_temp[i - frame_window:i + frame_window + 1],
                        axis=0)
                else:
                    E_map_enh_l_temp[i, :, :] = np.mean(E_map_enh_l[i - frame_window:i + frame_window + 1],
                        axis=0)


            # if layer_id_l == 0:
            E_map_enh_l = E_map_enh_l_temp.copy()

            E_min_l = E_map_enh_l.min()
            D_map_l = (E_map_enh_l > thresh).astype(int)
            D_map_all.append(D_map_l)
            # D_map0 = D_map.copy() # should be checked

            # print('Finding 3D connected components')
            # Mask, vcomp = find_components3d(D_map)

            comp_file = '%s/%s_component_%d.npz' % (vis_folder, s, layer_id_l)
            if os.path.isfile(comp_file) == False:
                print('Finding 3D connected components for layer %d' % layer_id_l)
                Mask_l, vcomp_l = find_components3d(D_map_l)
                np.savez(comp_file, Mask=Mask_l, vcomp=vcomp_l)
                print('Saving component file for layer %d: %s' % (layer_id_l, comp_file))
            else:
                npz_data = np.load(comp_file)
                Mask_l = npz_data['Mask']
                vcomp_l = npz_data['vcomp']
                print('Loading component file for layer %d: %s' % (layer_id_l, comp_file))


            # if layer_id_l == 0:
            if l == 0:
                beta = 1.0
            else:
                beta = 0.5
            # beta = 1.0
            kill_count = 0
            survival_count = 0
            D_map_l_2 = D_map_l.copy()
            vcomp_list_l = []
            for i in range(vcomp_l.shape[0]):
                mask_i = (Mask_l == i + 1)
                mean_size = vcomp_l[i, 0] * 1.0 / vcomp_l[i, 1]
                if vcomp_l[i, 1] < longevity or mean_size < beta*min_size:
                    D_map_l_2[mask_i] = 0
                    E_map_enh_l[mask_i] = E_min_l
                    kill_count += 1
                else:
                    vcomp_i = np.insert(vcomp_l[i, :], 0, i + 1)
                    vcomp_list_l.append(vcomp_i)
                    survival_count += 1
            print('Kill %d survive %d / %d' % (kill_count, survival_count, vcomp_l.shape[0]))
            Mask_all.append(Mask_l)

            D_map_l_2, E_map_enh_l, Mask_l_2, vcomp_list_l_2 = dilate(D_map_l_2, E_map_enh_l, Mask_l, vcomp_list_l)
            D_map_all_2.append(D_map_l_2)
            E_map_enh_all.append(E_map_enh_l)
            Mask_all_2.append(Mask_l_2)
            vcomp_list_all_2.append(vcomp_list_l_2)



        # combine all detection maps at different layers

        if len(layer_ids) > 1:
            # combine detections
            vcomp_list_2_base = vcomp_list_all_2[0]
            Mask_2_base = Mask_all_2[0]
            D_map_2_base = D_map_all_2[0]
            E_map_enh_base = E_map_enh_all[0]
            D_map_final = D_map_2_base.copy()
            E_map_final = E_map_enh_base.copy()
            for l in range(1, len(layer_ids)):

                D_map_final, E_map_final = combine(vcomp_list_2_base, Mask_2_base, D_map_final, E_map_final,
                        Mask_all[l], Mask_all_2[l], D_map_all_2[l], E_map_enh_all[l], fr_rate_2obj, min_size=min_size)

        else:
            D_map_final = D_map_l_2.copy()
            # E_max = E_map_enh_l.max()
            E_map_final = E_map_enh_l

        E_map_final = np.minimum(E_map_final, 2.0*thresh)


        anom_map_final_list.append(E_map_final)

        result_file = '%s/%s_final.npz' % (vis_folder, s)
        np.savez(result_file, Emap_enh=E_map_final)
        print('Saving %s' % result_file)
        for l in range(len(layer_ids)):
            layer_id_l = layer_ids[l]
            E_map_file_l = '%s/%s_E_map_%d.npz' % (vis_folder, s, layer_id_l)
            np.savez(E_map_file_l, E_map=anom_map_list[l])
            print('Saving %s' % E_map_file_l)

        time_end = time.time()
        time_test_novisual_total += time_end - time_start
        dualprint('Time %f (seconds)' % (time_end - time_start), flog)

        im_format = data_folder + '/' + s + '/' + dataholder.img_format
        gt_folder = data_folder + '/' + s + '_gt'

        _, gt_ext = os.path.splitext(dataholder.gt_format)
        gt_files = glob.glob('%s/*%s' % (gt_folder, gt_ext))
        gt_files.sort()



        if bsave == 1:
            video_vis_folder = '%s/%s' % (vis_folder, s)
            if os.path.isdir(video_vis_folder) == False:
                os.mkdir(video_vis_folder)
            fig_saved_folder = video_vis_folder
        else:
            fig_saved_folder = None
        plt.close('all')

        if bshow == 1 and bsave == 1:

            dualprint('Visualizing and saving...', flog)

            vis_data = empty_struct()
            vis_data.layer_ids = layer_ids
            vis_data.F0 = F_all[0]


            vis_data.D_map0 = D_map_all[0]
            vis_data.D_map_abs0 = D_map_all[-1]
            vis_data.D_map2 = D_map_all_2[0]
            vis_data.D_map_abs2 = D_map_all_2[-1]

            vis_data.F0_recon = F_recon_all[0]
            vis_data.Fl_recon = F_recon_all[-1]
            vis_data.M0_recon = M_recon_all[0]
            vis_data.Ml_recon = M_recon_all[-1]

            vis_data.F4 = F_all[-1]
            vis_data.M0 = M_all[0]
            vis_data.M4 = M_all[-1]

            vis_data.F0_diff = Delta_F_norm_all[0]
            vis_data.F4_diff = Delta_F_norm_all[-1]
            vis_data.M0_diff = Delta_M_norm_all[0]
            vis_data.M4_diff = Delta_M_norm_all[-1]

            vis_data.E_map = E_map_final
            vis_data.D_map = M_recon_all[-1]

            vis_data.D_map_2 = D_map_final

            visualize_save_v2(vis_data, thresh,
                           s, im_format, gt_files, vis_folder=fig_saved_folder, bshow=bshow,
                           frame_step=frame_step, pause_time = pause_time)

        plt.close()

    time_test_end = time.time()
    dualprint('Test time %f (seconds):' % (time_test_end - time_test_start), flog)
    dualprint('Test time (no visualization) %f (seconds):' % (time_test_novisual_total), flog)
    if beval>0:
        beta = 0.05

        deci_file = '%s/result_%s_beta%0.5f_deci_v1.pkl' % (vis_folder, dataholder.version, beta )
        print('Decision at the threshold %f' % thresh)

        GT = dataholder.get_groundtruth(data_str=test_str, resz = imsz)

        # load the ground-truth
        gt_frame = GT.sum(axis=(1, 2))

        S_norm = np.concatenate(anom_map_final_list, axis=0)
        FPR_pxl, TPR_pxl, MCR_pxl, deci = dataholder.evaluate_at_thresh(S_norm, gt_frame, GT,
                                                                        thresh, level=1,
                                                                        param={'beta': beta})
        deci_list = []
        count_start = 0
        for i in range(len(anom_map_final_list)):
            count_end = count_start + anom_map_final_list[i].shape[0]
            deci_list.append(deci[count_start: count_end])

            print('Video %d: count_start-count_end: %d %d = %d' % (
                i, count_start, count_end, anom_map_final_list[i].shape[0]))
            count_start = count_end

        pkl.dump({'thresh': thresh, 'FPR_pxl': FPR_pxl, 'TPR_pxl': TPR_pxl, 'MCR_pxl': MCR_pxl,
                  'deci': deci_list}, open(deci_file, 'wb'))
        print('Saved decision file %s for the threshold %f' % (deci_file, thresh))

        time_eval_start = time.time()
        # evalv1(data_str, imsz, vis_folder, beta)
        evalv1(data_str, resz, vis_folder, beta, test_str =test_str)
        time_eval_end = time.time()
        dualprint('Evaluation time %f (seconds):' % (time_eval_end - time_eval_start), flog)

    print('Finished.')
    flog.close()

if __name__ == "__main__":
    if len(sys.argv)>1:
        # dataset name
        data_str = sys.argv[1]

        # list of layers whose features are used to detect anomaly
        # e.g., "0-1-2" ==> use the low-level feature (0) and the high-level features at the first
        # and second layers
        #       "1" ==> only use the first hidden layer to detect anomalies
        #       "0-2"==> use the low-level feature (0) and the feature at the second hidden layer (2)
        layer_id_str = sys.argv[2]

        # the name of the folder containing trained CAEs and GANs
        cae_folder_name = sys.argv[3]

        # a file contains a list of testing videos
        test_str = sys.argv[4]

        # show the results every 5 frames on figures
        bshow = int(sys.argv[5])
        # save the results to a folder
        bsave = bshow
        # evaluate the detection results using the frame/pixel/dual-pixel levels
        beval = int(sys.argv[6])
    else:
        raise ValueError("Please provide the arguments")

    gan_layer0_folder_name = 'hvad-gan-layer0-v5-brox'
    layer_ids = [int(s) for s in layer_id_str.split('-')]
    print('data_str=%s' % data_str)
    print('layer_ids=',layer_ids)
    print('cae_folder_name = %s' % cae_folder_name)
    print('test_str = %s' % test_str)
    print('beval = %d' % beval)
    print('gan_layer0_folder_name = %s' % gan_layer0_folder_name)
    params = ParamManager()

    # experiment parameters
    data_range = [-1.0, 1.0]

    net_str = cae_folder_name.split(sep='-lrelu')
    net_str = net_str[0]
    net_str = net_str.replace('hvad-', '')

    # layer = 0
    layer_with_cluster = False

    bh5py = 1
    resz = [256, 256]
    fr_rate_2obj = 0.1 # best for UCSDped1, UCSDped2 and Avenue
    thresh = 0.8

    min_size = 50

    pause_time = 0.0001

    longevity = 30

    # folder_name = 'hvad-net%s-msz%dt%0.3fl%d-broxhier2t-v4-fr2obj%0.2f-ly%s-large-v1a-reshape-beta0.5-comb0.6' % (net_str, min_size,  thresh, longevity, fr_rate_2obj, layer_id_str)
    folder_name = 'hvad-net%s-t%0.3f-ly%s-large-v2-reshape' % (
    net_str, thresh, layer_id_str)
    frame_step = 5

    params.add('cae_folder_name', cae_folder_name, 'hvad')
    params.add('gan_layer0_folder_name', gan_layer0_folder_name, 'hvad')
    params.add('data_str', data_str, 'hvad')
    params.add('test_str', test_str, 'hvad')
    params.add('bshow', bshow, 'hvad')
    params.add('bsave', bsave, 'hvad')
    params.add('beval', beval, 'hvad')
    params.add('bh5py', bh5py, 'hvad')
    params.add('frame_step', frame_step, 'hvad')
    params.add('min_size', min_size, 'hvad')
    params.add('data_range', data_range, 'hvad')
    params.add('pause_time', pause_time, 'hvad')
    params.add('longevity', longevity, 'hvad')
    params.add('folder_name', folder_name, 'hvad')

    params.add('layer_ids', layer_ids, 'hvad')

    params.add('resz', resz, 'detector')
    params.add('thresh', thresh, 'detector')
    params.add('fr_rate_2obj', fr_rate_2obj, 'hvad')

    test_hvad(params)



