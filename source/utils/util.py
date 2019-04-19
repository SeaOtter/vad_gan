""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.
For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy as np
import os
import cv2
import math
from utils.anom_UCSDholderv1 import anom_UCSDholder

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

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
            # print(img_sz)
            # print(disp_img.shape)
            # print(data_display.shape)
            # print(c)
            disp_img[y_start:y_end, x_start:x_end, :] = np.reshape(data_display[c], img_sz)
            c += 1
    disp_img = disp_img[:-1, :-1, :]
    if num_dim == 2:
        disp_img = np.reshape(disp_img, [disp_img.shape[0], disp_img.shape[1]])
    return disp_img

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not
    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8') + X.min()
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype) + X.min()

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt) + X.min()

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def load_feat(feat_list, feat_folder, file_format, skip_frame=1):
    data_F = None
    for s in feat_list:
        # frm_folder = "%s/%s" % (data_folder, s)
        # feat_file = feat_file_format  % (
        # feat_folder, s, resz[0], resz[1], h, w, h_step, w_step, strfeature)
        frame_file = file_format % (feat_folder, s)
        if os.path.isfile(frame_file):
            print('Loading %s' % frame_file)
            F = np.load(frame_file)
            if skip_frame > 1:
                F = F[::skip_frame, :, :]
                print('Skipping frame:', F.shape)
            if data_F is None:
                data_F = F
            else:
                data_F = np.concatenate([data_F, F], axis=0)
        else:
            print('File %s doesn''t exists' % frame_file)
    return data_F



def load_feat_OF_mask(feat_list, feat_folder, file_format, skip_frame=1):
    data_F = None
    data_OF_mask = None
    for s in feat_list:
        # frm_folder = "%s/%s" % (data_folder, s)
        # feat_file = feat_file_format  % (
        # feat_folder, s, resz[0], resz[1], h, w, h_step, w_step, strfeature)
        frame_file = file_format % (feat_folder, s)
        if os.path.isfile(frame_file):
            print('Loading %s' % frame_file)
            F = np.load(frame_file)
            if skip_frame > 1:
                F = F[::skip_frame, :, :]
                print('Skipping frame:', F.shape)

            pix_diff = np.zeros(F.shape[:3])
            if len(F.shape) == 3:
                for i in range(F.shape[0] - 1):
                    pix_diff[i, :, :] = F[i + 1, :, :] - F[i, :, :]
            elif len(F.shape) == 4:
                for i in range(F.shape[0] - 1):
                    pix_diff[i, :, :] = np.mean(np.sqrt(
                                                            np.square(F[i + 1, :, :, 0] - F[i, :, :, 0])
                                                            + np.square(F[i + 1, :, :, 1] - F[i, :, :, 1])
                                                            + np.square(F[i + 1, :, :, 2] - F[i, :, :, 2])
                                                        ),
                                                axis=2)


            OF_Mask = np.abs(pix_diff) > 0.05

            kernel = np.ones((5, 5), np.uint8)
            for i in range(OF_Mask.shape[0]):
                img = OF_Mask[i, :, :].astype(np.uint8)
                img = cv2.dilate(img, kernel, iterations=1)
                OF_Mask[i, :, :] = img

            OF_Mask_reshape = np.reshape(OF_Mask, [*OF_Mask.shape, 1])
            if data_F is None:
                data_F = F
                data_OF_mask = OF_Mask_reshape
            else:
                data_F = np.concatenate([data_F, F], axis=0)
                data_OF_mask = np.concatenate([data_OF_mask, OF_Mask_reshape], axis=0)
        else:
            print('File %s doesn''t exists' % frame_file)
    return data_F, data_OF_mask


def convert01tom1p1(data):
    # [0, 1] => [-1, 1]
    return data * 2 - 1

def convertm1p1to01(data):
    # [-1, 1] => [0, 1]
    return (data + 1) / 2

def norm_data(data, shift=0.0, scale=1.0, epsilon = 1e-8):
    data_mean = np.mean(data, axis=0)
    data_std =  np.std(data, axis=0)
    data_norm = np.divide(data - data_mean, data_std + epsilon)*scale + shift

    return data_norm, data_mean, data_std
def norm_BroxOF_01(B, scale=0.3):
    B = B * scale + 0.5
    B = np.minimum(np.maximum(B, 0.0), 1.0)
    return B

def norm_filter_4_vis(filter):
    filter_mean = filter.mean()
    filter_std = filter.std()
    filter = (filter - filter_mean) / filter_std

    filter_min = filter.min()
    filter_max = filter.max()
    a = np.maximum(np.abs(filter_min), np.abs(filter_max))
    return (filter + a) * 0.5 / a

def factorize_number(n):
    i = int(math.floor(math.sqrt(n)))
    for k in range(i, 0, -1):
        if n % k == 0:
            j = n / k
            break
    j = int(j)
    return j, k

class empty_struct():
    def __init__(self):
        pass

def create_dataholder(data_str, imsz):
    if data_str == 'UCSDped2':
        dataholder = anom_UCSDholder(data_str, imsz)
    elif data_str == 'UCSDped1':
        dataholder = anom_UCSDholder(data_str, imsz)
    elif data_str.startswith('Avenue'):
        dataholder = anom_Avenueholder(data_str, imsz)
    elif data_str == 'Pako':
        dataholder = anom_Pakoholder(data_str, imsz)
    return dataholder

def print_np_info(A, s=''):
    print('%s: mean=%f min=%f max=%f' % (s, A.mean(), A.min(), A.max()))
    print('shape: ', A.shape)