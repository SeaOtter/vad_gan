import numpy as np
import os
from utils.read_list_from_file import read_list_from_file
def load_data(data_folder, feat_file_format, feat_folder, resz, strfeature, data_str='all'):
    train_list = read_list_from_file('%s/%s.lst' % (data_folder, data_str))
    data_train = None
    for s_tr in train_list:
        print('Video %s' % s_tr)

        frm_folder = "%s/%s" % (data_folder, s_tr)
        feat_file = feat_file_format % (feat_folder, s_tr, resz[0], resz[1], strfeature)
        print('Loading %s' % s_tr)
        feat = np.load(feat_file)

        if data_train is None:
            data_train = feat
        else:
            data_train = np.concatenate([data_train, feat], axis=0)

    return data_train
