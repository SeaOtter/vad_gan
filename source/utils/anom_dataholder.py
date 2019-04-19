from __future__ import absolute_import, division, print_function
from utils.dataholder import dataholder
from utils.data_config import data_config
from utils.read_list_from_file import read_list_from_file
import os
import numpy as np
class anom_dataholder(dataholder):
    def __init__(self, name, resz):
        DATAINFO = data_config(name)
        self.name = name
        self.data_folder = DATAINFO['data_folder']
        self.feat_folder = '%s/feat' % self.data_folder
        self.result_folder = '%s/result' % self.data_folder
        self.model_folder = '%s/model' % self.data_folder
        #self.num_train_videos = DATAINFO['num_train_videos']
        #self.num_test_videos = DATAINFO['num_test_videos']
        self.ext = DATAINFO['image_extension']
        self.thresh = DATAINFO['threshold']
        self.imsz = DATAINFO['imsz']
        self.resz = resz
        # self.resz = DATAINFO['resz']
        #self.train_folder_format =DATAINFO['train_folder_format']
        #self.test_folder_format = DATAINFO['train_test_format']
        self.train_str = DATAINFO['train_str']
        self.val_str = DATAINFO['val_str']
        self.test_str = DATAINFO['test_str']
        self.gt_format = DATAINFO['gt_format']
        self.img_format = DATAINFO['img_format']
        #self.train_list = read_list_from_file('%s/%s.lst' % (self.data_folder, self.train_str))
        #self.val_list = read_list_from_file('%s/%s.lst' % (self.data_folder, self.val_str))
        #self.test_list = read_list_from_file('%s/%s.lst' % (self.data_folder, self.test_str))

    def evaluate(self, anom_map_list, data_str='test'):
        pass
    def load_feature(self, feat_file_format, param, list_file_name='all', bconcate=True):

        data_folder = self.data_folder
        data_list = read_list_from_file('%s/%s.lst' % (data_folder, list_file_name))
        data = None

        for s in data_list:
            frm_folder = "%s/%s" % (data_folder, s)
            # feat_file = feat_file_format  % (
            # feat_folder, s, resz[0], resz[1], h, w, h_step, w_step, strfeature)
            format_param = [self.feat_folder, s]+ param
            feat_file = feat_file_format % tuple(format_param)

            if os.path.isfile(feat_file):
                print('Loading %s' % s)
                print('--> File: %s' % feat_file)
                datai = np.load(feat_file)
                if data is None:
                    if bconcate==True:
                        data = datai
                    else:
                        data = []
                        data.append(datai)
                else:
                    if bconcate==True:
                        data = np.concatenate([data, datai], axis=0)
                    else:
                        data.append(datai)

            else:
                print('File %s doesn''t exists' % s)
        return data