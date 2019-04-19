from collections import namedtuple
import socket
#import os
def data_config(data_str):
    print('Loading %s configuration' % data_str)
    DATA = namedtuple('DATA', 'UCSDped2_demo')
    DATASET = DATA(UCSDped2_demo = 'UCSDped2_demo')
    # computer_name = os.environ['COMPUTERNAME']
    computer_name = socket.gethostname()
    dataset_folder = None

    dataset_folder = 'data'
    temp_folder = 'temp'
    exp_folder = 'exp'
    DATAINFO = None

    if  data_str== DATASET.UCSDped2_demo:
        DATAINFO = {'data_folder': '%s/UCSD/UCSDped2' % dataset_folder,
                    'image_extension': 'tif', 'train_folder_format': 'Train%03d',
                    'test_folder_format': 'Test%03d', 'num_train_videos': 16,
                    'num_test_videos': 12, 'imsz':(240, 360),
                    'threshold': 0.01, 'train_str':'train', 'val_str':'val',
                    'test_str':'test', 'gt_format':'%03d.bmp', 'img_format':'%03d.tif'}

    DATAINFO['temp_folder'] = temp_folder
    DATAINFO['exp_folder'] = exp_folder
    return DATAINFO
