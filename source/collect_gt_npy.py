import numpy as np
import os
import cv2
import glob
import sys
from utils.data_config import data_config
from utils.anom_UCSDholderv1 import anom_UCSDholder
from utils.read_list_from_file import read_list_from_file
from matplotlib import pyplot as plt


if len(sys.argv)>1:
    # dataset name
    dataset = sys.argv[1]

else:
    dataset = "UCSDped2_demo"
dataholder = anom_UCSDholder(dataset, resz=None)
data_folder = dataholder.data_folder


debug=False
if debug:
    plt.figure()
    plt.show(block=False)
test_file = "%s/test.lst" % data_folder
test_list = read_list_from_file(test_file)
for i in range(len(test_list)):
    test_name = test_list[i]
    print('[%d+1/%d] Test video: %s' % (i, len(test_list), test_name))
    img_gt_folder = '%s/%s_gt' % (data_folder, test_name)
    img_files = glob.glob("%s/*.%s" % (img_gt_folder, 'bmp'))
    img_files.sort()

    for j in range(len(img_files)):
        img_gt_file = img_files[j]
        im = cv2.imread(img_gt_file, 0)

        if j==0:
            gt = np.zeros([len(img_files), im.shape[0], im.shape[1]], dtype=np.float64)

        gt[j, :, :] = im/255.0
        if debug:
            print(img_gt_file)
            plt.imshow(gt[j, :, :], cmap='gray')
            plt.pause(1)


    npy_file = '%s/%s_gt.npy' % (img_gt_folder, test_name)
    np.save(npy_file, gt)
    print('--->Saving as %s' % npy_file)

print('Finished')




