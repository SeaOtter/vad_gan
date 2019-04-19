from utils.sytem_config import system_config
SYSINFO = system_config()
if SYSINFO['display']==False:
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
if sys.version_info[0] < 3:
    import cPickle as pkl
else:
    import pickle as pkl
from utils.anom_UCSDholderv1 import anom_UCSDholder
from utils.read_list_from_file import read_list_from_file
import numpy as np
import os
def evalv1(data_str, imsz, vis_folder, beta, filename=None, test_str = 'test', anom_map_list = None):


    train_str = 'train'

    dataholder = anom_UCSDholder(data_str, imsz)
    data_folder = dataholder.data_folder


    if filename is None:
        res_file = '%s/result_%s_beta%0.5f_v1.pkl'  % (vis_folder, dataholder.version, beta)
    else:
        res_file = '%s/%s.pkl' % (vis_folder, filename)

    if anom_map_list is None:
        test_list = read_list_from_file('%s/%s.lst' % (data_folder, test_str))

        anom_map_list = []
        for s in test_list:
            print('Loading %s' % s)
            npzfiles = np.load('%s/%s_final.npz' % (vis_folder, s))
            Emap_enh  = npzfiles['Emap_enh']
            anom_map_list.append(Emap_enh)



    res = dataholder.evaluate(anom_map_list, beta, test_str)
    pkl.dump(res, open(res_file, 'wb'))

    if filename is None:
        txt_file = '%s/result_%s_beta%0.5f_v1.txt'  % (vis_folder, dataholder.version, beta)
    else:
        txt_file = '%s/%s.txt' % (vis_folder, filename)
    f = open(txt_file, 'wt')
    f.write('Frame Level: \n')
    f.write('AUC\t%f \n' % res['AUC_frame'] )
    f.write('EER\t%f \n' % res['MCR_frame_ERR'] )

    f.write('Pixel Level: \n')
    f.write('AUC\t%f \n' % res['AUC_pxl'] )
    f.write('EER\t%f \n' % res['MCR_pxl_ERR'] )

    f.write('Dual Pixel Level: \n')
    f.write('AUC\t%f \n' % res['AUC_dpxl'] )
    # f.write('EER %f \n' % res['MCR_dpxl_ERR'] )
    f.close()


    plt.figure()
    plt.subplot(2, 2, 1)
    # roc1 = plt.plot(res['FPR_frame'], res['TPR_frame'], 'x-r', label='Ours (AUC=%0.2f)' % res['AUC_frame'])
    roc1 = plt.plot(res['FPR_frame'], res['TPR_frame'], 'x-r', label='Ours (AUC=%0.2f, EER=%0.2f)'
                                                                     % (res['AUC_frame'], 100 * res['MCR_frame_ERR']))
    plt.title('Frame level')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis('equal')

    plt.subplot(2, 2, 2)
    # roc2 = plt.plot(res['FPR_pxl'], res['TPR_pxl'], 'x-r', label='Ours (AUC=%0.2f)' % res['AUC_pxl'])
    roc2 = plt.plot(res['FPR_pxl'], res['TPR_pxl'], 'x-r', label='Ours (AUC=%0.2f, EER=%0.2f)'
                                                                 % (res['AUC_pxl'], 100 * res['MCR_pxl_ERR']))
    plt.title('Pixel level')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis('equal')
    plt.suptitle('ROC curve')

    plt.subplot(2, 2, 3)

    roc3 = plt.plot(res['FPR_dpxl'], res['TPR_dpxl'], 'x-r', label='Ours (AUC=%0.2f)'
                                                                 % (res['AUC_dpxl']))
    plt.title('Dual pixel level')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis('equal')
    plt.suptitle('ROC curve')


    plt.savefig('%s/roc_%s_beta%0.5f_v1.pdf' % (vis_folder, dataholder.version, beta))
    plt.show(block=False)

    print('Output to %s' % vis_folder)
    print('Finished.')

if __name__ == "__main__":
    data_str = 'UCSDped2'
    imsz = [240, 360]
    vis_folder_names =   [

        'hvad-msz50-t1.200'
        ]

    if data_str == 'Avenue':
        dataholder = anom_Avenueholder(data_str, imsz)
    else:
        dataholder = anom_UCSDholder(data_str, imsz)
    data_folder = dataholder.data_folder
    exp_folder = '%s/result' % data_folder

    vbeta = [0.05]
    for vis_folder_name in vis_folder_names:

        vis_folder = '%s/%s' % (exp_folder, vis_folder_name)

        print('Vis folder name: %s' % vis_folder_name)

        for beta in vbeta:
            print('beta = %0.5f ' % beta)
            evalv1(data_str, imsz, vis_folder, beta)