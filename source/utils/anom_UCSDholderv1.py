from __future__ import absolute_import, print_function, division
from .anom_dataholder import anom_dataholder
from .read_list_from_file import read_list_from_file
import glob
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import auc
import os
class anom_UCSDholder(anom_dataholder):
    def __init__(self, name="UCSDped2", resz=None):
        super(anom_UCSDholder, self).__init__(name = name, resz = resz)
        self.version = "PAMI13" # self.version = "CVPR10"


    def compute_TPR_FPR(self, vbool, gtbool):
        vTP = [ x and y for (x,y) in zip(vbool, gtbool)]
        vFP = [ x and not y for (x,y) in zip(vbool, gtbool)]
        pos = gtbool.sum()
        neg = len(gtbool) - pos
        return sum(vTP)*1.0/pos, sum(vFP)*1.0/neg

    def evaluate_at_thresh(self, S_norm, gt_frame, GT, thresh, level, param=None):
        D_norm = (S_norm >= thresh).astype(int)
        D_frame = D_norm.sum(axis=(1, 2))
        gt_frame_bool = gt_frame > 0
        pos = gt_frame_bool.sum()
        neg = len(gt_frame_bool) - pos
        deci = np.zeros(gt_frame.shape)
        # deci[i] = 1 : True positive
        # deci[i] = -1: False positive
        # deci[i] = -2: False negative
        # deci[i] = 2: True negative
        if level == 0:

            D_frame_bool = D_frame > 0

            TP_frame, FP_frame = 0, 0
            MisClass_frame = 0

            for i in range(len(D_frame)):
                if D_frame[i] > 0:
                    if gt_frame_bool[i] == True:  # true positive
                        TP_frame += 1
                        deci[i] = 1
                    else:
                        FP_frame += 1
                        MisClass_frame+=1
                        deci[i] = -1
                else:
                    if gt_frame_bool[i] == True:
                        MisClass_frame += 1
                        deci[i] = -2
                    else:
                        deci[i] = 2


            TPR_frame = TP_frame / pos
            FPR_frame = FP_frame / neg
            MCR_frame = MisClass_frame * 1.0 / len(gt_frame_bool)
            TPR, FPR, MCR = TPR_frame, FPR_frame, MCR_frame


        elif level==1:

            I = (D_norm + GT)
            I_intersect = I >= 2
            I_frame = I_intersect.sum(axis=(1, 2))

            TP_pxl, FP_pxl = 0, 0
            MisClass_pxl = 0
            for i in range(len(D_frame)):
                if D_frame[i] > 0:
                    if self.version == "PAMI13":
                        if gt_frame_bool[i] == True and I_frame[i] * 1.0 / gt_frame[i] >= 0.4:
                            TP_pxl += 1
                            deci[i] = 1

                        if gt_frame_bool[i] == False:
                            FP_pxl += 1
                            MisClass_pxl += 1
                            deci[i] = -1

                    elif self.version == "CVPR10":
                        if gt_frame_bool[i] == True and I_frame[i] * 1.0 / gt_frame[i] >= 0.4:
                            TP_pxl += 1
                            deci[i] = 1
                        else:
                            MisClass_pxl += 1
                            FP_pxl += 1
                            deci[i] = -1

                else:
                    if gt_frame_bool[i] == True:
                        MisClass_pxl += 1
                        deci[i] = -2
                    else:
                        deci[i] = 2


            TPR_pxl = TP_pxl / pos
            FPR_pxl = FP_pxl / neg
            MCR_pxl = MisClass_pxl*1.0/len(gt_frame_bool)
            TPR, FPR, MCR = TPR_pxl, FPR_pxl, MCR_pxl
        elif level == 2:
            I = (D_norm + GT)
            I_intersect = I >= 2
            I_frame = I_intersect.sum(axis=(1, 2))

            TP, FP = 0, 0
            MisClass = 0
            beta = param['beta']
            for i in range(len(D_frame)):
                if D_frame[i] > 0:
                    # print('[%d] intersection %d / %d' % (i, I_frame[i] ,  D_frame[i]))
                    # print('beta=%0.5f %0.5f' % (beta, I_frame[i] * 1.0 / D_frame[i] ))
                    if self.version == "PAMI13":
                        if  gt_frame_bool[i] == True and \
                            I_frame[i] * 1.0 / gt_frame[i] >= 0.4 and \
                            I_frame[i] * 1.0 / D_frame[i] >= beta:
                            TP += 1
                            deci[i] = 1

                        if gt_frame_bool[i] == False:
                            FP += 1
                            MisClass += 1
                            deci[i] = -1

                    elif self.version == "CVPR10":
                        if gt_frame_bool[i] == True and \
                            I_frame[i] * 1.0 / gt_frame[i] >= 0.4 and \
                            I_frame[i] * 1.0 / D_frame[i] >= beta:
                            TP += 1
                            deci[i] = 1
                        else:
                            MisClass += 1
                            FP += 1
                            deci[i] = -1

                else:
                    if gt_frame_bool[i] == True:
                        MisClass += 1
                        deci[i] = -2
                    else:
                        deci[i] = 2


            TPR = TP / pos
            FPR = FP / neg
            MCR = MisClass*1.0/len(gt_frame_bool)

        return FPR, TPR, MCR, deci

    def recursive_EER(self, tplus, fplus, tminus, fminus, S_norm, gt_frame, GT, level, param=None, stopping_criteria=0.01):
        t = (tplus + tminus)*0.5
        FPR, TPR, MCR, _ = self.evaluate_at_thresh(S_norm, gt_frame, GT, t, level, param)

        f = FPR + TPR -1
        # print(tplus, tminus, fplus, fminus)
        if math.fabs(f) < stopping_criteria:
            return t, FPR, TPR, MCR
        if math.fabs(f - fplus)< stopping_criteria:
            return t, FPR, TPR, MCR
        if f > 0:
            return self.recursive_EER(t, f, tminus, fminus, S_norm, gt_frame, GT, level, param, stopping_criteria)
        else:# f < 0
            return self.recursive_EER(tplus, fplus, t, f, S_norm, gt_frame, GT, level, param, stopping_criteria)

    def find_ERR(self, vthresh, vFPR, vTPR, S_norm, gt_frame, GT, level, param=None, stopping_criteria=0.01):
        vf = np.array(vTPR) + np.array(vFPR) - 1

        for i in range(1, len(vthresh)):

            if vf[i - 1] * vf[i] <= 0:
                if vf[i - 1] < 0:
                    tminus = vthresh[i - 1]
                    fminus = vf[i - 1]
                    tplus = vthresh[i]
                    fplus = vf[i]
                else:  # f_frame[i-1]>0
                    tplus = vthresh[i - 1]
                    fplus = vf[i - 1]
                    tminus = vthresh[i]
                    fminus = vf[i]

                t, FPR, TPR, MCR = self.recursive_EER(tplus, fplus, tminus, fminus, S_norm, gt_frame, GT,
                                                      level, param, stopping_criteria=stopping_criteria)
                return t, FPR, TPR, MCR

    # def load_gt(self, data_str = 'test'):
    #     video_list = read_list_from_file('%s/%s.lst' % (self.data_folder, data_str))
    #
    #     _, gt_extension = os.path.splitext(self.gt_format)
    #     gt_pxl = []
    #     gt_frame = []
    #     for (i, s) in enumerate(video_list):
    #         frm_folder = '%s/%s_gt' % (self.data_folder, s)
    #
    #         gt_1file = '%s/%s_gt.npy' % (frm_folder, s)
    #         if os.path.isfile(gt_1file):
    #             V =np.load(gt_1file)
    #         else:
    #             # gt_files = glob.glob('%s/*.bmp' % (frm_folder))
    #             gt_files = glob.glob(frm_folder + '/*' + gt_extension)
    #             gt_files.sort()
    #             V = None
    #
    #             for (j, file) in enumerate(gt_files):
    #                 img = cv2.imread(file, 0)
    #                 im_resz = cv2.resize(img, (self.resz[1], self.resz[0]), interpolation=cv2.INTER_NEAREST)
    #                 img = img / 255.0
    #                 im_resz = im_resz/255.0
    #                 if V is None:
    #                     V = np.zeros([len(gt_files), self.resz[0], self.resz[1]])
    #                 # V[j, :, :] = img
    #                 V[j, :, :] = im_resz
    #
    #             np.save(gt_1file, V)
    #
    #
    #         v = np.zeros(V.shape[0])
    #         for j in range(V.shape[0]):
    #             if V[j, :, :].sum()>0:
    #                     v[j] = 1
    #         gt_pxl.append(V)
    #         gt_frame.append(v)
    #     return gt_pxl, gt_frame
    # def get_groundtruth(self, data_str='test', bresize = False):
    def get_groundtruth(self, data_str='test', resz = None):
        video_list = read_list_from_file('%s/%s.lst' % (self.data_folder, data_str))

        gt_list = []
        _, gt_extension = os.path.splitext(self.gt_format)
        for (i, s) in enumerate(video_list):
            print('%s: Loading ground-truth' % s)
            frm_folder = '%s/%s_gt' % (self.data_folder, s)
            gt_files = glob.glob(frm_folder + '/*' + gt_extension)
            gt_files.sort()
            gt_listi = []
            for (j, file) in enumerate(gt_files):
                img = cv2.imread(file, 0)
                if resz is not None:
                    # im_resz = cv2.resize(img, (self.resz[1], self.resz[0]), interpolation=cv2.INTER_NEAREST)
                    im_resz = cv2.resize(img, (resz[1], resz[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    im_resz = im_resz / 255.0
                    gt_listi.append(im_resz)
                else:
                    img = img/255.0
                    gt_listi.append(img)
            gt_list.append(gt_listi)
            print('-->%d frames' % len(gt_listi))

        GT = np.concatenate(gt_list, axis=0)
        del gt_list
        return GT

    def evaluate(self, anom_map_list, beta=0.1, data_str='test'):

        # S = np.concatenate(anom_map_list, axis=0)
        # del anom_map_list
        # val_min = S.min()
        # val_max = S.max()
        # delta = val_max - val_min
        # S_norm = (S - val_min) / delta
        # del S
        S_norm = np.concatenate(anom_map_list, axis=0)
        del anom_map_list
        val_min = S_norm.min()
        val_max = S_norm.max()
        delta = val_max - val_min
        S_norm -= val_min
        S_norm /= delta
        # S_norm = (S - val_min) / delta



        GT = self.get_groundtruth(data_str=data_str, resz = (S_norm.shape[1], S_norm.shape[2]))
        param = {'beta':beta}
        # load the ground-truth here

        gt_frame = GT.sum(axis=(1, 2))
        gt_frame_bool = gt_frame > 0
        # normalize the anom_map in to [0, 1]

        vTPR_pxl, vFPR_pxl = [], []
        vTPR_frame, vFPR_frame = [], []
        vTPR_dpxl, vFPR_dpxl = [], []

        thresh_step = 0.05
        # thresh_step = 0.01

        vthresh = np.arange(0, 1 + 2 * thresh_step, thresh_step)

        # h, bins = np.histogram(S_norm.flatten(), bins=1000, range=[0.0, 1.0])
        # h = h / h.sum()
        # h_cs = np.cumsum(h)
        # center = (bins[:-1] + bins[1:]) * 0.5
        #
        # ys = np.arange(0, 1 + 2 * thresh_step, thresh_step)
        # vthresh = []
        # for i in range(len(ys)):
        #     y = ys[i]
        #     for j in range(len(h_cs)):
        #         if y < h_cs[j]:
        #             break
        #     vthresh.append(center[j])

        #
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.bar(center, h, width=center[1] - center[0])
        # for i in range(len(vthresh)):
        #     plt.axvline(x=vthresh[i])
        #
        # plt.subplot(2, 2, 2)
        # plt.bar(center, h_cs, width=center[1] - center[0])
        # plt.show(block=False)

        for thresh in vthresh:
            # thresh = 0.2
            print('Evaluation threshold = %0.5f' % thresh)

            FPR_frame, TPR_frame, _, _ = self.evaluate_at_thresh(S_norm, gt_frame, GT, thresh, level=0, param=param)
            print('Frame level: TPR_frame = %0.5f and FPR_frame = %0.5f' % (TPR_frame, FPR_frame))
            vFPR_frame.append(FPR_frame)
            vTPR_frame.append(TPR_frame)

            FPR_pxl, TPR_pxl, _, _ = self.evaluate_at_thresh(S_norm, gt_frame, GT, thresh, level=1, param=param)
            print('Pixel level: TPR_pxl = %0.5f and FPR_pxl = %0.5f' % (TPR_pxl, FPR_pxl))
            vFPR_pxl.append(FPR_pxl)
            vTPR_pxl.append(TPR_pxl)

            FPR_dpxl, TPR_dpxl, _, _ = self.evaluate_at_thresh(S_norm, gt_frame, GT, thresh, level=2, param=param)
            print('Dual Pixel level: TPR_pxl = %0.5f and FPR_pxl = %0.5f' % (TPR_dpxl, FPR_dpxl))
            vFPR_dpxl.append(FPR_dpxl)
            vTPR_dpxl.append(TPR_dpxl)


        stopping_criteria = 0.0001
        # Frame level
        t_frame_EER, FPR_frame_ERR, TPR_frame_ERR, MCR_frame_ERR = \
            self.find_ERR(vthresh, vFPR_frame, vTPR_frame, S_norm, gt_frame, GT,
                          level=0, param=param, stopping_criteria=stopping_criteria)
        print('Frame MC Rate %0.5f at threshold = %0.5f FPR + TPR = %0.5f+%0.5f=%0.5f' \
              % (MCR_frame_ERR, t_frame_EER,  FPR_frame_ERR, TPR_frame_ERR, FPR_frame_ERR+ TPR_frame_ERR ))

        # Pixel level
        t_pxl_EER, FPR_pxl_ERR, TPR_pxl_ERR, MCR_pxl_ERR = \
            self.find_ERR(vthresh, vFPR_pxl, vTPR_pxl, S_norm, gt_frame, GT,
                          level=1, param=param, stopping_criteria=stopping_criteria)

        print('Pixel MC Rate %0.5f at threshold = %0.5f FPR + TPR = %0.5f+%0.5f=%0.5f' \
              % (MCR_pxl_ERR, t_pxl_EER, FPR_pxl_ERR, TPR_pxl_ERR, FPR_pxl_ERR + TPR_pxl_ERR))

        # Dual pixel level
        #
        # t_dpxl_EER, FPR_dpxl_ERR, TPR_dpxl_ERR, MCR_dpxl_ERR = \
        #     self.find_ERR(vthresh, vFPR_dpxl, vTPR_dpxl, S_norm, gt_frame, GT,
        #                   level=2, param=param, stopping_criteria=stopping_criteria)
        #
        # print('Dual Pixel MC Rate %0.5f at threshold = %0.5f FPR + TPR = %0.5f+%0.5f=%0.5f' \
        #       % (MCR_dpxl_ERR, t_dpxl_EER, FPR_dpxl_ERR, TPR_dpxl_ERR, FPR_dpxl_ERR + TPR_dpxl_ERR))



        auc1 = auc(vFPR_frame, vTPR_frame, reorder= True)
        auc2 = auc(vFPR_pxl, vTPR_pxl, reorder  = True)
        # auc3 = auc(vFPR_dpxl, vTPR_dpxl, reorder  = True)
        auc3 = auc(vFPR_dpxl, vTPR_dpxl, reorder=False) # should not reorder for dual pixel
        print('auc (frame level) = %0.5f auc (pixel level) = %0.5f auc (dual pixel level) = %0.5f' % (auc1, auc2, auc3))

        Dict = {'thresh': vthresh,
                'TPR_frame': vTPR_frame, 'FPR_frame': vFPR_frame, 'AUC_frame': auc1,
                't_frame_EER':t_frame_EER, 'FPR_frame_ERR':FPR_frame_ERR, 'TPR_frame_ERR':TPR_frame_ERR,
                'MCR_frame_ERR':MCR_frame_ERR,
                'TPR_pxl': vTPR_pxl, 'FPR_pxl': vFPR_pxl, 'AUC_pxl': auc2,
                't_pxl_EER':t_pxl_EER, 'FPR_pxl_ERR':FPR_pxl_ERR, 'TPR_pxl_ERR':TPR_pxl_ERR,
                'MCR_pxl_ERR':MCR_pxl_ERR,
                'TPR_dpxl': vTPR_dpxl, 'FPR_dpxl': vFPR_dpxl, 'AUC_dpxl': auc3
                # 't_dpxl_EER':t_dpxl_EER, 'FPR_dpxl_ERR':FPR_dpxl_ERR, 'TPR_dpxl_ERR':TPR_dpxl_ERR,
                # 'MCR_dpxl_ERR':MCR_dpxl_ERR
                }

        return Dict