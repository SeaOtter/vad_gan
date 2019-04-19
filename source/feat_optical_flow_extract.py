# # extracting optical flow images
import os
import sys
import subprocess
from utils.anom_UCSDholderv1 import anom_UCSDholder
if len(sys.argv) > 1:
    dataset = sys.argv[1]
    dataholder = anom_UCSDholder(dataset, resz=None)

    data_folder = dataholder.data_folder
    feat_folder = '%s/feat' % (data_folder)

    release_folder = os.path.dirname(os.path.realpath(__file__))
    # data_folder= '%s/data/UCSD/UCSDped2' % release_folder

    feat_folder = '%s/feat' % data_folder
    if os.path.isdir(feat_folder) == False:
        os.mkdir(feat_folder)
    set_name = 'all'
    ext = 'tif'
    subprocess.call(["matlab", "-nodisplay", "-r",  "addpath('%s/eccv2004Matlab'); extract_BroxOF('%s','%s','%s');exit();" % (release_folder, data_folder, set_name, ext)])


