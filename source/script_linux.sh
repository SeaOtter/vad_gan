export CUDA_VISIBLE_DEVICES=0

# parameter setting
batchsize=50
dataset=UCSDped2_demo

####################################################################
echo [1] EXTRACTING FEATURES

## extracting motion feature
python3 feat_optical_flow_extract.py $dataset

## extracting raw feature data
resz=[240,360]
python3 feat_raw_extract.py $dataset $resz

####################################################################
echo [2] TRAINING DENOISING AUTOENCODERS
# Denoising Autoencoder architecture
encoder=32-16-8
modeldir=hvad-32-16-8-release

disp_freq = 10
save_freq = 10
num_epochs = 10
device ='/gpu:0'

# for our experiments
#disp_freq=10
#save_freq=10
#num_epochs=500
#device =/device:GPU:0

# for our demo
disp_freq=10
save_freq=10
num_epochs=50
device='/device:GPU:0'

python3 train_hvad_CAEv5_brox_release.py 0 $dataset $batchsize $encoder $disp_freq $save_freq $num_epochs $device
python3 train_hvad_CAEv5_brox_release.py 1 $dataset $batchsize $encoder $disp_freq $save_freq $num_epochs $device

####################################################################
echo [3] EXTRACT HIGH-LEVEL REPRESENTATION FEATURES
python3 extract_high_feat_from_cae_brox_batch_v5_release.py $dataset 1 $batchsize $modeldir all $device
python3 extract_high_feat_from_cae_brox_batch_v5_release.py $dataset 2 $batchsize $modeldir all $device

####################################################################
echo [4] TRAINING CONDITIONAL GANS
#resizing raw features into 256 x 256 one that is the input size of cGANs
python3 feat_resize.py "UCSDped2_demo" [240,360] [256,256]

python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 0 False AtoB 1 $modeldir
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 0 False BtoA 1 $modeldir
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 1 False AtoB 1 $modeldir
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 1 False BtoA 1 $modeldir
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 2 False AtoB 1 $modeldir
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 2 False BtoA 1 $modeldir

####################################################################
echo [5] DETECTION: CALCULATING GENERATED FRAMES OF TESTING VIDEOS
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 0 2 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 1 2 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 0 1 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 1 1 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 0 0 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 1 0 False $modeldir hvad-gan-layer0-v5-brox

####################################################################
echo [6] DETECTION: FINDING ANOMALY OBJECTS
# detecting anomalies using features at all levels
python3 test_hvad_v5_brox_hier_2thesh_release_v4_largev2_reshape_release.py $dataset 0-1-2 $modeldir test 1 1

# detecting anomalies using low-level features + top-level features
#python3 test_hvad_v5_brox_hier_2thesh_release_v4_largev2_reshape_release.py $dataset 0-2 $modeldir test 1 1
# detecting anomalies using top-level features only
#python3 test_hvad_v5_brox_hier_2thesh_release_v4_largev2_reshape_release.py $dataset 2 $modeldir test 1 1
# detecting anomalies using low-level features only
#python3 test_hvad_v5_brox_hier_2thesh_release_v4_largev2_reshape_release.py $dataset 0 $modeldir test 1 1
