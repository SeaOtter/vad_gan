export CUDA_VISIBLE_DEVICES=0
batchsize=50


#dataset=Avenue_sz240x360fr5
#dataset=Avenue_sz240x360fr1
dataset=UCSDped2
#dataset=UCSDped1


#encoder=64-128-256-512-1024
#modeldir=hvad-64-128-256-512-1024-lrelu-k5-gamma0.00-denoise0.20-bn1-Adagrad-lr0.10-v5-brox

#encoder=32-64-128-256
#modeldir=hvad-32-64-128-256-lrelu-k5-gamma0.00-denoise0.20-bn1-Adagrad-lr0.10-v5-brox

encoder=32-16-8
modeldir=hvad-32-16-8-lrelu-k5-gamma0.00-denoise0.20-bn1-Adagrad-lr0.10-v5-brox
#encoder=32-16-8-8
#modeldir=hvad-32-16-8-8-lrelu-k5-gamma0.00-denoise0.20-bn1-Adagrad-lr0.10-v5-brox
#encoder=32-32-32-32
#modeldir=hvad-32-32-32-32-lrelu-k5-gamma0.00-denoise0.20-bn1-Adagrad-lr0.10-v5-brox


#python3 train_hvad_CAEv5_brox.py 0 $dataset $batchsize $encoder
#python3 train_hvad_CAEv5_brox.py 1 $dataset $batchsize $encoder

#python3 extract_high_feat_from_cae_brox_batch_v5.py $dataset 1 $batchsize $modeldir
#python3 extract_high_feat_from_cae_brox_batch_v5.py $dataset 2 $batchsize $modeldir

#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 4 False AtoB 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 4 False BtoA 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 3 False AtoB 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 3 False BtoA 1 $modeldir
##python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 0 False AtoB 1 $modeldir
##python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 0 False BtoA 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 1 False AtoB 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 1 False BtoA 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 2 False AtoB 1 $modeldir
#python3 train_hvad_GANv5_brox_largev2_reshape.py $dataset train 2 False BtoA 1 $modeldir

#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 0 4 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 1 4 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 0 3 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 1 3 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 0 2 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 1 2 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 0 1 False $modeldir hvad-gan-layer0-v5-brox
#python3 test_compute_recon_brox_v5_largev2_reshape.py $dataset 1 1 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5.py $dataset 0 0 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5.py $dataset 1 0 False $modeldir hvad-gan-layer0-v5-brox


