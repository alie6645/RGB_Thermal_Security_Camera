python pix2pix/pix2pix_train.py --dataroot ./pix2pix/facades --name facades_pix2pix --model pix2pix --direction BtoA --n_epochs 5 --n_epochs_decay 5

python test.py --dataroot ./pix2pix/facades --name facades_pix2pix --model pix2pix --direction BtoA

options, 

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/donpc/anaconda3/envs/pix2pix/lib/python3.12/site-packages/triton/backends/nvidia/lib"
python pix2pix/pix2pix_train.py

Preprocess RGB dataset images and test data folders. seperate processed photos into a train and test folder and point to it in the train script.