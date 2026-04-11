python pix2pix/pix2pix_train.py --dataroot ./pix2pix/facades --name facades_pix2pix --model pix2pix --direction BtoA --n_epochs 5 --n_epochs_decay 5

python test.py --dataroot ./pix2pix/facades --name facades_pix2pix --model pix2pix --direction BtoA

options, 