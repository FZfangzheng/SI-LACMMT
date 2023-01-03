set -ex
python train.py --dataroot ./datasets/maps_haikou --name maps_cyclegan_haikou --model cycle_gan --pool_size 50 --no_dropout
