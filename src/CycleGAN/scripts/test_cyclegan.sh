set -ex
python test.py --dataroot ./datasets/maps_haikou --name maps_cyclegan --model cycle_gan --phase test --no_dropout
