#!/bin/bash
nohup python -u train.py > output.log 2>&1 &
nohup python -u train.py --version V0 --device cuda:1 > output.log 2>&1 &
python -u train.py --version V0 --device cuda:1 | tee output.log

python -u train.py --version V0 --device cuda:1 --batch_size 50 --num_encoder_layers 1 --num_decoder_layers 1
python -u test.py --version V4 --device cuda:0 --batch_size 1 --num_encoder_layers 3 --num_decoder_layers 3
