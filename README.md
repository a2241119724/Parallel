```cmd
nohup python -u train.py > output.log 2>&1 &
sudo reptyr [pid] -T
#
python train.py --device cuda:0 --version V0 --lr 1e-3
python test.py --version v0
#
python train.py --device cuda:1 --version V1 --lr 1e-4
python test.py --version v1
```
torch==1.9

nohup python -u train.py --device cuda:0 --version V0 --lr 1e-3 > output_V0.log 2>&1 &
nohup python -u train.py --device cuda:1 --version V1 --lr 1e-4 > output_V1.log 2>&1 &

# test
nohup python -u train.py --is_scheduler --device cuda:0 --version V2 --lr 5e-5 > output_V2.log 2>&1 &
2.5e-4
1e-4
5e-5
3531753
nohup python -u train.py --num_encoder_layers 3 --num_decoder_layers 3 --device cuda:1 --version V3 --lr 1e-4 > output_V3.log 2>&1 &
