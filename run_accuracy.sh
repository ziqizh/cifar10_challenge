#!/bin/bash

#path="/data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/"
#python measure-converge.py --model-name m.1.model --ckpt-start 40000 --ckpt-end 50000 --ckpt-step 1000 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --gpuid 3
python measure-converge.py --model-name m.1.model --ckpt-end 79000 --ckpt-step 4000 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --gpuid 0
#python draw_converge.py --model-name m.1.model
#python measure-converge.py --model-name m.1.model.nlr
#python draw_converge.py --model-name m.1.model.nlr
#python measure-converge.py --model-name m.3.model --ckpt-end 79000 --gpuid 3
python measure-converge.py --model-name m.3.model --ckpt-start 40000 --ckpt-end 45000 --gpuid 0 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/
#python draw_converge.py --model-name m.3.model
#python measure-converge.py --model-name m.5.model --ckpt-end 32000 --ckpt-step 2000 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --gpuid 3
#python draw_converge.py --model-name m.5.model
#python measure-converge.py --model-name m.7.model --ckpt-end 15000 --ckpt-step 1000 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --gpuid 3
#python draw_converge.py --model-name m.7.model
#python measure-converge.py --model-name m.10.model --gpuid 3 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/
python measure-converge.py --model-name m.10.model --ckpt-start 40000 --ckpt-end 45000  --gpuid 0 --model-dir /data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/
#python draw_converge.py --model-name m.10.model
#python measure-converge.py --model-name m.10.model.nlr
#python draw_converge.py --model-name m.10.model.nlr

#m.10.model  m.10.model.nlr  m.1.model  m.1.model.nlr  m.3.model  m.5.model  m.7.model