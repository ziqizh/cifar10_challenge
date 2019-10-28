#!/bin/bash

python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.1.model
python draw_converge.py --model-name m.1.model
python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.1.model.nlr
python draw_converge.py --model-name m.1.model.nlr
python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.3.model
python draw_converge.py --model-name m.3.model
python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.5.model
python draw_converge.py --model-name m.5.model
python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.7.model
python draw_converge.py --model-name m.7.model
python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.10.model
python draw_converge.py --model-name m.10.model
python measure-converge.py --model-dir ./data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/ --model-name m.10.model.nlr
python draw_converge.py --model-name m.10.model.nlr

#m.10.model  m.10.model.nlr  m.1.model  m.1.model.nlr  m.3.model  m.5.model  m.7.model