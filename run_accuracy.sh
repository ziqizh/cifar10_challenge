#!/bin/bash

python measure-converge.py --model-name m.1.model --ckpt-end 79000
python draw_converge.py --model-name m.1.model
python measure-converge.py --model-name m.1.model.nlr
python draw_converge.py --model-name m.1.model.nlr
python measure-converge.py --model-name m.3.model
python draw_converge.py --model-name m.3.model
python measure-converge.py --model-name m.5.model --ckpt-end 32000 --ckpt-step 2000
python draw_converge.py --model-name m.5.model
python measure-converge.py --model-name m.7.model --ckpt-end 15000 --ckpt-step 1000
python draw_converge.py --model-name m.7.model
python measure-converge.py --model-name m.10.model
python draw_converge.py --model-name m.10.model
python measure-converge.py --model-name m.10.model.nlr
python draw_converge.py --model-name m.10.model.nlr

#m.10.model  m.10.model.nlr  m.1.model  m.1.model.nlr  m.3.model  m.5.model  m.7.model