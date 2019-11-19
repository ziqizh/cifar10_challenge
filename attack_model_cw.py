"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json

import tensorflow as tf

import cifar10_input
from pgd_attack import LinfPGDAttack

parser = argparse.ArgumentParser(description='TF CIFAR PGD')
parser.add_argument('--model-ckpt', default='/data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/m.10.model/checkpoint-44000',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--atta-loop', type=int, default=10,
                    help='ATTA attack measurement loop.')
parser.add_argument('--model-name', default='m.3.model',
                    help='model name')
parser.add_argument('--model-dir', default='./models/data-model/',
                    help='The dir of the saved model')
parser.add_argument('--ckpt-step', type=int, default=4000,
                    help='checkpoint step')
parser.add_argument('--ckpt', type=int, default=0,
                    help='checkpoint')
parser.add_argument('--ckpt-start', type=int, default=0,
                    help='checkpoint')
parser.add_argument('--ckpt-end', type=int, default=69000,
                    help='checkpoint')
parser.add_argument('--batch-size', type=int, default=128,
                    help='checkpoint')
parser.add_argument('--data-size', type=int, default=10000,
                    help='checkpoint')
args = parser.parse_args()

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# log_file = open(args.log_path, 'w')

if __name__ == '__main__':
    import json

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = args.model_dir + args.model_name
    data_path = config['data_path']

    model = Model('eval')
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start'],
                           config['loss_func'])
    saver = tf.train.Saver()

    cifar = cifar10_input.CIFAR10Data(data_path)

    # checkpoint starts from 0
    batch_size = args.batch_size
    data_size = args.data_size
    cur_ckpt = args.ckpt
    ckpt_step = args.ckpt_step
    ckpt_start = args.ckpt_start
    ckpt_end = args.ckpt_end
    # path = args.log_prefix + args.model_name + '.' + str(data_size) + ".log"
    #     print(path)
    # log_file = open(path, 'w')
    # log_f = open("data-log/temp.log", 'w')
    # log_loss = [[0 for x in range(args.atta_max_step + 1)] for y in range(args.atta_loop + 1)]

    with tf.Session() as sess:
        model_ckpt = args.model_ckpt
        saver.restore(sess, model_ckpt)

        total_nat_corr = 0
        total_adv_corr = 0
        nat_acc = 0
        adv_acc = 0
        # print(cifar.eval_data.xs.shape)
        for batch_start in range(0, data_size, batch_size):
            # print(batch_start)
            batch_end = min(batch_start + batch_size, data_size)
            # size = batch_end - batch_start
            # print(size)
            x_batch = cifar.eval_data.xs[batch_start:batch_end]
            y_batch = cifar.eval_data.ys[batch_start:batch_end]
            # x_batch, y_batch = cifar.eval_data.get_next_batch(batch_size, multiple_passes=True)
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            batch_s = x_batch.shape[0]
            # print(batch_s)

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            nat_corr = sess.run(model.num_correct, feed_dict=nat_dict)
            adv_corr = sess.run(model.num_correct, feed_dict=adv_dict)
            print("batch nat corr: {}, adv corr: {}".format(nat_corr, adv_corr))
            total_nat_corr += nat_corr
            total_adv_corr += adv_corr

        nat_acc = total_nat_corr / data_size
        adv_acc = total_adv_corr / data_size
        print("nat acc:     {}".format(nat_acc))
        print("adv acc:     {}".format(adv_acc))

