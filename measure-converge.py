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
parser.add_argument('--log-prefix', default='./data-log/measure/atta-loss-',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--atta-max-step', type=int, default=50,
                    help='ATTA attack step.')
parser.add_argument('--atta-loop', type=int, default=10,
                    help='ATTA attack measurement loop.')
parser.add_argument('--model-dir', default='./models/m_r100000',
                    help='The dir of the saved model')
parser.add_argument('--ckpt-step', type=int, default=1000,
                    help='checkpoint step')
parser.add_argument('--ckpt', type=int, default=79000,
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

    model_dir = args.model_dir
    data_path = config['data_path']
    #
    # model_file = tf.train.latest_checkpoint(model_dir)
    # if model_file is None:
    #   print('No model found')
    #   sys.exit()

    model = Model('train')
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start'],
                           config['loss_func'])
    saver = tf.train.Saver()

    cifar = cifar10_input.CIFAR10Data(data_path)

    idx_atta = 0
    cur_ckpt = args.ckpt
    log_loss = [[0 for x in range(args.atta_max_step + 1)] for y in range(args.atta_loop + 1)]
    print(os.path.join(model_dir, "checkpoint-" + str(cur_ckpt)))
    model_ckpt = os.path.join(model_dir, "checkpoint-" + str(cur_ckpt))

    with tf.Session() as sess:
        for batch_start in range(0, 512, 64):
            x_batch = cifar.train_data.xs[batch_start:batch_start + 64]
            y_batch = cifar.train_data.ys[batch_start:batch_start + 64]
            x_batch_adv = x_batch.copy()

            for i in range(args.atta_loop):
                model_number = i + 1
                x_batch_adv = x_batch.copy()

                saver.restore(sess, model_ckpt)

                x_batch_adv = attack.perturb(x_batch, y_batch, sess, log_loss[model_number], step=args.atta_max_step)

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            nat_loss = sess.run(model.mean_xent, feed_dict=nat_dict)
            loss = sess.run(model.mean_xent, feed_dict=adv_dict)

            print("adv loss:     {}".format(loss))
            print("nat-loss: {}".format(nat_loss))
            print("per:      {}%".format(loss / nat_loss * 100))


        for i in range(args.atta_loop):
            model_number = i + 1
            path = args.log_prefix + str(model_number) + ".log"
            print(path)
            log_file = open(path, 'w')
            for ii in range(args.atta_max_step):
                step = ii + 1
                log_file.write("{} {} {}\n".format(model_number, step, log_loss[model_number][step] / 8.0))
        log_file.close()
        idx_atta += 1
