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
parser.add_argument('--log-prefix', default='./data-log/measure-accuracy/',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--atta-loop', type=int, default=10,
                    help='ATTA attack measurement loop.')
parser.add_argument('--model-name', default='m.3.model',
                    help='model name')
parser.add_argument('--model-dir', default='./models/data-model/m.3.model/',
                    help='The dir of the saved model')
parser.add_argument('--ckpt-step', type=int, default=4000,
                    help='checkpoint step')
parser.add_argument('--ckpt', type=int, default=0,
                    help='checkpoint')
parser.add_argument('--batch-size', type=int, default=128,
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
    cur_ckpt = args.ckpt
    ckpt_step = args.ckpt_step
    path = args.log_prefix + args.model_name + ".log"
    #     print(path)
    log_file = open(path, 'w')
    log_f = open("data-log/temp.log", 'w')
    # log_loss = [[0 for x in range(args.atta_max_step + 1)] for y in range(args.atta_loop + 1)]

    with tf.Session() as sess:
        for cur_ckpt in range(0, 79000, ckpt_step):
            print(os.path.join(model_dir, "checkpoint-" + str(cur_ckpt)))
            model_ckpt = os.path.join(model_dir, "checkpoint-" + str(cur_ckpt))
            saver.restore(sess, model_ckpt)

            nat_acc = 0;
            adv_acc = 0;
            # print(cifar.eval_data.xs.shape)
            for batch_start in range(0, 10000, batch_size):
                # print(batch_start)
                # batch_end = min(batch_start + batch_size, 10000)
                # size = batch_end - batch_start
                # print(size)
                x_batch, y_batch = cifar.eval_data.get_next_batch(batch_size, multiple_passes=True)
                x_batch_adv = attack.perturb(x_batch, y_batch, sess, log_f)

                batch_s = x_batch.shape[0]
                # print(batch_s)

                nat_dict = {model.x_input: x_batch,
                            model.y_input: y_batch}
                adv_dict = {model.x_input: x_batch_adv,
                            model.y_input: y_batch}

                nat_accuracy = sess.run(model.accuracy, feed_dict=nat_dict)
                adv_accuracy = sess.run(model.accuracy, feed_dict=adv_dict)
                print("batch nat acc: {}, adv acc: {}".format(nat_accuracy, adv_accuracy))
                nat_acc += nat_accuracy * batch_s
                adv_acc += adv_accuracy * batch_s

            nat_acc /= 10000
            adv_acc /= 10000
            print("nat acc:     {}".format(nat_acc))
            print("adv acc:     {}".format(adv_acc))
            # for i in range(args.atta_loop):
            #     model_number = i + 1

            log_file.write("{} {} {}\n".format(cur_ckpt, nat_acc, adv_acc))
            cur_ckpt += ckpt_step

    log_file.close()
