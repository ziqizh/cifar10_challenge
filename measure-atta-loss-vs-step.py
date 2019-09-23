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
parser.add_argument('--log-prefix',  default='./data-log/measure/atta-loss-',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')                    
parser.add_argument('--atta-largest-step', type=int, default=20,
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

  atta_loop = [1, 2, 4, 6, 8, 10]

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

  with tf.Session() as sess:
      for batch_start in range(0, 512, 64):
          x_batch = cifar.train_data.xs[batch_start:batch_start+64]
          # mnist.train.images[0:500, :]
          y_batch = cifar.train_data.ys[batch_start:batch_start+64]
          x_batch_adv = x_batch.copy()
          for loop_size in atta_loop:
            path = args.log_prefix + str(atta_loop[idx_atta]) + "-" + str(batch_start) + ".log"
            print(path)
            log_file = open(path, 'w')
            print("Current loop size: {}".format(loop_size))
            for i in range(args.atta_largest_step):
              atta_step = i + 1
              x_batch_adv = x_batch.copy()
              cur_ckpt = args.ckpt - args.ckpt_step * (loop_size - 1)

              print(os.path.join(model_dir, "checkpoint-" + str(cur_ckpt)))

              for iteration in range(loop_size):
                model_ckpt = os.path.join(model_dir, "checkpoint-" + str(cur_ckpt))
                saver.restore(sess, model_ckpt)

                x_batch_adv = attack.perturb_transferbility(x_batch, x_batch_adv, y_batch, sess, step=atta_step)
                cur_ckpt += args.ckpt_step

              nat_dict = {model.x_input: x_batch,
                          model.y_input: y_batch}
              adv_dict = {model.x_input: x_batch_adv,
                          model.y_input: y_batch}

              nat_loss = sess.run(model.mean_xent, feed_dict=nat_dict)
              loss = sess.run(model.mean_xent, feed_dict=adv_dict)

              print("Attack iterations:     {}".format(i))
              print("adv loss:     {}".format(loss))
              print("nat-loss: {}".format(nat_loss))
              print("per:      {}%".format(loss / nat_loss * 100))

              log_file.write("{} {} {} {}\n".format(loop_size, atta_step, loss, nat_loss))
            log_file.close()
            idx_atta += 1
