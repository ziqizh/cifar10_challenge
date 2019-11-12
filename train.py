"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up logfile
# path = ["data-log/train/training-" + str(i) + ".log" for i in range(0, 51)]
# print(path)
#
# log_file = [open(path[i], 'w') for i in range(0, 51)]

# Setting up the data and the model

GPUID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)
data_size = 10000
with tf.Session() as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  path = "natural-training-log.txt"
  #     print(path)
  log_f = open(path, 'w')

  # Main training loop
  for ii in range(max_num_training_steps):
    # print(cifar.train_data)
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    # Compute Adversarial Perturbations
    # start = timer()
    # log_file = [0.0] * 52
    # # x_batch_adv = attack.perturb(x_batch, y_batch, sess, log_file, step=0)
    # end = timer()
    # training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    # adv_dict = {model.x_input: x_batch_adv,
    #             model.y_input: y_batch}

    # Output to stdout
    # if ii % num_output_steps == 0:
    #   nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
    #   # adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
    #   print('Step {}:    ({})'.format(ii, datetime.now()))
    #   print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
    #   # print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
    #   if ii != 0:
    #     print('    {} examples per second'.format(
    #         num_output_steps * batch_size / training_time))
    #     training_time = 0.0
    # Tensorboard summaries
    # if ii % num_summary_steps == 0:
    #   summary = sess.run(merged_summaries, feed_dict=adv_dict)
    #   summary_writer.add_summary(summary, global_step.eval(sess))

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
        saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)
        end = timer()
        training_time += end - start

        total_nat_corr = 0
        nat_acc = 0
        # print(cifar.eval_data.xs.shape)
        for batch_start in range(0, data_size, batch_size):
            # print(batch_start)
            batch_end = min(batch_start + batch_size, data_size)
            # size = batch_end - batch_start
            # print(size)
            x_batch = raw_cifar.eval_data.xs[batch_start:batch_end]
            y_batch = raw_cifar.eval_data.ys[batch_start:batch_end]
            # x_batch, y_batch = cifar.eval_data.get_next_batch(batch_size, multiple_passes=True)

            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}

            nat_corr = sess.run(model.num_correct, feed_dict=nat_dict)
            total_nat_corr += nat_corr

        nat_acc = total_nat_corr / data_size
        print("step: {} nat_acc: {} training time: {}".format(ii, nat_acc, training_time))
        log_f.write("{} {} {}\n".format(ii, nat_acc, training_time))
    else:
        end = timer()
        training_time += end - start



