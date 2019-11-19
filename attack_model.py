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
import numpy as np

import cifar10_input
from pgd_attack import LinfPGDAttack
from art.attacks import FastGradientMethod
from art.attacks import DeepFool
from art.attacks import AdversarialPatch
from art.attacks import HopSkipJump
from art.attacks import CarliniL2Method
from art.attacks import CarliniLInfMethod
from art.attacks import ProjectedGradientDescent
from art.classifiers import TFClassifier
from art.utils import load_cifar10

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


    model = Model('eval')
    logits = model.pre_softmax
    input_ph = model.x_input
    labels_ph = model.y_input
    loss = model.mean_xent
    saver = tf.train.Saver()

    # Setup the parameters
    epsilon = 0.031  # Maximum perturbation
    batch_size = 128

    model_ckpt = args.model_ckpt

    # (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    # x_test = x_test[0:20, :]
    # y_test = y_test[0:20]

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)
    x_test = cifar.eval_data.xs[0:50, :]
    y_test = cifar.eval_data.ys[0:50]
    # print(x_test.shape)
    # print(min_pixel_value)
    # print(max_pixel_value)


    with tf.Session() as sess:
        saver.restore(sess, model_ckpt)
        classifier = TFClassifier(input_ph=input_ph, logits=logits, sess=sess,
                                    loss=loss, output_ph=labels_ph)

        predictions = classifier.predict(x_test)
        print(x_test[0])
        # print(predictions)
        
        print(np.argmax(predictions, axis=1))
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

        # FGSM
        attack = FastGradientMethod(classifier=classifier, eps=epsilon, eps_step=epsilon/10)
        x_test_adv = attack.generate(x=x_test/255.0)

        predictions = classifier.predict(x_test_adv*255.0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))



        adv_crafter_deepfool = DeepFool(classifier, batch_size=batch_size, epsilon=epsilon)
        x_test_adv = adv_crafter_deepfool.generate(x=x_test/255.0)
        predictions = classifier.predict(x_test_adv*255.0)
        print(np.argmax(predictions, axis=1))
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
        # pgd 20
        adv_crafter_pgd_20 = ProjectedGradientDescent(classifier, eps=epsilon, eps_step=0.00775, max_iter=20, batch_size=batch_size)
        x_test_adv = adv_crafter_pgd_20.generate(x=x_test/255.0)
        # print(x_test_adv)
        predictions = classifier.predict(x_test_adv*255.0)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))

        # C&W 20
        # adv_crafter_cwinf = CarliniLInfMethod(classifier, eps=epsilon, learning_rate=epsilon/10, max_iter=20, batch_size=batch_size)
        # x_test_adv = adv_crafter_cwinf.generate(x=x_test/255.0)

        # predictions = classifier.predict(x_test_adv*255.0)
        # accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        # print('Accuracy after C&W attack: {}%'.format(accuracy * 100))
