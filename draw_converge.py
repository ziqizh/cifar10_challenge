import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='CIFAR ACCURACY')

parser.add_argument('--model-name', default='m.3.model',
                    help='model name.')

args = parser.parse_args()

# log_file = open(args.log_path, 'w')

if __name__ == '__main__':

    plt.switch_backend('agg')

    log1 = open('data-log/measure-accuracy/' + args.model_name + '.log')

    label1 = args.model_name + " Natural"
    label2 = args.model_name + " Adversarial"

    data1 = []

    log_lines1 = log1.readlines()

    for i in range(len(log_lines1)):
        data1.append([eval(j) for j in log_lines1[i].split(' ')])

    print(len(data1))

    x = np.array([i[0] for i in data1]) + 1

    nat_acc1 = np.array([i[1] for i in data1])
    adv_acc1 = np.array([i[2] for i in data1])

    current_palette = sns.color_palette()

    plt.plot(x, nat_acc1, color=current_palette[0], lw=2, label=label1)
    plt.plot(x, adv_acc1, color=current_palette[1], lw=2, label=label2)

    plt.xlabel("Training iterations", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.tick_params(labelsize=10)

    plt.legend(fontsize='x-large')

    plt.savefig('data-pic/' + args.model_name + '.png')
