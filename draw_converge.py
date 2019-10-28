import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.switch_backend('agg')

log1 = open('data-log/measure-accuracy/accuracy-m.3.log')



data1 = []
length = 79

log_lines1 = log1.readlines()

for i in range(length):
  data1.append([eval(j) for j in log_lines1[i].split(' ')])


print(len(data1))

x = np.array([i[0] for i in data1]) + 1

acc1 = np.array([i[1] for i in data1])

current_palette = sns.color_palette()

plt.plot(x, acc1, color=current_palette[0], lw=2)


plt.xlabel("Training iterations", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.tick_params(labelsize=10)

# plt.legend(fontsize='x-large')

plt.savefig('data-pic/m.3-accuracy.png')
