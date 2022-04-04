import csv
import matplotlib.pyplot as plt
import numpy as np

csv_path = '/home/jewel/Downloads/wandb_export_2022-04-05T12_07_10.298+08_00.csv'
f = csv.reader(open(csv_path, 'r'))

p16, p32, p64 = [], [], []

for idx, item in enumerate(f):
    if idx == 0:
        continue
    if i := item[1]:
        p16.append(float(i))
    if i := item[4]:
        p64.append(float(i))
    if i := item[7]:
        p32.append(float(i))

print(len(p16), len(p32), len(p64))


def ewma(data, alpha):
    out = np.array(data)
    for i in range(1, len(data)):
        out[i] = data[i] * (1. - alpha) + out[i - 1] * alpha
    return out


plt.plot(np.arange(len(p16)) * 16 * 128, ewma(p16, 0.95))
plt.plot(np.arange(len(p32)) * 32 * 128, ewma(p32, 0.95))
plt.plot(np.arange(len(p64)) * 64 * 128, ewma(p64, 0.95))
plt.xlabel('num_frames')
plt.ylabel('reward')
plt.legend(['p16', 'p32', 'p64'])
plt.show()
