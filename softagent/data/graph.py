import matplotlib.pyplot as plt
import csv
import numpy as np 
from scipy.interpolate import make_interp_spline

x = []
y = []

with open('dreamer_cloth_flatten_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')

    for i, row in enumerate(plots):
        if i == 0:
            continue
        x.append((float(row[1])))
        y.append((float(row[2])))
        print(row[1], row[2])

x = np.array(x)
y = np.array(y)


xnew = np.linspace(x.min(), x.max(), 50)
gfg = make_interp_spline(x, y, k=3)
y_new = gfg(xnew)
plt.plot(x, y, color='g', alpha=0.2)
plt.plot(xnew, y_new, color='g', label="Dreamer")
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Cloth Flatten')
plt.legend()
plt.show()

