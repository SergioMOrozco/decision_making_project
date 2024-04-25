import matplotlib.pyplot as plt
import csv
import numpy as np 
from scipy.interpolate import make_interp_spline


def get_dreamer_data(filename):
    x = []
    y = []
    with open(filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(plots):
            if i == 0:
                continue
            x.append((float(row[1])))
            y.append((float(row[2])))

    x = np.array(x)
    y = np.array(y)

    return x,y

def smooth_data(x,y):
    x_smoothed= np.linspace(x.min(), x.max(), 50)
    gfg = make_interp_spline(x, y, k=3)
    y_smoothed = gfg(x_smoothed)

    return x_smoothed,y_smoothed

def graph_dreamer_data(filename,color):
    x,y = get_dreamer_data(filename)
    x_smoothed, y_smoothed = smooth_data(x, y)


    plt.plot(x, y, color=color, alpha=0.2)
    plt.plot(x_smoothed, y_smoothed, color=color, label="Dreamer")

def main():
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Cloth Flatten')
    plt.legend()
    graph_dreamer_data('dreamer_cloth_flatten_1.csv', 'green')
    plt.show()

if __name__ == "__main__":
   main()
