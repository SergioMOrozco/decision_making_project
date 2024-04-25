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

def graph_dreamer_data(filename,color,ax,title):
    x,y = get_dreamer_data(filename)
    x_smoothed, y_smoothed = smooth_data(x, y)

    ax.set(xlabel='Episode', ylabel='Reward')
    ax.set_title(title)

    ax.plot(x, y, color=color, alpha=0.2)
    ax.plot(x_smoothed, y_smoothed, color=color, label="Dreamer")
    ax.legend(loc='upper right')

def main():
    fig, axs = plt.subplots(2)
    fig.suptitle('Results')
    graph_dreamer_data('dreamer_cloth_flatten_1.csv', 'green',axs[0], 'Cloth Flatten')
    graph_dreamer_data('dreamer_rope_flatten_1.csv', 'green',axs[1], 'Rope Flatten')
    plt.show()

if __name__ == "__main__":
   main()
