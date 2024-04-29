import matplotlib.pyplot as plt
import csv
import numpy as np 
from scipy.interpolate import make_interp_spline


def get_data_from_csv(filename,x_idx,y_idx):
    x = []
    y = []
    with open(filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(plots):
            if i == 0:
                continue
            x.append((float(row[x_idx])))
            y.append((float(row[y_idx])))

    x = np.array(x)
    y = np.array(y)

    return x,y

def smooth_data(x,y):
    x_smoothed= np.linspace(x.min(), x.max(), 50)
    gfg = make_interp_spline(x, y, k=3)
    y_smoothed = gfg(x_smoothed)

    return x_smoothed,y_smoothed

def graph_data(filename,color,ax, algorithm, title,x_idx,y_idx):
    x,y = get_data_from_csv(filename,x_idx,y_idx)

    x_smoothed, y_smoothed = smooth_data(x, y)

    ax.set(xlabel='Episode', ylabel='Reward')
    ax.set_title(title)

    ax.plot(x, y, color=color, alpha=0.2)
    ax.plot(x_smoothed, y_smoothed, color=color, label=algorithm)
    ax.legend(loc='upper right')

def main():
    fig, axs = plt.subplots(3)
    fig.tight_layout(pad=2.0)
    fig.suptitle('Results')
    graph_data('dreamer_cloth_flatten_1.csv', 'green',axs[0], "Dreamer", 'Cloth Flatten',1,2)
    graph_data('dreamer_rope_flatten_1.csv', 'green',axs[1], "Dreamer", 'Rope Flatten',1,2)
    graph_data('dreamer_cloth_fold_1.csv', 'green',axs[2], "Dreamer", 'Cloth Fold',1,2)

    graph_data('clothflatten_seed100/progress.csv', 'blue',axs[0], "Planet", 'Cloth Flatten',8,2)
    graph_data('ropeflatten_seed100/progress.csv', 'blue',axs[1], "Planet", 'Rope Flatten',27,8)
    graph_data('clothfold_seed100/progress.csv', 'blue',axs[2], "Planet", 'Cloth Folds',32,29)
    plt.show()

if __name__ == "__main__":
   main()
