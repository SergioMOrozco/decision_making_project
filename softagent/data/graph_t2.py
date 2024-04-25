import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import csv
import numpy as np 

# plots graph with standard deviation of error
def tsplot(ax, data,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw)
    ax.margins(x=0)

def plot_vanilla(data_list, min_len,title):

    sns.set_style("whitegrid", {'axes.grid' : True,
                                'axes.edgecolor':'black'

                                })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()

    colors = ["green", "blue"]
    labels = ["Dreamer", "Planet"]
    color_patch = []

    for color, label, data in zip(colors, labels, data_list):
        tsplot(ax,data, color=color)
        color_patch.append(mpatches.Patch(color=color, label=label))

    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes $(\\times10^6)$', fontsize=22)
    plt.ylabel('Average return', fontsize=22)
    plt.title(title, fontsize=28)

    ax = plt.gca()

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()

def get_dreamer_data(file_name):
    x = []
    y = []

    with open(file_name,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(plots):
            if i == 0:
                continue
            x.append((float(row[1])))
            y.append((float(row[2])))
            print(row[1], row[2])

    y = np.array(y).reshape((1,len(y)))
    return y

def main():
    y = get_dreamer_data('dreamer_cloth_flatten_1.csv')
    plot_vanilla([y], 300, "Cloth Flatten")

if __name__ == "__main__":
    main()
