import numpy as np
import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from implementation import compound_class_mapping as class_mapping, reverse_compound_class_mapping as reverse_class_mapping


# Import data
fnames = []
classes = []
pred_vecs = []
companies = []
with open('pred_vecs-dummy.csv', 'r', encoding='utf-8') as pred_vecs_file:
    content = pred_vecs_file.read().split('\n')[:-1]
    for item in content:
        components = item.split(',')
        filename, class_name, company = components[0:3]
        print(components[0:3])
        pred_vec = components[3:]
        fnames.append(filename)
        classes.append(class_mapping[class_name])
        companies.append(company)
        pred_vec_n = list(map(lambda s: float(s), pred_vec))
        pred_vecs.append(pred_vec_n)

X = np.array(pred_vecs)
y = np.array(classes)


def plot_with_labels(low_dim_embs, labels, companies, filename='table_preds_dummy.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(14, 9))  # in inches
    colors = ['r', 'g', 'b', 'orange', 'grey']
    colors = [
        'red',
        'firebrick',
        'lightcoral',
        'maroon',

        'olivedrab',
        'lawngreen',
        'g',
        'yellowgreen',

        'gold',
        'orange',
        'darkorange',
        'darkgoldenrod',

        'c',
        'darkcyan',
        'lightseagreen',
        'mediumaquamarine',

        'm',
        'fuchsia',
        'darkviolet',
        'darkorchid'
    ]

    taken_color_idxs = set()
    label_colors = dict()
    def get_color_dyn(label):
        if label in label_colors:
            return label_colors[label]
        company = label.split(':')[1]
        offset_begin = ['AGL', 'APA', 'CSL', 'RMD', 'TLS'].index(company) * 4
        for i in range(4):
            color_idx = offset_begin + i
            if color_idx not in taken_color_idxs:
                taken_color_idxs.add(color_idx)
                label_colors[label] = colors[color_idx]
                return label_colors[label]


    def get_color(label):
        company = label.split(':')[1]
        c_idx = {
            'AGL': 0,
            'APA': 1,
            'CSL': 2,
            'RMD': 3,
            'TLS': 4
        }[company]
        return colors[c_idx]

    def get_point_color(company):
        c_idx = {
            'AGL': 0,
            'APA': 4,
            'CSL': 8,
            'RMD': 12,
            'TLS': 16
        }[company]
        return colors[c_idx]

    def get_point_color2(label):
        c_idx = {
            'PRO': 0,
            'CAS': 4,
            'FIN': 8,
            'CHA': 12
        }[label[0:3]]
        return colors[c_idx]

    def get_label(label):
        company = "" #label.split(':')[1]
        return company

    legends = set()
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        company = companies[i]
        # plt.scatter(x, y, c=get_color(label), label=get_label(label)) # c=colors[class_mapping[label]], label=label)
        # plt.scatter(x, y, c=colors[class_mapping[label]], label=label)
        plt.scatter(x, y, c=get_point_color2(label), label=label)
        # plt.annotate(label,
        #              xy=(x, y),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
        label1 = get_label(label)
        # if label1 not in legends:
        #     print('#',label1,'#')
        #     legends.add(label1)
        #     plt.legend()

    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    pairs = zip(labels, handles)
    # pairs = sorted(pairs, key=lambda p: "{}{}".format(p[0].split(':')[1], p[0][:2]))
    pairs = sorted(pairs, key=lambda p: "{}{}".format(p[0], p[0][:2]))
    by_label = OrderedDict(pairs)
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(filename)
    print("plots saved in {0}".format(filename))

if __name__ == "__main__":
    tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=5000) #, method='exact')
    plot_only = len(X) #len(reverse_dictionary)
    low_dim_embs = tsne.fit_transform(X[:plot_only, :])
    labels = [reverse_class_mapping[i].split(":")[0] for i in y[:plot_only]]
    plot_with_labels(low_dim_embs, labels, companies)
    plt.show()