import numpy as np
import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from implementation import class_mapping, reverse_class_mapping


# Import data
fnames = []
classes = []
pred_vecs = []
with open('pred_vecs.csv', 'r', encoding='utf-8') as pred_vecs_file:
    content = pred_vecs_file.read().split('\n')[:-1]
    for item in content:
        components = item.split(',')
        filename, class_name = components[0:2]
        pred_vec = components[2:]
        fnames.append(filename)
        classes.append(class_mapping[class_name])
        pred_vec_n = list(map(lambda s: float(s), pred_vec))
        pred_vecs.append(pred_vec_n)

X = np.array(pred_vecs)
y = np.array(classes)


def plot_with_labels(low_dim_embs, labels, filename='table_preds.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(18, 18))  # in inches
    colors = ['r', 'g', 'b', 'orange']
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=colors[class_mapping[label]])
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    print("plots saved in {0}".format(filename))

if __name__ == "__main__":
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = len(X) #len(reverse_dictionary)
    low_dim_embs = tsne.fit_transform(X[:plot_only, :])
    labels = [reverse_class_mapping[i] for i in y[:plot_only]]
    plot_with_labels(low_dim_embs, labels)
    plt.show();