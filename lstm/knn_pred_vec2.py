import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import datasets

from implementation import class_mapping

n_neighbors = 15

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

iris = datasets.load_iris()
# X = iris.data[:, :2]
# y = iris.target
X = np.array(pred_vecs)
y = np.array(classes)
# print(X[0])
# print(y[0])

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#F4EE42'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#726F1C'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    a_min, a_max = X[:, 3].min() - 1, X[:, 3].max() + 1
    xx, yy, zz, aa = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h),
                                 np.arange(z_min, z_max, h),
                                 np.arange(a_min, a_max, h)
                                )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel(), aa.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("4-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()