import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from implementation import class_mapping
from test_row_split import generate_row_similarity


# Import data
def load_data():
    fnames = []
    classes = []
    pred_vecs = []
    with open('pred_vecs.csv', 'r', encoding='utf-8') as pred_vecs_file:
        content = pred_vecs_file.read().split('\n')[:-1]
        for item in content:
            components = item.split(',')
            filename, class_name = components[0:2]
            pred_vec = components[2:]
            fnames.append([filename, class_name])
            classes.append(class_mapping[class_name])
            pred_vec_n = list(map(lambda s: float(s), pred_vec))
            pred_vecs.append(pred_vec_n)

    X = np.array(pred_vecs)
    y = np.array(classes)
    return X, y, fnames

def train_clf(X, y):
    clf = KNeighborsClassifier()
    clf.fit(X, y)
    return clf

def get_neighbours(clf, query_x):
    indices, distances = clf.kneighbors(query_x)
    return indices, distances


if __name__ == "__main__":
    X, y, fnames = load_data()
    clf = train_clf(X, y)
    ds, indices = get_neighbours(clf, X[2].reshape(1, -1))
    print(indices, ds)
    for i in indices[0]:
        print(i, fnames[i])
    generate_row_similarity(fnames, indices[0])

