import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from implementation import compound_class_mapping
from test_row_split import generate_row_similarity


# Import data
def load_data():
    tables_map = dict()
    with open('classes.csv', 'r', encoding='utf-8') as classes_file:
        content = classes_file.read().split('\n')[:-1]
        for item in content:
            filename, class_str, company = item.split(',')
            tables_map[filename] = [class_str, company]
    fnames = []
    classes = []
    pred_vecs = []
    with open('pred_vecs.csv', 'r', encoding='utf-8') as pred_vecs_file:
        content = pred_vecs_file.read().split('\n')[:-1]
        for item in content:
            components = item.split(',')
            filename, class_name, o_comp = components[0:3]
            pred_vec = components[3:]
            company = tables_map[filename][1]
            fnames.append([filename, class_name, company, o_comp])
            classes.append(compound_class_mapping[class_name])
            pred_vec_n = list(map(lambda s: float(s), pred_vec))
            pred_vecs.append(pred_vec_n)

    X = np.array(pred_vecs)
    y = np.array(classes)
    return X, y, fnames

def train_clf(X, y):
    clf = KNeighborsClassifier(n_neighbors=6)
    clf.fit(X, y)
    return clf

def get_neighbours(clf, query_x):
    distances, indices = clf.kneighbors(query_x)
    return distances, indices


if __name__ == "__main__":
    X, y, fnames = load_data()
    clf = train_clf(X, y)
    ds, indices = get_neighbours(clf, X[53].reshape(1, -1))
    print(indices, ds)
    for i in indices[0]:
        print(i, fnames[i])
    generate_row_similarity(fnames, indices[0], ds[0])

