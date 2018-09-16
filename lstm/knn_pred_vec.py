import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV

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

X = np.array(pred_vecs)
y = np.array(classes)


################################################################################
def compute_accuracy(clf, data, target):
    y = clf.predict(data)
    score = accuracy_score(target, y)
    return score

################################################################################
def print_accuracy(clf, data1, target1, data2, target2):
    print ("- Training set", compute_accuracy(clf, data1, target1))
    print ("- Testing set", compute_accuracy(clf, data2, target2))

################################################################################
def print_gridsearch_summary(clf, parameters):
    print ("parameters:")
    print(parameters) 
    
    print("Best parameters set:")
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def tune_knn(data, target, rseed=42, verbose=1):
    print ("")
    print ("Tuning K-NN ")
    print ("--------------------------------------------------------------------------------")
    parameters = {
        'n_neighbors':  [1, 2, 3, 5]
    }

    classifier = KNeighborsClassifier()
    clf = GridSearchCV(classifier, parameters, verbose=verbose)
    clf.fit(data, target)

    print_gridsearch_summary(clf, parameters)

    return clf


verbosity = 1
data1, data2, target1, target2 = train_test_split(X, \
                                                      y, \
                                                      test_size=0.7, \
                                                      random_state=42)

clf_knn = tune_knn(data1, target1, verbose=verbosity)
print_accuracy(clf_knn, data1, target1, data2, target2)
