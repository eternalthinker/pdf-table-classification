import numpy as np
import pickle


#s = np.random.get_state()
#pickle.dump(s, open("random_seed.pkl", "wb"))

s = pickle.load(open("random_seed.pkl", "rb"))
np.random.set_state(s)
print(np.random.randint(1, 101))

state = pickle.load(open("random_seed.pkl", "rb"))
np.random.set_state(state)
rng_state = np.random.get_state()
np.random.set_state(rng_state)
a = list(range(1,11))
np.random.shuffle(a)
print(a)

with open('pred_vecs.csv', 'r', encoding='utf-8') as pred_vecs_file, \
     open('pred_vecs.csv', 'r', encoding='utf-8') as google_pred_vecs_file:

    pred_vecs_set = set()
    content = pred_vecs_file.read().split('\n')[:-1]
    for item in content:
        components = item.split(',')
        filename, class_name, company = components[0:3]
        pred_vecs_set.add(filename)
    
    google_pred_vecs_set = set()
    content = google_pred_vecs_file.read().split('\n')[:-1]
    for item in content:
        components = item.split(',')
        filename, class_name, company = components[0:3]
        google_pred_vecs_set.add(filename)

    print("P - G", len(pred_vecs_set - google_pred_vecs_set))
    print("G - P", len(google_pred_vecs_set - pred_vecs_set))
