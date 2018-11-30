import numpy as np
import pickle
import sys
import math

from gensim_load import load_google_embeddings

MAX_WORDS = 40
word2vec_array, word2vec_dict =  load_google_embeddings()
def convert_to_embedding(content):
    clean = content.lower()
    clean = clean.split()
    clean = clean[:MAX_WORDS]

    # Convert to index values
    int_value = []
    for word in clean:
        if word in word2vec_dict:
            index = word2vec_dict[word]
        else:
            index = 0
        int_value.append(index)

    # Zero padding
    int_value[len(int_value):MAX_WORDS] = [0] * (MAX_WORDS - len(int_value))
    embedding = []
    for idx in int_value:
        embedding += word2vec_array[idx].tolist()
    return embedding

def get_distance(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return math.sqrt(sum)
    
q = convert_to_embedding("Other income (expenses):")
r1 = convert_to_embedding("Other income, net:")
r2 = convert_to_embedding("Other, net (note 13)")

print("r1", get_distance(q, r1))
print("r2", get_distance(q, r2))







sys.exit(0) #4343434343

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

with open('pred_vecs-word.csv', 'r', encoding='utf-8') as pred_vecs_file, \
     open('pred_vecs-google.csv', 'r', encoding='utf-8') as google_pred_vecs_file:

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
