import numpy as np
import os
from load_table_data import read_data


def log(content):
    with open("log.txt", "a", encoding="utf-8") as logf:
        logf.write(str(content) + "\n")


def load_google_embeddings():
    if os.path.exists(os.path.join(os.path.dirname(__file__), "google_w2c_array.npy")):
        print('Loading pre-processed google embeddings')
        embeddings = np.load('google_w2c_array.npy')
        word_index_dict = np.load('google_w2c_dict.npy').item()
        return embeddings, word_index_dict
    valid_words = set(read_data())
    print('Loading google word2vec...')
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print('Loaded google word2vec')
    embeddings = [[]]
    word_index_dict = { 'UNK': 0 }
    i = 1
    for word in model.wv.vocab.keys():
        if word not in valid_words:
            continue
        value = model[word]
        embeddings.append(value)
        word_index_dict[word] = i
        i += 1
        if i % 100 == 0:
            print('Word', i)
    embeddings[0] = [0]*len(embeddings[1])
    embeddings = np.array(embeddings, np.float32)
    print('Processed google word2vec')
    np.save('google_w2c_array', embeddings)
    np.save('google_w2c_dict', word_index_dict)
    return embeddings, word_index_dict


def old__load_word2vec_embeddings(reverse_dictionary=None, word2vec=None):
    print('Extracting word2vec...')
    embeddings = [[]]
    word_index_dict = {'UNK': 0}
    if reverse_dictionary is None:
        reverse_dictionary = np.load("Idx2Word.npy").item()
    if word2vec is None:
        word2vec = np.load("CBOW_Embeddings.npy")
    i = 0
    for line in word2vec:
        vector = line
        word = reverse_dictionary[i]
        value = [float(x) for x in vector]
        embeddings.append(value)
        word_index_dict[word] = i
        i += 1
    embeddings[0] = [0]*len(embeddings[1])
    embeddings = np.array(embeddings, np.float32)

    #with open("embeddings.pkl", "wb") as embeddings_file:
    #    pickle.dump(embeddings, embeddings_file)
    #with open("wid.pkl", "wb") as wid_file:
    #    pickle.dump(word_index_dict, wid_file)
    return embeddings, word_index_dict


if __name__ == "__main__":
    load_google_embeddings()