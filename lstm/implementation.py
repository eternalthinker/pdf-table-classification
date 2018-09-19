import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string
import pickle # For testing
import re
from load_table_data import clean_table

batch_size = 10
max_words = 40

class_mapping = {
    "PROFIT_OR_LOSS" : 0,
    "FINANCIAL_POSITION": 1,
    "CHANGES_IN_EQUITY": 2,
    "CASH_FLOWS": 3
}

reverse_class_mapping = dict()
for class_name, class_num in class_mapping.items():
    reverse_class_mapping[class_num] = class_name

def load_data(word2vec_dict):
    if False: # os.path.exists(os.path.join(os.path.dirname(__file__), "tables.pkl")):
        print("loading pickled vectorized table...")
        with open("tables.pkl", "rb") as reviews_file:
            data = pickle.load(reviews_file)
            return data

    data = []
    classes = []
    fnames = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir, 'data/*.html'))
    i = 0

    stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", \
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", \
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", \
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", \
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", \
        "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", \
        "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", \
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", \
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

    tables_mapping = dict()
    with open('classes.csv', 'r', encoding='utf-8') as classes_file:
        content = classes_file.read().split('\n')[:-1]
        
        for item in content:
            filename, class_str = item.split(',')
            tables_mapping[filename] = class_mapping[class_str]

    for fname, table_class in tables_mapping.items():
        f = os.path.join('data', '{}.html'.format(fname))
        with open(f, 'r', encoding='utf-8') as openf:
            s = openf.read()
            s = clean_table(s)

            # Pre-processessing
            clean = s.lower()
            # clean = ''.join(c if c not in string.punctuation else ' ' for c in clean)
            clean = clean.split()
            # clean = [word for word in clean if word not in stopwords]
            clean = clean[:max_words]

            # Convert to index values
            int_value = []
            for word in clean:
                if word in word2vec_dict:
                    index = word2vec_dict[word]
                else:
                    index = 0
                int_value.append(index)

            # Zero padding
            int_value[len(int_value):max_words] = [0] * (max_words - len(int_value))
            
        data.append(int_value)
        classes.append(table_class)
        fnames.append(fname)

    data = np.array(data, dtype=np.float32)
    with open("tables.pkl", "wb") as reviews_file:
        pickle.dump(data, reviews_file)
    return data, classes, fnames


def load_word2vec_embeddings():
    print('Extracting word2vec...')
    embeddings = [[]]
    word_index_dict = {'UNK': 0}
    reverse_dictionary = np.load("Idx2Word.npy").item()
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

    with open("embeddings.pkl", "wb") as embeddings_file:
        pickle.dump(embeddings, embeddings_file)
    with open("wid.pkl", "wb") as wid_file:
        pickle.dump(word_index_dict, wid_file)
    return embeddings, word_index_dict


def define_graph(word2vec_embeddings_arr):
    vector_length = 40
    num_classes = 4
    num_lstm = 64
    num_layers = 1

    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")

    input_data = tf.placeholder(tf.int32, shape = [batch_size, vector_length], name="input_data")
    labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name="labels")

    embed_input_data = tf.nn.embedding_lookup(word2vec_embeddings_arr, input_data)
    lstm_stack = []
    for _ in range(num_layers):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_lstm)
        drop = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)
        lstm_stack.append(drop)
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_stack)
    cell = multi_cell
    # Single cell
    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_lstm) 
    # drop = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)
    # cell = drop
    value, final_state = tf.nn.dynamic_rnn(cell, embed_input_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([num_lstm, num_classes], dtype=tf.float32))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), dtype=tf.float32)
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    pred_prob = tf.nn.softmax(logits=prediction, name="pred_prob")

    pred_class = tf.argmax(prediction, 1, name="pred_class")
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss, prediction, correct_pred, pred_class, pred_prob
