import os
import sys
import tarfile
import glob
import string
import collections
import re
import numpy as np
from bs4 import BeautifulSoup
from load_table_data import clean_table


re_tags_with_attrs = re.compile(r"(<[a-z]+) .*?(/?>)")

def log(content):
    with open("log.txt", "a", encoding="utf-8") as logf:
        logf.write(str(content) + "\n")

def clean_table_dep(content):
    # Clean tag attributes and separate with space
    content = re.sub(re_tags_with_attrs, r"\1\2", content)
    content = re.sub(r"(<)", r" \1", content)
    content = re.sub(r"(>)", r"\1 ", content)
    # Trim delimiters around edges
    content = content[1:-1]

    # Clean up numbers
    content = re.sub(r"[\(]([0-9]+[,\.]?)+[0-9]+[\)]", r" ", content)
    content = re.sub(r" ([0-9]+[,\.])+[0-9]+ ", r" ", content)
    content = re.sub(r" [0-9]{1,3} ", r" ", content) # Keep years

    # Remove tags
    content = re.sub(r"<.*?>", r"", content)
    return content


def read_data(fnames, indices):
    print("READING CURRENT SUBSET OF DATA")

    data = []
    for idx in indices:
        fname = fnames[idx][0]
        f = os.path.join('data', '{}.html'.format(fname))
        with open(f, 'r', encoding='utf-8') as openf:
            table_content = openf.read()
            soup = BeautifulSoup(table_content, 'html.parser')
            row_tags = soup.find_all('tr')
            for row_tag in row_tags:
                clean = clean_table(str(row_tag))
                clean = clean.lower()
                # log(clean)
                clean = clean.split()
                data.extend(clean)
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def get_dataset(vocabulary_size, fnames, indices):
    if False: # os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed data, to reparse, delete 'data.npy'")
        data = np.load("data.npy")
        count = np.load("count.npy")
        dictionary = np.load("Word2Idx.npy").item()
        reverse_dictionary = np.load("Idx2Word.npy").item()
    else:
        vocabulary = read_data(fnames, indices)
        print('Data size', len(vocabulary))
        vocabulary_size = len(set(vocabulary))
        print('======================', vocabulary_size)
        # Step 2: Build the dictionary and replace rare words with UNK token.
        data, count, dictionary, reverse_dictionary =\
            build_dataset(vocabulary, vocabulary_size)

        #np.save("data", data)
        del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary, vocabulary_size

if __name__ == "__main__":
    pass
