import os
import sys
import tarfile
import glob
import string
import collections
import re
import numpy as np


re_tags_with_attrs = re.compile(r"(<[a-z]+) .*?(/?>)")

def clean_table(content):
    # Clean tag attributes and separate with space
    content = re.sub(re_tags_with_attrs, r"\1\2", content)
    content = re.sub(r"(<)", r" \1", content)
    content = re.sub(r"(>)", r"\1 ", content)
    # Trim delimiters around edges
    content = content[1:-1]

    # Clean up numbers
    content = re.sub(r"[\(]([0-9]+[,\.]?)+[0-9]+[\)]", r"()", content)
    content = re.sub(r" ([0-9]+[,\.])+[0-9]+ ", r" ", content)

    # Remove tags
    content = re.sub(r"<.*?>", r"", content)

    return content


def read_data():
    print("READING DATA")
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'data/*.html'))
    #file_list.extend(glob.glob(os.path.join(dir,
    #                                    'data/neg/*.txt')))
    print("Parsing %s files" % len(file_list))
    for f in file_list:
        basename = os.path.basename(f)
        filename, ext = os.path.splitext(basename)
        if not os.path.exists("data/{}.png".format(filename)):
            print(filename, f)
        with open(f, "r", encoding='utf-8') as openf:
            s = openf.read()
            s = clean_table(s)
            # no_punct = ''.join(c for c in s if c not in string.punctuation)
            s = s.lower()
            data.extend(s.split())

    print(data[:5])
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

def get_dataset(vocabulary_size):
    if False: # os.path.exists(os.path.join(os.path.dirname(__file__), "data.npy")):
        print("loading saved parsed data, to reparse, delete 'data.npy'")
        data = np.load("data.npy")
        count = np.load("count.npy")
        dictionary = np.load("Word2Idx.npy").item()
        reverse_dictionary = np.load("Idx2Word.npy").item()
    else:
        vocabulary = read_data()
        print('Data size', len(vocabulary))
        vocabulary_size = len(set(vocabulary))
        print('======================', vocabulary_size)
        # Step 2: Build the dictionary and replace rare words with UNK token.
        data, count, dictionary, reverse_dictionary =\
            build_dataset(vocabulary, vocabulary_size)

        np.save("data", data)
        np.save("count", count)
        np.save("Idx2Word", reverse_dictionary)
        np.save("Word2Idx", dictionary)
        del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary, vocabulary_size
