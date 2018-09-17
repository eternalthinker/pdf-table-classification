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

    # Read classes csv
    tables_mapping = dict()
    with open('classes.csv', 'r', encoding='utf-8') as classes_file:
        content = classes_file.read().split('\n')[:-1]
        
        for item in content:
            filename, class_str = item.split(',')
            if filename in tables_mapping:
                print("REPEAT: ", filename)
            tables_mapping[filename] = class_str
    print("Parsed %s CSV entries" % len(tables_mapping))

    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'data/*.html'))
    print("Parsing %s HTML files" % len(file_list))
    html_files = set()
    for f in file_list:
        basename = os.path.basename(f) # Only filename.ext part
        filename, ext = os.path.splitext(basename)
        html_files.add(filename)
        if filename not in tables_mapping:
            print('No CSV entry for:', filename, f)
        if not os.path.exists("data/{}.png".format(filename)):
            print('No image found for:', filename, f)
        with open(f, "r", encoding='utf-8') as openf:
            s = openf.read()
            s = clean_table(s)
            # no_punct = ''.join(c for c in s if c not in string.punctuation)
            s = s.lower()
            data.extend(s.split())

    file_list = glob.glob(os.path.join(dir,
                                        'data/*.png'))
    print("Parsing %s PNG files" % len(file_list))
    png_files = set()
    for f in file_list:
        basename = os.path.basename(f) # Only filename.ext part
        filename, ext = os.path.splitext(basename)
        png_files.add(filename)

    csv_files = set(tables_mapping.keys())
    print('CSV - HTML', csv_files - html_files)
    print('CSV - PNG', csv_files - png_files)
    print('HTML - CSV', html_files - csv_files)
    print('PNG - CSV', png_files - csv_files)
    print('PNG - HTML', png_files - html_files)
    print('HTML - PNG', html_files - png_files)
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

if __name__ == "__main__":
    read_data()
