import os
from bs4 import BeautifulSoup
import numpy as np
from sklearn.neighbors import NearestNeighbors

from load_table_data import clean_table
from implementation import load_word2vec_embeddings


MAX_WORDS = 10
word2vec_array, word2vec_dict = load_word2vec_embeddings()


def log(content):
    with open("log.txt", "a", encoding="utf-8") as logf:
        logf.write(str(content) + "\n")

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
    return int_value

def split_rows(table_content):
    soup = BeautifulSoup(table_content, 'html.parser')
    row_tags = soup.find_all('tr')
    rows_orig = []
    rows_embed = []
    for row_tag in row_tags:
        row_clean = clean_table(str(row_tag))
        embedding = convert_to_embedding(row_clean)
        rows_orig.append(str(row_tag))
        rows_embed.append(embedding)
    return rows_orig, rows_embed

def process_indices(fnames, indices):
    tables = dict()
    rows_orig = []
    rows_embed = []
    i = 0
    for idx in indices:
        fname = fnames[idx][0]
        tables[fname] = set()
        f = os.path.join('data', '{}.html'.format(fname))
        with open(f, 'r', encoding='utf-8') as openf:
            s = openf.read()
            t_rows_orig, t_rows_embed = split_rows(s)
            rows_orig += t_rows_orig
            rows_embed += t_rows_embed
            for _ in range(len(t_rows_orig)):
                tables[fname].add(i)
                i += 1
    return tables, rows_orig, rows_embed


def find_similar_rows(neigh, query):
    dist, ind = neigh.kneighbors(query)
    return dist, ind

def generate_row_similarity(fnames, neighbours_idxs):
    tables, rows_orig, rows_embed = process_indices(fnames, neighbours_idxs)
    
    nrows = [len(v) for v in tables.values()]
    avg = sum(nrows) // len(nrows)

    neigh = NearestNeighbors()
    neigh.fit(rows_embed)

    with open("test.html", "w") as html:
        html.write('''
            <style>
                tr:first-child {
                    background: yellow;
                }
            </style>
        ''')
        for row_idx in range(1, 15):
            query = np.array(rows_embed[row_idx]).reshape(1, -1)
            dist, ind = find_similar_rows(neigh, query)

            print(dist, ind)
            html.write("<table>")
            html.write(rows_orig[row_idx])
            for i in ind[0]:
                if i == row_idx:
                    continue
                html.write(rows_orig[i])
            html.write("</table><hr />")


if __name__ == "__main__":
    neighbours_idxs = [2, 0, 25, 53, 52] 
    generate_row_similarity(neighbours_idxs)


