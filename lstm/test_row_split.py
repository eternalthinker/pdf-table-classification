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
    table_to_rows_map = dict()
    row_to_table_map = dict()
    rows_orig = []
    subrows = []
    rows_embed = []
    i = 0
    for idx in indices:
        fname = fnames[idx][0]
        table_to_rows_map[fname] = set()
        f = os.path.join('data', '{}.html'.format(fname))
        with open(f, 'r', encoding='utf-8') as openf:
            s = openf.read()
            t_rows_orig, t_rows_embed = split_rows(s)
            rows_orig += t_rows_orig
            rows_embed += t_rows_embed
            for j in range(len(t_rows_orig)):
                table_to_rows_map[fname].add(i)
                row_to_table_map[i] = fname
                subrows.append(j+1)
                i += 1
    return table_to_rows_map, row_to_table_map, subrows, rows_orig, rows_embed


def find_similar_rows(neigh, query):
    dist, ind = neigh.kneighbors(query)
    return dist, ind

def add_td(tr_str, td_string):
    soup = BeautifulSoup(tr_str, 'html.parser')
    tr_tag = soup.tr
    new_td_tag = soup.new_tag("td", style="background: #facade")
    tr_tag.append(new_td_tag)
    new_td_tag.string = td_string
    return soup

def generate_row_similarity(fnames, neighbours_idxs):
    table_to_rows_map, row_to_table_map, subrows, rows_orig, rows_embed\
      = process_indices(fnames, neighbours_idxs)
    
    nrows = [len(v) for v in table_to_rows_map.values()]
    avg = sum(nrows) // len(nrows)

    neigh = NearestNeighbors(n_neighbors=10)
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
            # html.write(rows_orig[row_idx])
            soup = add_td(rows_orig[row_idx], "Table: {}, Row: {}".format(row_to_table_map[row_idx], subrows[row_idx]))
            html.write(str(soup))
            for i in ind[0]:
                if i == row_idx:
                    continue
                soup = add_td(rows_orig[i], "Table: {}, Row: {}".format(row_to_table_map[i], subrows[i]))
                orig_fname = row_to_table_map[row_idx]
                cur_fname = row_to_table_map[i]
                if orig_fname == cur_fname:
                    soup = add_td(str(soup), "SAME TABLE")
                html.write(str(soup))
            html.write("</table><hr />")


if __name__ == "__main__":
    neighbours_idxs = [2, 0, 25, 53, 52] 
    generate_row_similarity(neighbours_idxs)


