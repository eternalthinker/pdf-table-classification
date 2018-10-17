import os
from bs4 import BeautifulSoup
import numpy as np
from sklearn.neighbors import NearestNeighbors
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
import json

from load_table_data import clean_table
from implementation import load_word2vec_embeddings


MAX_WORDS = 10
word2vec_array, word2vec_dict = load_word2vec_embeddings()


def log(content):
    with open("log.txt", "w", encoding="utf-8") as logf:
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
    row_counts = []
    for idx in indices:
        fname = fnames[idx][0]
        table_to_rows_map[fname] = set()
        f = os.path.join('data', '{}.html'.format(fname))
        with open(f, 'r', encoding='utf-8') as openf:
            s = openf.read()
            t_rows_orig, t_rows_embed = split_rows(s)
            rows_orig += t_rows_orig
            rows_embed += t_rows_embed
            row_counts.append(len(t_rows_orig))
            for j in range(len(t_rows_orig)):
                table_to_rows_map[fname].add(i)
                row_to_table_map[i] = fname
                subrows.append(j)
                i += 1
    return table_to_rows_map, row_to_table_map, subrows, rows_orig, rows_embed, row_counts


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

def get_style(part_fname):
    fname = part_fname.split("_")[0]
    f = os.path.join('full_data', '{}.html'.format(fname))
    if not os.path.exists(f):
        print("[!] ----- File {} not found to extract style".format(f))
        return None
    with open(f, 'r', encoding='utf-8') as openf:
        raw_html = openf.read()
        soup = BeautifulSoup(raw_html, 'html.parser')
        style_tags = soup.find_all('style')
        style = str(style_tags[0])
        return style

def render_table(table_template, fname, class_str, company, script):
    table_config = {
        'table': {
            'script': script,
            'class_str': class_str,
            'company': company
        }
    }
    style = get_style(fname)
    table_config['table']['style'] = style or ""
    f = os.path.join('data', '{}.html'.format(fname))
    with open(f, 'r', encoding='utf-8') as openf:
        table_config['table']['content'] = openf.read()
    table = table_template.render(**table_config)
    return table

def generate_row_similarity(fnames, neighbours_idxs):
    cur_table_index = neighbours_idxs[0]
    table_to_rows_map, row_to_table_map, subrows, rows_orig, rows_embed, row_counts \
      = process_indices(fnames, neighbours_idxs)
    
    nrows = [len(v) for v in table_to_rows_map.values()]
    avg = sum(nrows) // len(nrows)

    neigh = NearestNeighbors(n_neighbors=15)
    neigh.fit(rows_embed)

    with open("test.html", "w", encoding="utf-8") as html, \
         open("output.html", "w", encoding="utf-8") as output_html:

        html.write('''
            <style>
                tr:first-child {
                    background: yellow;
                }
            </style>
        ''')

        row_sim_map = dict()
        for row_idx in range(row_counts[0]):
            query = np.array(rows_embed[row_idx]).reshape(1, -1)
            dist, ind = find_similar_rows(neigh, query)
            ind_neighbours_dist = \
                filter(lambda p: row_to_table_map[p[0]] != fnames[neighbours_idxs[0]][0], zip(ind[0], dist[0]))
            row_sim_map[row_idx] = list(map(lambda p: {
                'fname': row_to_table_map[p[0]],
                'row': subrows[p[0]],
                'dist': p[1]
            }, ind_neighbours_dist))

            # print(dist, ind)
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

        env = Environment(
            # loader=PackageLoader('static', 'ui', 'templates'),
            loader = FileSystemLoader(searchpath="./static/ui/templates/"),
            autoescape=None #select_autoescape(['html'])
        )
        table_template = env.get_template('table.html')
        main_template = env.get_template('index.html')

        fname, class_str, company = fnames[neighbours_idxs[0]]
        query_table = render_table(table_template, fname, class_str, company, False, )

        main_config = dict()
        main_config['query'] = {
            'table_html': query_table,
            'row_sim_map': json.dumps(row_sim_map)
        }
        neighbour_tables = []
        neighbour_fnames = []
        neighbour_indices = neighbours_idxs[1:]
        for i in neighbour_indices:
            fname, class_str, company = fnames[i]
            neighbour_fnames.append(fname)
            neighbour_table = render_table(table_template, fname, class_str, company, True)
            neighbour_tables.append({
                'content': neighbour_table,
                'fname': fname
            })
        #neighbour_tables = map(lambda s: { 'fname': s['fname'], 'content': "{}".format(s['content']) }, neighbour_tables)
        main_config['neighbours'] = {
            'content_list': neighbour_tables,
            'fnames': json.dumps(neighbour_fnames)
        }
        main_config['tables'] = {
            'table_list': fnames,
            'cur_table_index': cur_table_index
        }
        main_html = main_template.render(**main_config)
        output_html.write(main_html)
        return main_html



if __name__ == "__main__":
    neighbours_idxs = [2, 0, 25, 53, 52] 
    generate_row_similarity(neighbours_idxs)


