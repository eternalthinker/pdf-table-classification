from flask import Flask, render_template, flash, request, jsonify
# from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import find_neighbours as neigh
from test_row_split import generate_row_similarity
import query_util
 
# App config.
DEBUG = True
app = Flask(__name__, static_url_path='')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'topsy-kret'
app.config['STATIC_FOLDER'] = '.'

table_cluster_clf = None
fnames = None
X = None
cur_table_index = 5
parsed_tables_map = dict()


def init():
    global table_cluster_clf
    global fnames
    global X
    X, y, fnames = neigh.load_data()
    table_cluster_clf = neigh.train_clf(X, y)

def find_neighbour_tables(table_index):
    ds, indices = neigh.get_neighbours(table_cluster_clf, X[table_index].reshape(1, -1))
    return generate_row_similarity(fnames, indices[0], ds[0])
 
@app.route("/", methods=['GET', 'POST'])
def index():
    global parsed_tables_map
    global cur_table_index
    if request.method == "POST":
        cur_table_index = int(request.form['queryTableIdx'])
 
    main_html, parsed_tables_map = find_neighbour_tables(cur_table_index)
    return main_html

@app.route("/query", methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data['query']
    conds = query_util.parse_query(query)
    result_map = dict()
    for fname, rows in parsed_tables_map.items():
        result_row_idxs = []
        for i, row in enumerate(rows):
            new_row = row[:]
            for cond in conds:
                new_row = filter(cond, new_row)
            if len(list(new_row)) > 0:
                result_row_idxs.append(i)
        result_map[fname] = result_row_idxs[:]
    return jsonify(result_map)


 
if __name__ == "__main__":
    init()
    app.run()