from flask import Flask, render_template, flash, request
# from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import find_neighbours as neigh
from test_row_split import generate_row_similarity
 
# App config.
DEBUG = True
app = Flask(__name__, static_url_path='')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'topsy-kret'
app.config['STATIC_FOLDER'] = '.'

table_cluster_clf = None
fnames = None
X = None
cur_table_index = 11


def init():
    global table_cluster_clf
    global fnames
    global X
    X, y, fnames = neigh.load_data()
    table_cluster_clf = neigh.train_clf(X, y)

def find_neighbour_tables(table_index):
    ds, indices = neigh.get_neighbours(table_cluster_clf, X[table_index].reshape(1, -1))
    return generate_row_similarity(fnames, indices[0])
 
@app.route("/", methods=['GET', 'POST'])
def index():
    global cur_table_index
    if request.method == "POST":
        cur_table_index = int(request.form['queryTableIdx'])
 
    return find_neighbour_tables(cur_table_index)
 
if __name__ == "__main__":
    init()
    app.run()