# Using the data
The following files and directories are to be copied into `lstm/`  
* **`classes.csv`** Contains filenames containing individual table markup, their classification, and the company name.
* **`classes-orig.csv`** This is mostly the same as above, but will always preserve the original company names. When we only want to classify based on table type (2nd column), a quick way to do this without changing any code is, change the 3rd column in `classes.csv` to the same dummy value. So, in effect, even when 2nd and 3rd column are combined to form a single class, there is still only 4 classes. This original file is kept for those cases, when you classify into 4 classes by changing `classes.csv`, but still want a reference to original company names for analysis.
* **`data`** Contains extracted tables in single files. If the filename is `1234_x.html`, the original PDF will be `1234.pdf`
* **`full_data`** Contains original PDFs and the converted HTMLs in their complete form. This can be used in cases when you want to display the tables on a web UI. The style tags can be parsed from these HTMLs to preserve the original look of tables.

## Additional data
We use Google News pretrained word2vec embeddings in the training, in addition to custom trained vocabulary.  
This can be obtained at https://code.google.com/archive/p/word2vec/  
The archive is available at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing  
The file needs to be unzipped and copied to `lstm/`

# How to run
  
*Step 1*: Create embeddings from dataset:  
`python3 word2vec_cbow.py`
  
*Step 2*: Train LSTM neural net and create probability vectors for tables:  
`python3 train_and_test.py`  
This will create `pred_vecs.csv`, which is a list of probability distribution vectors corresponding to each table

*Step 3*: You can run the server after this step for visual analysis
`python3 server.py`
  
The server uses KNN behind the scenes to cluster table vectors. When you select a table from the UI, the nearest tables are queried and returned. See `find_neighbours.py`  
  
Furthermore, the rows in these tabled are sliced, converted into vectors, and clustered. The mapping to their original tables are kept track of. The server then finds nearest matches for each row and include this pre-processed information in the response to UI. This data can be used in row based similarity querying. That is, when a row is selected in the query table, similar rows are highlighted in the result tables. See `test_row_split.py`  
  
Tables are also parsed and loaded into memory as matrices, which enables querying them. See `table_util.py`

# Additional files and directories
**`fill_tables.py`** Parse individual HTML tables and load them into a Postgres database for similarity checking with database methods  
  
**`knn_pred_vec.py`** Prints KNN accuracy of `pred_vecs.csv`  
  
**`static/`** Web UI resources and Flask templates  
  
**`local_word2vec/`** Modules to do a custom word2vec, but only on the set of query table + result tables. Optionally used in current program logic  
  
**`plot_embeddings`** Plot tSNE 2D map of word2vec embeddings  
  
**`plot_table_preds`** Plot KNN cluster for prediction vectors 
  
**`random_seed.pkl`** The train-test split batches are picked at random. In case you want to test slightly varying factors in training but with the same test-train split, the randomness is loaded from this seed so that the data used remains constant and helps accurate comparison (For example, test using google word2vec vs custom word2vec). See commented lines with `pickle.dump()` call in `test.py` to generate new `random_seed.pkl`.  


  
  
