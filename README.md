# Usage
  
Create embeddings from dataset:  
`python3 word2vec_cbow.py`  
  
Train LSTM neural net and create probability vectors for tables:  
`python3 train_and_test.py`  

You can run the server after this step for visual analysis
`python3 server.py`
  
Programatically find neighbours given a particular table, and proceed to find similar rows among tables (used internally by server):  
`python3 find_neighbours.py`  
  
  
