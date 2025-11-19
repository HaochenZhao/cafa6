import pickle
import numpy as np
with open('./go_gnn_results/balanced_fixed/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(metadata.keys())

embeddings = np.load('./go_gnn_results/balanced_fixed/go_term_embeddings.npy')
print(embeddings.shape)