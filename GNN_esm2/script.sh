python GNN_esm2/preprocess_data.py
python GNN_esm2/precompute_embeddings.py
python GNN_esm2/train_scalable.py --exp_name my_experiment

# predict
python GNN_esm2/predict.py --checkpoint GNN_esm2/checkpoints/test2/best.pth --output_dir GNN_esm2/predictions --threshold 0.2 --top_k 1280
cd GNN_esm2/predictions
kaggle competitions submit -c cafa-6-protein-function-prediction -f submission.tsv -m "Message"