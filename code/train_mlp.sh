python train.py \
    --use_precomputed_embeddings \
    --embeddings_file embeddings/embeddings_t33_650M_mean.npz \
    --batch_size 64 \
    --lr 5e-6 \
    --num_epochs 500 \
    --save_dir checkpoints/esm2_650m_exp3