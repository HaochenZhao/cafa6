python predict.py \
    --checkpoint checkpoints/esm2_650m_exp1/best.pth \
    --use_precomputed_embeddings \
    --embeddings_file embeddings/embeddings_t33_650M_mean.npz \
    --output submissions/650m_1_submission.tsv \
    --threshold 0.05