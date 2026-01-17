OMP_NUM_THREADS=4 PYTHONPATH=../fairseq \
fairseq-hydra-train \
-cd conf/pretrain/mhubert1000_dual \
-cn large_noise_lrs2lrs3vox2avspmtedx_from_noise_avhubert
