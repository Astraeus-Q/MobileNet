export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--dataset_dir ./datasets \
--batch_size 128 \
--epochs 300 \
--lr 0.05  --wd 0.0005 \
--seed 0 \
--lr_scheduler \
--fig_name lr=0.05-lr_sche=cos-wd=0.0005-mixup.png \
--test
# --mixup \
# --wd 0.0005 \
# --fig_name lr=0.05-lr_sche-wd=0.0005-mixup.png \
