python main_PreFM_On_AVVP.py \
--mode test \
--model PreFM_On_AVVP_Net \
--model_name PreFM_On_AVVP \
--gpu 3 \
--val_batch_size 60 \
--pd_dir_test ./data_info/LLP/test_pd.csv \
--audio_dir ../dataset/LLP/data/CLAP \
--visual_dir ../dataset/LLP/data/CLIP \
--st_dir ../dataset/LLP/data/st \
--label_dir ../dataset/LLP/label/label \
--f_label_dir ../dataset/LLP/label/feature_label \
--hidden_dim 128 \
--nhead 8 \
--ff_dim 512 \
--num_layers 4 \
--norm_where post_norm \
--future_length 5 \

# bash scripts/test_PreFM_On_AVVP.sh | tee ./logs/PreFM_On_AVVP/1011res.txt