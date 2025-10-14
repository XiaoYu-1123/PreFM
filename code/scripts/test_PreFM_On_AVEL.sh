python main_PreFM_On_AVEL.py \
--mode test \
--model PreFM_On_AVEL_Net \
--model_name PreFM_On_AVEL \
--gpu 1 \
--val_batch_size 60 \
--pd_dir_test ./data_info/UnAV-100/test_pd.csv \
--audio_dir ../dataset/UnAV-100/data/CLAP \
--visual_dir ../dataset/UnAV-100/data/CLIP \
--label_dir ../dataset/UnAV-100/label/label_clip_clap \
--label_train ../dataset/UnAV-100/label/label_clip_clap/train \
--label_val ../dataset/UnAV-100/label/label_clip_clap/val \
--label_test ../dataset/UnAV-100/label/label_clip_clap/test \
--f_label_train ../dataset/UnAV-100/label/feature_label/train \
--f_label_val ../dataset/UnAV-100/label/feature_label/val \
--f_label_test ../dataset/UnAV-100/label/feature_label/test \
--hidden_dim 256 \
--nhead 8 \
--ff_dim 1024 \
--num_layers 4 \
--norm_where post_norm \
--future_length 5 \

# bash scripts/test_PreFM_On_AVEL.sh | tee ./logs/PreFM_On_AVEL/val_res.txt