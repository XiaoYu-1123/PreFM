python main_PreFM+_On_AVEL.py \
--mode test \
--model PreFM_plus_On_AVEL_Net \
--model_name PreFM+_On_AVEL \
--gpu 2 \
--val_batch_size 240 \
--pd_dir_test ./data_info/UnAV-100/test_pd.csv \
--audio_dir ../dataset/UnAV-100/data/onepeace \
--visual_dir ../dataset/UnAV-100/data/onepeace \
--true_label_dir ../dataset/UnAV-100/label/label_clip_clap \
--label_train ../dataset/UnAV-100/label/label_onepeace/train \
--label_val ../dataset/UnAV-100/label/label_onepeace/val \
--label_test ../dataset/UnAV-100/label/label_onepeace/test \
--f_label_train ../dataset/UnAV-100/label/feature_label_modified/train \
--f_label_val ../dataset/UnAV-100/label/feature_label_modified/val \
--f_label_test ../dataset/UnAV-100/label/feature_label_modified/test \
--hidden_dim 512 \
--nhead 8 \
--ff_dim 1024 \
--num_layers 4 \
--norm_where post_norm \
--future_length 5 \

# bash scripts/test_PreFM+_On_AVEL.sh | tee ./logs/PreFM+_On_AVEL/1011res.txt