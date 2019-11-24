# For WeiboNER
python -m main --train_file Weibo_TRAIN_FILE --dev_file Weibo_DEV_FILE --test_file Weibo_TEST_FILE --gat_nhead 5 --gat_layer 2 --strategy n --batch_size 10 --lr 0.001 --lr_decay 0.01 --use_clip False --optimizer SGD --droplstm 0 --dropout 0.6 --dropgat 0 --gaz_dropout 0.4 --norm_char_emb True --norm_gaz_emb False --random_seed 100
