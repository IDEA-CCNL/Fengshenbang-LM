python clip_finetune_flickr.py --batch_size 512 \
--num_gpus 1 \
--num_workers 20 \
--train_filename /shared_space/ccnl/mm_data/Flickr30k-CNA/train/flickr30k_cna_train.txt \
--val_filename /shared_space/ccnl/mm_data/Flickr30k-CNA/val/flickr30k_cna_val.txt \
--test_filename /shared_space/ccnl/mm_data/Flickr30k-CNA/test/flickr30k_cn_test.txt \
--train_root /shared_space/ccnl/mm_data/Flickr30k-CNA/flickr30k/images \
--val_root /shared_space/ccnl/mm_data/Flickr30k-CNA/flickr30k/images \
--test_root /shared_space/ccnl/mm_data/Flickr30k-CNA/flickr30k/images \

