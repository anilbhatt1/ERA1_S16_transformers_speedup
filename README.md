# Speeding up transformers

### Strategies used
    - One Cycle Policy
        - Refer train.py in pytorch_src folder (https://github.com/anilbhatt1/ERA1_S16_transformers_speedup/blob/master/pytorch_src/train.py)
    - Dynamic batching via collat_fn
        - Refer train.py in pytorch_src folder (https://github.com/anilbhatt1/ERA1_S16_transformers_speedup/blob/master/pytorch_src/train.py)
        - Refer get_ds(config) and collate_fn(batch) in train.py
        - Also, refer getitem function in dataset.py (https://github.com/anilbhatt1/ERA1_S16_transformers_speedup/blob/master/pytorch_src/dataset.py)
    - Used torch.cuda.amp.autocast(enabled=True) & associated scaler functions
    - Data cleaning 
        - Removed all English sentences with more than 150 "tokens"
        - Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
        - - Removed all English sentences with less than 20 "tokens"
- Trained in V100 GPU in google colab
- Time reduced 4 times. Previously it was taking 13 minutes now got reduced to 3.5 minutes per epoch for 
batch_size of 64
- Minimum loss achieved after training for 22 epochs : 1.778
- Training log for first 15 epochs are as follows. Minimum loss of 2.136 was achieved
```
len(sorted_ds) : 127085
len(filtered_sorted_ds) : 81559
len(train_ds) : 73403
len(val_ds) : 8156
Max length of source sentence: 149
Max length of target sentence: 157
Total parameters : 68145490
Total Epochs 0
Processing epoch  0: 100%|██████████| 1147/1147 [03:35<00:00,  5.33it/s, loss=4.485]
Total Epochs 1
Processing epoch  1: 100%|██████████| 1147/1147 [03:34<00:00,  5.35it/s, loss=3.825]
Total Epochs 2
Processing epoch  2: 100%|██████████| 1147/1147 [03:30<00:00,  5.46it/s, loss=3.098]
Total Epochs 3
Processing epoch  3: 100%|██████████| 1147/1147 [03:33<00:00,  5.36it/s, loss=2.888]
Total Epochs 4
Processing epoch  4: 100%|██████████| 1147/1147 [03:35<00:00,  5.31it/s, loss=2.794]
Total Epochs 5
Processing epoch  5: 100%|██████████| 1147/1147 [03:35<00:00,  5.32it/s, loss=2.503]
Total Epochs 6
Processing epoch  6: 100%|██████████| 1147/1147 [03:33<00:00,  5.36it/s, loss=2.505]
Total Epochs 7
Processing epoch  7: 100%|██████████| 1147/1147 [03:34<00:00,  5.35it/s, loss=2.413]
Total Epochs 8
Processing epoch  8: 100%|██████████| 1147/1147 [03:34<00:00,  5.34it/s, loss=2.390]
Total Epochs 9
Processing epoch  9: 100%|██████████| 1147/1147 [03:34<00:00,  5.35it/s, loss=2.275]
Total Epochs 10
Processing epoch  10: 100%|██████████| 1147/1147 [03:34<00:00,  5.35it/s, loss=2.184]
Total Epochs 11
Processing epoch  11: 100%|██████████| 1147/1147 [03:32<00:00,  5.39it/s, loss=2.195]
Total Epochs 12
Processing epoch  12: 100%|██████████| 1147/1147 [03:36<00:00,  5.29it/s, loss=2.229]
Total Epochs 13
Processing epoch  13: 100%|██████████| 1147/1147 [03:33<00:00,  5.37it/s, loss=2.274]
Total Epochs 14
Processing epoch  14: 100%|██████████| 1147/1147 [03:34<00:00,  5.35it/s, loss=2.136]
```
- After that restarted again and ran for 7 epochs by loading the weights using config['preload'] option
- Logs for those 7 epochs (15-22 epochs) are as below. Minimum loss of 1.778 was achieved.
```
len(sorted_ds) : 127085
len(filtered_sorted_ds) : 81559
len(train_ds) : 73403
len(val_ds) : 8156
Max length of source sentence: 149
Max length of target sentence: 157
Total parameters : 68145490
Model preloaded
Total Epochs 15
Processing epoch  15: 100%|██████████| 1147/1147 [03:42<00:00,  5.15it/s, loss=2.791]
Total Epochs 16
Processing epoch  16: 100%|██████████| 1147/1147 [03:40<00:00,  5.21it/s, loss=3.054]
Total Epochs 17
Processing epoch  17: 100%|██████████| 1147/1147 [03:40<00:00,  5.21it/s, loss=2.848]
Total Epochs 18
Processing epoch  18: 100%|██████████| 1147/1147 [03:42<00:00,  5.17it/s, loss=2.269]
Total Epochs 19
Processing epoch  19: 100%|██████████| 1147/1147 [03:41<00:00,  5.18it/s, loss=1.943]
Total Epochs 20
Processing epoch  20: 100%|██████████| 1147/1147 [03:41<00:00,  5.19it/s, loss=1.859]
Total Epochs 21
Processing epoch  21: 100%|██████████| 1147/1147 [03:41<00:00,  5.18it/s, loss=1.778]

```
- Colab Notebook reference : ERAV1_S16_Transformers_speedup_V2.ipynb
- Gdrive location for tensorboard runs: 
    - https://drive.google.com/drive/folders/1--0N21kIMTIUAPZkKTl1WLilwaoDxqBL
    - Epochs 1 - 15 : tmodel
    - Epochs 16 -22 : tmodel_2