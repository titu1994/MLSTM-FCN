import numpy as np
import os

base_path = '../data/JapaneseVowels/'
train_data_path = base_path + 'ae.train'
test_data_path = base_path + 'ae.test'
train_speaker_rows_path = base_path + 'size_ae.train'
test_speaker_rows_path = base_path + 'size_ae.test'

with open(train_data_path, 'r') as f:
    for i in range(20):
        lines = f.readline().split(' ')[:-1]
        print(lines)
    print('out of loop')
    lines = f.readline()
    print(lines)

    print('done')