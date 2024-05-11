# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import pandas as pd
import json
import os
import zipfile
import wget

# label = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_label.csv', delimiter=',', dtype='str')
# f = open("/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices.csv", "w")
# f.write("index,mid,display_name\n")
#
# label_set = []
# idx = 0
# for j in range(0, 5):
#     for i in range(0, 10):
#         cur_label = label[i][j]
#         cur_label = cur_label.split(' ')
#         cur_label = "_".join(cur_label)
#         cur_label = cur_label.lower()
#         label_set.append(cur_label)
#         f.write(str(idx)+',/m/07rwj'+str(idx).zfill(2)+',\"'+cur_label+'\"\n')
#         idx += 1
# f.close()
#

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# fill MDPS data

###########################

# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

for sample_type in ['1.0RPS','ES','LINE']:
    fold = 1
    base_path = f"./data/{exp_type}/audio/"

    meta = pd.read_csv('./data/{sample_type}/meta/{sample_type}_meta.csv')
    sample_count = len(meta)
    meta = meta.sample(frac=1, ignore_index=True, random_state=42)
    meta.iloc[:sample_count//5, 'fold'] = 0

    train_wav_list = []
    eval_wav_list = []
    for i in range(sample_count):
        cur_label = meta['category'][i]
        cur_path = meta['filename'][i]
        cur_fold = meta['fold'][i]
        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + cur_path, "labels": cur_label}
        if cur_fold == fold:
            train_wav_list.append(cur_dict)
        else:
            eval_wav_list.append(cur_dict)

    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open(f'./data/datafiles/mdps_train_data_{sample_type}_{fold}.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open(f'./data/datafiles/mdps_eval_data_{sample_type}_{fold}.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)

print('Finished MDPS Preparation')