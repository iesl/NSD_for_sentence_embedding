import torch
import os
import numpy as np
import threadpool

file_list = ['test_org.pt', 'test_shuffled.pt',
             'train.pt', 'val_org.pt', 'val_shuffled.pt']
split_list = ['corpus_00','corpus_01','corpus_02',
              'corpus_03','corpus_04','corpus_05',
              'corpus_06','corpus_07','corpus_08',]
data_dir = '/iesl/canvas/hanqingli/NSD_for_sentence_embedding/data/wiki2016_min100/5_5'
output_dir = os.path.join(data_dir,'all')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file_name in file_list:
    feature, target = None, None
    for split in split_list:
        split_dir = os.path.join(data_dir, split)
        file_dir = os.path.join(split_dir, file_name)
        data = torch.load(file_dir, map_location=torch.device('cpu'))
        # print(data)
        if feature is None:
            feature, target = data
        else:
            feature = torch.cat((feature, data[0]), dim=0)
            target = torch.cat((target, data[1]), dim=0)
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, 'wb') as f_out:
        torch.save([feature, target], f_out)
        print('{} finished'.format(file_name))
