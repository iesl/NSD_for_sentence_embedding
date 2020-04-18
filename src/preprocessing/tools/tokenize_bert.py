# import sys
import csv

import pandas as pd
from transformers import *

# bert_dir = '/iesl/canvas/hschang/language_modeling/pytorch-pretrained-BERT'
# bert_dir = '/mnt/nfs/scratch1/hschang/language_modeling/pytorch-pretrained-BERT'
# sys.path.insert(1, bert_dir)
# from pytorch_pretrained_bert import BertTokenizer, BertModel
#
# BERT_model_path = 'bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(BERT_model_path, cache_dir = bert_dir + '/cache_dir/', do_lower_case = False)
model_name = 'allenai/scibert_scivocab_cased'
lower_case = False
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)

# input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016.txt"
input_path = "/mnt/nfs/scratch1/purujitgoyal/NSD_for_sentence_embedding/data/raw/all_sentences.tsv"
# output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_bert_tok.txt"
output_path = "/mnt/nfs/scratch1/purujitgoyal/NSD_for_sentence_embedding/data/raw/all_sentences_bert_tok.txt"

f_out = open(output_path, 'w')

# with open(input_path) as f_in:
#     for line in f_in:
#         tokenized_text = bert_tokenizer.tokenize('[CLS] ' + line + ' [SEP]')
#         f_out.write(' '.join(tokenized_text) + '\n')
data = pd.read_csv(input_path, sep='\t', quoting=csv.QUOTE_NONE, header=None)
sentences = data[3]
for line in sentences:
    tokenized_text = bert_tokenizer.tokenize('[CLS] ' + line + ' [SEP]')
    f_out.write(' '.join(tokenized_text) + '\n')

f_out.close()
