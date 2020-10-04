#!/bin/bash

#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00
#SBATCH --mem=60000

# ~/anaconda3/bin/python ./src/testing/summarization/cnn_dm_pyrouge.py --input ./data/src/summ_inputs/abstract/ --checkpoint ./models/czi_neuro100-20200308-163129 --method_set ours+embs --dict ./data/dictionary_index --n_basis 10 --emsize 200

# ~/anaconda3/bin/python ./src/testing/summarization/cnn_dm_pyrouge.py --input ./data/src/summ_inputs/acl_ref_text/ --checkpoint ./models/acl_10-20200407-201039 --method_set ours+embs --dict ./data/processed/acl_lower_min100/dictionary_index --n_basis 10 --emsize 300 --outf_vis eval_log/summ_vis/map_ours_embs_n10.json
~/anaconda3/bin/python ./src/testing/summarization/cnn_dm_pyrouge.py --input ./data/src/summ_inputs/acl_full_text/ --checkpoint ./models/acl_10-20200407-201039 --method_set ours+embs --dict ./data/processed/acl_lower_min100/dictionary_index --n_basis 10 --emsize 300 --outf_vis eval_log/summ_vis/map_full_ours_embs_n10.json
# ~/anaconda3/bin/python ./src/testing/summarization/cnn_dm_pyrouge.py --input ./data/src/summ_inputs/acl_ref_text/ --checkpoint ./models/acl_100-20200407-205336 --method_set ours+embs --dict ./data/processed/acl_lower_min100/dictionary_index --n_basis 100 --emsize 300 --outf_vis eval_log/summ_vis/map_ours_embs_n100.json
# ~/anaconda3/bin/python ./src/testing/summarization/cnn_dm_pyrouge.py --input ./data/src/summ_inputs/acl_full_text/ --checkpoint ./models/acl_100-20200407-205336 --method_set ours+embs --dict ./data/processed/acl_lower_min100/dictionary_index --n_basis 100 --emsize 300 --outf_vis eval_log/summ_vis/map_full_ours_embs_n100.json
# ~/anaconda3/bin/python ./src/testing/summarization/cnn_dm_pyrouge.py --input ./data/src/summ_inputs/acl_ref_text/ --checkpoint ./models/acl_10-20200407-201039 --method_set bert --dict ./data/processed/acl_scibert_min100/dictionary_index --n_basis 10 --emsize 300 --outf_vis eval_log/summ_vis/map_bert_acl.json
