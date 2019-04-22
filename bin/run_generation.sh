#!/bin/bash
module load python3/current
#srun --partition=gpu --gres=gpu:1 --exclude="gpu-0-0" --cpus-per-task=2 --mem=20G  
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/val_glove_lc_elayer2_bsz200_ep5_linear --nlayers 2
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/Wacky-20190329-154753 --outf ./gen_log/val_updated_glove_lc_elayer2_bsz200_ep4_linear --nlayers 2
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/book-20190331-112352 --outf ./gen_log/val_glove_book_maxlc_elayer2_bsz200_ep2_linear --nlayers 2 --data ./data/processed/bookp1/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190414-143112 --outf ./gen_log/wiki2016_val_rnd_bsz200_ep1_1 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190415-145905 --outf ./gen_log/wiki2016_val_glove_bsz200_ep1 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190415-152214 --outf ./gen_log/wiki2016_val_glove_lc_bsz200_ep2_0 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190416-020305 --outf ./gen_log/wiki2016_val_updated_glove_bsz200_ep3_0 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190416-020848 --outf ./gen_log/wiki2016_val_word2vec_bsz200_ep2_0 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190416-020338 --outf ./gen_log/wiki2016_val_updated_word2vec_bsz200_ep2_0 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/basis_test.py --checkpoint ./models/wiki2016-20190414-132830 --outf ./gen_log/wiki2016_val_glove_bad_init_bsz200_ep1 --data ./data/processed/wiki2016_min100/
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org --checkpoint ./models/ --outf ./gen_log/SWCS_wiki2016_glove_lc_bsz200_ep2_0.json --max_sent_len 150
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org --checkpoint ./models/wiki2016-20190415-152214 --outf ./gen_log/STS_dev_wiki2016_glove_lc_bsz200_ep2_0.json --outf_vis gen_log/generated_STS_wiki2016_glove_lc.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org --checkpoint ./models/wiki2016-20190416-020305 --outf ./gen_log/STS_dev_wiki2016_updated_glove_maxlc_bsz200_ep3_0.json --outf_vis gen_log/generated_STS_wiki2016_updated_glove_maxlc.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org --checkpoint ./models/wiki2016-20190416-020848 --outf ./gen_log/STS_dev_wiki2016_word2vec_maxlc_bsz200_ep2_0.json --outf_vis gen_log/generated_STS_wiki2016_word2vec_maxlc.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org --checkpoint ./models/wiki2016-20190415-145905 --outf ./gen_log/STS_dev_wiki2016_glove_maxlc_bsz200_ep2_0.json --outf_vis gen_log/generated_STS_wiki2016_glove_maxlc.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org --checkpoint ./models/wiki2016-20190416-114532 --outf ./gen_log/STS_dev_wiki2016_glove_elayer2_maxlc_bsz200_ep2_0.json --outf_vis gen_log/generated_STS_wiki2016_glove_elayer2_maxlc.txt --nlayers 2 #need model_old_2
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/WiC_dataset/dev/dev.data_org --checkpoint ./models/wiki2016-20190415-152214 --outf ./gen_log/WiC_dev_wiki2016_glove_lc_elayer1_bsz200_ep2_0_lower.json --nlayers 1 --outf_vis gen_log/generated_WiC_wiki2016_glove_lower.txt
~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/WiC_dataset/dev/dev.data_org --checkpoint ./models/wiki2016-20190416-020848 --outf ./gen_log/WiC_dev_wiki2016_word2vec_lc_elayer1_bsz200_ep2_0_lower.json --nlayers 1 --outf_vis gen_log/generated_WiC_wiki2016_word2vec_lower.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org --checkpoint ./models/wiki2016-20190416-041600 --outf ./gen_log/SWCS_wiki2016_glove_maxlc_elayer1_bsz200_ep2_0_multi.json --nlayers 1 --max_sent_len 150 --outf_vis gen_log/generated_SCWS_wiki2016_glove_multi.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org --checkpoint ./models/wiki2016-20190416-100150 --outf ./gen_log/SWCS_wiki2016_word2vec_maxlc_elayer1_bsz200_ep2_0_multi.json --nlayers 1 --max_sent_len 150 --outf_vis gen_log/generated_SCWS_wiki2016_word2vec_multi.txt

#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org_lemma --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/SWCS_glove_lc_elayer2_bsz200_ep5_linear.json --nlayers 2 --max_sent_len 200
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org_lemma --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_linear_cosine.json --linear_mapping_dim -1 --nlayers 2 --outf_vis gen_log/generated_glove_linear.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org_lemma --checkpoint ./models/Wacky-20190329-043933 --outf ./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_posi_cosine.json --nlayers 2 --outf_vis gen_log/generated_glove_posi.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/WiC_dataset/dev/dev.data_org_lemma --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/WiC_dev_glove_lc_elayer2_bsz200_ep7_linear_cosine.json --linear_mapping_dim -1 --nlayers 2 --outf_vis gen_log/generated_WiC_glove_linear.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org_lemma --checkpoint ./models/Wacky-20190329-154753 --outf ./gen_log/STS_dev_updated_glove_lc_elayer2_bsz200_ep6_linear_cosine.json --nlayers 2 --linear_mapping_dim -1 --outf_vis gen_log/generated_updated_glove_cosine.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org_lemma --checkpoint ./models/Wacky-20190329-154753 --outf ./gen_log/STS_dev_updated_glove_lc_elayer2_bsz200_ep5_linear.json --nlayers 2 --outf_vis gen_log/generated_updated_glove.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/WiC_dataset/dev/dev.data_org_lemma --checkpoint ./models/Wacky-20190329-100602 --outf ./gen_log/WiC_dev_glove_lc_elayer2_bsz200_ep7_linear_cosine.json --nlayers 2 --linear_mapping_dim -1 --outf_vis gen_log/generated_WiC_glove_linear.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/WiC_dataset/dev/dev.data_org_lemma --checkpoint ./models/Wacky-20190329-043933 --outf ./gen_log/WiC_dev_glove_lc_elayer2_bsz200_ep7_posi_cosine.json --nlayers 2 --outf_vis gen_log/generated_WiC_glove_posi.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org_lemma --checkpoint ./models/Wacky-20190401-003001 --outf ./gen_log/SWCS_word2vec_lc_elayer1_bsz200_ep3.json --nlayers 1 --max_sent_len 200
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org_lemma --checkpoint ./models/Wacky-20190401-003001 --outf ./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep5_cosine.json --nlayers 1 --outf_vis gen_log/generated_STS_word2vec.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/WiC_dataset/dev/dev.data_org_lemma --checkpoint ./models/Wacky-20190401-003001 --outf ./gen_log/WiC_dev_word2vec_lc_elayer1_bsz200_ep5.json --nlayers 1 --outf_vis gen_log/generated_WiC_word2vec.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/STS/stsbenchmark/sts-dev_org_lemma --checkpoint ./models/Wacky-20190401-104403 --outf ./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep7_linear_cosine.json --nlayers 1 --linear_mapping_dim -1 --outf_vis gen_log/generated_STS_word2vec_linear.txt 
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org_lemma --checkpoint ./models/Wacky-20190405-013228 --outf ./gen_log/SWCS_glove_lc_elayer1_bsz200_ep.json --nlayers 1 --max_sent_len 150
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org_lemma --checkpoint ./models/Wacky-20190405-143756 --outf ./gen_log/SWCS_glove_lc_elayer1_bsz200_ep6_sgd_multi.json --nlayers 1 --max_sent_len 150 --linear_mapping_dim -1 --outf_vis gen_log/generated_SCWS_glove_multi_2.txt
#~/anaconda3/bin/python src/testing/sim/basis_test_from_sent.py --input ./dataset_testing/SCWS/scws_org_lemma --checkpoint ./models/Wacky-20190405-014701 --outf ./gen_log/SWCS_word2vec_lc_elayer1_bsz200_posi_ep6_multi.json --nlayers 1 --max_sent_len 150 --linear_mapping_dim -1 --outf_vis gen_log/generated_SCWS_word2vec_multi.txt
