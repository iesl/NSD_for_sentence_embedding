#!/bin/bash
module load python3/current
#srun --partition=gpu --gres=gpu:1 --exclude="gpu-0-0" --cpus-per-task=2 --mem=20G  
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --update_target_emb
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --optimizer Adam --lr 0.00001
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --nlayers 2
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --nlayers 2 --data ./data/processed/bookp1/ --save ./models/book
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --nlayers 1 --emb_file ./resources/Google-vec-neg300_filtered_wac_bookp1.txt
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --nlayers 2
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --nlayers 2 --L1_losss_B 0.4
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 2
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 2 --data ./data/processed/bookp1/ --save ./models/book
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 1 --data ./data/processed/bookp1/ --save ./models/book --update_target_emb
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 400 --optimizer Adam --lr 0.00001 --nlayers 2
~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --save ./models/wiki2016 --data ./data/processed/wiki2016 --emb_file ./resources/glove.840B.300d_filtered_wiki2016.txt
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --save ./models/wiki2016 --data ./data/processed/wiki2016 --emb_file ./resources/glove.840B.300d_filtered_wiki2016.txt --update_target_emb
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --save ./models/wiki2016 --data ./data/processed/wiki2016 --emb_file ./resources/Google-vec-neg300_filtered_wiki2016.txt
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --save ./models/wiki2016 --data ./data/processed/wiki2016 --emb_file ./resources/glove.840B.300d_filtered_wiki2016.txt --linear_mapping_dim=-1
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 2 --dropout 0.6 --dropouti 0.6
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 2 --update_target_emb
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 1 --emb_file ./resources/Google-vec-neg300_filtered_wac_bookp1.txt
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --optimizer Adam --lr 0.00001
#~/anaconda3/bin/python src/main.py --coeff_opt maxlc --batch_size 200 --optimizer Adam --lr 0.00001 --nlayers 2
#~/anaconda3/bin/python src/main.py --coeff_opt lc --batch_size 200 --update_target_emb
#~/anaconda3/bin/python src/main.py --coeff_opt max --update_target_emb --batch_size 200
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200  --lr2_divide 20 --clip 0.1
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --clip 0.1
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --lr2_divide 20
#~/anaconda3/bin/python src/main.py --coeff_opt lc --update_target_emb --batch_size 200 --lr 0.1
