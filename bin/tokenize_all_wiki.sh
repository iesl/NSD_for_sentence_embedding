#!/bin/bash

#SBATCH --ntasks=8
#sbatch --ntasks=9 --ntasks-per-node=1 --nodes=9 --cpus-per-task=16 --mem=100G ./bin/tokenize_all_wiki.sh

#sacct -j 89262 --format=JobID,Start,End,Elapsed,NCPUS,nodelist,JobName
INPUT_DIR='/iesl/canvas/hanqingli/NSD_for_sentence_embedding/data/wiki2016_min100/raw'
OUTPUT_DIR='/iesl/canvas/hanqingli/NSD_for_sentence_embedding/data/wiki2016_min100/5_5/'
#
#TOKENIZE_MODE=2
target_num_next=5
target_num_prev=5
for file_path in $INPUT_DIR/*; do
    file_name=`basename $file_path`
    output_path="${file_name}"
    echo --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda3/bin/python src/preprocessing/match_features_targets.py --data $file_path --save $OUTPUT_DI$output_path --target_num_prev $target_num_prev --target_num_next $target_num_next
    srun --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda3/bin/python src/preprocessing/match_features_targets.py --data $file_path --save $OUTPUT_DIR$output_path --target_num_prev $target_num_prev --target_num_next $target_num_next &
done
wait
