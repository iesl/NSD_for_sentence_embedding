#!/bin/bash

#srun --pty -p cpu --mem=16G --cpus-per-task=16 bash

INPUT_DIR='/iesl/canvas/hanqingli/NSD_for_sentence_embedding/data/wiki2016_min100/3_3'

output_dir=$INPUT_DIR/all
mkdir $output_dir
ln -s /iesl/canvas/hanqingli/NSD_for_sentence_embedding/data/wiki2016_min100/corpus_index INPUT_DIR/corpus_index
ln -s /iesl/canvas/hanqingli/NSD_for_sentence_embedding/data/wiki2016_min100/dictionary_index INPUT_DIR/dictionary_index

for file_path in $INPUT_DIR/*; do
    file_name=`basename $file_path`
    if [ $file_name = "all" ];then
      echo "skip $file_name"
      continue 
    elif [ $file_name = "corpus_00" ]; then
        cp $file_path/* $output_dir
        echo "backup finished"
        continue
    fi
    for file in $file_path/*; do
      name=`basename $file`
#      echo $output_dir
      cat $file >> $output_dir/$name
      echo "$file finished"
    done
done
wait
