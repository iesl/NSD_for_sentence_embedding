# NSD_for_sentence_embedding


1. How to preprocess the data (only require CPUs):

1.1 Finding the sentence boundary and tokenize the sentence by Spacy. 
Here is an example of doing this in parallel using slurm (assuming you have chunking the corpus into several pieces).
https://github.com/iesl/NSD_for_sentence_embedding/blob/master/bin/tokenize_all_wiki.sh
Your corpus format will be different from the wikipedia format, so you need to modify the code in src/preprocessing/tools/tokenize_wiki.py

1.2 Train a word embedding model on the corpus
If you don't know how to do it, you can refer to https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
Here is an example: src/preprocessing/train_word2vec.py
Save the w2v using save_word2vec_format (https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.save) function (using plan text format and the results should look like /mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/resources/glove.840B.300d_filtered_wiki2016.txt)

1.3 convert the corpus into tensors
change the path in
https://github.com/iesl/NSD_for_sentence_embedding/blob/master/bin/preprocessing.sh
and run it.


2. How to train a model:

2.1 uncomment
https://github.com/iesl/NSD_for_sentence_embedding/blob/master/bin/run_gpu_code.sh#L62
, change the path and run it with GPU.


3. How to test the model

To be continued

Some codes related to variational dropout and training are from https://github.com/salesforce/awd-lstm-lm and https://github.com/zihangdai/mos
The code of transformer comes from https://github.com/huggingface/pytorch-pretrained-BERT

src/testing/sim/tree_tagger comes from http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
src/testing/sim/treetagger-python comes from https://github.com/miotto/treetagger-python
src/testing/sim/SIF_pc_removal.py comes from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
src/testing/sim/Sinkhorn.py is modified from https://github.com/iesl/string-edit-distance/blob/dptam/src/python/entity_align/utils/Sinkhorn.py and https://github.com/gpeyre/SinkhornAutoDiff

The CNN/DailyMail is preprocessed using the codes which are modified from https://github.com/abisee/cnn-dailymail

Before releasing the code, remember to delete some testing scoring functions which are not used
remember to delete some redundant code such as still normalize target embedding for each batch.

L1_losss_B is lambda/2
