import sent2vec

sent2vec_model_path = '/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/resources/wiki_unigrams.bin'

ouput_path = 'resources/sent2vec_unigram.txt'
model = sent2vec.Sent2vecModel()
model.load_model(sent2vec_model_path)

uni_embs, vocab = model.get_unigram_embeddings()
print('vocab size ', len(vocab))
with open(ouput_path,'w') as f_out:
    for i in range(len(vocab)):
        f_out.write(vocab[i]+' ' + ' '.join(map(str,uni_embs[i]))+'\n')
