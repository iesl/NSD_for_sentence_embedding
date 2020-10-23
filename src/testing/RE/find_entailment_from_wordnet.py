from nltk.corpus import wordnet as wn
import sys

input_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/pattern_freq"
output_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/entailment_candidate_from_wordnet_pos"

pattern_d2_count = {}

with open(input_path) as f_in:
    for line in f_in:
        pattern, count = line.rstrip().split('\t')
        pattern_d2_count[pattern] = int(count)

#pattern_pair_info = []
pattern_pair_d2_info = {}
for idx, pattern in enumerate(pattern_d2_count):
    if idx % 1000 == 0:
        print(idx/float(len(pattern_d2_count)))
        sys.stdout.flush()
        #if idx > 10000:
        #    break
    w_seq = pattern.split()
    for i in range(1,len(w_seq)-1):
        w = w_seq[i]
        for sense in wn.synsets(w):
            for path in sense.hypernym_paths():
                for hypernym in path:
                    for phrase in [str(lemma.name()) for lemma in hypernym.lemmas()]:
                        if phrase == w:
                            continue
                        parent_pattern = ' '.join(w_seq[:i]) + ' ' + phrase.replace('_', ' ') + ' ' + ' '.join(w_seq[i+1:])
                        #print(parent_pattern)
                        if parent_pattern in pattern_d2_count and (pattern, parent_pattern) not in pattern_pair_d2_info:
                            info = [pattern_d2_count[pattern] * pattern_d2_count[parent_pattern], pattern, parent_pattern, pattern_d2_count[pattern], pattern_d2_count[parent_pattern], hypernym.pos()]
                            #print(info)
                            pattern_pair_d2_info[(pattern, parent_pattern)] = info
    #for w in w_seq[1:-1]:
pattern_pair_info_sorted = sorted(pattern_pair_d2_info.values(), key = lambda x: x[0], reverse=True)
with open(output_path,'w') as f_out:
    for info in pattern_pair_info_sorted:
        pattern, parent_pattern = info[1:3]
        if (parent_pattern, pattern) in pattern_pair_d2_info:
            continue
        f_out.write('\t'.join(map(str,info))+'\n')
