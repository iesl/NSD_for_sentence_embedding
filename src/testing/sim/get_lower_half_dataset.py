import numpy as np

#input_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-dev.csv'
#output_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-dev_lower.csv'
input_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-test.csv'
output_path = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-test_lower.csv'

def get_lower_half(score_list):
    sorted_ind = np.argsort(score_list)
    #print(score_list[sorted_ind[0]])
    lower_idx_list = list(sorted_ind[:int(len(sorted_ind)/2)])
    #print( len(lower_idx_set) )
    return lower_idx_list

output_list = []
score_list = []

with open(input_path) as f_in:
    for line in f_in:
        line = line.rstrip()
        fields = line.split('\t')
        output_list.append(line)
        score_list.append(float(fields[4]))

lower_idx_list = get_lower_half(score_list)
with open(output_path,'w') as f_out:
    for idx in lower_idx_list:
        f_out.write(output_list[idx]+'\n')
