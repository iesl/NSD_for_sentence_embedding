label_file = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/RE_entailment_labels.tsv"
all_file = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/entailment_candidate_from_wordnet_sorted_pool_pos"
output_file = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/RE_entailment_labels_all.tsv"

patterns_d2_label_pos = {}

with open(label_file) as f_in:
    for line in f_in:
        fields = line.rstrip().split('\t')
        label = fields[0]
        pos_tag = fields[1]
        s_pattern = fields[2]
        g_pattern = fields[3]
        patterns_d2_label_pos[(s_pattern, g_pattern)] = [label, pos_tag]

#end_patterns = ["$ARG1 leading , $ARG2", "$ARG1 deal , $ARG2"]
end_patterns = ["$ARG1 carried $ARG2", "$ARG1 broadcast $ARG2"]

with open(all_file) as f_in, open(output_file,'w') as f_out:
    for line in f_in:
        fields = line.rstrip().split('\t')
        s_pattern = fields[1]
        g_pattern = fields[2]
        if s_pattern == end_patterns[0] and g_pattern == end_patterns[1]:
            break
        if (s_pattern, g_pattern) in patterns_d2_label_pos:
            label, pos_tag = patterns_d2_label_pos[(s_pattern, g_pattern)]
        else:
            label = 'N'
            pos_tag = fields[0]
        f_out.write('\t'.join([label, pos_tag]+fields[1:])+'\n'  )
            
