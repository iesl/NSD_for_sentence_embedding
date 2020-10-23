input_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/train_sorted.ds"
output_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/pattern_freq"

pattern_d2_ep_set = {}

with open(input_path) as f_in:
    for line in f_in:
        e1, e2, pattern, dummy = line.rstrip().split('\t')
        ep_set = pattern_d2_ep_set.get(pattern,set())
        ep_set.add( (e1,e2) )
        pattern_d2_ep_set[pattern] = ep_set

pattern_d2_ep_count = {}
for pattern in pattern_d2_ep_set:
    pattern_d2_ep_count[pattern] = len(pattern_d2_ep_set[pattern])

pattern_count_sorted = sorted(pattern_d2_ep_count.items(), key = lambda x: x[1], reverse=True)
with open(output_path, 'w') as f_out:
    for pattern, count in pattern_count_sorted:
        f_out.write(pattern+'\t'+str(count)+'\n')
