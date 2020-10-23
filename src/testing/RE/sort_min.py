input_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/entailment_candidate_from_wordnet_pos"
output_path_min = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/entailment_candidate_from_wordnet_sorted_min_pos"
output_path_pool = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/entailment_candidate_from_wordnet_sorted_pool_pos"

all_data = []
g_d2_other = {}

with open(input_path) as f_in:
    for line in f_in:
        score, s_pattern, g_pattern, s_freq, g_freq, pos = line.rstrip().split('\t')
        all_data.append([pos, s_pattern, g_pattern, min(int(s_freq),int(g_freq)), s_freq, g_freq])
        if g_pattern[4] == '2':
            continue
        if g_pattern not in g_d2_other:
            g_d2_other[g_pattern] = []
        g_d2_other[g_pattern].append([pos, g_freq, s_pattern, s_freq, min(int(s_freq),int(g_freq))])
        
with open(output_path_min, 'w') as f_out:
    for fields in sorted(all_data, key=lambda x: x[3],reverse=True):
        f_out.write('\t'.join(map(str,fields))+'\n')

g_pattern_pool_list =[]

top_k = 5

for g_pattern in g_d2_other:
    pos_list, g_freq_list, s_pattern_list, s_freq_list, min_list =  zip(*g_d2_other[g_pattern])
    min_list_top = sorted(min_list, reverse=True)[:top_k]
    avg_min = sum(min_list_top) / float( len(min_list_top) )
    for count, (pos, g_freq, s_pattern, s_freq, min_score) in enumerate(g_d2_other[g_pattern]):
        g_pattern_pool_list.append([pos, s_pattern, g_pattern, min_score, int(s_freq), g_freq, avg_min])
        if count >= top_k:
            break
        #g_pattern_pool_list.append([s_pattern, g_pattern, min_score, int(s_freq), int(g_freq)])

with open(output_path_pool, 'w') as f_out:
    #for fields in sorted(g_pattern_pool_list, key=lambda x: (x[-1],x[1],x[-2]), reverse = True ):
    for fields in sorted(g_pattern_pool_list, key=lambda x: (x[-1],x[2],x[-3]), reverse = True ):
        f_out.write('\t'.join(map(str,fields))+'\n')
