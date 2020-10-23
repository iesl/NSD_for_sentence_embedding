import random

input_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/RE_entailment_labels_all.tsv"
out_val_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/RE_entailment_labels_val.tsv"
out_test_path = "/iesl/canvas/hschang/TAC_2016/data/training_source_data/RE_entailment_labels_test.tsv"

num_trial = 50
seed_count = 1

val_ratio = 0.2

def get_word_diff(gen, spec):
    extra_word = []
    for w in gen.split()[1:-1]:
        if w not in spec:
            extra_word.append(w)

    return tuple(extra_word)

#entailment_count = 0
is_entail_list = []
extra_d2_idx = {}
idx_l2_extra = []
output_line = []
with open(input_path) as f_in:
    for line in f_in:
        line = line.rstrip()
        output_line.append(line)
        label, pos_tag, spec, gen, score, spec_freq, gen_freq, avg_score = line.split('\t')
        if label == '>' or label == '<':
            #entailment_count += 1
            is_entail_list.append(1)
        else:
            is_entail_list.append(0)

        gen_words = get_word_diff(gen, spec)
        #spec_words = get_word_diff(spec, gen)
        #idx_l2_extra.append([gen_words,spec_words])
        idx_l2_extra.append([gen_words])
        
        idx = len(output_line) - 1
        idx_list = extra_d2_idx.get(gen_words, [])
        idx_list.append(idx)
        extra_d2_idx[gen_words] = idx_list
        #idx_list = extra_d2_idx.get(spec_words, [])
        #idx_list.append(idx)
        #extra_d2_idx[spec_words] = idx_list

def finding_closure(selected_idx_list, idx_l2_extra, extra_d2_idx):
    last_select_num = -1
    while len(selected_idx_list) != last_select_num:
        print(selected_idx_list)
        last_select_num = len(selected_idx_list)
        extra_list = [idx_l2_extra[idx][0] for idx in selected_idx_list]
        #extra_list += [idx_l2_extra[idx][1] for idx in selected_idx_list]
        selected_idx_list = set([idx for w in extra_list for idx in extra_d2_idx[w]])
    return list(selected_idx_list)

entailment_count = sum(is_entail_list)
print(entailment_count)
num_samples = len(output_line)
print(num_samples)
goal_val_ent_count = int(entailment_count * val_ratio)

min_diff = 100000
for i in range(num_trial):
    print("sampling ", seed_count)
    selected_idx_list = random.sample(range(num_samples), seed_count)
    selected_idx_list = finding_closure(selected_idx_list, idx_l2_extra, extra_d2_idx)
    selected_ent_num = len([idx for idx in selected_idx_list if is_entail_list[idx]])
    print(selected_ent_num / float(goal_val_ent_count))
    if selected_ent_num < goal_val_ent_count:
        seed_count += 1
    if selected_ent_num > goal_val_ent_count and seed_count > 1:
        seed_count -= 1
    diff = abs(selected_ent_num - goal_val_ent_count)
    if diff < min_diff:       
        best_round = i
        min_diff = diff
        best_selected_idx_list = selected_idx_list
print("best diff {} and round {}".format(min_diff, best_round))
with open(out_val_path, 'w') as f_out_val, open(out_test_path, 'w') as f_out_test:
    for j in range(num_samples):
        if j in best_selected_idx_list:
            f_out_val.write(output_line[j]+'\n')
        else:
            f_out_test.write(output_line[j]+'\n')
