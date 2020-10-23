import os
import glob

result_prefix = 'models/sentence_bert_lower_'
#result_prefix = 'models/sentence_bert_'

#file_list = os.listdir(result_prefix+'*')
file_list = glob.glob(result_prefix+'*')
output_list = []
for file_name in file_list:
    print(file_name)
    #if 'sentence_bert_lower_' in file_name:
    #    continue
    result_file_path = file_name + '/similarity_evaluation_results.csv' 
    with open(result_file_path) as f_in:
        last_line = f_in.readlines()[-1]
        #score = float(last_line.split(',')[2])
        score = (last_line.split(',')[2])
    fields = file_name.split('_')
    dataset = fields[-1].replace('.csv','')
    year = fields[-2]
    category = fields[-3]
    try: 
        trial_num = int(fields[-4])
    except:
        trial_num = 1
    output_list.append(','.join([str(trial_num), dataset, category, year, score]))
output_list = sorted(output_list)
print('\n'.join(output_list))
