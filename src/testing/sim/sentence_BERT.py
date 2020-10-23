from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers.readers import STSBenchmarkDataReader
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import models, losses
import math

cat_year_arr = [['question-question', 2016], ['belief', 2015], ['headlines', 2014], ['headlines', 2015], ['surprise.SMTnews', 2012], ['surprise.OnWN', 2012], ['OnWN', 2013], ['OnWN', 2014], ['FNWN', 2013], ['answers-students', 2015], ['plagiarism', 2016], ['tweet-news', 2014], ['SMTeuroparl', 2012], ['postediting', 2016]]


STS_all_folder = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS"
STSb_folder = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark"
#model_save_prefix = "./models/sentence_bert_"
model_save_prefix = "./models/sentence_bert_lower_3_"
use_lower_dataset = True
num_epochs = 10
train_batch_size = 25


def eval_sent_bert(train_category_year, test_file_name, output_dir_name):
    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.Transformer('bert-base-cased')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)


    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    #sts_reader = STSBenchmarkDataReader(STS_folder, normalize_scores=True)
    #train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
    sts_reader = STSBenchmarkDataReader(STS_all_folder, normalize_scores=True)
    #train_data = SentencesDataset(sts_reader.get_examples('sts_all_years_test',category_year=['question-question', 2016]), model)
    #train_data = SentencesDataset(sts_reader.get_examples('sts_all_years_test',category_year=['surprise.OnWN', 2012]), model)
    #train_data = SentencesDataset(sts_reader.get_examples('sts_all_years_test',category_year=['headlines', 2014]), model)
    train_data = SentencesDataset(sts_reader.get_examples('sts_all_years_test',category_year=train_category_year, rand_set_num=100), model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up

    sts_reader = STSBenchmarkDataReader(STSb_folder)
    #dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
    #dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-test.csv'), model=model)
    dev_data = SentencesDataset(examples=sts_reader.get_examples(test_file_name), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
             evaluator=evaluator,
             epochs=num_epochs,
             evaluation_steps=1000,
             warmup_steps=warmup_steps,
             output_path=output_dir_name
             )

for cat_year in cat_year_arr:
    model_save_path = model_save_prefix + cat_year[0] +'_' + str(cat_year[1]) + '_dev.csv'
    if use_lower_dataset:
        eval_sent_bert(cat_year, 'sts-dev_lower.csv', model_save_path)
    else:
        eval_sent_bert(cat_year, 'sts-dev.csv', model_save_path)
#eval_sent_bert(['headlines', 2014], 'sts-test.csv', model_save_path)
#eval_sent_bert(['plagiarism', 2016], 'sts-test.csv', model_save_prefix)
#eval_sent_bert(['question-question', 2016], 'sts-test.csv', model_save_path)


for cat_year in cat_year_arr:
    #model_save_path = model_save_prefix + '_'.join(cat_year) + '_test.csv'
    model_save_path = model_save_prefix + cat_year[0] +'_' + str(cat_year[1]) + '_test.csv'
    if use_lower_dataset:
        eval_sent_bert(cat_year, 'sts-test_lower.csv', model_save_path)
    else:
        eval_sent_bert(cat_year, 'sts-test.csv', model_save_path)
