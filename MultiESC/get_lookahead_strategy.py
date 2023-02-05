import argparse
import logging
import math
import os
import pickle
import random
import json

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support,mean_absolute_error
import sklearn.metrics
from transformers.trainer import Trainer
# from clss_trainer import MyTrainer as Trainer
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser
import copy
# from torch.utils.data.dataset import Dataset
from data.Datareader import get_stratege, read_pk, PredictFeedBackDataset
from MODEL.BertModelForFeedBack import BERTMODEL_LIST
from transformers import BertTokenizer
import warnings
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")
'''
get lookahead.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model', default='../MODEL/bert-base-uncased',
                        help='Pretrain model weight')
parser.add_argument('--output_dir', default='./output/',
                        help='The output directory where the model predictions and checkpoints will be written.')
parser.add_argument('--data_dir', default='./data/',
                        help='Path saved data')
parser.add_argument('--seed', default=42,
                        help='Path saved data')
parser.add_argument('--per_device_train_batch_size', default=16, type=int)
parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
# parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
parser.add_argument('--source_len', default=512, type=int)
parser.add_argument('--num_train_epochs', default=5, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--lr2', default=5e-5, type=float)
parser.add_argument('--evaluation_strategy', default="epoch", type=str)
parser.add_argument('--save_strategy', default="epoch", type=str)
parser.add_argument('--do_train', default=True)
parser.add_argument('--do_eval', default=True)
parser.add_argument('--do_predict', default=True)
parser.add_argument('--load_best_model_at_end', default=True)
parser.add_argument("--metric_for_best_model", default="micro_f1")
parser.add_argument("--model_type", default=6, type=int)
parser.add_argument("--save_total_limit", default=2, type=int)
parser.add_argument("--dataset_type", default=2, type=int)
parser.add_argument("--extend_data", default=1, type=int)
parser.add_argument("--no_origin", default=False, type=bool)
parser.add_argument("--cls", default=False, type=bool)
parser.add_argument("--extend_prefix", default='_beam2', type=str)
parser.add_argument("--no_history", default=False, type=bool) # 413是有history的
parser.add_argument("--output_file", default="aa.txt", type=str)


# parser.add_argument('--load_best_model_at_end', default=True)
args = parser.parse_args()
print(args.extend_data, args.output_dir)
# args.output_dir = f'{args.output_dir}/feedback_model'

strateges = get_stratege('../new_strategy.json', norm=True)
stratege2id = {v: k for k, v in enumerate(strateges)}
train_path = args.data_dir + 'train.txt'
val_path = args.data_dir + 'valid.txt'
test_path = args.data_dir + 'test.txt'
tokenizer = BertTokenizer.from_pretrained(args.pretrain_model, use_fast=False)
tokenizer.add_tokens(list(stratege2id.keys()))

Bertmodel = BERTMODEL_LIST[args.model_type]
BertDataset = PredictFeedBackDataset

model, loading_info = Bertmodel.from_pretrained(args.pretrain_model, num_labels=1, problem_type="regression",
                                                    output_loading_info=True)
sencond_parameters = loading_info['missing_keys']
model.resize_token_embeddings(len(tokenizer))
if args.extend_data == 1:
    print("we extend data", args.extend_data, type(args.extend_data))
    train_set = BertDataset(train_path, tokenizer, args.source_len, extend_path=f'./final_data/train_extend{args.extend_prefix}.pk', no_origin=args.no_origin,clss=args.cls)
else:
    train_set = BertDataset(train_path, tokenizer, args.source_len, clss=args.cls)
# eval_set = BertDataset(val_path, tokenizer, args.source_len, extend_path=f'./final_data/valid_extend.pk',clss=args.cls)
# test_set = BertDataset(test_path, tokenizer, args.source_len, extend_path=f'./final_data/test_extend.pk',clss=args.cls)
eval_set = BertDataset(val_path, tokenizer, args.source_len,clss=args.cls)
test_set = BertDataset(test_path, tokenizer, args.source_len,clss=args.cls)

def tmp_socre(result):
    return {"ab": 1.0}

def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def compute_metrics_with_bart_result_withexcept(predict, generate_labels, language_scores, first_scores, passed_dataset, history2one=None, history2two=None, beita=0.1,erfa=0.1, no_history=False,save_file=None):
    # labels = result.label_ids
    preds = predict[:,0]
    # print("preds", preds.shape)
    tmp_dataset = passed_dataset
    assert len(language_scores) == len(tmp_dataset), print(f"language score: {len(language_scores)}, tmp_dataset: {len(tmp_dataset)}")
    # assert len(tmp_dataset) == len(labels), print(f"dataset_len: {len(tmp_dataset)}, predict_labels: {len(labels)}")
    assert len(tmp_dataset) == len(preds), print("tmp_dataset: ", len(tmp_dataset), "len_preds: ", len(preds))
    assert len(tmp_dataset) % len(generate_labels) == 0, print(f"dataset_len: {len(tmp_dataset)}, generateed_labels_len: {len(generate_labels)}")
    sequence_num = len(tmp_dataset) // len(generate_labels)
    dataset_len = len(generate_labels)
    label_list = []
    preds_list = []
    len_list = []
    tmp_num = 0
    recall_upbound_num = 0
    right_num = 0.
    acc_list = np.zeros(5)
    stage_list = []
    tot_feedback = 0.
    for i in range(dataset_len):
        tmp_generated_list = []
        tmp_preds = []
        tmp_language = []
        first_list = []
        result_dic = defaultdict(float)
        tmp_dic1 = tmp_dataset.total_data[i*sequence_num]
        if 'stage' in tmp_dic1:
            stage = tmp_dic1['stage']
            stage_list.append(stage)
        # for nnn,mmm in tmp_dic1.items():
        #     print(nnn, mmm)
        tot_first_score = tmp_dic1['tot_first_score']
        tmp_history2one = history2one.get('#'.join(tmp_dic1['history_strategy'][-3:]), np.zeros(len(stratege2id)))
        tmp_history2two = history2two.get('#'.join(tmp_dic1['history_strategy'][-3:]), {})
        history2one_list = []
        history2two_list = []

        for j in range(sequence_num):
            tmp_dic = tmp_dataset.total_data[i*sequence_num + j]

            if len(tmp_dic['next_strategy']) < 1:
                continue
            if tmp_dic['next_strategy'][0] not in stratege2id.keys():
                print(i, j, tmp_dic['next_strategy'])
                print(generate_labels[i])
                continue
            tmp_generated_list.append(tmp_dic['next_strategy'][0])
            tmp_preds.append(float(preds[i * sequence_num + j]))
            tmp_language.append(language_scores[i * sequence_num + j])
            # result_dic[tmp_dic['next_strategy'][0]] = first_scores[i * sequence_num + j]
            # result_dic[tmp_dic['next_strategy'][0]] = 0.1
            assert int(100*tot_first_score[stratege2id[tmp_generated_list[-1]]]) == int(100*first_scores[i * sequence_num + j]),print(
                stratege2id[tmp_generated_list[-1]], tot_first_score[stratege2id[tmp_generated_list[-1]]],first_scores[i * sequence_num + j]
            )
            first_list.append(first_scores[i * sequence_num + j])
            history2one_list.append(tmp_history2one[stratege2id[tmp_dic['next_strategy'][0]]])
            history2two_list.append(tmp_history2two.get('#'.join(tmp_dic['next_strategy'][:2]), 0.))

        # 这两个结合起来可以看做是 future scores
        # print("tmp_preds: ", tmp_preds)
        # print("tmp_language: ", tmp_language)

        tmp_preds = np.array(tmp_preds)/np.sum(tmp_preds)
        tmp_language = np.array(tmp_language)/ np.sum(tmp_language)

        one_sum = np.sum(tmp_history2one)

        two_sum = np.sum(history2two_list)
        if one_sum != 0:
            tmp_history2one = np.array(tmp_history2one) / one_sum
        if two_sum != 0:
            history2two_list = np.array(history2two_list) / two_sum

        if no_history:
            # history2one_list = np.zeros_like(history2one_list)
            tmp_history2one = np.zeros_like(tmp_history2one)
            history2two_list = np.zeros_like(history2two_list)

        for strategy in stratege2id.keys():
            result_dic[strategy] = tot_first_score[stratege2id[strategy]] * erfa + tmp_history2one[stratege2id[strategy]]

        h_score = defaultdict(list)
        for x, y, z in zip(tmp_generated_list, tmp_preds, tmp_language+history2two_list):
            h_score[x].append((y, z))
        tot_score = defaultdict(int)
        for x in h_score.keys():
            feedback, probs = zip(*h_score[x])
            # feedback = np.array(feedback) / np.sum(feedback)
            # tot_score[x] = np.sum(np.array(feedback) * np.array(probs))
            probs = np.array(probs) / np.sum(probs)
            tot_score[x] = np.sum(np.array(probs) * np.array(feedback))
        for k in result_dic.keys():
            result_dic[k] += beita * tot_score[k]

        sorted_generted_item = sorted(result_dic.items(), key=lambda x: x[1], reverse=True)
        generted_item = sorted_generted_item[0][0]
        # 计算topk_acc.
        acc_candidate = [x[0] for x in sorted_generted_item]

        for tmp_index in range(5):
            if generate_labels[i] in acc_candidate[:tmp_index+1]:
                acc_list[tmp_index] += 1

        for k, v in enumerate(tmp_generated_list):
            if generted_item == v:
                tot_feedback += preds[i*sequence_num+k]

        if generate_labels[i] in tmp_generated_list:
            recall_upbound_num += 1
        if generate_labels[i] == tmp_generated_list[0]:
            right_num += 1

        preds_list.append(generted_item)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(generate_labels, preds_list, average='macro')
    precision_, recall_, micro_f1, _ = precision_recall_fscore_support(generate_labels, preds_list, average='micro')
    precision_, recall_, weighted_f1, _ = precision_recall_fscore_support(generate_labels, preds_list, average='weighted')

    acc_list = list(acc_list / dataset_len)
    # print("tot_feedback: ",tot_feedback)
    dic = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'feedback': tot_feedback / dataset_len,
    }
    dic.update({f"acc_{i+1}": acc_list[i] for i in range(5)})
    # for k,v in dic.items():
    #     print(k,v)
    if save_file is not None:
        with open(save_file,'wb') as f:
            pickle.dump(preds_list, f)
        print(f"save preds_list in {save_file}")
        print("preds length is : ", len(preds_list))
    # with open('stage.pk','wb') as f:
    #     pickle.dump(stage_list, f)
    return dic

def computing_metrics(predict, test_path1, real_label_path, passed_dataset, beita, erfa, no_history,save_file=None):
    history2one = read_pk('./data/history2one.pk')
    history2two = read_pk('./data/history2two.pk')
    test_extend = read_pk(test_path1)
    real_label = read_pk(real_label_path)
    # print(len(test_extend['two']), len(real_label))
    language_score, first_score, generate_label = [], [], []
    for dic in test_extend['two']:
        language_score.append(dic['language_score'])
        first_score.append(dic['first_score'])
    return compute_metrics_with_bart_result_withexcept(predict, generate_labels=real_label, language_scores=language_score,
                                                       first_scores=first_score, passed_dataset=passed_dataset,
                                                       history2one=history2one, history2two=history2two, beita=beita,
                                                       erfa=erfa, no_history=no_history, save_file=save_file)

def test_one(trainer1):
    pre_fix = './final_data'
    # save_path='./final_data/predicted_strategy'
    save_file = f'{pre_fix}/multiesc_predicted_strategy.pk'
    for i in [6]:
        # print(i)
        test_path1 = f'{pre_fix}/test_extend_beam{i}.pk'
        saved = f'{pre_fix}/beam{i}_feedback.pk'
        with open(saved, 'rb') as ff:
            predict = pickle.load(ff)
        passed_dataset = BertDataset(test_path, tokenizer, args.source_len, extend_path=test_path1, clss=args.cls)
        # aa = get_bert_feedback_predictor(saved, test_path1, f'./data/test_extend_label.pk', passed_dataset, beita=beita, erfa=erfa)
        aa = computing_metrics(predict, test_path1, f'{pre_fix}/test_extend_label.pk', passed_dataset, beita=1.0, erfa=0.4,no_history=args.no_history,save_file=save_file)
        print("beam num: ", i)
        print(aa)



def predict_cases(trainer1, save_path, passed_dataset1):
    predict = trainer1.predict(passed_dataset1)
    with open(save_path, 'wb') as f:
        pickle.dump(predict.predictions, f)

def predict_feedback(trainer1):
    pre_fix = './final_data'
    for i in [2,3,4,5,6]:
        print(i)
        test_path1 = f'{pre_fix}/test_extend_beam{i}.pk'
        passed_dataset1 = BertDataset(test_path, tokenizer, args.source_len, extend_path=test_path1, clss=args.cls)
        predict_cases(trainer1, f'{pre_fix}/beam{i}_feedback.pk', passed_dataset1)

from transformers.optimization import AdamW, Adafactor
def get_optimer(model, second_parameter, train_parser):
    # decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in second_parameter],
            "lr": args.lr2,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in second_parameter],
            "lr": args.learning_rate
        },
    ]
    optimizer_cls = Adafactor if train_parser.adafactor else AdamW
    if train_parser.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (train_parser.adam_beta1, train_parser.adam_beta2),
            "eps": train_parser.adam_epsilon,
        }
    # optimizer_kwargs["lr"] = train_parser.learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

def get_trainer(args, pretrain_path='./output/no_origin_cls/extend=1model=6no_origin=True'):
    args.pretrain_model = pretrain_path
    args.output_dir = './output/test'
    training_args1 = HfArgumentParser(TrainingArguments).parse_dict(vars(args))[0]
    model1 = Bertmodel.from_pretrained(args.pretrain_model)
    trainer1 = Trainer(
        model=model1,
        args=training_args1,
        tokenizer=tokenizer,
        compute_metrics=tmp_socre,
        train_dataset=train_set,
        eval_dataset=eval_set,
    )
    return trainer1

if __name__ == '__main__':
    os.environ["WANDB_DISABLED"] = "true"
    fix_random(args.seed)
    # os.system.wa
    # pre = './output/new_beam2'
    pretrained_model_path="./final_output/feedback_model"
    trainer1 = get_trainer(args, pretrained_model_path)
    predict_feedback(trainer1)
    test_one(trainer1)
