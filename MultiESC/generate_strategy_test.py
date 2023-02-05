import argparse
import copy
import json
import logging
import os
import random
import time
from collections import defaultdict
import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM, set_seed)
from strategy_trainer import Seq2SeqTrainer
from transformers.trainer_utils import is_main_process
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import Counter
from sklearn.metrics import accuracy_score
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
# from modeling_cpt import CPTModel, CPTForConditionalGeneration
from transformers import BartTokenizer, BartModel,BartConfig
from MODEL.MultiSource import BART_MODEL
from data.Datareader import GenerateDataset2 as BartDataset, get_stratege
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default='../MODEL/bart-base',type=str)
# parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr2",default=1e-4,type=float)
# parser.add_argument("--batch_size",default='50',type=str)
# parser.add_argument("--epoch",default='5',type=str)
parser.add_argument("--do_train",default=True)
parser.add_argument("--do_eval",default=True)
parser.add_argument("--do_predict",default=True)
parser.add_argument("--train_file",default="./data/train.txt",type=str)
parser.add_argument("--validation_file",default="./data/valid.txt",type=str)
parser.add_argument("--test_file",default="./data/test.txt",type=str)
parser.add_argument("--output_dir",default="./output/",type=str)
parser.add_argument("--saved_dir",default="./output/",type=str)
parser.add_argument("--per_device_train_batch_size", default=16, type=int)
parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--warmup_ratio", default=0.0, type=float)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--generation_max_length", default=4, type=int) # 这里可以改
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--save_total_limit", default=5, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--metric_for_best_model", default="acc1",type=str)
parser.add_argument("--greater_is_better", default=True)
parser.add_argument("--evaluation_strategy", default="epoch",type=str)  # 注意一下这个地方
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--ignore_pad_token_for_loss", default=True)
parser.add_argument("--predict_with_generate", default=True)

parser.add_argument("--data_type", default=4, type=int)
parser.add_argument("--model_type", default=0, type=int) # 0 norm bart  2 hierarchical bart
parser.add_argument("--sen_num", default=64, type=int)
parser.add_argument("--with_cause",action="store_true")
parser.add_argument("--not_pretrain", action="store_true")
parser.add_argument("--config_path", default='../../MODEL/transformer_config', type=str)

# data_type :{ 0: norm_bert   1: hie_bert  2: norm_strategy 3: hie_strategy  4: norm_sentence  5: hie_sentence, 6.two seqence}

parser.add_argument("--with_strategy",action="store_true")
# save_strategy="epoch",load_best_model_at_end=True
args = parser.parse_args()

arg_dict = args.__dict__
print(arg_dict)
logger = logging.getLogger(__name__)

train_parser = HfArgumentParser(Seq2SeqTrainingArguments)

def read_pk(pre_path):
    with open(pre_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pk(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def set_log(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

print("args.model_name_or_path: ", args.model_name_or_path)


###################
# Dataset and model ready
###################
strategys = get_stratege('../new_strategy.json', norm=True)
strategy_list = [v for k,v in enumerate(strategys)]
BartForConditionalGeneration = BART_MODEL[args.model_type]
tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.add_tokens(strategy_list)
# model = BartForConditionalGeneration(BartConfig.from_pretrained(args.model_name_or_path))



###################
# vaildation and test metrics
###################
import nltk
import metric
def clac_metric2(decoder_preds, decoder_labels,no_glove=False):
    # ref_list = []
    # hyp_list = []
    acc1, acc2, acc3 = 0.,0.,0.
    tot1, tot2, tot3 = 1., 1., 1.
    label, predict = [], []
    for ref, hyp in zip(decoder_labels, decoder_preds):
        ref = ref.split()
        hyp = hyp.split()
        if len(hyp) >= 1:
            if ref[0] == hyp[0]:
                acc1 += 1
            tot1 += 1
            label.append(ref[0])
            predict.append(hyp[0])
        else:
            if random.random()<0.1:
                print("error: we predict nothing")
        if len(hyp) >= 2:
            if ref[0] in hyp[:2]:
                acc2 += 1
            tot2 += 1
        if len(hyp) >= 3:
            if ref[0] in hyp[:3]:
                acc3 += 1
            tot3 += 1
    metric_res = {
        "acc1": acc1 / tot1,
        "acc2": acc2 / tot2,
    }

    sk_acc = accuracy_score(label, predict)
    metric_res["sk_acc"] = sk_acc
    precision_, recall_, macro_f1, _ = precision_recall_fscore_support(label, predict, average='macro')
    precision, recall, micro_f1, _ = precision_recall_fscore_support(label, predict, average='micro')
    precision_, recall_, weighted_f1, _ = precision_recall_fscore_support(label, predict, average='weighted')
    ca_precision_, ca_recall_, ca_f1, _ = precision_recall_fscore_support(label, predict)
    # macro_auc = roc_auc_score(label, predict,average='macro')
    # micro_auc = roc_auc_score(label, predict,average='micro')
    # weighted_auc = roc_auc_score(label, predict,average='weighted')
    metric_res['micro_f1'] = micro_f1
    metric_res['macro_f1'] = macro_f1
    metric_res['weighted_f1'] = weighted_f1
    for i in range(len(ca_f1)):
        metric_res[f'f1_{i}'] = ca_f1[i]
    # metric_res['precision'] = precision
    # metric_res['recall'] = recall
    return metric_res


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

# 加上bleu的评测
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        print("preds_0: ", len(preds[0]))
    # print("one: before decoder")
    # print("decoder_pred: ", preds[0:5])
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    my_metric = clac_metric2(decoder_preds=decoded_preds, decoder_labels=decoded_labels)

    x = random.sample(range(len(decoded_labels)), 5)
    print("first is preds")
    print(decoded_preds[x[0]], "####", decoded_labels[x[0]])
    print(decoded_preds[x[1]], "####", decoded_labels[x[1]])
    return my_metric

###################
# optimazer and lr
###################
from transformers.trainer_pt_utils import get_parameter_names
from torch import nn
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

####################
## predict
####################
def calc_score(pred_list, label_list):
    if len(pred_list) == 0:
        return 1.0
    if ''.join(pred_list) == ''.join(label_list):
        return 5.0
    elif pred_list[0] == label_list[0]:
        return 4.0
    else:
        return 1.0

def get_bart_predict_withscore(train_dataset1, path='./data/extend_data.pk', sequence_num=4, pretrain_model=None, args=None):
    if pretrain_model is not None:
        args.pretrain_model = pretrain_model
    print('predict using model in ', pretrain_model)
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    args.output_dir = './output/test1'

    # from MODEL.MultiSource import BartForConditionalGeneration as BartModel1
    BartModel1 = BART_MODEL[args.model_type]
    model = BartModel1.from_pretrained(args.pretrain_model)
    model.config.num_return_sequences = sequence_num
    training_args = HfArgumentParser(Seq2SeqTrainingArguments).parse_dict(vars(args))[0]

    trainer1 = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset1,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    predict_dic = trainer1.predict(train_dataset1, num_beams=sequence_num, generate_sequence_num=sequence_num, output_score=True)
    predict_tuple = predict_dic.predictions
    need_item = [train_dataset1.tokenizer.convert_tokens_to_ids(strategy) for strategy in strategy_list]

    predict_ids, predict_scores, all_scores = predict_tuple
    predict_scores = np.exp(predict_scores)
    all_scores = np.exp(all_scores)

    test_label = predict_dic.label_ids

    decoded_preds = tokenizer.batch_decode(predict_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(test_label, skip_special_tokens=True)

    dataset_len = len(train_dataset1)
    preds = []
    two_time_data = []
    labels = []
    for i in range(dataset_len):
        tmp_data = train_dataset1.__getitem__(i)
        tmp_str_data = train_dataset1.total_data[i]
        if i < 100:
            xx = tokenizer.decode(tmp_data['labels'], skip_special_tokens=True)
            yy = tokenizer.decode(test_label[i], skip_special_tokens=True)
            assert ''.join(xx) == ''.join(yy), print(xx, "###", yy)
        tmp_list = decoded_preds[i*sequence_num:(i+1)*sequence_num]
        preds.append(tmp_list[0].split()[0])
        assert isinstance(decoded_labels[i][0], str), print(type(decoded_labels[i]), decoded_labels[i])
        labels.append(decoded_labels[i].split()[0])

        for j, tmp in enumerate(tmp_list):
            two_time_data.append({"history": tmp_str_data['history'], "next_strategy": tmp.split(),
                                  "feedback": calc_score(tmp.split(), decoded_labels[i].split()),
                                  "language_score": predict_scores[i*sequence_num + j],
                                  "first_score": all_scores[i*sequence_num + j][predict_ids[i*sequence_num+j][1]],
                                  "tot_first_score": all_scores[i*sequence_num + j][need_item],
                                  "history_strategy": tmp_str_data['history_strategy'],
                                  "stage": tmp_str_data['stage'],
                                  })
    one, two = [], []

    for tmp_dic in two_time_data:
        two.append(tmp_dic['feedback'])
    print(f"two_time: {Counter(two)}")
    print("acc: ", accuracy_score(labels, preds))
    # random_sample = random.sample(two_time_data, 5)
    for tmp_sample in two_time_data[:5]:
        print(tmp_sample['next_strategy'], tmp_sample['language_score'])
    import pickle
    print("two_data: ", two_time_data[0])
    with open(path, 'wb') as f:
        pickle.dump({"two": two_time_data}, f)
    # 获取每个生成的label
    label_path = path.split('.pk')[0]+'_label.pk'
    label_path2 = path.split('_beam')[0]+'_label.pk'
    score_path = path.split('.pk')[0]+'_score.pk'
    with open(label_path, 'wb') as f:
        pickle.dump(labels, f)
    with open(label_path2, 'wb') as f:
        pickle.dump(labels, f)
    with open(score_path,'wb') as f:
        pickle.dump(predict_scores, f)

def make_differ_beam_dataset(pre_path, need_beam, saved_path, index_num=6):
    beam6 = read_pk(pre_path)['two']
    dataset_length = len(beam6) // index_num
    # assert len(beam6) % 6 == 0, print(len(beam6), dataset_length)
    new_dataset = []
    for i, item in enumerate(beam6):
        if i % index_num < need_beam:
            new_dataset.append(item)
    assert len(new_dataset) == need_beam * dataset_length, print(len(new_dataset))
    with open(saved_path, 'wb') as f:
        pickle.dump({"two": new_dataset}, f)

def predict(tmp_index=2, saveed_model=None):
    max_target_length = args.generation_max_length - 1
    train_dataset = BartDataset(args.data_type, args.train_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    valid_dataset = BartDataset(args.data_type, args.validation_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause)
    get_bart_predict_withscore(train_dataset, path=f'./final_data/train_extend_beam{tmp_index}.pk',args=args,sequence_num=tmp_index, pretrain_model=saveed_model)
    get_bart_predict_withscore(valid_dataset, path=f'./final_data/valid_extend_beam{tmp_index}.pk',args=args,sequence_num=tmp_index, pretrain_model=saveed_model)
    get_bart_predict_withscore(test_dataset, path=f'./final_data/test_extend_beam{tmp_index}.pk', args=args, sequence_num=tmp_index, pretrain_model=saveed_model)
    for i in range(1,tmp_index):
        print(i)
        make_differ_beam_dataset('./final_data/train_extend_beam8.pk', i, f'./final_data/train_extend_beam{i}.pk', index_num=tmp_index)
        make_differ_beam_dataset('./final_data/valid_extend_beam8.pk', i, f'./final_data/valid_extend_beam{i}.pk', index_num=tmp_index)
        make_differ_beam_dataset('./final_data/test_extend_beam8.pk', i, f'./final_data/test_extend_beam{i}.pk', index_num=tmp_index)

if __name__ == '__main__':
    #
    '''
    CUDA_VISIBLE_DEVICES=0,1 python generate_strategy_norm.py --data_type=3 --model_type=1  --output_dir=./output  --learning_rate=2e-5  --num_train_epochs=15 --lr2=2e-5 --with_cause --with_strategy
    '''
    #'./output/bart_wh'
    predict(7, args.saved_dir)