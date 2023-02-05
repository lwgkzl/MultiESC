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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default='../MODEL/bart-base',type=str)
# parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr2",default=1e-4,type=float)
parser.add_argument("--do_train",default=True)
parser.add_argument("--do_eval",default=True)
parser.add_argument("--do_predict",default=True)
parser.add_argument("--train_file",default="./data/train.txt",type=str)
parser.add_argument("--validation_file",default="./data/valid.txt",type=str)
parser.add_argument("--test_file",default="./data/test.txt",type=str)
parser.add_argument("--output_dir",default="./output/",type=str)
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

parser.add_argument("--with_strategy",action="store_true")
# save_strategy="epoch",load_best_model_at_end=True
args = parser.parse_args()

arg_dict = args.__dict__
print(arg_dict)
logger = logging.getLogger(__name__)


train_parser = HfArgumentParser(Seq2SeqTrainingArguments)

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

###################
# training
###################
def train(args):
    # assert isinstance(args.use_pretrain,bool),print(type(args.use_pretrain))
    training_args = train_parser.parse_dict(vars(args))[0]
    sencond_parameters = []
    if args.not_pretrain:
        model = BartForConditionalGeneration(BartConfig.from_pretrained(args.model_name_or_path))
        print('we do not use pretrain parameters')
    else:
        model, loading_info = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, output_loading_info=True)
        sencond_parameters = loading_info['missing_keys']

        print("we use pretrain")
        # assert False
    my_optim = get_optimer(model, sencond_parameters, training_args)
    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = args.generation_max_length
    max_target_length = args.generation_max_length - 1
    assert isinstance(args.with_strategy, bool), print("with_strategy's type is: ", type(args.with_strategy))
    train_dataset = BartDataset(args.data_type, args.train_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    valid_dataset = BartDataset(args.data_type, args.validation_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    set_log(training_args)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(my_optim, None),
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    predict_metrics1 = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=args.generation_max_length,
                                        num_beams=1)
    predict_metrics2 = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=args.generation_max_length,
                                        num_beams=2)
    predict_metrics3 = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=args.generation_max_length,
                                        num_beams=3)
    predict_metrics4 = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=args.generation_max_length,
                                       num_beams=4)
    aa = [predict_metrics1, predict_metrics2, predict_metrics3, predict_metrics4]
    for i in range(4):
        tmp_i = i+1
        print(f"beam={tmp_i}, predict_metrics: ", aa[i])
        print('###')

    # print("beam=4, predict_metrics: ", predict_metrics)
    # print("beam=1, predict_metrics2: ", predict_metrics2)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return predict_metrics1, predict_metrics4
if __name__ == '__main__':
    '''
    CUDA_VISIBLE_DEVICES=0,1 python generate_strategy_norm.py --data_type=3 --model_type=1  --output_dir=./output  --learning_rate=2e-5  --num_train_epochs=15 --lr2=2e-5 --with_cause --with_strategy
    '''

    train(args)
