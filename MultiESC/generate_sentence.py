import argparse
import copy
import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict
import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM, set_seed)
from strategy_trainer import Seq2SeqTrainer
from transformers.trainer_utils import is_main_process
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
# from modeling_cpt import CPTModel, CPTForConditionalGeneration
from transformers import BartTokenizer, BartModel, BartConfig, GPT2Tokenizer, BlenderbotSmallTokenizer
from MODEL.MultiSource import BART_MODEL
from data.Datareader import GenerateDataset2 as BartDataset, get_stratege,fix_random

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
parser.add_argument("--per_device_train_batch_size", default=16, type=int)
parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--warmup_ratio", default=0.0, type=float)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--generation_max_length", default=64, type=int)  # 这里可以改
parser.add_argument("--seed", default=3407, type=int)
parser.add_argument("--save_total_limit", default=3, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--metric_for_best_model", default="ppl",type=str)
parser.add_argument("--greater_is_better", default=False)
parser.add_argument("--evaluation_strategy", default="epoch",type=str)  # 注意一下这个地方
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--ignore_pad_token_for_loss", default=True)
parser.add_argument("--predict_with_generate", default=True)

parser.add_argument("--data_type", default=4, type=int)
parser.add_argument("--model_type", default=0, type=int)
parser.add_argument("--sen_num", default=64, type=int)
parser.add_argument("--with_cause",action="store_true")
parser.add_argument("--lookahead",action="store_true")
parser.add_argument("--not_pretrain", action="store_true")
parser.add_argument("--config_path", default='../../MODEL/transformer_config', type=str)

parser.add_argument("--with_strategy",action="store_true")
args = parser.parse_args()
fix_random(args.seed)
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
if args.model_type == 3:
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path)
elif args.model_type == 4:
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.sep_token = "[SEP]"
else:
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

tokenizer.add_tokens(strategy_list)
# model = BartForConditionalGeneration(BartConfig.from_pretrained(args.model_name_or_path))



###################
# vaildation and test metrics
###################
import nltk
import metric
def clac_metric(decoder_preds, decoder_labels, no_glove=False):
    ref_list = []
    hyp_list = []
    for ref, hyp in zip(decoder_labels, decoder_preds):
        ref = ' '.join(nltk.word_tokenize(ref.lower()))
        hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)

    from metric import NLGEval
    metric = NLGEval(no_glove=no_glove)
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    return metric_res


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    # if len(preds) == 0:
    labels = [label.strip() for label in labels]
    return preds, labels

# 加上bleu的评测
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print()
    if isinstance(preds, tuple):
        preds = preds[0]
    # print("one: before decoder")
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    x = random.choice(range(len(decoded_labels)))
    print("preds: ", decoded_preds[x])
    print("label: ", decoded_labels[x])
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print("process_preds: ", decoded_preds[x])
    print("process_label: ", decoded_labels[x])
    my_metric = clac_metric(decoder_preds=decoded_preds, decoder_labels=decoded_labels)
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
        # if args.model_type == 4:
        #     model, loading_info = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, output_loading_info=True)
        #     print(type(model))
        # else:
        model, loading_info = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, output_loading_info=True)
        sencond_parameters = loading_info['missing_keys']
        if args.model_type == 4:
            model.config.pad_token_id = tokenizer.unk_token_id

        # for k,v in loading_info.items():
        #     print(k,v)
        # print("model parameter: ",[x for x,y in model.named_parameters()])
        print("we use pretrain")
        # assert False
    my_optim = get_optimer(model, sencond_parameters, training_args)
    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = args.generation_max_length
    max_target_length = args.generation_max_length
    assert isinstance(args.with_strategy, bool), print("with_strategy's type is: ", type(args.with_strategy))
    train_dataset = BartDataset(args.data_type, args.train_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    valid_dataset = BartDataset(args.data_type, args.validation_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=args.lookahead)
    # # test_dataset2 = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
    #                            max_target_len=max_target_length, with_strategy=args.with_strategy,
    #                            sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=False)
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
    predict_metrics = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                       num_beams=4)

    predict_metrics2 = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                        num_beams=1)
    # predict_wo_metric = trainer.evaluate(test_dataset2, metric_key_prefix="predict", max_length=max_target_length,
    #                                     num_beams=1)
    print("beam=4, predict_metrics: ", predict_metrics)
    print("beam=1, predict_metrics: ", predict_metrics2)
    # print("beam=1, wo_look metrics: ", predict_wo_metric)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return predict_metrics, predict_metrics2

if __name__ == '__main__':
    start_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    metric1, metric4 = defaultdict(list), defaultdict(list)
    # args.output_dir = os.path.join(args.output_dir,
    #                                f'modeltype={args.model_type}#data_type={args.data_type}#with_strategy={args.with_strategy}#add_cause={args.with_cause}')

    if not os.path.exists(args.output_dir):
        print("new a _dir: ", args.output_dir)
        os.makedirs(args.output_dir)

    beam4, beam1 = train(args)
    for k in beam1.keys():
        metric1[k].append(beam1[k])
        metric4[k].append(beam4[k])
    for k in metric1.keys():
        print(f"beam1_{k}", metric1[k], "mean: ", np.mean(metric1[k]), "std: ", np.std(metric1[k]))
    for k in metric1.keys():
        print(f"beam4_{k}", metric4[k], "mean: ", np.mean(metric4[k]), "std: ", np.std(metric4[k]))
    end_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(start_time, end_time)

