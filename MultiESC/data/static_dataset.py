import pickle

from Datareader import load_json
from collections import defaultdict, Counter
# train_data = load_json('train.txt')
# valid_data = load_json('valid.txt')
# test_data = load_json('test.txt')

def static_feedback(data_list):
    feedback_list = []
    for case_example in data_list:
        dialog = case_example['dialog']
        for sen in dialog:
            if 'feedback' in sen and sen['feedback'] is not None:
                feedback_list.append(sen['feedback'])
    tmp_dict = Counter(feedback_list)
    return {k: tmp_dict.get(k, 0)/len(feedback_list) for k in ['1','2','3','4','5']}

from transformers import BartTokenizer
from Datareader import GenerateDataset2, get_stratege
import numpy as np
MODEL_PATH = '../MODEL/bart-base'


tot_strategy = get_stratege('../new_strategy.json',norm=True)
strategy2id = {v:k for k,v in enumerate(tot_strategy)}
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
train_data_set = GenerateDataset2(4,'./data/train.txt',tokenizer,None,add_cause=True, with_strategy=True)
valid_data_set = GenerateDataset2(4,'./data/valid.txt',tokenizer,None,add_cause=True, with_strategy=True)
test_data_set = GenerateDataset2(4,'./data/test.txt',tokenizer,None,add_cause=True, with_strategy=True)


### 转移到未来一个strategy
def static_transfor_probs(tot_data, history_length):
    result_ans = {}
    for tmp_dic in tot_data:
        history_strategy = tmp_dic['history_strategy']
        next_strategy = tmp_dic['strategy']
        tmp_key = '#'.join(history_strategy[-history_length:])
        if tmp_key not in result_ans:
            result_ans[tmp_key] = np.zeros(len(tot_strategy))
        result_ans[tmp_key][strategy2id[next_strategy]] += 1

    new_result_ans = {}
    for k, v in result_ans.items():
        new_result_ans[k] = v / np.sum(v)
    return new_result_ans


def clac_train_test_overage(train_probs, test_probs):
    cover_time = 0.
    right_time = 0.
    for k, v in test_probs.items():
        if k in train_probs.keys():
            cover_time += 1
            if np.argmax(train_probs[k]) == np.argmax(v):
                right_time += 1
    print("covere_rate: ", cover_time / len(test_probs))
    print("right_rate: ", right_time / len(test_probs))


def clac_test_acc(train_probs, test_tot_data, history_length):
    right_num = 0.
    covered = 0.
    for tmp_dic in test_tot_data:
        history_strategy = tmp_dic['history_strategy']
        next_strategy = tmp_dic['strategy']
        tmp_key = '#'.join(history_strategy[-history_length:])
        if tmp_key in train_probs:
            covered += 1
            xx = sorted(train_probs[tmp_key], reverse=True)
            first_predict = np.argmax(train_probs[tmp_key])
            if xx[0] - xx[1] < 0.25:
                covered -= 1
                continue
            if first_predict == strategy2id[next_strategy]:
                right_num += 1
    print("cover_rate: ", covered / len(test_tot_data))
    print("acc: ", right_num / len(test_tot_data))
    print("covered acc: ", right_num / (covered+1))


def see_one():
    for i in range(1, 6):
        print(i)
        train_probs = static_transfor_probs(train_data_set.total_data + valid_data_set.total_data, i)
        test_probs = static_transfor_probs(test_data_set.total_data, i)
        # clac_train_test_overage(train_probs, test_probs)
        clac_test_acc(train_probs, test_data_set.total_data, i)
        print("############\n\n\n")

def static_transfor_probs_two(tot_data, history_length):
    result_ans = {}
    for tmp_dic in tot_data:
        history_strategy = tmp_dic['history_strategy']
        future_strategy = tmp_dic['future_strategy'].split()
        next_strategy = '#'.join(future_strategy[:2])
        tmp_key = '#'.join(history_strategy[-history_length:])
        if tmp_key not in result_ans:
            result_ans[tmp_key] = defaultdict(int)
        result_ans[tmp_key][next_strategy] += 1

    new_result_ans = {}
    for k, v in result_ans.items():
        t_sum = np.sum(list(v.values()))
        tmp_result = {}
        for t_k,t_v in v.items():
            tmp_result[t_k] = t_v / t_sum
        new_result_ans[k] = tmp_result.copy()
    return new_result_ans

import random
def clac_test_acc_two(train_probs, test_tot_data, history_length):
    right_num = 0.
    covered = 0.
    for tmp_dic in test_tot_data:
        history_strategy = tmp_dic['history_strategy']
        future_strategy = tmp_dic['future_strategy'].split()
        next_strategy = '#'.join(future_strategy[:2])
        gt_strategy = tmp_dic['strategy']
        tmp_key = '#'.join(history_strategy[-history_length:])
        if tmp_key in train_probs:
            covered += 1
            predict_strategy = sorted(train_probs[tmp_key].items(), key=lambda x: x[1], reverse=True)[:4]
            tmp_result = defaultdict(int)
            for k_tuple in predict_strategy:
                tmp_result[k_tuple[0].split('#')[0]] += float(k_tuple[1])
            predict_ans = sorted(tmp_result.items(), key=lambda x:x[1], reverse=True)[0][0]
            if predict_ans == gt_strategy:
                right_num += 1
            if random.random() < 0.005:
                for k_tuple in predict_strategy:
                    print(k_tuple)
                print(predict_ans, '####', gt_strategy)
                print(train_probs[tmp_key].get(next_strategy, -1), np.max(list(train_probs[tmp_key].values())))

    print('covered_rate: ', covered / len(test_tot_data))
    print("acc: ", right_num / len(test_tot_data))
    print("covered acc: ", right_num / covered)
    return {
        "covered_rate": covered / len(test_tot_data),
        "acc": right_num / len(test_tot_data),
        "covered acc": right_num / covered,
    }


history2one = static_transfor_probs(train_data_set.total_data + valid_data_set.total_data, 3)
history2two = static_transfor_probs_two(train_data_set.total_data + valid_data_set.total_data, 3)

with open('./data/history2one.pk', 'wb') as f:
    pickle.dump(history2one, f)

with open('./data/history2two.pk', 'wb') as f:
    pickle.dump(history2two, f)

# def see_two():
#     tot_see = []
#     for i in range(1, 6):
#         print(i)
#         train_probs = static_transfor_probs_two(train_data_set.total_data + valid_data_set.total_data, i)
#         tmp_see = clac_test_acc_two(train_probs, test_data_set.total_data, i)
#         print("############\n\n")
#         tot_see.append(tmp_see)
#     for i in range(1,6):
#         print(i)
#         print(tot_see[i-1])
#
# see_one()