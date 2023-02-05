import copy
import json
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer
from collections import defaultdict
from sklearn.metrics import accuracy_score
def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data


def read_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_pk(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def norm_strategy(strategy):
    norm_str = "-".join(strategy.split())
    return "@["+norm_str+"]"


def get_stratege(file_path, norm=False):
    with open(file_path,'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d.replace('[','').replace(']','') for d in data]
    if norm:
        data = [norm_strategy(d) for d in data]
    print('strategy: ', data)

    return data


def _norm(x):
    return ' '.join(x.strip().split()).lower()

class EmotionalIndex():
    def __init__(self, tokener:BartTokenizer, divid_num=8):
        self.vad_dict = self.get_vad_dict()
        self.special_token = list(tokener.special_tokens_map.values()) # 64
        self.symbol = ['?',',','.',':','!','\'','"'] # 65
        self.divid_num = divid_num


    def get_vad_dict(self, path='./data/NRC_VAD.txt'):
        vad_dict = {}
        num_list = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                sp_line = line.split()
                if len(sp_line) > 4:
                    continue
                vad_dict[sp_line[0].strip().lower()] = sp_line[1:]
                num_list.append(len(sp_line))
        # print("vad_dict_len: ", len(vad_dict))
        return vad_dict

    def get_value2index(self, one_item):
        for i in range(1, self.divid_num+1):
            if i * 0.125 >= float(one_item):
                return i-1

    def get_vector2index(self, one_list):
        return self.get_value2index(one_list[0]) * self.divid_num + self.get_value2index(one_list[1])

    def get_one_sentence(self, sentence):
        emotion_list = []
        for s in sentence:
            s = s.replace('Ġ', '')
            if s in self.special_token:
                emotion_list.append(self.divid_num * self.divid_num)
            elif s in self.symbol:
                emotion_list.append(self.divid_num * self.divid_num + 1)
            else:
                # ss = s.replace('Ġ', '')
                if s in self.vad_dict:
                    emotion_list.append(self.get_vector2index(self.vad_dict[s]))
                else:
                    emotion_list.append(self.divid_num * self.divid_num +2)
        assert  len(emotion_list) == len(sentence)
        return emotion_list

######################
#  predict feedback
######################

class PredictFeedBackDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_source_len, each_sentence_length=32, sentence_num=64, extend_path=None, no_origin=True, clss=False):
        super(PredictFeedBackDataset, self).__init__()
        self.max_source_len = max_source_len
        self.tokenizer = tokenizer
        self.each_length = each_sentence_length
        self.sentence_num = sentence_num
        self.strategy_num = 2
        self.cls = clss
        self.emotion_index = EmotionalIndex(tokenizer)
        data = load_json(file_path)
        self.total_data = []
        for case_example in data:
            dialog = case_example['dialog']
            situation = case_example['situation']
            emotion_type = case_example['emotion_type']
            problem_type = case_example['problem_type']
            history = [_norm(emotion_type) + self.tokenizer.sep_token + _norm(
                problem_type) + self.tokenizer.sep_token + _norm(situation)]
            tmp_strategy_list = []
            for index, tmp_dic in enumerate(dialog):
                text = _norm(tmp_dic['text'])
                dialog_len = len(text)
                if index == 0 and tmp_dic['speaker'] != 'sys':
                    history[0] = text + self.tokenizer.sep_token + history[0]
                    continue
                if tmp_dic['speaker'] != 'sys' and tmp_dic['feedback'] is not None:
                    tmp_len = min(len(tmp_strategy_list), self.strategy_num)
                    self.total_data.append({
                        "history": history[:1-2*tmp_len].copy(),
                        "next_strategy": tmp_strategy_list[-tmp_len:],
                        'feedback': int(tmp_dic['feedback'])-1,
                        "stage": 5*index//dialog_len,
                    })
                if tmp_dic['speaker'] == 'sys':
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    tmp_strategy_list.append(tmp_stratege)
                    history.append(tmp_stratege + self.tokenizer.sep_token + text)
                else:
                    cause = tmp_dic['cause']
                    if cause is not None:
                        history.append(cause + self.tokenizer.sep_token + text)
                    else:
                        history.append(text)

        if extend_path is not None:
            with open(extend_path,'rb') as f:
                tmp_data = pickle.load(f)['two']
            choosed_index = defaultdict(list)
            for k, one_item in enumerate(tmp_data):
                choosed_index[int(one_item['feedback'])].append(k)
            four_len = len(choosed_index[4]+choosed_index[5])
            # print(four_len, len(choosed_index[1]))
            one_index = choosed_index[1]
            after_data = [v for k, v in enumerate(tmp_data) if k in one_index+choosed_index[4]+choosed_index[5]]
            if 'train' not in file_path:
                print('length: ', len(self.total_data))
                self.total_data = tmp_data
                # # print(tmp_data[0])
                #
                # for i,tmp_dd in enumerate(tmp_data):
                #     # assert "stage" in tmp_dd, print(tmp_dd,i)
                #     self.total_data.append(tmp_dd.copy())
            else:
                if no_origin:
                    self.total_data = []
                self.total_data.extend(after_data)

        if 'train' in file_path:
            # self.total_data = self.total_data[:100]
            x = random.randint(1, 34)
            print(x)
            print(len(self.total_data))
            xx = self.__getitem__(x)
            print("tot_data: ",self.total_data[x])
            # print(xx)
            # print(self.total_data[x]['new_strategy_list'])
            print(self.tokenizer.decode(xx['input_ids']))
            if "history_ids" in xx.keys():
                for t in xx['history_ids']:
                    print(self.tokenizer.decode(t))

    def truncat(self, history, max_len, flag='train'):  # label 不应该加cls标记， 因为decoder_inputs会自动加上 详见shift_tokens_right
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_ids = input_ids[-max_len:]
        input_ids[0] = self.tokenizer.cls_token_id
        return self.padding_sentence(input_ids, max_len)

    def padding_sentence(self, input_list, max_len):
        return input_list + [self.tokenizer.pad_token_id] * max(max_len - len(input_list), 0)

    def get_history_tensor(self, history):
        tensor_history = []
        for sentence in history:
            tensor_history.append(torch.tensor(self.truncat([sentence], max_len=self.each_length)).int())
        tensor_history = tensor_history[-self.sentence_num:]
        for i in range(max(0, self.sentence_num - len(tensor_history))):
            tensor_history.append(torch.fill_(torch.zeros(self.each_length), self.tokenizer.pad_token_id).int())
        ans = torch.cat([tmp_t.unsqueeze(0) for tmp_t in tensor_history], dim=0)
        return ans, ans != self.tokenizer.pad_token_id

    def get_str_form(self, tmp_str, tmp_len):
        str_list = tmp_str.split()[:tmp_len]
        return ' '.join(str_list)


    def get_model_input(self, tmp_dic):
        next_strategy = tmp_dic['next_strategy']
        tmp_history = tmp_dic['history']
        tmp_history[-1] = self.get_str_form(tmp_history[-1], self.each_length-3)
        history_ids, history_mask = self.get_history_tensor(tmp_history)
        vads = []
        for sentence_id in history_ids:
            sentence = self.tokenizer.convert_ids_to_tokens(sentence_id)
            assert len(sentence) == len(sentence_id), print(sentence, sentence_id, len(sentence), len(sentence_id))
            vads.append(np.array(self.emotion_index.get_one_sentence(sentence)))
        input_ids = torch.tensor(self.truncat([' '.join(next_strategy)], self.strategy_num + 2))  # 取决于模型是否使用它
        ans = {"history_ids": history_ids, "history_attention_mask": history_mask, 'input_ids': input_ids, "vads":vads}
        if "feedback" in tmp_dic:
            if self.cls:
                ans['labels'] = int(tmp_dic['feedback'])
            else:
                feedback = float(tmp_dic['feedback'])
                if feedback > 1.0:
                    feedback += 2.0
                ans['labels'] = feedback
        return ans

    def __getitem__(self, item):
        tmp_dic = self.total_data[item]
        return self.get_model_input(tmp_dic)

    def __len__(self):
        return len(self.total_data)


###################
# 生成策略 或者 生成句子
# model_type :{ 0: norm_bert   1: hie_bert  2: norm_strategy 3: hie_strategy  4: norm_sentence  5: hie_sentence.}
###################

class GenerateDataset2(Dataset):

    def __init__(self, model_type, file_path, tokenizer: BartTokenizer, strategy2id=None, max_source_len=510, max_target_len=4, each_sentence_length=32, sentence_num=64, add_cause=False, with_strategy=False, lookahead=False):
        super(GenerateDataset2, self).__init__()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.tokenizer = tokenizer
        self.emotion_index = EmotionalIndex(tokenizer)
        self.each_length = each_sentence_length
        self.sentence_num = sentence_num
        self.strategy2id = strategy2id
        self.model_type = model_type
        self.function_dict = {
            0: self.get_norm_bert_input,
            1: self.get_hierarchical_bert_input,
            2: self.get_norm_strategy_generate_input,
            3: self.get_hierarchical_strategy_generate_input,
            4: self.get_norm_sentence_generate_input,
            5: self.get_hierarchical_sentence_generate_input,
            6: self.get_hierarchical_sentence_generate_input2,
            7: self.get_sequicity_sentence_generate_input,
            8: self.get_hierarchical_sentence_generate_input_add_emotion,
            9: self.get_norm_gpt_generate_input,
        }
        data = load_json(file_path)
        self.sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token is not None else " "
        self.total_data = []
        self.is_train = 'train' in file_path
        if 'test' in file_path and self.model_type > 4:
            gt_strategy = read_pk('./final_data/test_extend_label.pk')
            if lookahead is not True:
                print("do not use lookahead! ")
                predict_strategy = read_pk('./final_data/wo_lookahead_predicted.pk')
            else:
                print('use lookahead! ')
                predict_strategy = read_pk('./final_data/multiesc_predicted_strategy.pk')
            print("acc: ", accuracy_score(gt_strategy, predict_strategy))
        predict_strategy_index = 0
        for case_example in data:
            dialog = case_example['dialog']
            # history = []
            dialog_len = len(dialog)
            emotion_type = case_example['emotion_type']
            problem_type = case_example['problem_type']
            situation = case_example['situation']
            # history = [_norm(emotion_type+' '+problem_type)]
            tot_strategy = []
            for index, tmp_dic in enumerate(dialog):
                if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                    tot_strategy.append(norm_strategy(tmp_dic['strategy']))
            history = [_norm(emotion_type) + self.sep_token + _norm(
                problem_type) + self.sep_token + _norm(situation)]

            tmp_strategy_list = []
            # vad_list = [np.zeros(3)]
            for index, tmp_dic in enumerate(dialog):
                text = _norm(tmp_dic['text'])
                if index == 0 and tmp_dic['speaker'] != 'sys':
                    # vad_list[0] = np.array(tmp_dic['vad'])
                    history[0] = text + self.sep_token + history[0]
                    continue
                if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    save_s = [x for x in tot_strategy[len(tmp_strategy_list):]].copy()
                    assert len(save_s) > 0, print(tot_strategy, tmp_strategy_list)
                    tmp_history = copy.deepcopy(history)
                    response = text
                    if with_strategy and self.model_type > 4:
                        if 'test' in file_path:
                            # tmp_history[-1] = tmp_history[-1] + self.sep_token + predict_strategy[predict_strategy_index]
                            if self.model_type == 8:
                                # response = predict_strategy[predict_strategy_index] + " " + text
                                tmp_history.append(predict_strategy[predict_strategy_index])
                            else:
                                tmp_history[-1] = tmp_history[-1] + self.sep_token + predict_strategy[predict_strategy_index]
                            predict_strategy_index += 1
                        else:
                            if self.model_type == 8:
                                tmp_history.append(tmp_stratege)
                                # response = tmp_stratege + " " + text
                            else:
                                tmp_history[-1] = tmp_history[-1] + self.sep_token + tmp_stratege
                    self.total_data.append({
                        "history": tmp_history,
                        "strategy": tmp_stratege,
                        "history_strategy": tmp_strategy_list,
                        "response": response,
                        "future_strategy": ' '.join(save_s),
                        "stage": 5 * index // dialog_len,
                        # 'vad': vad_list.copy(),
                    })
                    tmp_strategy_list.append(tmp_stratege)
                if tmp_dic['speaker'] == 'sys':
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    # vad_list.append(np.zeros(3))
                    if with_strategy:
                        tmp_sen = tmp_stratege + self.sep_token + text
                        history.append(tmp_sen)
                    else:
                        history.append(text)
                else:
                    # vad_list.append(np.array(tmp_dic['vad']))
                    if add_cause:
                        cause = tmp_dic['cause']
                        if cause is not None:
                            history.append(cause + self.sep_token + text)
                    else:
                        history.append(text)
        if 'test' in file_path and with_strategy and self.model_type > 4:
            assert len(self.total_data) == predict_strategy_index, print("tot_data: ",len(self.total_data), "predict_index", predict_strategy_index)
        x = random.randint(1, 50)
        print(x)
        xx = self.__getitem__(x)
        if 'train' in file_path:
            # self.total_data = self.total_data[:100]
            if len(xx['input_ids'].size()) < 2:
                # print(xx)
                print(self.tokenizer.decode(xx['input_ids']))
                print("ans: ",self.tokenizer.decode(xx['labels']))
            if "history_ids" in xx.keys():
                for index in range(len(xx['history_ids'])):
                    print(self.tokenizer.convert_ids_to_tokens(xx['history_ids'][index]))
                    if 'vads' in xx:
                        print(xx['vads'][index])
        else:
            print("ans: ", self.tokenizer.decode(xx['labels']))
    def generate_truncat(self, history, max_len, flag='no_label'):  # label 不应该加cls标记， 因为decoder_inputs会自动加上 详见shift_tokens_right
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            if self.tokenizer.sep_token_id is not None:
                input_ids.append(self.tokenizer.sep_token_id)
        input_ids[-1] = self.tokenizer.eos_token_id
        if flag == 'label':
            # input_ids.append(self.tokenizer.eos_token_id)
            input_ids = input_ids[:max_len]
            input_ids[-1] = self.tokenizer.eos_token_id
        else:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            # input_ids.append(self.tokenizer.eos_token_id)
            input_ids = input_ids[-max_len:]
            input_ids[0] = self.tokenizer.bos_token_id
        return self.padding_sentence(input_ids, max_len)

    def predict_truncat(self, history, max_len):
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_ids = input_ids[:max_len]
        return self.padding_sentence(input_ids, max_len)

    def padding_sentence(self, input_list, max_len):
        return input_list + [self.tokenizer.pad_token_id] * max(max_len - len(input_list), 0)

    def padding_vad(self, vad_list, max_len):
        return np.concatenate([vad_list, np.zeros((max(max_len-len(vad_list), 0), 3))], 0)

    def get_history_tensor(self, history):
        tensor_history = []
        for sentence in history:
            if self.model_type < 2:
                tensor_history.append(torch.tensor(self.predict_truncat([sentence], max_len=self.each_length)).int())
            else:
                tensor_history.append(torch.tensor(self.generate_truncat([sentence], max_len=self.each_length)).int())
        tensor_history = tensor_history[-self.sentence_num:]
        for i in range(max(0, self.sentence_num - len(tensor_history))):
            tensor_history.append(torch.fill_(torch.zeros(self.each_length), self.tokenizer.pad_token_id).int())
        ans = torch.cat([tmp_t.unsqueeze(0) for tmp_t in tensor_history], dim=0)
        return ans, ans != self.tokenizer.pad_token_id

    def get_str_form(self, tmp_str, tmp_len):
        str_list = tmp_str.split()[:tmp_len]
        return ' '.join(str_list)

    def get_norm_bert_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        input_ids = torch.tensor(self.predict_truncat(tmp_history, self.max_source_len))
        labels = torch.tensor(self.strategy2id[tmp_dic['strategy']], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    # 需要加user state.
    def get_hierarchical_bert_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        assert len(tmp_history) % 2 == 1, print(tmp_history)
        combine_history = []
        for i in range(len(tmp_history) // 2):
            tmp_str = self.get_str_form(tmp_history[i * 2],
                                        self.each_length // 2)+' '+self.sep_token+' '+self.get_str_form(
                tmp_history[i * 2 + 1], self.each_length // 2)
            combine_history.append(tmp_str)
        combine_history.append(self.get_str_form(tmp_history[-1], self.each_length))
        assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        input_ids = torch.tensor(self.predict_truncat(tmp_history, self.max_source_len))
        labels = torch.tensor(self.strategy2id[tmp_dic['strategy']], dtype=torch.long)
        vads = torch.tensor(self.padding_vad(tmp_dic['vad'], self.sentence_num))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_mask": history_mask,
            "vad": vads,
            "labels": labels,
        }

    def get_norm_strategy_generate_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        # assert len(tmp_history) % 2 == 1, print(tmp_history)
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['future_strategy']], self.max_target_len, flag='label'), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_hierarchical_strategy_generate_input(self, tmp_dic):
        combine_history = tmp_dic['history']
        # assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        vads = []
        for sentence_id in history_ids:
            sentence = self.tokenizer.convert_ids_to_tokens(sentence_id)
            assert len(sentence) == len(sentence_id), print(sentence, sentence_id, len(sentence), len(sentence_id))
            vads.append(np.array(self.emotion_index.get_one_sentence(sentence)))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['future_strategy']], self.max_target_len, flag='label'), dtype=torch.long)        # print('history_ids', history_ids.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            "vads": torch.tensor(vads),
        }


    def get_norm_sentence_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_norm_gpt_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['strategy'] + ' ' + tmp_dic['response']], self.max_target_len, flag='label'), dtype=torch.long)
        if not self.is_train:
            return {
                "input_ids": input_ids,
                "attention_mask": input_ids != self.tokenizer.pad_token_id,
                "labels": torch.cat((input_ids, labels))
            }
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": input_ids
        }



    def get_hierarchical_sentence_generate_input(self, tmp_dic):
        combine_history = tmp_dic['history']
        # combine_history = tmp_history
        # assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        # input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'), dtype=torch.long)
        return {
            "input_ids": history_ids,
            "attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            # "vad": vads,
        }

    def get_hierarchical_sentence_generate_input2(self, tmp_dic):
        combine_history = tmp_dic['history']
        # combine_history = tmp_history
        # assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        # assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'), dtype=torch.long)
        # print('history_ids', history_ids.size())
        # print('vads', vads.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            # "vad": vads,
        }

    def get_sequicity_sentence_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['strategy'] + ' ' + tmp_dic['response']], self.max_target_len, flag='label'), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_hierarchical_sentence_generate_input_add_emotion(self, tmp_dic):
        combine_history = tmp_dic['history']
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        vads = []
        for sentence_id in history_ids:
            sentence = self.tokenizer.convert_ids_to_tokens(sentence_id)
            assert len(sentence) == len(sentence_id), print(sentence, sentence_id, len(sentence), len(sentence_id))
            vads.append(np.array(self.emotion_index.get_one_sentence(sentence)))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        # vads = torch.tensor(vads)
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'), dtype=torch.long)
        # print('history_ids', history_ids.size())
        # print('vads', vads.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            "vads": torch.tensor(vads),
        }

    def see_one_item(self, item):
        xx = self.__getitem__(item)
        if "history_ids" in xx:
            for x in zip(xx['history_ids'], xx['history_attention_mask']):
                print(self.tokenizer.decode(x[0]))
        print(self.tokenizer.decode(xx['labels']))

    def __getitem__(self, item):
        tmp_dic = self.total_data[item]
        # print(tmp_dic)
        return self.function_dict[self.model_type](tmp_dic)

    def __len__(self):
        return len(self.total_data)


class GenerateDataset3(Dataset):

    def __init__(self, model_type, file_path, tokenizer, strategy2id=None, max_source_len=512, max_target_len=4, each_sentence_length=32, sentence_num=32, add_cause=False, with_strategy=False):
        super(GenerateDataset3, self).__init__()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.tokenizer = tokenizer
        self.each_length = each_sentence_length
        self.sentence_num = sentence_num
        self.strategy2id = strategy2id
        self.model_type = model_type
        self.emotion_index = EmotionalIndex(tokenizer)
        self.function_dict = {
            0: self.get_norm_bert_input,
            1: self.get_hierarchical_bert_input,
            2: self.get_norm_strategy_generate_input,
            3: self.get_hierarchical_strategy_generate_input,
            4: self.get_norm_sentence_generate_input,
            5: self.get_hierarchical_sentence_generate_input,
        }
        data = load_json(file_path)
        self.total_data = []

        for case_example in data:
            dialog = case_example['dialog']
            # history = []
            emotion_type = case_example['emotion_type']
            problem_type = case_example['problem_type']
            situation = case_example['situation']
            # history = [_norm(emotion_type+' '+problem_type)]
            tot_strategy = []
            for index, tmp_dic in enumerate(dialog):
                if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                    tot_strategy.append(norm_strategy(tmp_dic['strategy']))
            history = [_norm(emotion_type) + self.tokenizer.sep_token + _norm(
                problem_type) + self.tokenizer.sep_token + _norm(situation)]

            tmp_strategy_list = []
            vad_list = [np.zeros(3)]
            for index, tmp_dic in enumerate(dialog):
                text = _norm(tmp_dic['text'])
                if index == 0 and tmp_dic['speaker'] != 'sys':
                    vad_list[0] = np.array(tmp_dic['vad'])
                    history[0] = text + self.tokenizer.sep_token + history[0]
                    continue
                if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    save_s = [x for x in tot_strategy[len(tmp_strategy_list):]].copy()
                    assert len(save_s) > 0, print(tot_strategy, tmp_strategy_list)
                    tmp_history = copy.deepcopy(history)
                    if with_strategy:
                        tmp_history[-1] = tmp_history[-1] + self.tokenizer.sep_token + tmp_stratege
                    self.total_data.append({
                        "history": tmp_history,
                        "strategy": tmp_stratege,
                        "response": text,
                        "future_strategy": ' '.join(save_s),
                        'vad': vad_list.copy(),
                    })
                    tmp_strategy_list.append(tmp_stratege)
                if tmp_dic['speaker'] == 'sys':
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    vad_list.append(np.zeros(3))
                    if with_strategy:
                        history.append(tmp_stratege + self.tokenizer.sep_token + text)
                    else:
                        history.append(text)
                else:
                    vad_list.append(np.array(tmp_dic['vad']))
                    if with_strategy:
                        cause = tmp_dic['cause']
                        if cause is not None:
                            history.append(cause + self.tokenizer.sep_token + text)
                    else:
                        history.append(text)
        x = random.randint(1, 340)
        print(x)
        xx = self.__getitem__(x)
        if len(xx['input_ids'].size()) < 2:
            print(self.tokenizer.decode(xx['input_ids']))

    def generate_truncat(self, history, max_len, flag='train'):  # label 不应该加cls标记， 因为decoder_inputs会自动加上 详见shift_tokens_right
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids[-1] = self.tokenizer.eos_token_id
        if flag == 'test':
            input_ids = input_ids[:max_len]
            input_ids[-1] = self.tokenizer.eos_token_id
        else:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            input_ids = input_ids[-max_len:]
            input_ids[0] = self.tokenizer.bos_token_id
        return self.padding_sentence(input_ids, max_len)

    def predict_truncat(self, history, max_len):
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_ids = input_ids[-max_len:]
        input_ids[0] = self.tokenizer.cls_token_id
        return self.padding_sentence(input_ids, max_len)

    def padding_sentence(self, input_list, max_len):
        return input_list + [self.tokenizer.pad_token_id] * max(max_len - len(input_list), 0)

    def padding_vad(self, vad_list, max_len):
        return np.concatenate([vad_list, np.zeros((max(max_len-len(vad_list), 0), 3))], 0)

    def get_history_tensor(self, history):
        tensor_history = []
        for sentence in history:
            if self.model_type < 2:
                tensor_history.append(torch.tensor(self.predict_truncat([sentence], max_len=self.each_length)).int())
            else:
                tensor_history.append(torch.tensor(self.generate_truncat([sentence], max_len=self.each_length)).int())
        tensor_history = tensor_history[-self.sentence_num:]
        for i in range(max(0, self.sentence_num - len(tensor_history))):
            tensor_history.append(torch.fill_(torch.zeros(self.each_length), self.tokenizer.pad_token_id).int())
        ans = torch.cat([tmp_t.unsqueeze(0) for tmp_t in tensor_history], dim=0)
        return ans, ans != self.tokenizer.pad_token_id

    def get_str_form(self, tmp_str, tmp_len):
        str_list = tmp_str.split()[:tmp_len]
        return ' '.join(str_list)

    def get_norm_bert_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        input_ids = torch.tensor(self.predict_truncat(tmp_history, self.max_source_len))
        labels = torch.tensor(self.strategy2id[tmp_dic['strategy']], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    # 需要加user state.
    def get_hierarchical_bert_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        assert len(tmp_history) % 2 == 1, print(tmp_history)
        combine_history = []
        for i in range(len(tmp_history) // 2):
            tmp_str = self.get_str_form(tmp_history[i * 2],
                                        self.each_length // 2)+' '+self.tokenizer.sep_token+' '+self.get_str_form(
                tmp_history[i * 2 + 1], self.each_length // 2)
            combine_history.append(tmp_str)
        combine_history.append(self.get_str_form(tmp_history[-1], self.each_length))
        assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        input_ids = torch.tensor(self.predict_truncat(tmp_history, self.max_source_len))
        labels = torch.tensor(self.strategy2id[tmp_dic['strategy']], dtype=torch.long)
        vads = torch.tensor(self.padding_vad(tmp_dic['vad'], self.sentence_num))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_mask": history_mask,
            "vad": vads,
            "labels": labels,
        }

    def get_norm_strategy_generate_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        assert len(tmp_history) % 2 == 1, print(tmp_history)
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['future_strategy']], self.max_target_len, flag='test'), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_hierarchical_strategy_generate_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        assert len(tmp_history) % 2 == 1, print(tmp_history)
        combine_history = []
        for i in range(len(tmp_history) // 2):
            tmp_str = self.get_str_form(tmp_history[i * 2],
                                        self.each_length // 2)+' '+self.tokenizer.sep_token+' '+self.get_str_form(
                tmp_history[i * 2 + 1], self.each_length // 2)
            combine_history.append(tmp_str)
        combine_history.append(self.get_str_form(tmp_history[-1], self.each_length))
        assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        # input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['future_strategy']], self.max_target_len, flag='test'), dtype=torch.long)
        return {
            "input_ids": history_ids,
            "attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            "vad": vads,
        }

    def get_norm_sentence_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='test'), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_hierarchical_sentence_generate_input(self, tmp_dic):
        combine_history = tmp_dic['history']
        # combine_history = tmp_history
        assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='test'), dtype=torch.long)
        # print('history_ids', history_ids.size())
        # print('vads', vads.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            # "vad": vads,
        }

    def see_one_item(self, item):
        xx = self.__getitem__(item)
        if "history_ids" in xx:
            for x in zip(xx['history_ids'], xx['history_attention_mask']):
                print(self.tokenizer.decode(x[0]))
        print(self.tokenizer.decode(xx['labels']))

    def __getitem__(self, item):
        tmp_dic = self.total_data[item]
        return self.function_dict[self.model_type](tmp_dic)

    def __len__(self):
        return len(self.total_data)
