from __future__ import division, print_function
import argparse
import logging
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import sklearn.metrics as mtc
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW, SchedulerType, get_scheduler
import json
import sys
import torch.nn as nn
import torch.nn.functional as F
import copy
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.initialization import *
from utils.example import Example
from xpinyin import Pinyin
from utils.evaluator import Evaluator
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)
Example.configuration('../data', train_path='../data/train.json', word2vec_path='../word2vec-768.txt')
def anti_noise_prediction(predictions):
    p = Pinyin()
    select_pos_set = {'poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', '终点名称', '终点修饰', '终点目标', '途经点名称'}
    select_others_set = {'请求类型': [Example.label_vocab.request_map_dic, Example.label_vocab.request_pinyin_set], \
        '出行方式' : [Example.label_vocab.travel_map_dic, Example.label_vocab.travel_pinyin_set], \
        '路线偏好' : [Example.label_vocab.route_map_dic, Example.label_vocab.route_pinyin_set], \
        '对象' :  [Example.label_vocab.object_map_dic, Example.label_vocab.object_pinyin_set], \
        '页码' : [Example.label_vocab.page_map_dic, Example.label_vocab.page_pinyin_set], \
        '操作' : [Example.label_vocab.opera_map_dic, Example.label_vocab.opera_pinyin_set], \
        '序列号' : [Example.label_vocab.ordinal_map_dic, Example.label_vocab.ordinal_pinyin_set]   }

    modify_num = 0
    for i, pred in enumerate(predictions):
        pred_length = len(pred)
        if pred_length > 0 :
            for j in range(pred_length):
                tmp_pred = pred[j]
                split_result = tmp_pred.split('-')
                tmp_pinyin = p.get_pinyin(split_result[2], ' ')
                if split_result[1] != 'value' :
                    if split_result[1] in select_pos_set :
                        map_dic, pinyin_set = Example.label_vocab.poi_map_dic, Example.label_vocab.poi_pinyin_set
                    else :
                        [map_dic, pinyin_set] = select_others_set[split_result[1]]

                    standard_output = get_standard_output (map_dic, pinyin_set, tmp_pinyin)
                    modify_pred = split_result[0] + '-' + split_result[1] + '-' + standard_output
                    if standard_output != split_result[2] :
                        modify_num += 1
                    predictions[i][j] = modify_pred
    print ("modify_num == ", modify_num)                    
    return  predictions            

def get_standard_output (map_dic, pinyin_set, tmp_pinyin) :
    if tmp_pinyin in pinyin_set :
        standard_output = map_dic[tmp_pinyin]
    else :
        max_similarity = 0
        most_similar_pinyin = ''
        for standard_pinyin in iter(pinyin_set) :
            similarity = get_pinyin_similarity(standard_pinyin, tmp_pinyin)
            if similarity > max_similarity :
                max_similarity = similarity
                most_similar_pinyin = standard_pinyin
        if max_similarity == 0 : 
            standard_output = '无'
        else :
            standard_output = map_dic[most_similar_pinyin]
    return standard_output
            

def get_pinyin_similarity(standard_pinyin, tmp_pinyin) :
    standard_set = set (standard_pinyin.split(' '))
    tmp_set = set (tmp_pinyin.split(' '))

    inter_set = standard_set & tmp_set
    similarity = len (inter_set) / (len (standard_set) + len (tmp_set) )
    return similarity




class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class SpeechProcessor:
    def get_train_examples(self, data_dir):
        return self.read_json(os.path.join(data_dir, "train.json"), "train")
    def get_dev_examples(self, data_dir):
        return self.read_json(os.path.join(data_dir, "development.json"), "dev")
    def get_test_examples(self, data_dir):
        return self.read_json(data_dir, "test")
    def get_labels(self):
        return ["O", "B-inform-poi名称", "B-inform-poi修饰", "B-inform-poi目标", "B-inform-起点名称", "B-inform-起点修饰",
        "B-inform-起点目标", "B-inform-终点名称", "B-inform-终点修饰", "B-inform-终点目标", "B-inform-途经点名称", "B-inform-请求类型",
        "B-inform-出行方式", "B-inform-路线偏好", "B-inform-对象", "B-inform-操作", "B-inform-序列号", "B-inform-页码", "B-inform-value",
        "B-deny-poi名称", "B-deny-poi修饰", "B-deny-poi目标", "B-deny-起点名称", "B-deny-起点修饰",
        "B-deny-起点目标", "B-deny-终点名称", "B-deny-终点修饰", "B-deny-终点目标", "B-deny-途经点名称", "B-deny-请求类型",
        "B-deny-出行方式", "B-deny-路线偏好", "B-deny-对象", "B-deny-操作", "B-deny-序列号", "B-deny-页码", "B-deny-value",
        "I-inform-poi名称", "I-inform-poi修饰", "I-inform-poi目标", "I-inform-起点名称", "I-inform-起点修饰",
        "I-inform-起点目标", "I-inform-终点名称", "I-inform-终点修饰", "I-inform-终点目标", "I-inform-途经点名称", "I-inform-请求类型",
        "I-inform-出行方式", "I-inform-路线偏好", "I-inform-对象", "I-inform-操作", "I-inform-序列号", "I-inform-页码", "I-inform-value",
        "I-deny-poi名称", "I-deny-poi修饰", "I-deny-poi目标", "I-deny-起点名称", "I-deny-起点修饰",
        "I-deny-起点目标", "I-deny-终点名称", "I-deny-终点修饰", "I-deny-终点目标", "I-deny-途经点名称", "I-deny-请求类型",
        "I-deny-出行方式", "I-deny-路线偏好", "I-deny-对象", "I-deny-操作", "I-deny-序列号", "I-deny-页码", "I-deny-value",]
    def read_json(self, input_file, set_type):
        datas = json.load(open(input_file, 'r', encoding='utf-8'))
        examples = []
        labels = []
        index = 0
        for data in datas:
            for utt in data:
                raw_text, tags, label = self.parse_utt(utt, set_type, 'asr')
                guid = '%s-%s'%(set_type, index)
                text_a = raw_text
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, labels=tags))
                index += 1
                labels.append(label)
                if set_type == 'train':
                    raw_text, tags, label = self.parse_utt(utt, set_type, 'manual')
                    guid = '%s-%s'%(set_type, index)
                    text_a = raw_text
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, labels=tags))
                    index += 1
                    labels.append(label)
        return examples, labels
    def parse_utt(self, utt: dict, set_type, text_type):
        if set_type == 'train' and text_type == 'asr':
            raw_text = utt['asr_1best'].lower()
        elif set_type == 'train' and text_type == 'manual':
            raw_text = utt['manual_transcript'].lower()
        else:
            raw_text = utt['asr_1best'].lower()
        slots = {}
        if 'semantic' in utt.keys():
            for label in utt['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    slots[act_slot] = label[2]
        tags = ['O'] * len(raw_text)
        for slot in slots:
            value = slots[slot]
            bidx = raw_text.find(value)
            if bidx != -1:
                tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                tags[bidx] = f'B-{slot}'
        labels = [f'{slot}-{value}' for slot, value in slots.items()]
        return raw_text, tags, labels

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        encoded_inputs = tokenizer(example.text_a,
                                   max_length=max_seq_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_token_type_ids=True,)
        labels = example.labels
        word_ids = encoded_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        input_ids = encoded_inputs["input_ids"]
        input_mask = encoded_inputs["attention_mask"]
        segment_ids = encoded_inputs["token_type_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % ' '.join(tokens))
            logger.info("input_ids: %s" % ' '.join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % ' '.join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % ' '.join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids)
        )
    return features


class Metrics:
    @staticmethod
    def acc(predictions, labels):
        return mtc.accuracy_score(labels, predictions)

    @staticmethod
    def mcc(predictions, labels):
        return mtc.matthews_corrcoef(labels, predictions)

    @staticmethod
    def f1(predictions, labels, average="micro"):
        return mtc.f1_score(labels, predictions, average=average)

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''
    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')

        self.reduction = reduction

        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.

        target = target * (target != self.ignore_index).long()

        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))

        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy

        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss
class BertForNer(nn.Module):
    def __init__(self, model_type, cache_dir, num_labels,):
        super(BertForNer,self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(model_type, cache_dir = cache_dir, num_labels = num_labels,return_dict=True, classifier_dropout = 0.1)
        # self.bert = AutoModel.from_pretrained(model_type, cache_dir = cache_dir, return_dict = True)
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(768, num_labels)
        # self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        # self.loss = FocalLoss(gamma=1)
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels=labels)
        # sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        # final_logits = self.classifier(sequence_output)
        # loss = self.loss(final_logits.view(-1, final_logits.shape[-1]), labels.view(-1))
        final_logits = outputs.logits
        loss = outputs.loss
        return final_logits, loss
def main():
    parser = argparse.ArgumentParser()

    # Data config.
    parser.add_argument("--data_dir", type=str, default="../data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="speech",
                        help="Name of the training task.")
    parser.add_argument("--model_type", type=str, default="bert-base-chinese",
                        help="Type of BERT-like pre-trained language models.")
    parser.add_argument("--load_model_path", type=str, default="",
                        help="Trained model path to load if needed.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--test_dir", type=str, default="../data/development.json")
    # Training config.
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--eval_on", type=str, default="dev",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=32,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")

    args = parser.parse_args()

    processors = {
        "speech": SpeechProcessor
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", "Unsupported"))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))
    elif args.do_test:
        torch.save(args, os.path.join(args.output_dir, "test_args.bin"))

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_type,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)
    
    if args.do_test:
        test_examples, test_labels = processor.get_test_examples(args.test_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        if not args.load_model_path:
            print("You need to specify model path when testing")
            return
        model = BertForNer(model_type=args.model_type, cache_dir=cache_dir, num_labels=num_labels)
        model.load_state_dict(torch.load(args.load_model_path))
        model.to(device)
        model.eval()
        all_predictions = []
        for batch in tqdm(test_dataloader, desc="Test"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, tmp_test_loss = model(input_ids = input_ids, attention_mask = input_mask,
                                                token_type_ids = segment_ids, labels = label_ids)
            logits = logits.detach().cpu().numpy()
            tmp_predictions = np.argmax(logits, axis=2).tolist()
            for i, sub_pred in enumerate(tmp_predictions):
                    pred_label = []
                    for pred in sub_pred:
                        pred_label.append(label_list[pred])
                    sub_input_ids = input_ids[i]
                    sub_tokens = tokenizer.convert_ids_to_tokens(sub_input_ids)
                    idx_buff, tag_buff, pred_tuple = [], [], []
                    for ii, label in enumerate(pred_label):
                        if sub_input_ids[ii] == 101:
                            continue
                        if sub_input_ids[ii] == 102:
                            break
                        if (label == 'O' or label.startswith('B')) and len(tag_buff) > 0:
                            slot = '-'.join(tag_buff[0].split('-')[1:])
                            value = ''.join([sub_tokens[j] for j in idx_buff])
                            idx_buff, tag_buff = [], []
                            pred_tuple.append(f'{slot}-{value}')
                            if label.startswith('B'):
                                idx_buff.append(ii)
                                tag_buff.append(label)
                        elif label.startswith('I') or label.startswith('B'):
                            idx_buff.append(ii)
                            tag_buff.append(label)
                    if len(tag_buff) > 0:
                        slot = '-'.join(tag_buff[0].split('-')[1:])
                        value = ''.join([sub_tokens[j] for j in idx_buff])
                        pred_tuple.append(f'{slot}-{value}')
                    all_predictions.append(pred_tuple)
        all_predictions = anti_noise_prediction(all_predictions)
        # evaluator = Evaluator()
        # metrics = evaluator.acc(all_predictions, test_labels)
        # print(metrics['acc'])
        pred_id = 0
        test_data = json.load(open(args.test_dir, 'r', encoding='utf-8'))
        pred_test_data = copy.deepcopy(test_data)
        for data in pred_test_data:
            for utt in data:
                tmp_pred = []
                for pred in all_predictions[pred_id]:
                    pred_list = pred.split('-')
                    tmp_pred.append(pred_list)
                utt['pred'] = tmp_pred
                pred_id += 1
        pred_json = json.dumps(pred_test_data, indent=4, ensure_ascii=False)
        with open(os.path.join(args.data_dir, "test.json"), 'w', encoding='utf-8') as f:
            f.write(pred_json)
        return
    
    if args.do_train:
        train_examples, train_labels = processor.get_train_examples(os.path.join(args.data_dir))
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        if args.load_model_path:
            model = BertForNer(model_type=args.load_model_path, cache_dir=cache_dir, num_labels=num_labels)                                                        
        else:
            model = BertForNer(model_type=args.model_type, cache_dir=cache_dir, num_labels=num_labels)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_scheduler(name=args.lr_scheduler_type,
                                  optimizer=optimizer,
                                  num_warmup_steps=args.max_train_steps * args.warmup_proportion,
                                  num_training_steps=args.max_train_steps)

        if args.do_eval:
            eval_examples, dev_labels = processor.get_dev_examples(os.path.join(args.data_dir))
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        progress_bar = tqdm(range(args.max_train_steps))
        global_step = 0
        best_acc = 0
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            train_loss = 0
            num_train_examples = 0
            train_steps = 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # outputs = model(input_ids=input_ids,
                #                 attention_mask=input_mask,
                #                 token_type_ids=segment_ids,
                #                 labels=label_ids)
                # loss = outputs.loss
                logits, loss = model(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                train_loss += loss.item()
                num_train_examples += input_ids.size(0)
                train_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    progress_bar.update(1)

                if global_step >= args.max_train_steps:
                    break

            

            if args.do_eval:
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                eval_loss = 0
                num_eval_examples = 0
                eval_steps = 0
                all_predictions, all_labels = [], dev_labels
                # print(all_labels)
                for batch in tqdm(eval_dataloader, desc="Evaluation"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    with torch.no_grad():
                        # outputs = model(input_ids=input_ids,
                        #                 attention_mask=input_mask,
                        #                 token_type_ids=segment_ids,
                        #                 labels=label_ids)
                        # tmp_eval_loss = outputs.loss
                        # logits = outputs.logits
                        logits, tmp_eval_loss = model(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                labels=label_ids)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to("cpu").numpy()
                    eval_loss += tmp_eval_loss.mean().item()
                    tmp_predictions = np.argmax(logits, axis = 2).tolist()
                    for i, sub_pred in enumerate(tmp_predictions):
                        pred_label = []
                        for pred in sub_pred:
                            pred_label.append(label_list[pred])
                        sub_input_ids = input_ids[i]
                        sub_tokens = tokenizer.convert_ids_to_tokens(sub_input_ids)
                        idx_buff, tag_buff, pred_tuple = [], [], []
                        for ii, label in enumerate(pred_label):
                            if sub_input_ids[ii] == 101:
                                continue
                            if sub_input_ids[ii] == 102:
                                break
                            if (label == 'O' or label.startswith('B')) and len(tag_buff) > 0:
                                slot = '-'.join(tag_buff[0].split('-')[1:])
                                value = ''.join([sub_tokens[j] for j in idx_buff])
                                idx_buff, tag_buff = [], []
                                pred_tuple.append(f'{slot}-{value}')
                                if label.startswith('B'):
                                    idx_buff.append(ii)
                                    tag_buff.append(label)
                            elif label.startswith('I') or label.startswith('B'):
                                idx_buff.append(ii)
                                tag_buff.append(label)
                        if len(tag_buff) > 0:
                            slot = '-'.join(tag_buff[0].split('-')[1:])
                            value = ''.join([sub_tokens[j] for j in idx_buff])
                            pred_tuple.append(f'{slot}-{value}')
                        
                            # if label.startswith('B'):
                            #     if sub_input_ids[ii] != 101 and sub_input_ids[ii] != 0 and sub_input_ids[ii]!=102:
                            #         sub_value = sub_tokens[ii]
                            #         sub_slot = label[2:]
                            # elif label.startswith('I'):
                            #     if sub_input_ids[ii] != 101 and sub_input_ids[ii] != 0 and sub_input_ids[ii]!=102:
                            #         sub_value += sub_tokens[ii]
                            # elif label.startswith('O') and sub_value!="" and sub_slot!="":
                            #     tmp_prediction.append("%s-%s"%(sub_slot, sub_value))
                            #     sub_value = ""
                            #     sub_slot = ""
                        all_predictions.append(pred_tuple)
                        
                    num_eval_examples += input_ids.size(0)
                    eval_steps += 1
                evaluator = Evaluator()
                all_predictions = anti_noise_prediction(all_predictions)
                metrics = evaluator.acc(all_predictions, all_labels)
                eval_acc, eval_f1 = metrics['acc'], metrics['fscore']
                loss = train_loss / train_steps
                eval_loss = eval_loss / eval_steps
                model_to_save = model.module if hasattr(model, "module") else model
                output_model_file = os.path.join(args.output_dir, "best_pytorch_model_for_{}.bin".format(task_name))
                if eval_acc > best_acc:
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_acc = eval_acc
                result = {
                    "global_step": global_step,
                    "loss": loss,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "eval_f1": eval_f1
                }

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, 'a') as writer:
                    logger.info("***** Eval results *****")
                    writer.write(
                        "Epoch %s: global step = %s | loss = %.3f | eval score = %.2f | eval acc = %.3f | Dev fscore(p/r/f): (%.2f/%.2f/%.2f) \n"
                        % (str(epoch),
                           str(result["global_step"]),
                           result["loss"],
                           result["eval_acc"],
                           result["eval_loss"],
                           eval_f1['precision'],
                           eval_f1['recall'],
                           eval_f1['fscore']))
                    for key in sorted(result.keys()):
                        logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))


if __name__ == "__main__":
    main()