from __future__ import absolute_import, division, print_function
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
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AdamW, SchedulerType, get_scheduler
import json
import sys


install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from utils.evaluator import Evaluator

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        return self.read_json(os.path.join(data_dir, "test_unlabelled.json"), "test")
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
                raw_text, tags, label = self.parse_utt(utt, set_type)
                guid = '%s-%s'%(set_type, index)
                text_a = raw_text
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, labels=tags))
                index += 1
                labels.append(label)
        return examples, labels
    def parse_utt(self, utt: dict, set_type):
        if set_type == 'train':
            raw_text = utt['asr_1best']
        else:
            raw_text = utt['asr_1best']
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

    # Training config.
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--eval_on", type=str, default="dev",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=64,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.,
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
            model = AutoModelForTokenClassification.from_pretrained(args.load_model_path,
                                                                    num_labels=num_labels,
                                                                    return_dict=True,
                                                                    cache_dir=cache_dir)
        else:
            model = AutoModelForTokenClassification.from_pretrained(args.model_type,
                                                                    num_labels=num_labels,
                                                                    return_dict=True,
                                                                    cache_dir=cache_dir)
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
            if args.eval_on == "dev":
                eval_examples, dev_labels = processor.get_dev_examples(os.path.join(args.data_dir))
            else:
                eval_examples, dev_labels = processor.get_test_examples(os.path.join(args.data_dir))
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
                outputs = model(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                labels=label_ids)
                loss = outputs.loss

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
                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        labels=label_ids)
                        tmp_eval_loss = outputs.loss
                        logits = outputs.logits

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
                        sub_value = ""
                        sub_slot = ""
                        tmp_prediction = []
                        for ii, label in enumerate(pred_label):
                            if label.startswith('B'):
                                if sub_input_ids[ii] != 101 and sub_input_ids[ii] != 0 and sub_input_ids[ii]!=102:
                                    sub_value = sub_tokens[ii]
                                    sub_slot = label[2:]
                            elif label.startswith('I'):
                                if sub_input_ids[ii] != 101 and sub_input_ids[ii] != 0 and sub_input_ids[ii]!=102:
                                    sub_value += sub_tokens[ii]
                            elif label.startswith('O') and sub_value!="" and sub_slot!="":
                                tmp_prediction.append("%s-%s"%(sub_slot, sub_value))
                                sub_value = ""
                                sub_slot = ""
                        all_predictions.append(tmp_prediction)

                
                    # tmp_predictions = np.argmax(logits, axis=2).reshape(-1).tolist()
                    # tmp_labels = label_ids.reshape(-1).tolist()
                    # all_predictions.extend([p for p, l in zip(tmp_predictions, tmp_labels) if l != -100])
                    # all_labels.extend([l for l in tmp_labels if l != -100])
                    num_eval_examples += input_ids.size(0)
                    eval_steps += 1
                # print(all_predictions)
                evaluator = Evaluator()
                metrics = evaluator.acc(all_predictions, all_labels)
                eval_acc, eval_f1 = metrics['acc'], metrics['fscore']
                loss = train_loss / train_steps
                eval_loss = eval_loss / eval_steps
                # eval_acc = mtc.f1_score(all_labels,
                #                         all_predictions,
                #                         labels=list(range(1, num_labels)),
                #                         average="micro") * 100
                model_to_save = model.module if hasattr(model, "module") else model
                output_model_file = os.path.join(args.output_dir, "best_pytorch_model_for_{}.bin".format(task_name))
                if eval_acc > best_acc:
                    torch.save(model_to_save.state_dict(), output_model_file)
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