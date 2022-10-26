'''
!pip install easynmt
!pip install paddlepaddle==2.3.2
!pip install paddlenlp==2.4.1
'''

##### model train scope
import os
import sys
##### attach PaddleNLP/examples/few_shot/p-tuning to import
sys.path.insert(0, "p-tuning/")

import argparse
import os
import sys
import random
import time
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from model import ErnieForPretraining, ErnieMLMCriterion
from data import create_dataloader, transform_fn_dict
from data import convert_example, convert_chid_example
from evaluate import do_evaluate, do_evaluate_chid

from collections import namedtuple
import pandas as pd

def read_fn(data_file):
    examples = []
    with open(data_file, 'r', encoding="utf-8") as f:
        for line in f:
            text, label = line.rstrip().split("\t")
            example = {"sentence1": text, "label": label}
            # 如果有 2 列文本，则
            # example = {"sentence1": text1, "sentence2": text2，"label": label}
            examples.append(example)
    return examples

def transform_ethos(example, label_normalize_dict=None, is_test=False):
    if is_test:
        # Label: ['种族', '宗教', '性别', '民族', '取向', '残疾']
        example["label_length"] = 2
        return example
    else:
        origin_label = str(example["label"])

        # Normalize some of the labels, eg. English -> Chinese
        example['text_label'] = label_normalize_dict[origin_label]

        del example["label"]

        return example

parser = argparse.ArgumentParser()

parser.add_argument("--task_name", required=True, type=str, help="The task_name to be evaluated")
parser.add_argument("--p_embedding_num", type=int, default=1, help="number of p-embedding")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000, help="Inteval steps to save checkpoint")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")

###args = parser.parse_args()

#### toolkit to transform argument parser to dict or namedtuple
#### or use open source toolkit: https://github.com/roee30/datargs
#### !pip install datargs
def parse_parser_add_arg(parser, as_named_tuple = False):
    args_df = pd.DataFrame(
    pd.Series(parser.__dict__["_actions"]).map(
    lambda x:x.__dict__
    ).values.tolist())
    args_df = args_df.explode("option_strings")
    args_df["option_strings"] = args_df["option_strings"].map(
    lambda x: x[2:] if x.startswith("--") else x
    ).map(
        lambda x: x[1:] if x.startswith("-") else x
        )
    args_df = args_df[["option_strings", "default"]]
    args = dict(args_df.values.tolist())
    if as_named_tuple:
        args_parser_namedtuple = namedtuple("args_config", args)
        return args_parser_namedtuple(**args)
    return args_df

def transform_named_tuple_to_dict(N_tuple):
    return dict(map(
    lambda x: (x, getattr(N_tuple, x))
    ,filter(lambda x: not x.startswith("_") ,dir(N_tuple))
))

def transform_dict_to_named_tuple(dict_, name = "NamedTuple"):
    args_parser_namedtuple = namedtuple(name, dict_)
    return args_parser_namedtuple(**dict_)

reformat_parser_df = parse_parser_add_arg(parser, as_named_tuple=False)
reformat_parser_tuple = parse_parser_add_arg(parser, as_named_tuple=True)
reformat_parser_dict = transform_named_tuple_to_dict(reformat_parser_tuple)

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

#### change args scope
reformat_parser_dict["device"] = "cpu"
reformat_parser_dict["task_name"] = "ethos_unique_zh"
#### batch_size
#### learning_rate
reformat_parser_dict["batch_size"] = 4
reformat_parser_dict["learning_rate"] = 1e-6
reformat_parser_dict["epochs"] = 50

args = transform_dict_to_named_tuple(reformat_parser_dict)
####args.device

paddle.set_device(args.device)
rank = paddle.distributed.get_rank()
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()

set_seed(args.seed)

label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")

label_norm_dict = None
with open(label_normalize_json, 'r', encoding="utf-8") as f:
    label_norm_dict = json.load(f)

convert_example_fn = convert_example if args.task_name != "chid" else convert_chid_example
evaluate_fn = do_evaluate if args.task_name != "chid" else do_evaluate_chid

'''
train_ds, dev_ds, public_test_ds = load_dataset("fewclue",
                                                    name=args.task_name,
                                                    splits=("train_0", "dev_0",
                                                            "test_public"))
'''
train_ds = load_dataset(read_fn ,data_file \
                                = "ethos_unique_train.tsv", lazy = False)
dev_ds = load_dataset(read_fn ,data_file \
                                = "ethos_unique_dev.tsv", lazy = False)
public_test_ds = load_dataset(read_fn ,data_file \
                                = "ethos_unique_test.tsv", lazy = False)


# Task related transform operations, eg: numbert label -> text_label, english -> chinese
transform_fn = partial(transform_ethos,
                           label_normalize_dict=label_norm_dict)

# Some fewshot_learning strategy is defined by transform_fn
# Note: Set lazy=True to transform example inplace immediately,
# because transform_fn should only be executed only once when
# iterate multi-times for train_ds
train_ds = train_ds.map(transform_fn, lazy=False)
dev_ds = dev_ds.map(transform_fn, lazy=False)
public_test_ds = public_test_ds.map(transform_fn, lazy=False)

model = ErnieForPretraining.from_pretrained('ernie-3.0-medium-zh')
tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

if args.task_name != "chid":
    # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
    batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
        ): [data for data in fn(samples)]
else:
    # [src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_labels_ids]
    batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
            Stack(dtype="int64"
                  ),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

trans_func = partial(convert_example_fn,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         p_embedding_num=args.p_embedding_num)

train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

dev_data_loader = create_dataloader(dev_ds,
                                        mode='eval',
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

public_test_data_loader = create_dataloader(public_test_ds,
                                                mode='eval',
                                                batch_size=args.batch_size,
                                                batchify_fn=batchify_fn,
                                                trans_fn=trans_func)

if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
    state_dict = paddle.load(args.init_from_ckpt)
    model.set_dict(state_dict)
    print("warmup from:{}".format(args.init_from_ckpt))

mlm_loss_fn = ErnieMLMCriterion()
rdrop_loss = paddlenlp.losses.RDropLoss()

num_training_steps = len(train_data_loader) * args.epochs

lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

global_step = 0
tic_train = time.time()
#### default 10 epochs
for epoch in range(1, args.epochs + 1):
    model.train()
    for step, batch in enumerate(train_data_loader, start=1):

        src_ids = batch[0]
        token_type_ids = batch[1]
        masked_positions = batch[2]
        masked_lm_labels = batch[3]

        prediction_scores = model(input_ids=src_ids,
                                      token_type_ids=token_type_ids,
                                      masked_positions=masked_positions)

        if args.rdrop_coef > 0:
            prediction_scores_2 = model(input_ids=src_ids,
                                            token_type_ids=token_type_ids,
                                            masked_positions=masked_positions)
            ce_loss = (
                    mlm_loss_fn(prediction_scores, masked_lm_labels) +
                    mlm_loss_fn(prediction_scores_2, masked_lm_labels)) * 0.5
            kl_loss = rdrop_loss(prediction_scores, prediction_scores_2)
            loss = ce_loss + kl_loss * args.rdrop_coef
        else:
            loss = mlm_loss_fn(prediction_scores, masked_lm_labels)

        global_step += 1
        if global_step % 10 == 0 and rank == 0:
            print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, 10 /
                       (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

    dev_accuracy, total_num = evaluate_fn(model, tokenizer, dev_data_loader,
                                              label_norm_dict)
    print("epoch:{}, dev_accuracy:{:.3f}, total_num:{}".format(
            epoch, dev_accuracy, total_num))
    test_accuracy, total_num = evaluate_fn(model, tokenizer,
                                               public_test_data_loader,
                                               label_norm_dict)
    print("epoch:{}, test_accuracy:{:.3f}, total_num:{}".format(
            epoch, test_accuracy, total_num))

    if rank == 0:
        save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_param_path = os.path.join(save_dir, 'model_state.pdparams')
        paddle.save(model.state_dict(), save_param_path)
        tokenizer.save_pretrained(save_dir)

##### 10 epochs
##### 6 categories random as 16 (train size 120, test dev as 40)
##### max 53
##### incr with more epochs (16, 16) -> (20, 12) -> (35, 25) >> -> (65, 53)

##### 6 categories random as 16 (train size 160, test dev as 20)
##### not incr perform worse.

tokenizer = AutoTokenizer.from_pretrained("checkpoint/model_1500/")

save_param_path = os.path.join("checkpoint/model_1500/", "model_state.pdparams")
assert os.path.exists(save_param_path)
model_state_dict = paddle.load(save_param_path)
model = ErnieForPretraining.from_pretrained('ernie-3.0-medium-zh')
model.load_dict(model_state_dict)
model.eval()

def transform_fn_eval(example):
    example["label_length"] = 2
    return example

with open("label_normalized/ethos_unique_zh.json", "r") as f:
    label_normalize_dict = json.load(f)

normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]
label_length = len(normed_labels[0])

normed_labels
'''
['性别', '种族', '取向', '宗教', '残疾', '民族']
'''

##### prediction scope
input_dict =  {'sentence1': '同性恋真恶心',}
src_ids, token_type_ids, mask_positions = convert_example(
    transform_fn_eval(
       input_dict
    ),
    tokenizer
    , is_test = True)
src_ids, token_type_ids, mask_positions = list(
    map(lambda x: x ,
        map(paddle.to_tensor,
            map(lambda y: [y] ,[src_ids, token_type_ids, mask_positions])
                        )
       ))

prediction_probs = model.predict(input_ids=src_ids,
                                         token_type_ids=token_type_ids,
                                         masked_positions=mask_positions)

batch_size = len(src_ids)
vocab_size = prediction_probs.shape[1]

# prediction_probs: [batch_size, label_lenght, vocab_size]
prediction_probs = paddle.reshape(prediction_probs,
                                          shape=[batch_size, -1,
                                                 vocab_size]).numpy()

# [label_num, label_length]
label_ids = np.array(
            [tokenizer(label)["input_ids"][1:-1] for label in normed_labels])

y_pred = np.ones(shape=[batch_size, len(label_ids)])

# Calculate joint distribution of candidate labels
for index in range(label_length):
    y_pred *= prediction_probs[:, index, label_ids[:, index]]

# Get max probs label's index
y_pred_index = np.argmax(y_pred, axis=-1)

y_pred_index
'''
[2]
'''
