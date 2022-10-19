####
'''
pip install setfit==0.3.0
pip install easynmt
'''
####

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from easynmt import EasyNMT

import pandas as pd
import seaborn as sns
import numpy as np
import json

from datasets import load_dataset

import jieba
def repeat_to_one_f(x):
    req = None
    for token in jieba.lcut(x):
        #print("req :", req)

        if len(set(token)) == 1:
            token = token[0]
        if req is None:
            req = token
        else:

            if token in req:
                continue
            else:
                while req.endswith(token[0]):
                    token = token[1:]
                req = req + token
    return req.strip()

def repeat_to_one_fb(x):
    return sorted(map(repeat_to_one_f, [x, "".join(jieba.lcut(x)[::-1])]),
                 key = len
                 )[0]

repeat_to_one = repeat_to_one_fb

dataset = load_dataset("ethos", "multilabel")

ds_df = pd.DataFrame(list(dataset["train"]))
all_eng_text_list = ds_df["text"].values.tolist() + ds_df.columns.tolist()
with open("ethos_eng.json", "w") as f:
    json.dump(all_eng_text_list, f)

trans_model = EasyNMT('opus-mt')
trans_model.translate(
    "Who are you ?", source_lang="en", target_lang = "zh"
)
pool = trans_model.start_multi_process_pool(["cpu"] * 5)

print(len(all_eng_text_list))
req = all_eng_text_list
trans_list = trans_model.translate_multi_process(pool ,req,
       source_lang="en", target_lang = "zh")

trans_model.stop_multi_process_pool(pool)

ds_df = pd.DataFrame(list(zip(*[req, trans_list])))
ds_df.columns = ["en", "zh"]
ds_df["zh"] = ds_df["zh"].map(repeat_to_one)
ds_df.to_csv("ethos_en_zh.csv", index = False)

en_zh_dict = dict(ds_df.values.tolist())
#### add zh_text as field
dataset = dataset.map(
    lambda x: {"zh_text": en_zh_dict[x["text"]]}
)
dataset = dataset.remove_columns("text")
dataset = dataset.rename_column(
    "zh_text", "text"
)

features = dataset["train"].column_names
features.remove("text")
features

num_samples = 8
samples = np.concatenate(
    [
        np.random.choice(np.where(dataset["train"][f])[0], num_samples)
        for f in features
    ]
)

def encode_labels(record):
    return {"labels": [record[feature] for feature in features]}

dataset = dataset.map(encode_labels)

train_dataset = dataset["train"].select(samples)
eval_dataset = dataset["train"].select(
    np.setdiff1d(np.arange(len(dataset["train"])), samples)
)


from setfit import SetFitModel

####model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
#### "xlm-roberta-base" not cos
#### "sentence-transformer/paraphrase-multilingual-mpnet-base-v2" cos
model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest")

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitTrainer

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_epochs=1,
    num_iterations=20,
    column_mapping={"text": "text", "labels": "label"},
)

#### model.from_pretrained
model.save_pretrained("ethos_zh_model")

for_pred = [
        "Jewish people often don't eat pork.",
        "Is this lipstick suitable for people with dark skin?"
    ]
zh_for_pred = trans_model.translate(for_pred, source_lang="en", target_lang = "zh")
zh_for_pred
'''
['犹太人经常不吃猪肉', '这口红适合皮肤黑的人吗?']
'''

preds = model(
 zh_for_pred
)
preds

pd.DataFrame(preds, columns=list(map(lambda x: en_zh_dict[x], features)), index = zh_for_pred)
'''

暴力	定向 -通用	性别	种族	民族原籍	残疾	宗教、	性取向和
犹太人经常不吃猪肉	0	0	0	0	0	0	1	0
这口红适合皮肤黑的人吗?	0	1	0	1	0	0	0	0
'''
