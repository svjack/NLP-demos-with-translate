#### data prepare scope

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

if __name__ == "__main__":
    trans_model = EasyNMT('opus-mt')
    trans_model.translate(
        "Who are you ?", source_lang="en", target_lang = "zh"
    )

    pool = trans_model.start_multi_process_pool(["cpu"] * 10)

    print(len(all_eng_text_list))
    req = all_eng_text_list
    trans_list = trans_model.translate_multi_process(pool ,req,
           source_lang="en", target_lang = "zh")

    trans_model.stop_multi_process_pool(pool)
    '''
    req = all_eng_text_list
    trans_list = trans_model.translate(req,
           source_lang="en", target_lang = "zh")
    '''

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

    train_df = pd.DataFrame(list(dataset["train"]))
    unique_train_df = train_df[
        train_df.apply(
            lambda x: x.iloc[:-1].sum() == 1,axis = 1
        )
    ]

    #### drop tiny category
    req_features = unique_train_df.iloc[:, :-1].sum(axis = 0).sort_values(ascending = False).iloc[:-2].index.tolist()
    req_features
    '''
    ['race',
     'religion',
     'gender',
     'national_origin',
     'sexual_orientation',
     'disability']
    '''

    unique_train_df = unique_train_df[req_features + ["text"]]
    dict(map(lambda x: (x, en_zh_dict[x]), req_features))
    '''
    {'race': '种族',
     'religion': '宗教、',
     'gender': '性别',
     'national_origin': '民族原籍',
     'sexual_orientation': '性取向和',
     'disability': '残疾'}
    '''
    #### to character length 2
    '''
    {'race': '种族',
     'religion': '宗教',
     'gender': '性别',
     'national_origin': '民族',
     'sexual_orientation': '取向',
     'disability': '残疾'}
     '''

    unique_train_df["cate_index"] = unique_train_df.apply(
        lambda x: x.iloc[:-1].map(int).tolist().index(1) if 1 in x.iloc[:-1].map(int).tolist() else np.nan, axis = 1
    )
    unique_train_df = unique_train_df.dropna()
    unique_train_df["cate_index"] = unique_train_df["cate_index"].map(int)
    unique_train_df["cate"] = unique_train_df["cate_index"].\
    map(lambda x: unique_train_df.columns.tolist()[x])

    d = {'race': '种族',
     'religion': '宗教',
     'gender': '性别',
     'national_origin': '民族',
     'sexual_orientation': '取向',
     'disability': '残疾'}

    dict(map(lambda t2: (str(t2[0]), d[t2[1]]) ,unique_train_df[["cate_index" ,"cate"]].values.tolist()))
    '''
    {'2': '性别', '0': '种族', '4': '取向', '1': '宗教', '5': '残疾', '3': '民族'}
    '''
    en_d = {'race': '种族',
     'religion': '宗教',
     'gender': '性别',
     'national_origin': '民族',
     'sexual_orientation': '取向',
     'disability': '残疾'}

    #### https://github.com/PaddlePaddle/PaddleNLP/tree/0a618c70f95eeea29ac084d6cf16d26fad289dd5/examples/few_shot/p-tuning
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

    #### ptuning.py
    #### transform_fn_dict use label_norm_dict

    #### data.py
    #### transform_fn () functions (transform_bustm)
    '''
    def transform_bustm(example, label_normalize_dict=None, is_test=False):
        if is_test:
            # Label: ["很"， "不"]
            example["label_length"] = 1
            return example
        else:
            origin_label = str(example["label"])

            # Normalize some of the labels, eg. English -> Chinese
            example['text_label'] = label_normalize_dict[origin_label]

            del example["label"]

            return example
    '''

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

    unique_train_df.to_csv("ethos_unique.csv", index = False)

    #### dataset format req
    unique_train_df[["text", "cate_index"]].applymap(str).to_csv("ethos_unique.tsv",
                                        sep = "\t", header = None, index = False)
    #### !head -n 3 ethos_unique.tsv
    from paddlenlp.datasets import load_dataset
    ethos_load_local = load_dataset(read_fn ,data_file \
                                    = "ethos_unique.tsv", lazy = False)

    unique_train_df_list = np.array_split(
        unique_train_df[["text", "cate_index"]].applymap(str).sample(frac = 1.0),
        10
    )
    assert len(unique_train_df_list) == 10

    #### 120 -> 40 -> 39
    train_ = pd.concat(unique_train_df_list[:6], axis = 0)
    dev_ = pd.concat(unique_train_df_list[6:8], axis = 0)
    test_ = pd.concat(unique_train_df_list[8:], axis = 0)
    '''
    #### 160 -> 20 -> 19
    train_ = pd.concat(unique_train_df_list[:8], axis = 0)
    dev_ = pd.concat([unique_train_df_list[8]], axis = 0)
    test_ = pd.concat([unique_train_df_list[9]], axis = 0)
    list(map(len, [train_, dev_, test_]))
    '''

    train_.to_csv("ethos_unique_train.tsv",
                    sep = "\t", header = None, index = False)
    dev_.to_csv("ethos_unique_dev.tsv",
                    sep = "\t", header = None, index = False)
    test_.to_csv("ethos_unique_test.tsv",
                    sep = "\t", header = None, index = False)

    train_ds = load_dataset(read_fn ,data_file \
                                    = "ethos_unique_train.tsv", lazy = False)
    dev_ds = load_dataset(read_fn ,data_file \
                                    = "ethos_unique_dev.tsv", lazy = False)
    public_test_ds = load_dataset(read_fn ,data_file \
                                    = "ethos_unique_test.tsv", lazy = False)

    import shutil
    import json
    if os.path.exists("label_normalized"):
        shutil.rmtree("label_normalized")
    os.mkdir("label_normalized")
    with open("label_normalized/ethos_unique_zh.json", "w", encoding = "utf-8") as f:
        json.dump(
            dict(map(lambda t2: (str(t2[0]), d[t2[1]]) ,unique_train_df[["cate_index" ,"cate"]].values.tolist()))
        , f)
