# SetFit Text classification multilabel demo in Chinese

 [setfit](https://github.com/huggingface/setfit) use sentencetransformer in few shot learning Scene. This is a brief demo of it use in dataset translate into Chinese.

 code located in [setfit_text_classification_multilabel_zh.py](https://github.com/svjack/NLP-demos-with-translate/blob/main/setfit_text_classification_multilabel_zh.py)

 Use dataset: [ethos](https://huggingface.co/datasets/ethos) is a onlinE haTe speecH detectiOn dataSet contains a dataset for hate speech detection on social media platforms, called Ethos.

 We use the multilabel version of it to run a multilabel classification (which contain different classes as  `violence`,
 `directed_vs_generalized`,
 `gender`,
 `race`,
 `national_origin`,
 `disability`,
 `religion`,
 `sexual_orientation`)

and use [EasyNMT](https://github.com/UKPLab/EasyNMT) to translate the source text corpus into Chinese. (classes map to 暴力	定向-通用	性别	种族	民族原籍	残疾	宗教	性取向)

The model output seems reasonable after training.
