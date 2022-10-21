# P_Tuning Text classification multilabel demo in Chinese

a Paddle demo use [P_Tuning](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot/p-tuning) about few shot learning.

## Installtion
```bash
pip install -r requirements.txt
```

<!--
 code located in [setfit_text_classification_multilabel_zh.py](https://github.com/svjack/NLP-demos-with-translate/blob/main/setfit_text_classification_multilabel_zh.py)
-->

 Use dataset: [ethos](https://huggingface.co/datasets/ethos) is a onlinE haTe speecH detectiOn dataSet contains a dataset for hate speech detection on social media platforms, called Ethos.

 We use the multilabel version of it (which contain different classes as  `violence`,
 `directed_vs_generalized`,
 `gender`,
 `race`,
 `national_origin`,
 `disability`,
 `religion`,
 `sexual_orientation`) only use (`gender`,
 `race`,
 `national_origin`,
 `disability`,
 `religion`,
 `sexual_orientation`) and samples with unique label.

use [EasyNMT](https://github.com/UKPLab/EasyNMT) to translate the source text corpus into Chinese. (classes map to 性别 种族	民族 宗教	残疾 性取向)

The model output seems reasonable after training.

For reproduce the output, cd into ethos_to_p_tuning path and run [ethos_to_p_tuning_data.py](https://github.com/svjack/NLP-demos-with-translate/blob/main/ethos_to_p_tuning/ethos_to_p_tuning_data.py) (this will produce the dataset used by PaddleNLP dataset interface), then run [ethos_to_p_tuning.py](https://github.com/svjack/NLP-demos-with-translate/blob/main/ethos_to_p_tuning/ethos_to_p_tuning.py) (will train a model with epoch num 50 and do some single sentence evaluations)
