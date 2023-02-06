# MultiESC: Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning
This repository contains the codes and data used in our EMNLP'2022 paper: [Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning](https://arxiv.org/abs/2210.04242).

## Data and Environment Setup
### Environment
Refer to new_torch15.yml to set up the environment. You could use the following command.

```bash
conda env create -f new_torch15.yml 
```

### Data
All the data are provided in ./MultiESC/data/{train, valid, test}.txt. 

### Download the pretrained models

bert-base:  [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)

bart-base:  [https://huggingface.co/facebook/bart-base](https://huggingface.co/facebook/bart-base)

Download them and save in the folder ./MODEL
### Download metric code

Link:  https://drive.google.com/file/d/1AFE2B7dYw9mU4rLEN4k7BMrtOxIlhXYh/view?usp=sharing

Download it and save in the folder ./MultiESC/metric

## Implementation Process

1. Train the strategy sequence predictor. The model will be saved in ./final_output/bart_output.

```bash
CUDA_VISIBLE_DEVICES=0,1 python generate_strategy_norm_train.py --data_type=3 --model_type=1  --output_dir=./final_output/bart_output  --learning_rate=2e-5  --num_train_epochs=15 --lr2=2e-5 --with_cause --with_strategy
```

2. Data augmentation for training the feedback predictor. The augmented data will be save in ./final_data/{train， valid,test}_extend_beam*.pk.

```bash
CUDA_VISIBLE_DEVICES=0,1 python generate_strategy_test.py --data_type=3 --model_type=1  --output_dir=./output --saved_dir=./final_output/bart_output  --learning_rate=2e-5  --num_train_epochs=15 --lr2=2e-5 --with_cause --with_strategy
```

3. Train the feedback predictor. The model will be saved in ./final_output/feedback_model.

```bash
CUDA_VISIBLE_DEVICES=0,2 python feedback_model_train.py
```

4. Use our proposed lookahead strategy planning method to predict the strategy for the upcoming turn. The output will be saved in ./final_data/multiesc_predicted_strategy.pk.

```bash
CUDA_VISIBLE_DEVICES=0 python3 get_lookahead_strategy.py
```

5. Generate the responses, using the predicted strategies. The model will be saved in ./final_output/whlookahead_generate.

```bash
CUDA_VISIBLE_DEVICES=0,1 python generate_sentence.py  --data_type=8  --output_dir=./final_output/whlookahead_generate  --learning_rate=5e-5 --lr2=1e-4 --num_train_epochs=15  --with_cause --with_strategy --model_type=1 --lookahead
```


## Cite
If you use our codes or your research is related to our work, please kindly cite our paper:
```bib
@inproceedings{cheng-etal-2022-improving,
    title = "Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning",
    author = "Cheng, Yi  and
      Liu, Wenge  and
      Li, Wenjie  and
      Wang, Jiashuo  and
      Zhao, Ruihui  and
      Liu, Bang  and
      Liang, Xiaodan  and
      Zheng, Yefeng",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    pages = "3014--3026",
}
```
