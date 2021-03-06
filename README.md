

Codes for the paper **Structural Pre-training for Dialogue Comprehension** 

### Instruction 

#### Dataset

Since the datasets are quite large that exceed the Github file size limit, we only upload part of the data as examples. Do not forget to change to the data directory after you download the full data.
1. Datasets can be download from [Ubuntu dataset](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0),  [Douban dataset](https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0), and [ECD dataset](https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view?usp=sharing).
2. Unzip the dataset and put data directory into `data/`.

#### SVO pre-annotation

Use the scripts `svo_annotate.py` and `svo_combine.py` to generate the pre-annotated SVO files for quick access in model training.

*Note: To avoid being killed due to large memory use, we separete the annotation with <u>start_id</u> and <u>end_id</u> in `svo_annotate.py`, and combine those annotated files with  `svo_combine.py`. 

#### NUP pre-training 

The steps to further pre-training BERT with NUP strategy is introduced as follows. We also provide the language model trained on Ubuntu training set. Our trained nup language model on Ubuntu training set can be accessed here. 

https://www.dropbox.com/s/d1earb9ta6drqoy/ubuntu_nup_bert_base.zip?dl=0

You can unzip the model and put it into `ubuntu_nup_bert_base` directory then use it during model training.

The following scripts take the sbr objective training as example.

1. Run `make_lm_data.py` to process the original training data format into a single file with one sentence(utterance) per line, and one blank line between documents(dialog context).  

```
python nup_lm_finetuning/make_lm_data.py \
--data_file ../data/ubuntu_data/train.txt \
--output_file data/ubuntu_data/lm_train.txt
```

2. Use `pregenerate_training_data_sbr.py` to pre-process the data into training examples following the NUP methodology.

```
python nup_lm_finetuning/pregenerate_training_data_sbr.py \
--train_corpus ../data/ubuntu_data/lm_train.txt \
--bert_model bert-base-uncased \
--do_lower_case \
--output_dir ../data/ubuntu_data/ubuntu_sbr \
--epochs_to_generate 1 \
--max_seq_len 512
```

3. Train on the pregenerated data using `finetune_on_pregenerated_sbr.py`, and pointing it to the folder created by `pregenerate_training_data.py` .

```
python nup_lm_finetuning/finetune_on_pregenerated_sbr.py \
--pregenerated_data ../data/ubuntu_data/ubuntu_sbr \
--bert_model bert-base-uncased \
--train_batch_size 64 \
--do_lower_case \
--output_dir ubuntu_finetuned_lm_sbr \
--epochs 1
```

#### Model training

1. Train a model

   Change the `--bert_model` parameter to the path of the pretrained language model if need. Example as ` ubuntu_finetuned_lm` for Ubuntu dataset.

   An example:

```
python run_bert_sbr.py \
--data_dir data/ubuntu_data \
--task_name ubuntu \
--train_batch_size 64 \
--eval_batch_size 64 \
--max_seq_length 384 \
--max_utterance_num 20 
--bert_model bert-base-uncased \ 
--cache_flag ubuntu \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--do_train \   # set to do_eval when evaluation on test set
--do_lower_case \
--output_dir experiments/ubuntu
```

### Requirements

Python 3.6 + Pytorch 1.0.1 



