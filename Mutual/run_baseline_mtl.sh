export CUDA_VISIBLE_DEVICES="0"
python run_MDFN_mtl.py \
--data_dir datasets/mutual \
--model_name_or_path bert-base-uncased \
--model_type bert \
--task_name mutual \
--output_dir experiments/mutual_bert_base_baseline_mtl \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 6 \
--eval_batch_size 6 \
--learning_rate 4e-6 \
--num_train_epochs 3 \
--gradient_accumulation_steps 1 \
--local_rank -1