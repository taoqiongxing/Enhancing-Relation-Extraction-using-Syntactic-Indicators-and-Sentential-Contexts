论文：Enhancing Relation Extraction using Syntactic Indicators and Sentential Contexts

###训练

```shell
export BERT_BASE_DIR=/文件路径/uncased_L-24_H-1024_A-16
export GLUE_DIR=/文件路径/glue_data

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=/路径/output/
```

###测试

```shell
export BERT_BASE_DIR=/文件路径/uncased_L-24_H-1024_A-16
export GLUE_DIR=/文件路径/glue_data
export TRAINED_CLASSIFIER=/路径/output/

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/路径/output/
```
