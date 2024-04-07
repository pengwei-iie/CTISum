# MatchSum

## Dependencies
- Python 3.7
- [torch](https://github.com/pytorch/pytorch) 1.4.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 0.5.0
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
- [rouge](https://github.com/pltrdy/rouge) 1.0.1
- [transformers](https://github.com/huggingface/transformers) 4.30.2

	
All code only supports running on Linux.

## Process Data

### Step 1. Convert general/attack dataset to the *jsonl* format
First you need to convert general/attack dataset to the *jsonl* format, and make sure to include *text* and *summary* fields. Second, you should use BertExt or other methods to select some important sentences from each document and get an *index.jsonl* file.
```
python data_preprocess.py    -mode format_general_to_data_and_index    -use_bert_basic_tokenizer false    -n_cpus 1    -raw_path ./data/general/tokenize    -save_path ./data/general/jsonl_data 
```
```
python data_preprocess.py    -mode format_attack_to_data_and_index    -use_bert_basic_tokenizer false    -n_cpus 1    -raw_path ./data/attack/tokenize    -save_path ./data/attack/jsonl_data 
```


### Step 2. get candidate summaries for each documen
```
python get_candidate.py    --tokenizer=bert    --data_path ./data/general/jsonl_data/data    --index_path ./data/general/jsonl_data/index    --write_path ./data/general/jsonl_data/processed_data
```
```
python get_candidate.py    --tokenizer=bert    --data_path ./data/attack/jsonl_data/data    --index_path ./data/attack/jsonl_data/index    --write_path ./data/attack/jsonl_data/processed_data
```
## Model Training
```
CUDA_VISIBLE_DEVICES=0    python train_matching.py    --mode=train    --encoder=bert    --raw_path=./data/general/jsonl_data/processed_data    --save_path ./bert_general    --gpus=0
```
```
CUDA_VISIBLE_DEVICES=0    python train_matching.py    --mode=train    --encoder=bert    --raw_path=./data/attack/jsonl_data/processed_data    --save_path ./bert_attack    --gpus=0
```
## Model Evaluation
```
CUDA_VISIBLE_DEVICES=0    python train_matching.py    --mode=test    --encoder=bert    --raw_path=./data/general/jsonl_data/processed_data    --save_path=./bert_general/2024-01-19-19-28-14    --gpus=0
```
```
CUDA_VISIBLE_DEVICES=0    python train_matching.py    --mode=test    --encoder=bert    --raw_path=./data/attack/jsonl_data/processed_data    --save_path=./bert_attack/2024-01-20-08-30-08    --gpus=0
```
## Note
*The code and data released here are used for the matching model. Before the matching stage, we use BertExt to prune meaningless candidate summaries
