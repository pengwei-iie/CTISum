# BertSumExt

## Process Data
**Python version**: This code is in Python3.8

**Package Requirements**: torch==2.1.2 pytorch_transformers==1.2.0 tensorboardX==2.6.2.2 multiprocess==0.70.15 pyrouge==0.1.3

All code only supports running on Linux.

## Process Data

### Step 1. Convert general/attack csv file to txt file
```
python segment_data.py    -raw_path ./raw_general/raw_generl    -save_path ./raw_general/per_general
```
```
python segment_data.py    -raw_path ./raw_attack/raw_attack     -save_path ./raw_attack/raw_attack
```

### Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-4.5.5/stanford-corenlp-4.5.5.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.5.5` directory. 


### Step 3. Sentence Splitting and Tokenization
```
python preprocess.py    -mode tokenize_attack      -raw_path ./raw_general/per_general     -save_path ./raw_general/tokenize
```
```
python preprocess.py    -mode tokenize_attack      -raw_path ./raw_attack/per_attack       -save_path ./raw_attack/tokenize
```

### Step 4. Format to Simpler Json Files
```
python preprocess.py    -mode format_general_to_lines      -use_bert_basic_tokenizer false     -n_cpus 1     -raw_path ./raw_general/tokenize    -save_path ./raw_general/json_data
```
```
python preprocess.py    -mode format_attack_to_lines      -use_bert_basic_tokenizer false     -n_cpus 1     -raw_path ./raw_attack/tokenize    -save_path ./raw_attack/json_data
```

### Step 5. Format to PyTorch Files
```
python preprocess.py    -mode format_to_bert      -raw_path  ./raw_general/json_data     -save_path  ./raw_general/bert_data     -lower    -n_cpus 1     -log_file ./logs/preprocess.log
```
```
python preprocess.py    -mode format_to_bert      -raw_path  ./raw_attack/json_data      -save_path  ./raw_attack/bert_data      -lower    -n_cpus 1     -log_file ./logs/preprocess.log
```

## Model Training
```
python train.py    -task ext    -mode train    -bert_data_path ./raw_general/bert_data    -model_path ./models/general_models    -log_file ./logs/ext_bert_attack    -ext_dropout 0.1    -report_every 10    -save_checkpoint_steps 10    -batch_size 2    -train_steps 10   -accum_count 1    -warmup_steps 1    -lr 5e-5    -use_interval true    -max_pos 512    -visible_gpus -1
```
```
python train.py    -task ext    -mode train    -bert_data_path ./raw_attack/bert_data    -model_path ./models/attack_models    -log_file ./logs/ext_bert_attack    -ext_dropout 0.1    -report_every 10    -save_checkpoint_steps 10    -batch_size 2    -train_steps 10   -accum_count 1    -warmup_steps 1    -lr 5e-5    -use_interval true    -max_pos 512    -visible_gpus -1

```

## Model Evaluation
```
python train.py    -task ext   -mode validate    -bert_data_path ./raw_general/bert_data    -model_path ./models/general_models      -result_path ./results/ext_bert_general    -temp_dir ./temp    -log_file ./logs/val_ext_bert_general    -sep_optim true     -use_interval true    -visible_gpus -1     -max_pos 512    -max_length 200     -alpha 0.95     -min_length 50    -batch_size 2    -test_batch_size 2
```
```
python train.py    -task ext   -mode validate    -bert_data_path ./raw_attack/bert_data    -model_path ./models/attack_models      -result_path ./results/ext_bert_general    -temp_dir ./temp    -log_file ./logs/val_ext_bert_attack    -sep_optim true     -use_interval true    -visible_gpus -1     -max_pos 512    -max_length 200     -alpha 0.95     -min_length 50    -batch_size 2    -test_batch_size 2
```

