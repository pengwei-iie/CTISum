# BART-large Model


python BART-large.py    --model_name_or_path bart-large-cnn    --train_file data/general/train.csv    --validation_file data/general/test.csv    --source_prefix "summarize: "    --output_dir tmp/tst-summarization --per_device_train_batch_size=2    --per_device_eval_batch_size=2    --text_column text    --summary_column summary    --with_tracking    --num_beams=4   
