# Reproduction for submission
Here we choose to reproduce the model in neurips_submission_2.

# How to Reproduce it

1. Training Data

We uploaded the processed data to hugging face, so you to download the training data from our hugging face space:  
```
cd ./data-prepare && python get_dataset.py && cd ..
```

Also if you want to reproduce the production of data, then the entire process of downloading open source data and processing data is:
```
cd ./data-prepare && python generate_dataset.py && cd ..
```
Note: It's not recommanded to do so bacuase it will take a few hours.



2. Train

Use Dockerfile.train directly, or:

```
sh ./LLaMA-Effcient-Tuning/train_qwen.sh
```

When the training is completed, the result file of the lora model is in ```./output```.

3. Inference

Modify line 26 of the file neurips_submission_2/main.py in our github, and set lora_path to the output in step 2.