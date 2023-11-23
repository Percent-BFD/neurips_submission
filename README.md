# NeurIPS Competition Submission

Our team name is Percent_bfd.  
This repo contains all the code for the 'NeurIPS Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day' competition, including model training reproduction and submission evaluation.


## Submission Evaluation

We provide 2 submissions to run eval, both for the A100 tracks.

The folder 'neurips_submission_1' contains the eval docker file for the first submission. And the model weights are uploaded to huggingface [Percent-BFD/nips_qwen14b_lora_v9](https://huggingface.co/Percent-BFD/nips_qwen14b_lora_v9).

The folder 'neurips_submission_2' contains the eval docker file for the second submission. And the model weights are uploaded to huggingface [Percent-BFD/nips_qwen14b_lora_v7](https://huggingface.co/Percent-BFD/nips_qwen14b_lora_v7).


## Model Reproduction

The folder 'docker_train' contains the train docker file to reproduce the model. Additionally, the datasets preparation scripts can be found in its subfolder.

Our training datasets is uploaded to huggingface [Percent-BFD/nips_data_v7](https://huggingface.co/datasets/Percent-BFD/nips_data_v7).