from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("Percent-BFD/nips_data_v7")
print(dataset)

def list_dict_to_json(input,output_path):
    df = pd.DataFrame(input)
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    return


list_dict_to_json(dataset['train'],'../LLaMA-Effcient-Tuning/data_for_fintune/nips_data_add_v7.json')
