from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import os
import io
import random
import tqdm
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode,encoding='utf-8')
    return f
#
def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default,ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f
def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
result_dataset = []

mmlu = []
mmlu_philosophy = load_dataset('cais/mmlu',name = 'philosophy',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_biology = load_dataset('cais/mmlu',name = 'high_school_biology',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_chemistry = load_dataset('cais/mmlu',name = 'high_school_chemistry',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_computer_science = load_dataset('cais/mmlu',name = 'high_school_computer_science',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_european_history = load_dataset('cais/mmlu',name = 'high_school_european_history',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_geography = load_dataset('cais/mmlu',name = 'high_school_geography',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_government_and_politics = load_dataset('cais/mmlu',name = 'high_school_government_and_politics',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_macroeconomics = load_dataset('cais/mmlu',name = 'high_school_macroeconomics',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_mathematics = load_dataset('cais/mmlu',name = 'high_school_mathematics',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_microeconomics = load_dataset('cais/mmlu',name = 'high_school_microeconomics',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_physics = load_dataset('cais/mmlu',name = 'high_school_physics',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_psychology = load_dataset('cais/mmlu',name = 'high_school_psychology',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_statistics = load_dataset('cais/mmlu',name = 'high_school_statistics',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_us_history = load_dataset('cais/mmlu',name = 'high_school_us_history',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_high_school_world_history = load_dataset('cais/mmlu',name = 'high_school_world_history',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_moral_disputes = load_dataset('cais/mmlu',name = 'moral_disputes',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu_moral_scenarios = load_dataset('cais/mmlu',name = 'moral_scenarios',split='auxiliary_train', cache_dir='./dataset_dir')
mmlu.append(mmlu_philosophy)
mmlu.append(mmlu_high_school_biology)
mmlu.append(mmlu_high_school_chemistry)
mmlu.append(mmlu_high_school_computer_science)
mmlu.append(mmlu_high_school_european_history)
mmlu.append(mmlu_high_school_geography)
mmlu.append(mmlu_high_school_government_and_politics)
mmlu.append(mmlu_high_school_macroeconomics)
mmlu.append(mmlu_high_school_mathematics)
mmlu.append(mmlu_high_school_microeconomics)
mmlu.append(mmlu_high_school_physics)
mmlu.append(mmlu_high_school_psychology)
mmlu.append(mmlu_high_school_statistics)
mmlu.append(mmlu_high_school_us_history)
mmlu.append(mmlu_high_school_world_history)
mmlu.append(mmlu_moral_disputes)
mmlu.append(mmlu_moral_scenarios)


#处理mmlu数据集的格式
for dataset in mmlu:
    for item in dataset:
        insturction='Question: This question refers to the following information.\n'
        choices=''
        for idx,choice in enumerate(item['choices']):
            choices+=chr(ord('A')+idx)+'.'+choice+'\n'
        choices+='Answer:'
        result_dataset.append({
            'instruction':insturction,
            'input':item['question']+'\n'+choices,
            'output':chr(ord('A')+item['answer']),
            'data_source':'mmlu'
        })
result_dataset = random.sample(result_dataset,int(len(result_dataset)/10))
#
bigbench = []
bigbench_analytic_entailment = load_dataset('tasksource/bigbench',name = 'analytic_entailment', cache_dir='./dataset_dir')
bigbench_causal_judgment = load_dataset('tasksource/bigbench',name = 'causal_judgment', cache_dir='./dataset_dir')
bigbench_emoji_movie = load_dataset('tasksource/bigbench',name = 'emoji_movie', cache_dir='./dataset_dir')
bigbench_empirical_judgments = load_dataset('tasksource/bigbench',name = 'empirical_judgments', cache_dir='./dataset_dir')
bigbench_known_unknowns = load_dataset('tasksource/bigbench',name = 'known_unknowns', cache_dir='./dataset_dir')
bigbench_logical_deduction = load_dataset('tasksource/bigbench',name = 'logical_deduction', cache_dir='./dataset_dir')
bigbench_strange_stories = load_dataset('tasksource/bigbench',name = 'strange_stories', cache_dir='./dataset_dir')
bigbench_snarks = load_dataset('tasksource/bigbench',name = 'snarks', cache_dir='./dataset_dir')
bigbench_dark_humor_detection = load_dataset('tasksource/bigbench',name = 'dark_humor_detection', cache_dir='./dataset_dir')
bigbench.append(bigbench_analytic_entailment)
bigbench.append(bigbench_causal_judgment)
bigbench.append(bigbench_emoji_movie)
bigbench.append(bigbench_empirical_judgments)
bigbench.append(bigbench_known_unknowns)
bigbench.append(bigbench_logical_deduction)
bigbench.append(bigbench_strange_stories)
bigbench.append(bigbench_snarks)
bigbench.append(bigbench_dark_humor_detection)

#处理bigbench数据的格式
for dataset in bigbench:
    if len(dataset)>3:
        for item in dataset:
            insturction = 'Question: \n'
            choices = ''
            for idx, choice in enumerate(item['multiple_choice_targets']):
                choices += chr(ord('A') + idx) + '.' + choice + '\n'
            choices += 'Answer:'
            result_dataset.append({
                'instruction': insturction,
                'input': item['inputs'] + '\n' + choices,
                'output': chr(ord('A') + item['multiple_choice_targets'].index(item['targets'][0])),
                'data_source': 'bigbench'
            })
    else:
        for index in dataset:
            for item in dataset[index]:
                insturction = 'Question: \n'
                choices = ''
                for idx, choice in enumerate(item['multiple_choice_targets']):
                    choices += chr(ord('A') + idx) + '.' + choice + '\n'
                choices += 'Answer:'
                result_dataset.append({
                    'instruction': insturction,
                    'input': item['inputs'] + '\n' + choices,
                    'output': chr(ord('A') + item['multiple_choice_targets'].index(item['targets'][0])),
                    'data_source': 'bigbench'
                })
print(len(result_dataset))
#
# other_data = []
#
truthful_qa = load_dataset('truthful_qa',name = 'multiple_choice', cache_dir='./dataset_dir')
if len(truthful_qa)>3:
    for item in truthful_qa:
        insturction = 'Question: \n'
        choices = ''
        for idx, choice in enumerate(item['mc1_targets']['choices']):
            choices += chr(ord('A') + idx) + '.' + choice + '\n'
        choices += 'Answer:'
        result_dataset.append({
            'instruction': insturction,
            'input': item['question'] + '\n' + choices,
            'output': chr(ord('A') + item['mc1_targets']['labels'].index(1)),
            'data_source': 'truthful_qa'
        })
else:
    for index in truthful_qa:
        for item in truthful_qa[index]:
            insturction = 'Question: \n'
            choices = ''
            for idx, choice in enumerate(item['mc1_targets']['choices']):
                choices += chr(ord('A') + idx) + '.' + choice + '\n'
            choices += 'Answer:'
            result_dataset.append({
                'instruction': insturction,
                'input': item['question'] + '\n' + choices,
                'output': chr(ord('A') + item['mc1_targets']['labels'].index(1)),
                'data_source': 'truthful_qa'
            })

print(len(result_dataset))
temp_result_dataset = []
dailymail = load_dataset('cnn_dailymail',name = '3.0.0',split='train', cache_dir='./dataset_dir')
for item in dailymail:
    insturction = '	### Article: \n'
    temp_result_dataset.append({
        'instruction': insturction,
        'input': item['article'] + '\n',
        'output': 'Summarize the above article in 3 sentences.' + item['highlights'],
        'data_source': 'dailymail'
    })
temp_result_dataset = random.sample(temp_result_dataset,int(0.1*len(temp_result_dataset)))
result_dataset.extend(temp_result_dataset)

print(len(result_dataset))
gsm = load_dataset('gsm8k',name='main',split='train', cache_dir='./dataset_dir')
if len(gsm)<=3:
    for index in gsm:
        for item in gsm[index]:
            insturction = 'Q: \n'
            result_dataset.append({
                'instruction': insturction,
                'input': item['question'] + '\n',
                'output': 'A:' + item['answer'],
                'data_source': 'gsm'
            })
else:
    for item in gsm:
        insturction = 'Q: \n'
        result_dataset.append({
            'instruction': insturction,
            'input': item['question'] + '\n',
            'output': 'A:' + item['answer'],
            'data_source': 'gsm'
        })

print(len(result_dataset))
# bbq = load_dataset('lighteval/bbq_helm',name = 'all',split='test', cache_dir='./dataset_dir')
# for item in bbq:
#     insturction = 'The following are multiple choice questions (with answers). \n'
#     choices = ''
#     for idx, choice in enumerate(item['choices']):
#         choices += chr(ord('A') + idx) + '.' + choice + '\n'
#     choices += 'Answer:'
#     result_dataset.append({
#         'instruction': insturction,
#         'input': 'Passage:' + item['context'] + '\n' + item['question'] + '\n' + choices,
#         'output': chr(ord('A') + item['gold_index']),
#         'data_source': 'bbq'
#     })
#
# print(len(result_dataset))
lima = load_dataset('GAIR/lima', cache_dir='./dataset_dir')
#这个数据集测试集没有标注，因此只用训练集来训练
for i in range(10):
    if len(lima)>3:
        for item in lima:
            insturction = ''
            # 多轮对话大概有20条，我们把他们也变成单轮的。
            for i in range(len(item['conversations'])):
                result_dataset.append({
                    'instruction': insturction,
                    'input': 'Question:' + item['conversations'][2 * i] + '\n',
                    'output': 'Answer:' + item['conversations'][2 * i + 1],
                    'data_source': 'lima'
                })
    else:
        for item in lima['train']:
            insturction = ''
            #多轮对话大概有20条，我们把他们也变成单轮的。
            for i in range(int(len(item['conversations'])/2)):
                result_dataset.append({
                    'instruction': insturction,
                    'input': 'Question:' + item['conversations'][2*i] + '\n',
                    'output': 'Answer:' + item['conversations'][2*i+1],
                    'data_source': 'lima'
                })


databricks_dolly = load_dataset('databricks/databricks-dolly-15k', cache_dir='./dataset_dir')
if len(databricks_dolly)>3:
    for item in databricks_dolly:
        insturction = item['instruction'] + '\n'
        result_dataset.append({
            'instruction': insturction,
            'input': item['context'] + '\n',
            'output': item['response'],
            'data_source': 'databricks_dolly'
        })
else:
    for index in databricks_dolly:
        for item in databricks_dolly[index]:
            insturction = item['instruction'] + '\n'
            result_dataset.append({
                'instruction': insturction,
                'input': item['context'] + '\n',
                'output': item['response'],
                'data_source': 'databricks_dolly'
            })
print(len(result_dataset))
#oasst1这里面包含多个语种的，因为我只想训练英文的，所以找了个只有英文的oasst1
oasst1 = load_dataset('OpenAssistant/oasst1', cache_dir='./dataset_dir')
oasst1_en = load_dataset('Photolens/oasst1-en', cache_dir='./dataset_dir')
for index in oasst1_en:
    for item in oasst1_en[index]:
        if len(item['messages']) != 2:
            continue
        insturction = ''
        result_dataset.append({
            'instruction': insturction,
            'input': item['messages'][0]['content'] + '\n',
            'output': item['messages'][1]['content'],
            'data_source': 'oasst1_en'
        })


# print(len(result_dataset))
# #相比于原数据而言，把title什么的过滤掉
summarize_from_feedback_tldr_3_filtered = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', cache_dir='./dataset_dir')
if len(summarize_from_feedback_tldr_3_filtered)<=3:
    for index in summarize_from_feedback_tldr_3_filtered:
        for item in summarize_from_feedback_tldr_3_filtered[index]:
            insturction = 'Summarize: \n'
            result_dataset.append({
                'instruction': insturction,
                'input': item['post'] + '\n',
                'output': 'A:' + item['summary'],
                'data_source': 'MetaMathQA'
            })
else:
    for item in summarize_from_feedback_tldr_3_filtered:
        insturction = 'Summarize: \n'
        result_dataset.append({
            'instruction': insturction,
            'input': item['post'] + '\n',
            'output': 'A:' + item['summary'],
            'data_source': 'MetaMathQA'
        })
random.shuffle(result_dataset)
result_dataset = random.sample(result_dataset,int(len(result_dataset)*0.67))
jdump(result_dataset,'../LLaMA-Effcient-Tuning/data_for_fintune/nips_data_add_v7.json')
print(len(result_dataset))