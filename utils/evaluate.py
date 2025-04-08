# !/usr/bin/env python
# -*-coding:utf-8 -*-


import argparse
import codecs
import json
import sys
import re
import numpy as np

from tqdm import tqdm
from typing import Union

from nltk.tokenize import word_tokenize
import string

import pkg_resources
# 获取 math_evaluation 的版本号
math_evaluation_version = pkg_resources.get_distribution("math_evaluation").version
print(f"math_evaluation version: {math_evaluation_version}")
if math_evaluation_version == '0.0.1':
    # 如果版本是0.0.1，则导入旧版本的模块和函数
    from math_evaluation.core.evaluations import is_equiv
    from math_evaluation.core.preprocess import *
    from math_evaluation.core.metamath_util import is_equiv as metamath_is_equiv
elif '0.2' in math_evaluation_version:
    # 如果版本是0.2，则导入新版本的模块和函数
    from math_evaluation import is_equiv
else:
    raise ImportError("不支持的 math_evaluation 版本")

from utils import load_jsonl, write_jsonl

from transformers import AutoModelForCausalLM, AutoTokenizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def math_is_equiv(grt: Union[str, list[str]], prd: str):
  if isinstance(grt, list):
    for g in grt:
      if is_equiv(g, prd):
        return True
    return False
  else:
    return is_equiv(grt, prd)

def parse_args():
    parser = argparse.ArgumentParser(description="String Match based Eval Math Result")
    parser.add_argument("--model", type=str, default="/data/qldu/wyang/workspace/20241211_inference")
    parser.add_argument("--input_file", type=str, default="/data/qldu/wyang/workspace/20241211_inference_time_scaling/scaling_v1/process_data/inference/M0/QwQ-32B-Preview/test_default_prompt/math_500.jsonl.prediction.with_QwQ-32B-Preview.temp_0.0_top-p_1_top-k_-1_n-sample_1_max-tokens_4096_seed_42_template_QwQ_length_control_-1.jsonl", help="input file.")
    parser.add_argument("--save_file", action="store_true", help="save file.")
    parser.add_argument("--debug", type=bool, default=False, help="debug mode.")
    args = parser.parse_args()
    return args

# def extract_answer(data, prefix: str = "Final Answer: "):
#     data_split = data.split(prefix)
#     if len(data_split) < 2:
#         return "None"
#     # return data_split[-1].strip().split("</p>")[0].strip().strip("\\n").strip("$")
#     return data_split[-1].strip().split("</p>")[0].strip().strip("\\n").strip("$").split("$")[0]

# def extract_answer(text):
#     # 正则表达式匹配最后的数字答案
#     match = re.search(r'Final Answer[^\d]*([\d\.]+)', text)
#     if match:
#         return float(match.group(1))  # 提取数字并转换为浮动类型
#     else:
#         return None  # 如果没有匹配到，返回 None

def extract_answer(text):
    # fine all the matches
    match = re.findall(r'\\boxed\{(.+)\}', text)
    if len(match) > 0:
        return match[-1]
    else:
        return None

def length_word(data):
    response_len_words = []
    
    for response in tqdm(data):
        response_len_words.append(count_words(response))
    
    return response_len_words

def count_words(text) -> int:
    # Count the number of words
    # while excluding punctuations
    return len([word for word in word_tokenize(text) if word not in string.punctuation])
       
def length_token(data, tokenizer):

    response_len_tokens = []
    for response in tqdm(data):
        tokens = tokenizer.tokenize(response)
        response_len_tokens.append(len(tokens))

    return response_len_tokens

def label_preprocess(data):
    try:
        return data.split("####")[-1].strip()
    except:
        return data
    
def main(args):
    print("args: {{{")
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")
    print("}}}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.input_file, 'r') as fin:
        data_list = fin.readlines()

    right_count = 0
    total_count = 0

    total_tokens = 0

    update_data_list = []

    for idx, item_json_string in enumerate(tqdm(data_list)):
        item = json.loads(item_json_string)
        total_count += 1
        # import pdb; pdb.set_trace()
        # ref_str = item["final_answer"] if "final_answer" in item else item["answer"]
        ref_str = item["answer"]

        # if item['unique_id'] =="test/precalculus/927.json":
        #     print(item['question'])
        #     import pdb; pdb.set_trace()

        item['pred_answer'] = []
        item['verification'] = []
        

        for response in item["response"]:
            answer = extract_answer(response)

            item['pred_answer'].append(answer)
            # import pdb; pdb.set_trace()

            ref_answer = label_preprocess(ref_str)
            
            predicted_answer = answer

            # import pdb; pdb.set_trace()
            whether_right = math_is_equiv(ref_answer, predicted_answer)

            if whether_right:
                item['verification'].append("True")

            else:
                item['verification'].append("False")


        item['sampling_acc'] = item['verification'].count("True") / len(item['verification'])
        
        # item['math_eval_version'] = str(math_evaluation_version)

        if item['sampling_acc'] != 0:
            right_count += 1
        total_tokens += item["n_steps"]

        update_data_list.append(item)


    all_response  = []
    for item in update_data_list:
        for response in item['response']:
            all_response.append(response)

    new_update_data_list = []

    # import pdb; pdb.set_trace()


    if args.save_file:
        write_jsonl(update_data_list, args.input_file + ".eval.jsonl")

    right_rate = (right_count / total_count) * 100
    print(f"Total Questions: {total_count}")
    print(f"Right Answer Count: {right_count}")
    print(f"Right Rate: {right_rate:.2f}%")

    print(f"Average steps: {total_tokens / total_count:.2f}")
    # 按照level进行统计avg length, avg sampling_acc
    level_dict = {}

    # import pdb; pdb.set_trace()

    # for item in update_data_list:
    #     level = item['level']
    #     if level not in level_dict:
    #         level_dict[level] = {
    #             'level': level,
    #             'sampling_acc': [],
    #             'response_avg_len_words': [],
    #             'response_avg_len_tokens': [],
    #             'response_std_len_words': [],
    #         }

    #     level_dict[level]['sampling_acc'].append(item['sampling_acc'])
    #     level_dict[level]['response_avg_len_words'].append(item['response_avg_len_words'])
    #     level_dict[level]['response_avg_len_tokens'].append(item['response_avg_len_tokens'])
    #     level_dict[level]['response_std_len_words'].append(item['response_std_len_words'])

    # # import pdb; pdb.set_trace()    
    
    # for level, value in level_dict.items():
    #     value['overall_avg_sampling_acc'] = sum(value['sampling_acc']) / len(value['sampling_acc'])
    #     value['overall_avg_response_len_words'] = sum(value['response_avg_len_words']) / len(value['response_avg_len_words'])
    #     value['overall_avg_response_len_tokens'] = sum(value['response_avg_len_tokens']) / len(value['response_avg_len_tokens'])
    #     value['overall_std_response_len_words'] = sum(value['response_std_len_words']) / len(value['response_std_len_words'])

    # level_dict = dict(sorted(level_dict.items(), key=lambda x: x[0]))
    
    # for level, value in level_dict.items():
    #     print(f"level: {level}, overall_avg_sampling_acc: {value['overall_avg_sampling_acc']:.2f}, overall_avg_response_len_words: {value['overall_avg_response_len_words']:.2f}, overall_std_response_len_words: {value['overall_std_response_len_words']:.2f}, overall_avg_response_len_tokens: {value['overall_avg_response_len_tokens']:.2f}")


if __name__ == "__main__":
   args = parse_args()
   main(args)
