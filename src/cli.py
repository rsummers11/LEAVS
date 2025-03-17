# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import time
import re
import os
from tqdm import tqdm
from filelock import FileLock
from joblib import Parallel, delayed
import pandas as pd
import nltk
import ast
nltk.download('punkt_tab')
nltk.download('punkt')
from random import randint
import os
import socket
import sys
import subprocess
import json

# Function to get pip libraries and versions
def get_pip_libraries():
    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    packages = result.stdout.strip().split('\n')
    return {pkg.split('==')[0]: pkg.split('==')[1] for pkg in packages if '==' in pkg}

# Function to get conda libraries and versions
def get_conda_libraries():
    result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
    conda_versions = {}
    for line in result.stdout.strip().split('\n')[3:]:  # Skip the first 3 lines (header)
        parts = line.split()
        if len(parts) >= 2:
            package, version = parts[0], parts[1]
            conda_versions[package] = version
    return conda_versions

class ProgressParallel(Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

from retry import retry

class Open(object):
    @retry((FileNotFoundError, IOError, OSError), delay=1, backoff=2, max_delay=10, tries=100)
    def __init__(self, file_name, method):
        self.file_obj = open(file_name, method)
    def __enter__(self):
        return self.file_obj
    def __exit__(self, type, value, traceback):
        self.file_obj.close()

@retry((FileNotFoundError, IOError, OSError), delay=1, backoff=2, max_delay=10, tries=100)
def get_lock(lock):
    lock.acquire()

import joblib
import contextlib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def parse_yes_no(sentence, fn_inputs):
    sentence = sentence.lower()
    print(sentence, flush=True)
    if sentence[:3]=='yes':
        return 1
    else:
        return 0

def parse_severity(sentence, fn_inputs):
    sentence = sentence.lower().replace('"','').strip()
    if sentence not in ['mild','moderate','severe']:
        return -1
    return sentence

def parse_percentage(sentence, fn_inputs, threshold_percentage = 90):
    try:
        sentence = int(re.sub(r'\D', '', sentence))
    except ValueError:
        return 1
    if sentence>=threshold_percentage:
        return 1
    else:
        return 0

def parse_percentage_number(sentence, fn_inputs):
    try:
        sentence = re.sub(r'\D', '', sentence)
    except ValueError:
        return 101
    if len(sentence)==0:
        return 101
    try:
        sentence = int(sentence)
    except ValueError:
        return 101
    return sentence

class Node:
    def __init__(self, data, subdata, max_new_tokens=2, parse_sentence=parse_yes_no):
        self.subdata = subdata
        self.data = data
        if not isinstance(max_new_tokens, (list, tuple)):
            max_new_tokens = [max_new_tokens for i in range(len(data))]
        self.max_new_tokens = max_new_tokens
        self.parse_sentence = parse_sentence
    def __len__(self):
        return len(self.subdata)
    def __getitem__(self, index):
        if self.subdata is None:
            return index
        return self.subdata[index]

def stream_output(output_stream):
    for outputs in output_stream:
        output_text = outputs["text"]
    return "".join(output_text)
    
def get_node_output(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args, return_conv=False):

    from fastchat.conversation import get_conv_template
    from fastchat.model.model_adapter import (
        get_conversation_template,
        get_generate_stream_function,
    )

    from fastchat.utils import get_context_length

    model_path = model_name
    generate_stream_func = get_generate_stream_function(model, model_name)
    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type
    repetition_penalty = args.repetition_penalty

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    context_len = get_context_length(model.config)

    if type(current_node)==list:
        return [get_node_output(current_node[node_index], fn_inputs, tokenizer, model, model_name, do_tasks, args) if do_tasks[node_index] else -1 for node_index in range(len(current_node)) ]
    if type(current_node)==Node:
        def new_chat():
            if args.conv_template:
                conv = get_conv_template(args.conv_template)
            else:
                conv = get_conversation_template(model_path)
            system_message="A chat between a radiologist and an artificial intelligence assistant trained to understand radiology reports and any synonyms and word equivalency of findings and medical terms that may appear in the report. The assistant gives helpful structured answers to the radiologist.",
            conv.set_system_message(system_message)
            conv.messages = conv.messages[:0]
            return conv
        if 'conv_input' in fn_inputs:
            conv = fn_inputs['conv_input']
        else:
            conv = new_chat()
        
        for prompt_index in range(len(current_node.data)):
            
            inp =f"{current_node.data[prompt_index](report_ = fn_inputs['report_'], sentence_ = fn_inputs['sentence_'], label_ = fn_inputs['label_'])}"
            if len(inp)==0:
                conv.append_message(conv.roles[1], fn_inputs['sentence_'])
            else:
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
            
                prompt = conv.get_prompt()
                
                if is_codet5p:  # codet5p is a code completion model.
                    prompt = inp
                gen_params = {
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": args.temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": current_node.max_new_tokens[prompt_index],
                    "stop": conv.stop_str,
                    "stop_token_ids": conv.stop_token_ids,
                    "echo": False,
                    'do_sample': False,
                }
                output_stream = generate_stream_func(            
                    model,
                    tokenizer,
                    gen_params,
                    device = args.device,
                    context_len=context_len,
                    judge_sent_end=args.judge_sent_end,)
                            
                outputs = stream_output(output_stream)

                conv.update_last_message(outputs.strip())
        print(fn_inputs['id'], fn_inputs['organ'], conv, flush=True)
        answer = current_node.parse_sentence(conv.messages[-1][-1], fn_inputs)
        current_node = current_node[answer]
        if type(answer)==list:
            if return_conv:
                return current_node, fn_inputs['conv']
            else:
                return current_node
        if return_conv:
            if 'conv' in fn_inputs:
                fn_inputs['conv'] = fn_inputs['conv'] + conv
            else:
                fn_inputs['conv'] = conv
        return get_node_output(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args)
    else:
        if return_conv:
            return current_node, fn_inputs['conv']
        else:
            return current_node

async def get_full_answer(request_outputs):
    to_return = []
    async for request_output in request_outputs:
        if request_output.finished:
            to_return = request_output
    return to_return

#run asynchronous function to generate a model output from the loop that is running on the background
def generate(prompt, sampling_params, engine):
    from vllm.utils import random_uuid 
    import asyncio
    request_id = random_uuid()
    assert engine is not None
    
    loop =  engine._background_loop_unshielded.get_loop()
    a = engine.generate(prompt, sampling_params, str(request_id))
    request_outputs = asyncio.run_coroutine_threadsafe(get_full_answer(a), loop)
    request_outputs = request_outputs.result()
    return request_outputs

def get_node_output_vllm(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args, return_conv=False):
    import importlib.util
    import sys
    module_name = 'vllm'

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        # Use importlib.util.find_spec to locate the module
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ImportError(f"Module {module_name} not found")

        # Load the module from the found specification
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    repetition_penalty = args.repetition_penalty

    if type(current_node)==list:
        return [get_node_output_vllm(current_node[node_index], fn_inputs, tokenizer, model, model_name, do_tasks, args) if do_tasks[node_index] else -1 for node_index in range(len(current_node)) ]
    if type(current_node)==Node:
        if 'conv_input' in fn_inputs:
            messages = fn_inputs['conv_input']
        else:
            system_message="A chat between a radiologist and an artificial intelligence assistant trained to understand radiology reports and any synonyms and word equivalency of findings and medical terms that may appear in the report. The assistant gives helpful structured answers to the radiologist."
            messages = []
            messages.append({"role":"system","content":system_message})
        for prompt_index in range(len(current_node.data)):
            
            inp =f"{current_node.data[prompt_index](report_ = fn_inputs['report_'], sentence_ = fn_inputs['sentence_'], label_ = fn_inputs['label_'])}"

            if len(inp)==0:
                messages.append({"role": "assistant", "content": fn_inputs['sentence_']})
            else:

                messages.append({"role": "user", "content": inp})
                prompt =  tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                assert(len(prompt)<=8192*2)
                
                prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # assert(args.temperature==0)
                if args.top_k is not None:
                    sampling_params = module.SamplingParams(temperature=args.temperature, 
                                                            repetition_penalty = repetition_penalty, top_k = args.top_k,
                                                            max_tokens = current_node.max_new_tokens[prompt_index],
                                                            skip_special_tokens = False)
                else:
                    sampling_params = module.SamplingParams(temperature=args.temperature, 
                                                            repetition_penalty = repetition_penalty,
                                                            max_tokens = current_node.max_new_tokens[prompt_index],
                                                            skip_special_tokens = False)
                
                outputs = generate(prompt, sampling_params, model) 
                
                outputs = outputs.outputs[0].text

                messages.append({"role": "assistant", "content": outputs.strip()})
        print(fn_inputs['id'], fn_inputs['organ'], messages, flush=True)
        answer = current_node.parse_sentence(messages[-1]['content'], fn_inputs)
        print(answer, flush=True)
        current_node = current_node[answer]
        if return_conv:
            if 'conv' in fn_inputs:
                fn_inputs['conv'] = fn_inputs['conv'] + messages
            else:
                fn_inputs['conv'] = messages
        if type(answer)==list:
            if return_conv:
                return current_node, fn_inputs['conv']
            else:
                return current_node
        return get_node_output_vllm(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args)
    else:
        if return_conv:
            return current_node, fn_inputs['conv']
        else:
            return current_node

def get_prompt_tree_maplez(short_answers, include_report_in_final_prompt, cot_for_uncertain, get_node_output_fn, heart_labels, label_set_mimic_complex, label_set_mimic_generic):
    assert(not include_report_in_final_prompt)
    assert(not cot_for_uncertain)
    
    first_part = 'Consider in your answer: 1) radiologists might skip some findings because of their low priority 2) explore all range of probabilities, giving preference to non-round probabilities 3) medical wording synonyms, subtypes of abnormalities 4) radiologists might express their uncertainty using words such as "or", "possibly", "can\'t exclude", etc..' 
    number_prompt =  Node([lambda report_, sentence_, label_: f'{first_part}. Given the complete CT report "{sentence_}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 80, parse_percentage_number)
    
    number_prompt_present =  Node([lambda report_, sentence_, label_: f'{first_part} Given the complete CT report "{sentence_}", consistent with the radiologist observing "{label_set_mimic_generic[label_]}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 80, parse_percentage_number)
    
    number_prompt_absent =  Node([lambda report_, sentence_, label_: f'{first_part} Given the complete CT report "{sentence_}", consistent with the radiologist stating the absence of evidence "{label_set_mimic_generic[label_]}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 80, parse_percentage_number)

    inner_prompts = Node([lambda report_, sentence_, label_:f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_generic[label_]}" as stable or unchanged. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                [Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist stated the absence of evidence of "{label_set_mimic_generic[label_]}". ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                    [[-2, number_prompt, '', -1],
                    [0, number_prompt, '', -1]], max_new_tokens=(1 if short_answers else [1600,1])),
                Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if "{label_set_mimic_generic[label_]}" might be present. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                    [Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_generic[label_]}" as normal. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                        [[-3,101,'',-1],
                        [0, number_prompt_absent, '', -1]]
                        , max_new_tokens=(1 if short_answers else [1600,1])),
                    [1, number_prompt_present, -1, -1]]
                , max_new_tokens=(1 if short_answers else [1600,1]))], max_new_tokens=(1 if short_answers else [1600,1]))
    number_inner_prompts_solo = Node([lambda report_, sentence_, label_:f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_generic[label_]}" as stable or unchanged. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
            [number_prompt,
            Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if "{label_set_mimic_generic[label_]}" might be present. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                [Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_generic[label_]}" as normal. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                    [101,
                    number_prompt_absent]
                    , max_new_tokens=(1 if short_answers else [1600,1])),
                number_prompt_present]
            , max_new_tokens=(1 if short_answers else [1600,1]))], max_new_tokens=(1 if short_answers else [1600,1]))
    prompts_all = Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist observed possible presence of evidence of "{label_set_mimic_generic[label_]}". ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
            [inner_prompts,
            [Node([lambda report_, sentence_, label_: f'Given the complete CT report "{sentence_}", consistent with the radiologist observing "{label_set_mimic_generic[label_]}", estimate from the report wording how likely another radiologist is to observe "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
                [-1,
                1], 80, parse_percentage),
            number_inner_prompts_solo,
            -1,
            -1
            ]], max_new_tokens=(1 if short_answers else [1600,1]))
    return prompts_all

def get_prompt_tree_maplez_negative(short_answers, include_report_in_final_prompt, cot_for_uncertain, get_node_output_fn, heart_labels, label_set_mimic_complex, label_set_mimic_generic):
    assert(not include_report_in_final_prompt)
    assert(not cot_for_uncertain)
    
    first_part = 'Consider in your answer: 1) radiologists might skip some findings because of their low priority 2) explore all range of probabilities, giving preference to non-round probabilities 3) medical wording synonyms, subtypes of abnormalities 4) radiologists might express their uncertainty using words such as "or", "possibly", "can\'t exclude", etc..' 
    number_prompt =  Node([lambda report_, sentence_, label_: f'{first_part}. Given the complete CT report "{sentence_}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 80, parse_percentage_number)
    
    number_prompt_present =  Node([lambda report_, sentence_, label_: f'{first_part} Given the complete CT report "{sentence_}", consistent with the radiologist observing "{label_set_mimic_generic[label_]}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 80, parse_percentage_number)
    
    number_prompt_absent =  Node([lambda report_, sentence_, label_: f'{first_part} Given the complete CT report "{sentence_}", consistent with the radiologist stating the absence of evidence "{label_set_mimic_generic[label_]}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_generic[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 80, parse_percentage_number)

    inner_prompts = Node([lambda report_, sentence_, label_:f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_generic[label_]}" as stable or unchanged. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                [Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist stated the absence of evidence of "{label_set_mimic_generic[label_]}". ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                    [[-2, number_prompt, '', -1],
                    [0, number_prompt, '', -1]], max_new_tokens=(1 if short_answers else [1600,1])),
                Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if "{label_set_mimic_generic[label_]}" might be present. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                    [Node([lambda report_, sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_generic[label_]}" as normal. ' + ('Respond only with "Yes" or "No".' if short_answers else 'Let\'s think step by step. Finish with a conclusion of the answer.')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),
                        [[-3,101,'',-1],
                        [0, number_prompt_absent, '', -1]]
                        , max_new_tokens=(1 if short_answers else [1600,1])),
                    [1, number_prompt_present, -1, -1]]
                , max_new_tokens=(1 if short_answers else [1600,1]))], max_new_tokens=(1 if short_answers else [1600,1]))
    prompts_all = inner_prompts
    return prompts_all

find_first_match = lambda lst, s: next(iter(re.finditer(r'\b(?:' + '|'.join(map(re.escape, lst)) + r')\b', s)), None).group(0) if s else None

def get_prompt_tree_leavs(short_answers, include_report_in_final_prompt, cot_for_uncertain, get_node_output_fn, heart_labels, label_set_mimic_complex, label_set_mimic_generic):
    from global_ import organ_denominations
    assert(not cot_for_uncertain)
    prompts_all = Node([lambda report_, sentence_, label_: "In your answer: 1) consider medical wording synonyms and subtypes of findings 2) consider abbreviations of the medical vocabulary 3) consider that radiologists may mention abnormal findings about a body region even if they say it is normal 4) consider both nonspecific patterns and specific conditions 5) when discussing focal versus diffuse, always reason which one is more likely for each finding 6) analyze each sentence individually and focus on the presence or absence of findings in each one, without letting one sentence influence the interpretation of others 7) consider that a single sentence may provide mentions to more than one finding.\n" + (f'In the following subpart of a CT report "\n' + "".join([f"{sent_index+1}. {sent}\n" for sent_index, sent in enumerate(nltk.sent_tokenize(sentence_))]) + f'", what can be concluded about "{label_set_mimic_generic[label_]}"? ' if not include_report_in_final_prompt else f'In the following full CT report "{report_}" and the subpart "\n' + "".join([f"{sent_index+1}. {sent}\n" for sent_index, sent in enumerate(nltk.sent_tokenize(sentence_))]) + f'", what can be concluded about "{label_set_mimic_generic[label_]}"? ') + \
    f'''(A) Findings of this category are negative or very unlikely: The report has language explicitly denying the presence of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(B) A direct reference to a finding in this category is missing: the report does not have any mentions of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(C) A finding of this category is mentioned as possible: the report contains explicit language saying that "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''" may be present, but does not give full confidence given that the radiologist expressed their uncertainty using words such as "or", "possibly", "can\'t exclude", etc.
(D) A finding of this category is mentioned as very likely or positive: the report has explicit language directly affirming the presence of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(E) The report has ambiguous language in the mention of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''", comparing it to a previous report, and the information from the previous report is necessary to make conclusions about the presence or absence of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(F) The report has a mention to "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''" that is not specific to the ''' + find_first_match(organ_denominations, re.sub(r'\(.*\)', '', label_set_mimic_generic[label_])) + ''' or one of its parts, but only for a broad anatomical area. Therefore, you cannot be confident that there is a finding specifically in or about the ''' + find_first_match(organ_denominations, re.sub(r'\(.*\)', '', label_set_mimic_generic[label_])) + ''' or one of its subparts being evaluated.
''' + ('''Answer only with the letter corresponding to the correct answer. You must choose the best answer among A, B, C, D, E, F. Answer:"(''' if short_answers else f'''\nWhat is the capital letter corresponding to the correct answer? You must choose the best answer among A, B, C, D, E, F. Let's think step by step. Explain what each sentence and finding from the report subpart medically means, then proceed to relate it to "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''". Finish with a conclusion of the answer.''')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize your answer with only the letter corresponding to the correct answer. Answer:"(']), [[0,-1,-1,-1],[1,-1,-1,-1],[-3,-1,-1,-1],[-2,-1,-1,-1],[-1,-1,-1,-1]], 20 if short_answers else [1600,20], parse_multiple_answers_certainty)
    return prompts_all

def get_prompt_tree_leavs_negative(short_answers, include_report_in_final_prompt, cot_for_uncertain, get_node_output_fn, heart_labels, label_set_mimic_complex, label_set_mimic_generic):
    assert(not cot_for_uncertain)
    from global_ import organ_denominations
    prompts_all = Node([lambda report_, sentence_, label_: "In your answer: 1) consider medical wording synonyms and subtypes of findings 2) consider abbreviations of the medical vocabulary 3) consider that radiologists may mention abnormal findings about a body region even if they say it is normal 4) consider both nonspecific patterns and specific conditions 5) when discussing focal versus diffuse, always reason which one is more likely for each finding 6) analyze each sentence individually and focus on the presence or absence of findings in each one, without letting one sentence influence the interpretation of others 7) consider that a single sentence may provide mentions to more than one finding.\n" + (f'In the following subpart of a CT report "\n' + "".join([f"{sent_index+1}. {sent}\n" for sent_index, sent in enumerate(nltk.sent_tokenize(sentence_))]) + f'", what can be concluded about "{label_set_mimic_generic[label_]}"? ' if not include_report_in_final_prompt else f'In the following full CT report "{report_}" and the subpart "\n' + "".join([f"{sent_index+1}. {sent}\n" for sent_index, sent in enumerate(nltk.sent_tokenize(sentence_))]) + f'", what can be concluded about "{label_set_mimic_generic[label_]}"? ') + \
    f'''(A) Findings of this category are negative or very unlikely: The report has language explicitly denying the presence of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(B) A direct reference to a finding in this category is missing: the report does not have any mentions of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(C) The report has ambiguous language in the mention of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''", comparing it to a previous report, and the information from the previous report is necessary to make conclusions about the presence or absence of "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''".
(D) The report has a mention to "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''" that is not specific to the ''' + find_first_match(organ_denominations, re.sub(r'\(.*\)', '', label_set_mimic_generic[label_])) + ''' or one of its parts, but only for a broad anatomical area. Therefore, you cannot be confident that there is a finding specifically in or about the ''' + find_first_match(organ_denominations, re.sub(r'\(.*\)', '', label_set_mimic_generic[label_])) + ''' or one of its subparts being evaluated.
''' + ('''Answer only with the letter corresponding to the correct answer. You must choose the best answer among A, B, C, D. Answer:"(''' if short_answers else f'''\nWhat is the capital letter corresponding to the correct answer? You must choose the best answer among A, B, C, D. Let's think step by step. Explain what each sentence and finding from the report subpart medically means, then proceed to relate it to "''' + re.sub(r'\(.*\)', '', label_set_mimic_generic[label_]).replace('  ', ' ').strip() + f'''". Finish with a conclusion of the answer.''')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize your answer with only the letter corresponding to the correct answer. Answer:"(']), [[0,-1,-1,-1],[1,-1,-1,-1],[-3,-1,-1,-1],[-2,-1,-1,-1],[-1,-1,-1,-1]], 20 if short_answers else [1600,20], parse_multiple_answers_certainty_negative)
    return prompts_all

def parse_multiple_answers_all(sentence_answer, fn_inputs):
    pattern = r'[A-K](?![a-z])'

    # Find matches
    matches = re.findall(pattern, sentence_answer)
    if matches is None or len(matches)==0:
        return [0]
    indices = [ord(letter) - ord('A') for letter in matches]
    return indices

def parse_multiple_answers_certainty(sentence_answer, fn_inputs):
    pattern = r'[A-F](?![a-z])'

    # Find matches
    matches = re.findall(pattern, sentence_answer)
    if matches is None or len(matches)==0:
        return -2
    dict_answers = [0,-2,-1,1,-3,-3]
    indices = [dict_answers[ord(letter) - ord('A')] for letter in matches][0]
    return indices

def parse_multiple_answers_certainty_negative(sentence_answer, fn_inputs):
    pattern = r'[A-D](?![a-z])'

    # Find matches
    matches = re.findall(pattern, sentence_answer)
    if matches is None or len(matches)==0:
        return -2
    dict_answers = [0,-2,-3,-3]
    indices = [dict_answers[ord(letter) - ord('A')] for letter in matches][0]
    return indices

def parse_one_sentence(sentence_answer, fn_inputs):
    to_return = parse_yes_no(sentence_answer, fn_inputs)
    return to_return

def parse_sentences(sentence_answer, fn_inputs, next_prompt, get_node_output_fn):
    
    model = fn_inputs['model']
    model_name = fn_inputs['model_name']
    tokenizer = fn_inputs['tokenizer']
    do_tasks = fn_inputs['do_tasks']
    args = fn_inputs['args']
    report = fn_inputs['sentence_']
    label = fn_inputs['label_'] 
    if 'sentence_outputs_fast' in fn_inputs:
        sentence_outputs_fast = fn_inputs['sentence_outputs_fast'] 
    else:
        sentence_outputs_fast = None
    real_excerpts = nltk.sent_tokenize(report)
    prompts_all = next_prompt
    to_return = []
    if sentence_outputs_fast is not None:
        sentence_outputs_fast_corrected = []
        for sentence_output_fast in sentence_outputs_fast:
            split_sentences = nltk.sent_tokenize(sentence_output_fast)
            sentence_outputs_fast_corrected = sentence_outputs_fast_corrected + split_sentences
        sentence_outputs_fast = sentence_outputs_fast_corrected
        def equals_func(a, b):
            a = a.lower().strip().rstrip(".").strip()
            b = b.lower().strip().rstrip(".").strip()
            return Levenshtein.distance(a, b)<=max(min(10,len(a)-1,len(b)-1),0) or \
                        (len(a)>10 and a in b)
    for one_sentence in real_excerpts:
        if sentence_outputs_fast is not None:
            import Levenshtein
            continue_loop = False
            for sentence_already_included in sentence_outputs_fast:
                if equals_func(sentence_already_included, one_sentence) or \
                    (len(one_sentence.strip().rstrip(".").strip())>10 and one_sentence.strip().rstrip(".").strip().lower() in sentence_already_included.lower().strip().rstrip(".").strip()):
                    to_return.append(one_sentence)
                    continue_loop = True
                    break
            if continue_loop:
                continue
        a = {'report_':report, 'sentence_':one_sentence, 'label_':label}
        a['model'] = fn_inputs['model']
        a['model_name'] = fn_inputs['model_name']
        a['tokenizer'] = fn_inputs['tokenizer']
        a['do_tasks'] = fn_inputs['do_tasks']
        a['args'] = fn_inputs['args']
        a['id'] = fn_inputs['id']
        a['organ'] = fn_inputs['organ']
        
        sentence_outputs = get_node_output_fn(prompts_all, a, tokenizer, model, model_name, do_tasks, args)
        
        if sentence_outputs:
            to_return.append(one_sentence)
    if sentence_outputs_fast is not None:
        for item2 in sentence_outputs_fast:
            if not any(equals_func(item2, item1) for item1 in to_return):
                to_return.append(item2)
    return to_return
                  
def main(args, tokenizer, model):

    result_root = 'results/' + args.result_root
    os.makedirs(result_root, exist_ok=True)
    args.timestamp = time.strftime("%Y%m%d-%H%M%S")

    random_number = randint(0, 99999)
    args.id = f"{random_number:05}"

    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID')
    args.command = ' '.join(sys.argv)
    # Get pip and conda versions
    pip_versions = get_pip_libraries()
    conda_versions = get_conda_libraries()

    # Dynamically add pip and conda libraries to `args`
    for pkg, version in pip_versions.items():
        setattr(args, f'pip_{pkg}', version)

    for pkg, version in conda_versions.items():
        setattr(args, f'conda_{pkg}', version)
    
    # Convert args to dictionary
    args_dict = vars(args)

    # Save to a JSON file
    with open(result_root + f'/args{args.timestamp}_{args.id}.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    report_label_file = result_root + "/parsing_results_llm.csv"
    model_name = args.model
    label_set_mimic_complex = None
    label_set_mimic_generic = None

    short_answers = not args.use_cot
    include_report_in_final_prompt = args.include_report_in_final_prompt
    if args.labels_abnormality=='organs':
        from global_ import organ_denominations        

        if args.definition in ['individual', 'multiple']:
            type_denominations = ['normal', 'absent', 'adjacent', 'quality', 'postsurgical', 'anatomy', 'device', 'enlarged', 'atrophy', 'diffuse', 'focal']
            type_sentences_no_description = [lambda organ:f'the finding that the {organ} is normal', lambda organ:f'the finding that the {organ} is not present', 
                lambda organ:f'an adjacent, extrinsic finding for the {organ}, or a finding in contact to it', lambda organ:f'''the finding that the {organ} or any of its parts\n - lack imaging quality,\n - lack contrast opacification\n - or are not fully imaged,\n and therefore has limited evaluation''', 
                lambda organ:f'the finding that the {organ} or any of its parts have postsurgical changes', lambda organ:f'an anatomical finding for the {organ}', lambda organ:f'the finding that the {organ} or any of its parts have support device', lambda organ:f'the finding that the {organ} is enlarged', lambda organ:f'the finding that the {organ} has atrophy',
                lambda organ:f'a type of diffuse finding in the {organ} or any of its parts', lambda organ:f'a type of focal finding in the {organ} or any of its parts']

            if not args.include_finding_description:
                type_sentences = type_sentences_no_description
            else:
                type_sentences = [type_sentences_no_description[0], 
                    lambda organ: type_sentences_no_description[1](organ) + f' (the full {organ} is not in the body)', 
                    lambda organ:type_sentences_no_description[2](organ) + f' (finding is out of the {organ}, but touching it)', 
                    lambda organ: type_sentences_no_description[3](organ) +''' (any finding about the acquisition process and not about the patient\'s body)''', 
                    lambda organ: type_sentences_no_description[4](organ) + ' (not support devices)', 
                    lambda organ: type_sentences_no_description[5](organ) + f' (uncommonly seen displacements, relative positionings, or shapes of the {organ} or one of its parts)', 
                    type_sentences_no_description[6],
                    type_sentences_no_description[7],
                    type_sentences_no_description[8],
                    lambda organ: type_sentences_no_description[9](organ) + ' (findings that usually, but not necessarily, do not have a well-defined border or shape for measurement, or typically affect large regions in the body region)', 
                    lambda organ: type_sentences_no_description[10](organ) + f' (findings inside the body region that usually, but not necessarily, can be measured from its (sometimes approximate) borders)']

            abnormality_denominations = [f'{organ}_{type_}' for organ in organ_denominations for type_ in type_denominations]
        else:
            type_denominations = ['']
            type_sentences = [lambda organ:'']
            abnormality_denominations = organ_denominations

        label_set_mimic_complex = [f"{type_sentence(organ)} (consider that findings about the {organ} in a CT report can be split into:\n" + ';\n'.join(['- ' + type_sentence_all(organ) for type_sentence_all in type_sentences[:-1]]) + ';\n - or '+ type_sentences[-1](organ) + ')'
                        for organ in organ_denominations for type_sentence in type_sentences]
        label_set_mimic_generic = [f"{type_sentence(organ)}"
                        for organ in organ_denominations for type_sentence in type_sentences] 
        label_set_finding_name = label_set_mimic_generic
        organ_set_simple = [f"any explicit radiological finding that can be inferred to be in or about the {organ} or any of its parts"
                            for organ in organ_denominations]
        if args.include_finding_types_sentence:
            organ_set_sentences = [f"any explicit radiological finding that can be inferred to be in or about the {organ} or any of its parts; include, for example, but not limited to, these findings:" + ';\n'.join(['- ' + type_sentence(organ) for type_sentence in type_sentences])
                            for organ in organ_denominations]
        else:
            organ_set_sentences = organ_set_simple
                    
    elif args.labels_abnormality=='maplez':
        abnormality_denominations = ['lung lesion', 'liver lesion', 'kidney lesion', 'adrenal gland abnormality', 'pleural effusion']
        abnormality_dict = abnormality_denominations
    if label_set_mimic_complex is None:
        label_set_mimic_complex = abnormality_dict
    if label_set_mimic_generic is None:
        label_set_mimic_generic = abnormality_dict

    tasks = ['labels','probability','location','severity']
    do_tasks = args.do_tasks
    heart_labels = range(len(label_set_mimic_generic))
    do_labels = args.do_labels
    
    

    if args.single_file is not None or args.test_list is not None:
        if args.single_file is not None:
            file_paths = args.single_file
        else:
            file_paths = []
            with open(args.test_list, 'r') as input_file:
                for line in input_file:
                    # Remove leading and trailing whitespace and newline characters
                    file_paths.append(line.strip())
        data = {'filepath': [], 'anonymized_report': [], 'study_id': [], 'subject_id': []}

        # Loop through the file paths and extract the data
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                file_content = file.read()
            
            # Extract study_id from the filename without extension
            study_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Append data to lists
            data['filepath'].append(file_path)
            data['anonymized_report'].append(file_content)
            data['study_id'].append(study_id)
            data['subject_id'].append('')

        # Create a DataFrame from the data
        chexpert_df = pd.DataFrame(data)
    else:
        chexpert_df = pd.read_csv(args.reports_csv)
        chexpert_df['subject_id'] = chexpert_df['image1']
        chexpert_df['study_id'] = chexpert_df['image1']
    if args.end_index is None:
        chexpert_df = chexpert_df.iloc[args.start_index:,:]
    else:
        chexpert_df = chexpert_df.iloc[args.start_index:args.end_index,:]

    report_label_file = report_label_file
    report_label_lockfile = f"{report_label_file}.lock"
    report_label_lock = FileLock(report_label_lockfile, timeout = 300)

    report_sentences_file = result_root + '/sentences.csv'
    report_sentences_lockfile = f"{report_sentences_file}_.lock"
    report_sentences_lock = FileLock(report_sentences_lockfile, timeout = 300)

    report_timing_file = result_root + '/timing.csv'
    report_timing_lockfile = f"{report_timing_file}_.lock"
    report_timing_lock = FileLock(report_timing_lockfile, timeout = 300)

    if args.definition=='multiple':
        report_ma_file = result_root + '/ma.csv'
        report_ma_lockfile = f"{report_ma_file}_.lock"
        report_ma_lock = FileLock(report_ma_lockfile, timeout = 300)
    
    get_lock(report_label_lock)
    header_text_labels = '","'.join(type_denominations)
    try:
        if not os.path.isfile(report_label_file):
            with Open(report_label_file, "w") as f:
                f.write(f'subjectid_studyid,organ,report,type_annotation,"{header_text_labels}"\n')
    finally:
        report_label_lock.release()
    
    get_lock(report_sentences_lock)
    try:
        if not os.path.isfile(report_sentences_file):
            with Open(report_sentences_file, "w") as f:
                f.write(f'subjectid_studyid,organ,report,sentences\n')
    finally:
        report_sentences_lock.release()

    get_lock(report_timing_lock)
    try:
        if not os.path.isfile(report_timing_file):
            with Open(report_timing_file, "w") as f:
                f.write(f'subjectid_studyid,organ,type_annotation,time\n')
    finally:
        report_timing_lock.release()
    start_full_script_time = time.time()

    if args.definition=='multiple':
        header_text_labels = 'ma'
        get_lock(report_ma_lock)
        try:
            if not os.path.isfile(report_ma_file):
                with Open(report_ma_file, "w") as f:
                    f.write(f'subjectid_studyid,organ,report,ma\n')
        finally:
            report_ma_lock.release()

    get_lock(report_label_lock)
    try:
        starting_state = pd.read_csv(report_label_file, sep = ',')
    finally:
        report_label_lock.release()

    get_lock(report_sentences_lock)
    try:
        starting_state_sentences = pd.read_csv(report_sentences_file, sep = ',')
    finally:
        report_sentences_lock.release()

    if args.use_vllm:
        get_node_output_fn = get_node_output_vllm
    else:
        get_node_output_fn = get_node_output
    if args.prompt_to_use=='maplez':
        get_prompt_tree_fn = lambda *args2, **kwargs: get_prompt_tree_maplez(not args.use_cot, args.include_report_in_final_prompt, args.cot_for_uncertain, get_node_output_fn, *args2, **kwargs) 
    elif args.prompt_to_use=='leavs':
        get_prompt_tree_fn = lambda *args2, **kwargs: get_prompt_tree_leavs(not args.use_cot, args.include_report_in_final_prompt, args.cot_for_uncertain, get_node_output_fn, *args2, **kwargs) 
    if args.go_through_negative_after_multiple:
        if args.prompt_to_use=='leavs':
            get_prompt_tree_fn_negative = lambda *args2, **kwargs: get_prompt_tree_leavs_negative(not args.use_cot, args.include_report_in_final_prompt, args.cot_for_uncertain, get_node_output_fn, *args2, **kwargs) 
        elif args.prompt_to_use=='maplez':
            get_prompt_tree_fn_negative = lambda *args2, **kwargs: get_prompt_tree_maplez_negative(not args.use_cot, args.include_report_in_final_prompt, args.cot_for_uncertain, get_node_output_fn, *args2, **kwargs) 
    
    urgency_prompt = Node([lambda report_, sentence_, label_: (f'''Given the complete report "{report_}", the sentences "{sentence_}" mention "{label_set_finding_name[label_]}". ''' if include_report_in_final_prompt else f'''In the following subpart of a CT report "{sentence_}" there is a mention to "{label_set_finding_name[label_]}".''') + 
                            f''' Classify the urgency of "{label_set_finding_name[label_]}" in this report:
(A) normal, expected, or chronic - no communication to other doctors necessary - no action or treatment needed
(B) low urgency, incidental, unexpected - clinical communication within days - no action now, but could cause problems to the patient long term
(C) medium urgency, significant - clinical communication within hours - clinically significant observations explaining an acute presentation and requiring treatment
(D) high urgency, critical - clinical communication within minutes - findings that could lead to death if not promptly communicated to the ordering clinician and requiring immediate emergency treatment\n''' + \
    ('''Answer only with the letter corresponding to the correct answer. Answer:"(''' if short_answers else f'''What is the capital letter corresponding to the correct answer? You must choose the best answer among A, B, C, D. Let's think step by step. Explain what each sentence and finding from the report subpart medically means, then proceed to relate it to the finding urgency. Avoid unnecessary conservatism: do not assign a higher urgency unless a clinically significant or acute abnormality is present. Finish with a conclusion of the answer.''')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize your answer with only the letter corresponding to the correct answer. Answer:("']), None, 60 if short_answers else [1600,60], parse_multiple_answers_all)
    
    multiple_prompt = Node([lambda report_, sentence_, label_: f'''Consider in your answer: 1) medical wording synonyms, subtypes of abnormalities 2) abbreviations of the medical vocabulary. ''' + (f'''Given the complete CT report "{report_}", for what kind of finding about the {organ_denominations[label_]} there is any information (to affirm or deny) in in this subpart "{sentence_}"?''' if include_report_in_final_prompt else f'''For what kind of finding about the {organ_denominations[label_]} there is any information in this subpart of a CT report "{sentence_}"?''') + 
            f''' The information that:\n''' + \
                ''.join([f'({chr(index_type + ord("A"))}) ' + type_sentence(organ_denominations[label_]) + ',\n' for index_type,type_sentence in enumerate(type_sentences)]) + \
                (f'''({chr(len(type_sentences) + ord("A"))}) the {organ_denominations[label_]} is not mentioned in the report\nAnswer only with a list of the capital letters corresponding to the correct answers, each between parentheses. Answer:"''' if short_answers else f'''Include a list of the capital letters corresponding to the correct answers in your answer. Explain what each sentence and finding from the report subpart medically means, then proceed to relate it to "{organ_set_simple[label_]}". Let's think step by step. Finish with a conclusion of the answer.''')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize your answer with only a list of the capital letters corresponding to the correct answers, each between parentheses. Answer:"']), None, 60 if short_answers else [1600,60], parse_multiple_answers_all)     
    
    def parse_subpart(sentence_answer, fn_inputs):
        import Levenshtein
        label = fn_inputs['label_'] 
        
        sentence_answers_p2 = re.compile(r'- "(.*?)"(.*?)\n').findall((sentence_answer+'\n').lower())
        sentence_answers_p2 = [capture[0] for capture in sentence_answers_p2 if "uninformative" not in capture[1].lower()]
        combined_subparts = []
        combined_subparts.extend(sentence_answers_p2)
        combined_subparts = list(set(combined_subparts))
        combined_subparts.sort(key=lambda x: sentence_answer.lower().strip().find(x))
        real_excerpts = []
        for sentence_answer in combined_subparts:
            if Levenshtein.distance(sentence_answer, organ_set_sentences[label])>10 and not (sentence_answer in organ_set_sentences[label]):
                real_excerpts.append(sentence_answer)
        return real_excerpts

    if args.split_report!="no":
        if args.split_report=="fast" or args.split_report=="fast_extensive":
            prompts_filter_sentence = Node([lambda report_, sentence_, label_: f'Given the CT report "{report_}", is there any subpart of the report that can be used to infer information about the presence, absence, or stability of "{organ_set_sentences[label_]}"? If yes, extensively quote all of those sentences between double quotes (""). Include all implicit and explicit sentences related to the "{organ_set_sentences[label_]}" from all of the parts of the report. List the sentences from your answer employing dash items ("-"). Indicate if each sentence provides any information about the "{organ_set_sentences[label_]}" right after the sentence, using "Informative" or "Uninformative".'],None, 4000, lambda a,b: parse_subpart(a,b))
        if args.split_report=='extensive' or args.split_report=="fast_extensive":
            in_prompts = Node([lambda report_, sentence_, label_: f'''Given the CT report "{report_}", use logical deductive reasoning to infer if the sentence "{sentence_}" can be used to infer any information about the presence or absence of "{organ_set_sentences[label_]}" solely based on the information in the sentence itself.''' + ('Respond only with "Yes" or "No".' if short_answers else f'''Let's think step by step. Explain what the sentence from the report medically mean''' +(f''', then proceed to relate it to each of the finding types listed, one by one.''' if args.include_finding_types_sentence else '.') + f''' Finish with a conclusion of the answer to the question: Can the sentence be used to infer any information about the presence or absence of "any explicit radiological finding that can be inferred to be in or about the {organ_denominations[label_]} or any of its parts"?.''')] + ([] if short_answers else [lambda report_, sentence_, label_: 'Summarize if the answer to the radiology question was positive or negative. Respond only with "Yes" or "No".']),None, max_new_tokens = [1600,1], 
                    parse_sentence = parse_one_sentence)
            if args.split_report=="fast_extensive":
                prompts_filter_sentence_0 = prompts_filter_sentence
            prompts_filter_sentence = Node([lambda report_, sentence_, label_: ''],None, max_new_tokens = 0, parse_sentence = lambda a,b: parse_sentences(a,b, in_prompts, get_node_output_fn))
    
    def run_one_report(idx, row, organ_index, model_list):
        subject_id = row['subject_id']
        study_id = row['study_id']
        skip_labels = False
        if f"{row['subject_id']}_{row['study_id']}" in starting_state['subjectid_studyid'].values:
            starting_state_filtered_by_id = starting_state[starting_state['subjectid_studyid']==f"{row['subject_id']}_{row['study_id']}"]
            starting_state_this_urgency = starting_state_filtered_by_id[starting_state_filtered_by_id['type_annotation']=='urgency']
            if organ_denominations[organ_index] in starting_state_this_urgency['organ'].values:
                return
            if organ_denominations[organ_index] in starting_state_filtered_by_id['organ'].values:
                skip_labels = True
        model = model_list[0]
        report = row['anonymized_report']
        
        report = str(report).replace('\r', '\n').replace('\t', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
            
        report = report.strip().replace('____','thing').replace('XXXX','thing').replace('  ',' ')
        
        if f"{row['subject_id']}_{row['study_id']}" in starting_state_sentences['subjectid_studyid'].values and organ_denominations[organ_index] in starting_state_sentences['organ'][starting_state_sentences['subjectid_studyid']==f"{row['subject_id']}_{row['study_id']}"].values:
            line_text = starting_state_sentences[starting_state_sentences['subjectid_studyid']==f"{row['subject_id']}_{row['study_id']}"]
            line_text = line_text[line_text['organ']==organ_denominations[organ_index]]['sentences'].values[0]
            selected_sentences = [ast.literal_eval(line_text)]
        else:
            start_time_sentences = time.time()
            selected_sentences = []
            if not do_labels[organ_index]:
                selected_sentences.append([])
            else:
                if args.split_report!="no":
                    prompt_dict = {'report_':report, 'sentence_':report, 
                            'label_':organ_index, 
                            'tokenizer': tokenizer, 
                            'model': model, 
                            'model_name': model_name, 
                            'args':args, 
                            'do_tasks': do_tasks,
                        'label_set_mimic_generic': organ_set_sentences, 
                        'label_set_mimic_complex': organ_set_sentences, 
                            'heart_labels': heart_labels, 
                            'id':f"{row['subject_id']}_{row['study_id']}",
                            'organ':organ_denominations[organ_index],
                            }
                    if args.split_report=='fast_extensive':
                        sentence_outputs_fast = get_node_output_fn(prompts_filter_sentence_0, prompt_dict, tokenizer, model, model_name, do_tasks, args)
                        prompt_dict.update({'sentence_outputs_fast':sentence_outputs_fast})
                    sentence_outputs = get_node_output_fn(prompts_filter_sentence, prompt_dict, tokenizer, model, model_name, do_tasks, args)
                    selected_sentences.append(sentence_outputs)
                else:
                    selected_sentences.append([report])
            assert(len(selected_sentences)==1)

            line_text = ','.join(['"'+str(value).replace('"','""').replace('\n',' ')+'"' for value in selected_sentences])
            get_lock(report_sentences_lock)
            try:
                with Open(report_sentences_file, "a") as f:
                    skipped_report = report.replace('"','""').replace('\n',' ')
                    skipped_line_text = line_text
                    f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},"{skipped_report}",{skipped_line_text}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
            finally:
                report_sentences_lock.release()

            end_time_sentences = time.time()
            get_lock(report_timing_lock)
            try:
                with Open(report_timing_file, "a") as f:
                    f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},sentences,{end_time_sentences-start_time_sentences}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
            finally:
                report_timing_lock.release()
        selected_sentences_by_organ = selected_sentences
        if not skip_labels:
            start_time_labels = time.time()
            all_report_outputs = [] 
            all_convs = []
            if args.definition=='multiple':
                all_ma_outputs = []
                selected_sentences_multiple = selected_sentences
                selected_sentences = []
                if len(selected_sentences_multiple[0])>0:
                    selected_report = '. '.join(selected_sentences_multiple[0]).replace('..', '.').replace('..', '.')
                    current_node = multiple_prompt
                    sentence_outputs, conv = get_node_output_fn(current_node, {'report_':report, 'sentence_':selected_report, 
                    'label_':organ_index, 
                    'tokenizer': tokenizer, 
                    'model': model, 
                    'model_name': model_name, 
                    'args':args, 
                    'do_tasks': do_tasks,
                    'label_set_mimic_generic': organ_set_sentences, 
                    'label_set_mimic_complex': organ_set_sentences, 
                    'heart_labels': heart_labels,
                    'id':f"{row['subject_id']}_{row['study_id']}",
                        'organ':organ_denominations[organ_index], 
                    }, tokenizer, model, model_name, do_tasks, args, return_conv = True)
                    for type_index in range(len(type_denominations)):
                        if type_index in sentence_outputs:
                            selected_sentences.append([selected_report])
                            all_convs.append(conv)
                        else:
                            if args.go_through_negative_after_multiple:
                                selected_sentences.append([selected_report,'0'])
                                all_convs.append([])
                            else:
                                selected_sentences.append([])
                                all_convs.append([])

                else:
                    sentence_outputs = []
                    for type_index in range(len(type_denominations)):
                        selected_sentences.append([])
                        all_convs.append([])
                all_ma_outputs.append(sentence_outputs)
                
                line_text = ','.join(['"'+str([type_denominations[value] for value in listvalue]).replace('"','""').replace('\n',' ')+'"' for listvalue in all_ma_outputs])
                get_lock(report_ma_lock)
                try:
                    with Open(report_ma_file, "a") as f:
                        skipped_report = report.replace('"','""').replace('\n',' ')
                        skipped_line_text = line_text
                        f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},"{skipped_report}",{skipped_line_text}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
                finally:
                    report_ma_lock.release()
            if not do_labels[organ_index]:
                for type_index in range(len(type_denominations)):
                    all_report_outputs.append([-1,-1,-1,-1])
            else:
                sentence_outputs = []
                for type_index in range(len(type_denominations)):
                    sentence_index = type_index if args.definition=='multiple' else 0
                    if len(''.join(selected_sentences[sentence_index]).strip())>0:   
                        if args.go_through_negative_after_multiple and selected_sentences[sentence_index][-1]=='0':
                            selected_report = '. '.join(selected_sentences[sentence_index][:-1]).replace('..', '.').replace('..', '.')           
                            current_node = get_prompt_tree_fn_negative(heart_labels, label_set_mimic_complex, label_set_mimic_generic)
                        else:    
                            selected_report = '. '.join(selected_sentences[sentence_index]).replace('..', '.').replace('..', '.')           
                            
                            current_node = get_prompt_tree_fn(heart_labels, label_set_mimic_complex, label_set_mimic_generic)
                        fn_inputs = {'report_':report, 'sentence_':selected_report, 
                                    'label_':type_index + organ_index*len(type_denominations), 
                                    'tokenizer': tokenizer, 
                                    'model': model, 
                                    'model_name': model_name, 
                                    'args':args, 
                                    'do_tasks': do_tasks,
                                    'label_set_mimic_generic': label_set_mimic_generic, 
                                    'label_set_mimic_complex': label_set_mimic_complex, 
                                    'heart_labels': heart_labels, 
                                    'id':f"{row['subject_id']}_{row['study_id']}",
                                    'organ':organ_denominations[organ_index],
                                    }
                        if (args.include_multiple_conv and args.definition=='multiple') and not (args.go_through_negative_after_multiple and selected_sentences[sentence_index][-1]=='0') :
                            fn_inputs['conv_input'] = all_convs[sentence_index]
                        sentence_outputs = get_node_output_fn(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args)
                        all_report_outputs.append(sentence_outputs)
                    else:
                        all_report_outputs.append([-2,-1,-1,-1])

            zipped_all_report_outputs = [list(row) for row in zip(*all_report_outputs)]
            if len(zipped_all_report_outputs)>0:
                for index_task, task in enumerate(tasks):
                    if not do_tasks[index_task]:
                        continue
                    report_outputs = zipped_all_report_outputs[index_task]
                    if task=='location':
                        report_outputs = [x[1:] if (len(x)>0 and x[0]==';') else x for x in report_outputs]
                        report_outputs = [-1 if x == '' else x for x in report_outputs]

                    line_text = ','.join([str(value) for value in report_outputs])
                    get_lock(report_label_lock)
                    try:
                        with Open(report_label_file, "a") as f:
                            skipped_report = report.replace('"','""').replace('\n',' ')
                            skipped_line_text = line_text.replace('"','""').replace('\n',' ')
                            f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},"{skipped_report}",{task},{skipped_line_text}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
                    finally:
                        report_label_lock.release()

            end_time_labels = time.time()
            get_lock(report_timing_lock)
            try:
                with Open(report_timing_file, "a") as f:
                    f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},labels,{end_time_labels-start_time_labels}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
            finally:
                report_timing_lock.release()
            all_report_outputs = zipped_all_report_outputs
        else:
            row_labels_this_organ = starting_state_filtered_by_id[starting_state_filtered_by_id['organ']==organ_denominations[organ_index]][type_denominations].values
            assert(len(row_labels_this_organ)==1)
            all_report_outputs = [row_labels_this_organ[0],[],[],[]]
        start_time_urgency = time.time()
        urgency_output = []
        if not do_labels[organ_index]:
            for type_index in range(len(type_denominations)):
                urgency_output.append(-1)
        else:
            for type_index in range(len(type_denominations)):
                # sentence_index = type_index if args.definition=='multiple' else 0
                sentence_index = 0
                if all_report_outputs[0][type_index] in [1,-1]:   
                    selected_report = '. '.join(selected_sentences_by_organ[sentence_index]).replace('..', '.').replace('..', '.')           
                    current_node = urgency_prompt
                    sentence_outputs = get_node_output_fn(current_node, {'report_':report, 'sentence_':selected_report, 
                        'label_':type_index + organ_index*len(type_denominations), 
                        'tokenizer': tokenizer, 
                        'model': model, 
                        'model_name': model_name, 
                        'args':args, 
                        'do_tasks': do_tasks,
                        'label_set_mimic_generic': label_set_mimic_generic, 
                        'label_set_mimic_complex': label_set_mimic_complex, 
                        'heart_labels': heart_labels, 
                        'id':f"{row['subject_id']}_{row['study_id']}",
                        'organ':organ_denominations[organ_index],
                        }, tokenizer, model, model_name, do_tasks, args)
                    urgency_output.append(sentence_outputs[0])
                else:
                    urgency_output.append(-1)
        
        line_text = ','.join([str(value) for value in urgency_output])
        get_lock(report_label_lock)
        try:
            with Open(report_label_file, "a") as f:
                skipped_report = report.replace('"','""').replace('\n',' ')
                skipped_line_text = line_text.replace('"','""').replace('\n',' ')
                f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},"{skipped_report}",urgency,{skipped_line_text}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
        finally:
            report_label_lock.release()
        end_time_urgency = time.time()
        get_lock(report_timing_lock)
        try:
            with Open(report_timing_file, "a") as f:
                f.write(f'{subject_id}_{study_id},{organ_denominations[organ_index]},urgency,{end_time_urgency-start_time_urgency}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
        finally:
            report_timing_lock.release()
                  
    with tqdm_joblib(tqdm(desc="Parsing reports", total=len(chexpert_df)*len(organ_denominations))) as progress_bar:
        try:
            Parallel(n_jobs=args.n_jobs, batch_size = 1, require='sharedmem')(delayed(run_one_report)(idx,row, organ_index, [model]) for idx, row in chexpert_df.reset_index().iterrows() for organ_index in range(len(organ_denominations)))
        finally:
            end_full_script_time = time.time()
            get_lock(report_timing_lock)
            try:
                with Open(report_timing_file, "a") as f:
                    f.write(f'-,-,full,{end_full_script_time-start_full_script_time}\n'.encode('ascii', errors='replace').decode('ascii').replace('?', ' '))
            finally:
                report_timing_lock.release()