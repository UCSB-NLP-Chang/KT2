import os
import sys
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.LLM_tools.gpt_call import call_gpt
from preprocess.LLM_tools.prompt import translation_system_prompt, translation_user_prompt

from preprocess.preprocess_config.config import (
    data_dir,
    translation_dir,
    handle_fails,
    times_of_attempt,
    dataset,
)

def translate_question(handle_fails=handle_fails, times_of_attempt=times_of_attempt):
    
    files = f'{data_dir}/question_info.json'

    all_questions = json.load(open(files, 'r', encoding='utf-8'))

    questions = []
    
    for i in all_questions.keys():
        if dataset == 'MOOCRadar':
            questions.append(all_questions[i]['content'].strip() + '\n Option: ' + str(all_questions[i]['options']))
        else:
            questions.append(all_questions[i]['content'].strip())
    
    question_dict = {}
    questions_to_retry = []
    if handle_fails:
        questions = json.load(open(f'{translation_dir}/fail_translation_questions.json', 'r', encoding='utf-8'))
        # remove duplicates
        questions = list(set([q.strip() for q in questions]))
        
    if os.path.exists(os.path.join(translation_dir, f'{dataset}_question_dict.json')):
        question_dict = json.load(open(os.path.join(translation_dir, f'{dataset}_question_dict.json'), 'r', encoding='utf-8'))
    
    print(len(question_dict))
    print(len(questions))

    q_index = 1
    for q in questions:
        print(f"Processing question {q_index} of {len(questions)}")
        q_index += 1
        if q in question_dict:
            print(f"Question {q} already exists in question_dict. Skipping...")
            continue

        system_prompt = translation_system_prompt()
        user_prompt = translation_user_prompt(q)

        attempt = 0
        while attempt < times_of_attempt:
            try:
                llm_output = call_gpt(system_prompt, user_prompt)
                question_dict[q] = llm_output

                break

            except Exception as e:
                print(e)
                attempt += 1

        if attempt == times_of_attempt:
            if q not in questions_to_retry:
                questions_to_retry.append(q)
                print(q)
            continue

    # mkdir if not exists
    if not os.path.exists(translation_dir):
        os.makedirs(translation_dir)

    output_file = os.path.join(translation_dir, f'{dataset}_question_dict.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(question_dict, f, ensure_ascii=False, indent=4)

    with open(f'{translation_dir}/fail_translation_questions.json', 'w') as f:
        json.dump(questions_to_retry, f, ensure_ascii=False, indent=4)

    print('Fail questions: ', len(questions_to_retry))

if __name__ == '__main__':
    translate_question()
    print("Done!")