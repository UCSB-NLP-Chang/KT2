import os
import json
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.LLM_tools.gpt_call import call_gpt
from preprocess.LLM_tools.prompt import fix_translate_system_prompt, fix_translate_user_prompt, translation_check_system_prompt, translation_check_user_prompt

from preprocess.preprocess_config.config import (
    translation_dir,
    check_dir,
    handle_fails,
    times_of_attempt,
    dataset
)

def main(handle_fails=handle_fails, times_of_attempt=times_of_attempt):
    question_path = f'{translation_dir}/{dataset}_question_dict.json'
    question_dict = json.load(open(question_path, 'r'))

    wrong_translation_path = f'{translation_dir}/translation_check/translation_check_false.json'
    with open(wrong_translation_path, 'r', encoding='utf-8') as f:
        wrong_translations = json.load(f)
        questions = [item['question'] for item in wrong_translations]
    
    all_fix_results = []
    all_fail_results = []
    all_check_results = []

    # Handle fails in the previous attempts
    if handle_fails:
        with open(f'{check_dir}/fail_fix_questions.json', 'r') as f:
            questions = json.load(f)

    if os.path.exists(f'{check_dir}/translation_fix_true.json'):
        with open(f'{check_dir}/translation_fix_true.json', 'r') as f:
            all_fix_results = json.load(f)
    if os.path.exists(f'{check_dir}/translation_fix_false.json'):
        with open(f'{check_dir}/translation_fix_false.json', 'r') as f:
            all_fail_results = json.load(f)
    
    existing_questions = [i['question'] for i in all_fix_results] + [i['question'] for i in all_fail_results]
    
    temp = []
    temp.extend(questions)
    temp.extend([item['question'] for item in all_fix_results])

    i = 0
    for q in [item['question'] for item in wrong_translations]:
        if q not in temp:
            print(f"Question {q} not found in temp")
            questions.append(q)
            i +=1

    # check if the question is in the question_dict
    for q in questions:
        if q not in question_dict:
            print(f"Question {q} not found in question_dict")
            breakpoint()
    
    fail_questions = []
    questions = questions
    index = 1
    for q in questions:

        print(f"Processing question {index} of {len(questions)}")
        index += 1

        if q in existing_questions:
            print(f"Skipping {q} because it has already been checked")
            continue

        original_q = q
        translation = question_dict[q]
        # Analyze the translation
        attempt = 0
        while attempt <= times_of_attempt:
            try:
                user_prompt = translation_check_user_prompt(original_q,translation)
                system_prompt = translation_check_system_prompt()
                llm_output = call_gpt(system_prompt, user_prompt, deployment_name='gpt-4o')

                llm_output = llm_output.replace("\\","\\\\")
                # Parse the GPT-4o output
                result = llm_output
                result_parsed = json.loads(result)
                
                if result_parsed["correct_translation"] == True or result_parsed["correct_translation"] == "True" or result_parsed["correct_translation"] == "true":
                    all_fix_results.append({'question': original_q,
                                             'translated_question': translation})
                    all_check_results.append(True)
                    print('fixed!')
                    break

                else:
                    reason = result_parsed['explanation']                  

                if attempt == times_of_attempt:
                    break

                attempt += 1

                # Fix the translation
                system_prompt = fix_translate_system_prompt()
                user_prompt = fix_translate_user_prompt(original_q,translation,reason)
                # Call the Llama 3.2 with the prompt

                llm_output = call_gpt(system_prompt, user_prompt, deployment_name='gpt-4o')

                def fix_json_string(raw_json_str):
                    pattern = r'(".*?")\s*:\s*"(.*?)"(?=,|\s*})' 
                    matches = re.findall(pattern, raw_json_str, flags=re.DOTALL)
                    
                    for key, val in matches:
                        fixed_val = val.replace('"', r'\"') 
                        fixed_val = re.sub(r'(?<!\\)(\\")', r'\1', fixed_val)
                        fixed_val = re.sub(r'(?<!\\)(?<!\\\\)(?<!\\")"', r'\\"', fixed_val)
                        raw_json_str = raw_json_str.replace(f'{key}: "{val}"', f'{key}: "{fixed_val}"')
                    
                    return raw_json_str
                
                clean_output = fix_json_string(llm_output)
                result_parsed  = json.loads(clean_output)

                translation = result_parsed

            except Exception as e:
                print(e)
                attempt += 1

        if attempt == times_of_attempt and (result_parsed["correct_translation"] == False or result_parsed["correct_translation"] == "False" or result_parsed["correct_translation"] == "false") :
            fail_questions.append(q)
            all_fail_results.append({'question': original_q,
                                     'translated_question': translation,
                                     'reason': reason})
            all_check_results.append(False)
            print('failed!')
    
    # Determine the output file path
    output_true_file = os.path.join(check_dir, f'translation_fix_true.json')
    output_false_file = os.path.join(check_dir, f'translation_fix_false.json')
    fix_question_path = os.path.join(check_dir, f'fix_question_dict.json')

    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_true_file)):
        os.makedirs(os.path.dirname(output_true_file))
    if not os.path.exists(os.path.dirname(output_false_file)):
        os.makedirs(os.path.dirname(output_false_file))
    if not os.path.exists(os.path.dirname(fix_question_path)):
        os.makedirs(os.path.dirname(fix_question_path))

    # Write the result to a txt file
    with open(output_true_file, 'w', encoding='utf-8') as f:
        json.dump(all_fix_results, f, ensure_ascii=False, indent=4)
    with open(fix_question_path, 'w', encoding='utf-8') as f:
        json.dump(question_dict, f, ensure_ascii=False, indent=4)
    with open(output_false_file, 'w', encoding='utf-8') as f:
        json.dump(all_fail_results, f, ensure_ascii=False, indent=4)


    # print accuracy
    print(f"Total questions: {len(all_check_results)}")
    if len(all_check_results) > 0:
        print(f"Accuracy: {sum(all_check_results) / len(all_check_results)}")
    print(f"Failed questions: {len(fail_questions)}")

    with open(f'{check_dir}/fail_fix_questions.json', 'w') as f:
        json.dump(fail_questions, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
    print("Done!")