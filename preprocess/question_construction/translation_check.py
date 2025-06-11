
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.LLM_tools.gpt_call import call_gpt
from preprocess.LLM_tools.prompt import translation_check_system_prompt, translation_check_user_prompt

from preprocess.preprocess_config.config import (
    translation_dir,
    check_dir,
    handle_fails,
    times_of_attempt,
    dataset,
)

def main(handle_fails=handle_fails, times_of_attempt=times_of_attempt):
    question_path = f'{translation_dir}/{dataset}_question_dict.json'
    question_dict = json.load(open(question_path, 'r'))
    questions = list(question_dict.keys())

    # Determine the output file path
    output_true_file = os.path.join(check_dir, f'translation_check_true.json')
    output_false_file = os.path.join(check_dir, f'translation_check_false.json')

    # Create the directory if it doesn't exist
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
 
    fail_questions = []
    all_true_results = []
    all_false_results = []
    all_check_results = []

    # Read the existing content if the file already exists
    if os.path.exists(f'{check_dir}/translation_check_true.json'):
        with open(f'{check_dir}/translation_check_true.json', 'r', encoding='utf-8') as f:
            all_true_results = json.load(f)
        with open(f'{check_dir}/translation_check_false.json', 'r', encoding='utf-8') as f:
            all_false_results = json.load(f)
    
    print(len(all_true_results), len(all_false_results))
    
    existing_questions = [i['question'] for i in all_true_results] + [i['question'] for i in all_false_results]
    
    # Handle fails in the previous attempts
    if handle_fails:
        with open(f'{check_dir}/fail_check_questions.json', 'r') as f:
            questions = json.load(f)


    q_index = 1
    for q in questions:

        print(f"Checking question {q_index} of {len(questions)}")
        q_index += 1

        if q in existing_questions:
            print(f"Skipping {q} because it has already been checked")
            continue

        original_q = q
        translated_q = question_dict[q]

        system_prompt = translation_check_system_prompt()
        user_prompt = translation_check_user_prompt(original_q,translated_q)

        # Call the gpt-4o-mini with the prompt

        attempt = 0
        while attempt < times_of_attempt:
            import time
            time.sleep(2)
            try:
                llm_output = call_gpt(system_prompt, user_prompt)


                import re

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

                formatted_result = {
                    "question": original_q,
                    "translated_question": translated_q,
                    "explanation": result_parsed["explanation"],
                    "correct_translation": result_parsed["correct_translation"]
                }
                if result_parsed["correct_translation"] == True or result_parsed["correct_translation"] == "true" or result_parsed["correct_translation"] == "True":
                    all_true_results.append(formatted_result)
                else:
                    all_false_results.append(formatted_result)
                all_check_results.append(result_parsed["correct_translation"])
                break
                    
            except Exception as e:
                print(e)
                attempt += 1

        if attempt == times_of_attempt:
            fail_questions.append(q)

    # Write the result to a json file
    with open(output_true_file, 'w', encoding='utf-8') as f:
        json.dump(all_true_results, f, ensure_ascii=False, indent=4)
    with open(output_false_file, 'w', encoding='utf-8') as f:
        json.dump(all_false_results, f, ensure_ascii=False, indent=4)


    # print accuracy
    print(f"Total questions: {len(all_check_results)}")
    if len(all_check_results) > 0:
        print(f"Accuracy: {sum(all_check_results) / len(all_check_results)}")
    print(f"False questions: {len(all_false_results)}")

    with open(f'{check_dir}/fail_check_questions.json', 'w') as f:
        json.dump(fail_questions, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
    print("Done!")