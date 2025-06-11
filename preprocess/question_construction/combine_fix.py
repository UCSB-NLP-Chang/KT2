import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.preprocess_config.config import (
    translation_dir,
    check_dir,
    dataset,
)

def main():
    question_path = f'{translation_dir}/{dataset}_question_dict.json'
    all_questions = json.load(open(question_path, 'r'))
        
    
    if not os.path.exists(f'{check_dir}/translation_fix_true.json'):
        print(f"File does not exist!")
    
    with open(f'{check_dir}/translation_fix_true.json', 'r', encoding='utf-8') as f:
        question_dict = json.load(f)
    
    # Convert list of dicts to dict using 'question' as key
    question_dict = {item['question']: item for item in question_dict}

    new_dict = dict()

    success = 0

    for key in all_questions.keys():
        if key in question_dict.keys():
            new_dict[key] = question_dict[key]['translated_question']
            success += 1
        else:
            new_dict[key] = all_questions[key]

    print(f"Translation fixed: {success}")

    with open(f'{translation_dir}/final_translated_question_dict.json', 'w', encoding='utf-8') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()