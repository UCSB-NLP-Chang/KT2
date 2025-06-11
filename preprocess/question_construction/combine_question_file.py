import os
import json
from preprocess.preprocess_config.config import (
    translation_dir,
    output_dir,
    data_dir,
    dataset
)

def main():
    question_path = f'{data_dir}/question_info.json'
    all_questions = json.load(open(question_path, 'r'))
        
    
    if not os.path.exists(f'{translation_dir}/final_translated_question_dict.json'):
        print(f"File does not exist!")
        # translate_question()
    
    question_dict = json.load(open(f'{translation_dir}/final_translated_question_dict.json', 'r', encoding='utf-8'))
    difficulty_dict = json.load(open(f'{data_dir}/question_difficulty_map.json', 'r', encoding='utf-8'))

    new_dict = dict()

    # Process the data in row

    for key in all_questions.keys():
        current_kc = None
        data_row = all_questions[key]


        question = data_row['content'].strip()
        option = str(data_row['options'])

        if dataset == 'MOOCRadar':
            question_key = question + '\n Option: ' + option
        else:
            question_key = question.strip()

        try:
            if key not in difficulty_dict.keys():
                continue
            translated_question = question_dict[question_key]        
            new_dict[key] = dict()
            new_dict[key]['content'] = translated_question
            new_dict[key]['difficulty'] = difficulty_dict[key]
            new_dict[key]['kc'] = all_questions[key]['kc_routes'][0].split('----')[-1]

        except Exception as e:
            print(e)
            breakpoint()

    output_file = os.path.join(output_dir, f'translated_question_info.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
    print("Done!")