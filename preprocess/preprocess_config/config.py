import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('./preprocess/preprocess_config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

api_key = config['api_key']
api_version = config['api_version']
api_base = config['api_base']
deployment_name = config['deployment_name']

dataset = config['dataset']
times_of_attempt = config['times_of_attempt']
handle_fails = config['handle_fails']

data_dir = os.path.join(config['data_dir'],dataset)
MOOCRadar_dir = os.path.join(config['data_dir'],'MOOCRadar')
translation_dir = os.path.join(data_dir,'translated_data')
check_dir = os.path.join(translation_dir,'translation_check')
output_dir = os.path.join(config['output_dir'],dataset)

dataset_dir = config['dataset_dir']
graph_dir = config['graph_dir']
