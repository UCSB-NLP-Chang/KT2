from openai import OpenAI, AzureOpenAI
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.preprocess_config.config import (
    deployment_name,
    api_version,
    api_base,
    api_key,
)

def call_gpt(system_prompt, user_prompt,deployment_name=deployment_name):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint = api_base,
    )


    response = client.chat.completions.create(
        model= deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the GPT-4 output
    gpt_output = response.model_dump_json(indent=4)

    # Parse the GPT-4 output
    predictions = json.loads(gpt_output)['choices'][0]['message']['content'].strip()

    return predictions