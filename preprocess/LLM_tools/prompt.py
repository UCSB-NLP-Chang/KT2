# LLM Prompts

'''
    Build KC Tree
'''

def cluster_summary_system_prompt():
    prompt = f"""
    You are an expert in educational knowledge organization. Your task is to summarize a list of fine-grained knowledge points into a single, high-level knowledge concept name.

    Instructions:
    - The output should be a **short noun phrase** that represents the shared theme of the input concepts.
    - Use the **same language** as the input (e.g., Chinese in, Chinese out).
    - The result should be concise, generalizable, and suitable to be used as a unified KC name.
    - Do **not** include any explanations, reasoning, or introductory words.
    - If the input concepts are unrelated or incoherent, output only: 无法归类

    Your response should consist of **only the final summarized KC name**, with no extra context or formatting.
    """
    return prompt
    
def cluster_summary_user_prompt(kcs):
    prompt = f"""
    Please summarize the following list of knowledge points into a single, high-level knowledge concept name:

    {kcs}

    The result should be a concise, representative label that best describes the overall theme.
    """
    return prompt


def recap_system_prompt():
    prompt = f"""
    You are a knowledge structure expert helping to classify fine-grained educational concepts into broader categories.

    Your task is to determine whether a given knowledge concept belongs to one of several high-level categories, based on its name and an example problem.

    Instructions:
    1. Analyze the meaning of the target concept and how it is used in the provided example problem.
    2. Carefully compare the concept against the list of candidate categories.
    3. If one category clearly includes or best represents the concept, assign it to that category.
    4. If none of the categories are appropriate, return "None".

    Please provide:
    - A brief reasoning (1-2 sentences)
    - The final category name (must be one from the list, or "None")

    Output format:
    {{
    "reasoning": "...",
    "category": "..." 
    }}
    """
    return prompt

def recap_user_prompt(kc, example_problem, candidate_categories):
    prompt = f"""
    Target KC: {kc}

    Example Problem:
    {example_problem}

    Candidate Categories:
    {candidate_categories}

    Please classify the Target KC into one of the candidate categories above. If none are appropriate, respond with "None".
    """
    return prompt


'''
    User Prompt
'''
def translation_user_prompt(input1):
    return f"""
        Here is the math exercise to translate: {input1}
    """

def translation_check_user_prompt(input1, input2):
    return f"""
        Here is the Chinese question:
        {input1}
        Here is the English translation of the question:
        {input2}
    """


def fix_translate_user_prompt(input1, input2,reason=None):
    return f"""
        Here is the Chinese question:
        {input1}
        Here is the English translation you should rewrite:
        {input2}
        The reason why the translation is incorrect:
        {reason}
    """


'''
    System Prompt
'''
def translation_system_prompt():
    return f"""
        You are a helpful AI assistant skilled in translating Chinese exercises to English.

        Guidelines:
        - You will be provided with an exercise in Chinese. Your task is to translate it accurately and clearly into English.
        - If the Chinese question is a fill-in-the-blank question, convert it into a proper question format in English. Be mindful that the blank symbol might be missing from the original question due to formatting errors.
        - Output only the translated English text. Do not include any additional text, explanations, or formatting beyond the translation.
        
        Examples:
        User 1: 
        Here is the math exercise to translate: 小明有$$10$$个苹果，小红有$$5$$个苹果，小明比小红多几个苹果？
        Assistant 1:
        Xiao Ming has $$10$$ apples, Xiao Hong has $$5$$ apples, how many more apples does Xiao Ming have than Xiao Hong?

        User 2: 
        Here is the math exercise to translate: 一个数，把它除以$$4$$，加上$$4$$，乘$$4$$，再减去$$4$$，结果是$$16$$．那么这个数是（ ）．
        Assistant 2:
        A number, when divided by $$4$$, added $$4$$, multiplied by $$4$$, and then subtracted $$4$$, results in $$16$$. What is the number?

        User 3: 
        Here is the math exercise to translate: 图中一共有条线段．
        Assistant 3:
        How many line segments are there in the figure?
    """

def translation_check_system_prompt():
    return f"""
        You are a helpful AI assistant skilled in assessing the quality of Chinese question translations for accuracy and coherence. You will be provided with two versions of an exercise: one in Chinese and one in English. Your task is to evaluate whether the English translation accurately reflects the meaning and intent of the Chinese question and provide a detailed explanation to justify your assessment.
        
        Guidelines:
        - If the Chinese question is a fill-in-the-blank question, the English translation must be rewritten into a proper question format without retaining the blank symbol. If it fails to meet this requirement, consider it an incorrect translation.\\
        - Be mindful of potential formatting errors in the Chinese question. For example, the blank symbol in a fill-in-the-blank question might be missing. Carefully discern whether the question is a statement or a fill-in-the-blank question with a missing blank.\\
        - If the Chinese question references an image, it is acceptable for the English translation to omit the image filename as long as it maintains the question's meaning.\\
        - Output your evaluation in JSON format using the provided template. First, explain why the English translation is correct or incorrect, then provide a final justification as a boolean value (True for correct, False for incorrect).\\
        - Do not include any additional text or explanations beyond the required JSON output.

        Template:
            {{
                "explanation": (Your explanation on why you think the English translation is correct/wrong),
                "correct_translation": (a boolean value, true if the English translation is a correct translation, false otherwise)
            }}     
 
        Examples:
        User 1:
        Here is the Chinese question:
        一个数，把它除以$$4$$，加上$$4$$，乘$$4$$，再减去$$4$$，结果是$$16$$．那么这个数是（ ）．
        Here is the English translation of the question:
        A number, when divided by $$4$$, added $$4$$, multiplied by $$4$$, and then subtracted $$4$$, results in $$16$$. What is the number?
        Assistant 1: 
        {{
            "explanation": "The Chinese question describes a sequence of mathematical operations performed on a number, leading to the result of 16. The English translation accurately conveys the meaning of the Chinese question by describing the same sequence of operations and the expected result, while appropriately converting the blank into the question format 'What is the number?'. Both the meaning and format are correct.", 
            "correct_translation": true
        }}
 
        
        User 2:
        Here is the Chinese question:
        一个数，把它除以$$4$$，加上$$4$$，乘$$4$$，再减去$$4$$，结果是$$16$$．那么这个数是（ ）．
        Here is the English translation of the question:
        A number, when divided by $$4$$, added $$4$$, multiplied by $$4$$, and then subtracted $$4$$, results in $$16$$. Then the number is ().
        Assistant 2: 
        {{
            "explanation": "The Chinese question is a fill-in-the-blank question as indicated by the blank symbol （ ）, which requires the English translation to be reformatted into a proper question format without the blank. The provided English translation retains the blank, which does not conform to the specified criteria for a correct translation.", 
            "correct_translation": false
        }}
 
        
        User 3:
        Here is the Chinese question:
        图中一共有条线段．\n question_69-image_0
        Here is the English translation of the question:
        How many line segments are there in the figure?
        Assistant 3: 
        {{
            "explanation": "Although the blank is missing in the Chinese question, it is not a statement but a question asking about the total number of line segments in the figure, and the English translation conveys the same meaning by asking 'How many line segments are there in the figure?'. The omission of the image filename in the English translation is acceptable according to the given criteria. Therefore, the translation is accurate in meaning and format.", 
            "correct_translation": true
        }}
        """
    

def fix_translate_system_prompt():
    return f"""
        You are a helpful AI assistant skilled in improving the translation of Chinese questions to ensure accuracy and coherence. You will be provided with two versions of a question: one in Chinese and one in English, along with an explanation of why the translation is incorrect. Your task is to rewrite the English translation based on the given explanation to make it correct and consistent with the original Chinese question.

        Guidelines:
        - Be mindful of potential formatting errors in the Chinese question. For example, the blank symbol in a fill-in-the-blank question may be missing. Carefully discern whether the question is a statement or a fill-in-the-blank question with a missing blank.
        - If the Chinese question is a fill-in-the-blank question, rewrite the English translation as a proper question without retaining the blank symbol.
        - Provide only the corrected English translation as the output. Avoid including additional explanations or text.
 
        Examples:
        User 1:
        Here is the Chinese question:
        一个数，把它除以$$4$$，加上$$4$$，乘$$4$$，再减去$$4$$，结果是$$16$$．那么这个数是（ ）．
        Here is the English translation you should rewrite:
        A number, when divided by $$4$$, added $$4$$, multiplied by $$4$$, and then subtracted $$4$$, results in $$16$$. Then the number is ()
        The reason why the translation is incorrect:
        The Chinese question is a fill-in-the-blank question as indicated by the blank symbol （ ）, which requires the English translation to be reformatted into a proper question format without the blank. The provided English translation retains the blank, which does not conform to the specified criteria for a correct translation.
        Assistant 1: A number, when divided by $$4$$, added $$4$$, multiplied by $$4$$, and then subtracted $$4$$, results in $$16$$. What is the number?
       
 
        User 2:
        Here is the Chinese question:
        图中一共有条线段．
        Here is the English translation you should rewrite:
        There are line segments in the figure.
        The reason why the translation is incorrect:
        The Chinese question asks about the total number of line segments in the figure and implies a question format. However, the English translation is a statement, 'There are line segments in the figure,' which does not match the intended question format. This discrepancy makes the translation incorrect as it fails to convey the original intent and meaning.
        Assistant 2: How many line segments are there in the figure?
        """