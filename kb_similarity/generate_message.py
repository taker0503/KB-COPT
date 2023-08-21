import os
import pandas as pd
from nltk import tokenize
import openai
import random
import pandas as pd
from tqdm import tqdm
import SemanticSearch

INITIAL_TEMPLATE = """
[개요 시작]
{개요}
[개요 끝]

당신은 고객의 구매 행동 과정, AIDA / AIDMA / AISAS 법칙 및 소비자 심리학과 같은 검증된 카피라이팅 공식에 경험이 풍부한 카피라이팅 전문가입니다. 관심을 끄는 헤드라인, 마음을 사로잡는 리드, 설득력 있는 클릭 유도문안을 작성하는 요령이 있습니다. 
이를 활용하여 위에 제공된 개요를 바탕으로 국민은행의 상품, 서비스 및 이벤트에 대한 독특하고 혁신적인 마케팅 메세지를 만들어주세요. 

다음 지침을 따르십시오
1. 아래 제공된 예시들을 참고하여 더 발전된 마케팅 메세지를 만드십시오. 
2. 위에 제공된 개요에 없는 내용은 포함하지 마시오.
3. 창의력을 발휘하여 트렌드에 맞춘 마케팅 문구를 개발하시오. 
4. 고객들의 편의를 위해 가독성을 향상시키십시오.

[예시 시작]
{예시}
[예시 끝]
"""

instructions_dict = {
        'initial' : INITIAL_TEMPLATE
    }

def set_openai_key(key):
    openai.api_key = key


def ask_question(text, overview, api_type, top_k, temperature, max_gpt_token=1200):
    instruction = instructions_dict['initial']
    chatgpt_messages, top1_example = prepare_chatgpt_message(
        instruction,
        text,
        top_k,
        overview
    )
    response, n_tokens = call_chatgpt(chatgpt_messages, model=api_type, temperature= temperature, max_tokens=max_gpt_token)

    return response, top1_example
        

def prepare_chatgpt_message(task_prompt, text, top_k, overview):
    overview_all = f'-목적: {text}'+ overview
    top1_example = 'X'
    input_prompt = task_prompt.replace('{개요}', overview_all)
    if top_k =='No example':
        topk_example = '예시 없음'

    else:
        topk_example = SemanticSearch.return_topk_documents(text, top_k)
        st_idx = topk_example.find('[예시 2]')
        top1_example = topk_example[:st_idx]

    input_prompt = input_prompt.replace('{예시}', topk_example)
    messages = [{"role": "system", "content": input_prompt}]
    return messages, top1_example

def call_chatgpt(chatgpt_messages, max_tokens, model, temperature):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temperature, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def generate_messages(
        text:str ='input query',
        overview:str = 'overview',
        top_k: str = '1',
        api_type:str='gpt-3.5-turbo',
        temperature: float = 0
        ):
    #개인 api key 입력
    open_ai_key = 'chatgpt api key'
    set_openai_key(open_ai_key)
    response, top1_example = ask_question(text, overview, api_type, top_k, temperature)

    return response, top1_example
