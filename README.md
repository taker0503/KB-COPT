# KB-COPT

## Overview
KB국민은행 AI 챌린지 
ChatGPT를 활용한 마케팅 카피라이터 AI
1) 설문조사를 활용한 데이터 수집 및 gpt api를 통한 STS/NLI 데이터셋 구성
2) 직접 구성한 데이터셋으로 Sentence Embedding model finetuning (STS, NLI)
3) 코사인 유사도를 기반으로 입력 쿼리와 가장 유사한 Top-k documents 반환
4) Top-K documents를 few-shot으로 넣어주어 최적의 마케팅 메시지 생성

## Model
huggingface에서 model 로드

```python
from transformers import AutoTokenizer, AutoModel
MODEL_NAME = 'TaeLeeKyung/KoSimCSE-roberta-multitask-marketing-lms'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
```

## Usage
```bash
$ cd kb_similarity
$ generate_msg.py
```

## Demo
<img src="/Users/user/Desktop/Copt.png">
