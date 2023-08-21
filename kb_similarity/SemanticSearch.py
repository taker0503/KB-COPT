import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoModel, AutoTokenizer
from bert import BERT

def return_topk_documents(text, top_k):
    data_path = '../Dataset/kb_dataset_deidentified_oneline_overview_all.csv'
    model_path = 'output/training_sts_all_continue_training/'

    df = pd.read_csv(data_path)
    model = BERT(AutoModel.from_pretrained(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 각 메시지별 한줄 개요 코퍼스 생성
    corpus = df.overview.to_list()
    inputs_corpus = convert_to_tensor(corpus, tokenizer)
    corpus_embeddings = model.encode(inputs_corpus)

    # faiss 활용 cosine similarity 계산
    Index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    vectors = corpus_embeddings.detach().numpy()
    vectors = vectors.copy(order='C')
    faiss.normalize_L2(vectors)	
    Index.add(vectors)

    # 새로 만들 마케팅 메시지 예시 Query
    query = text

    # 벡터 사이의 거리 기반 K-nearest neighbor 방식으로 예시 샘플 DB에서 가장 유사한 sentence 벡터 반환

    if top_k == 'Top 1':
        top_k = 1
    elif top_k == 'Top 2':
        top_k = 2
    elif top_k == 'Top 3':
        top_k = 3

    query_embedding = model.encode(convert_to_tensor(query, tokenizer))
    query_embedding = query_embedding.detach().numpy()

    faiss.normalize_L2(query_embedding)
    distances, indexes = Index.search(query_embedding, top_k)
    top_k_document_indexes = indexes[0]

    example_message = [f'\n[예시 {idx+1}]\n{example}' for idx, example in enumerate(df.iloc[top_k_document_indexes].deidentified_text.values)]
    example_msg = ''.join(example_message)
    return example_msg


def convert_to_tensor(corpus, tokenizer):
    inputs = tokenizer(corpus,
                       truncation=True,
                       return_tensors="pt",
                       max_length=50,
                       padding="max_length")
    
    embedding = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
        
    inputs = {'source': torch.LongTensor(embedding),
              'token_type_ids': torch.LongTensor(token_type_ids),
              'attention_mask': attention_mask}
    
    return inputs
