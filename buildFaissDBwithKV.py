import faiss
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import numpy as np
import os,gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_faiss_index_and_mapping(
    dataset_name='squad',
    split='train',
    embedding_model_name='all-MiniLM-L6-v2',
    llm_model_name='facebook/opt-1.3b',  # 사용할 LLM 모델 이름
    index_file='context_index.faiss',
    id_to_text_file='id_to_text_mapping.json',
    kvcache_dir='./kvcache',  # KV 캐시를 저장할 디렉토리
    batch_size=8,
    normalize_embeddings=True
):
    # KV 캐시 디렉토리 생성
    os.makedirs(kvcache_dir, exist_ok=True)

    # 1. 데이터셋 로드
    print(f"Loading {dataset_name} dataset...")
    with open("./dataset/msquad_balance.json", 'r', encoding='utf-8') as f:
        squad_dataset = json.load(f)
    dataset = sorted(squad_dataset, key=lambda x: x['id'])[:2000]
    print(f"Dataset loaded: {len(dataset)} examples.")

    # 2. 컨텍스트 추출 및 중복 제거
    print("Extracting and deduplicating contexts...")
    contexts = list(set(example['context'] for example in dataset))
    print(f"Total unique contexts: {len(contexts)}")

    # 3. ID-텍스트 매핑 생성
    print("Creating ID-to-text mapping...")
    id_to_text = {idx: context for idx, context in enumerate(contexts)}
    
    # ID-텍스트 매핑 파일 저장
    with open(id_to_text_file, 'w', encoding='utf-8') as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)
    print(f"ID-to-text mapping saved to {id_to_text_file}")

    # 4. 임베딩 모델 로드
    print(f"Loading embedding model ({embedding_model_name})...")
    embed_model = SentenceTransformer(embedding_model_name)
    embed_model.eval()
    if torch.cuda.is_available():
        embed_model = embed_model.to('cuda')
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Embedding model loaded on {device}")

    # 5. 컨텍스트 임베딩 생성
    print("Generating embeddings for contexts...")
    embeddings = []
    for i in tqdm(range(0, len(contexts), batch_size)):
        batch_contexts = contexts[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = embed_model.encode(
                batch_contexts,
                batch_size=len(batch_contexts),
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings
            )
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # 6. Faiss 인덱스 생성
    print("Building Faiss index...")
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (for normalized vectors)
    index.add(embeddings)
    print(f"Faiss index built with {index.ntotal} vectors.")

    # 7. 인덱스 저장
    faiss.write_index(index, index_file)
    print(f"Faiss index saved to {index_file}")

    # 8. LLM 모델 및 토크나이저 로드
    print(f"Loading LLM model ({llm_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_auth_token=ACCESS_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=ACCESS_TOKEN
    )
    model.eval()
    print("LLM model loaded.")

    # 9. 컨텍스트에 대한 KV 캐시 생성 및 저장
    print("Generating and saving KV caches for contexts...")
    for idx, context in tqdm(id_to_text.items()):
        # 이미 처리된 경우 스킵
        kvcache_path = os.path.join(kvcache_dir, f'kvcache_{idx}.pt')
        if os.path.exists(kvcache_path):
            continue  # 이미 존재하면 스킵

        # 컨텍스트 토크나이즈
        inputs = tokenizer(
            f"Context: {context}",
            return_tensors='pt',
            max_length=512  # 필요한 경우 길이 조정
        )
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values  # 튜플 형태

        # past_key_values를 CPU로 이동하고 저장
        past_key_values_cpu = []
        for layer in past_key_values:
            layer_cpu = tuple(tensor.cpu() for tensor in layer)
            past_key_values_cpu.append(layer_cpu)

        # KV 캐시를 파일로 저장
        torch.save(past_key_values_cpu, kvcache_path)

    print(f"KV caches saved to directory: {kvcache_dir}")
    print("All done!")

if __name__ == "__main__":
    ACCESS_TOKEN = "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"
    model_size = ["opt-1.3b", 
                  "opt-2.7b", 
                  "opt-6.7b"]  # 사용할 모델 크기 리스트
    for msize in model_size:
        LLM_MODEL_NAME = f"facebook/{msize}"
        kvcache_dir = f"./kvcaches/{msize}"
        build_faiss_index_and_mapping(llm_model_name=LLM_MODEL_NAME, kvcache_dir=kvcache_dir)
        LLM_MODEL_NAME = None
        gc.collect()
        torch.cuda.empty_cache()
