import torch, gc
import time
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import faiss
from sentence_transformers import SentenceTransformer
import json
from collections import OrderedDict
from tqdm import tqdm
import os
import torch.nn.functional as F

# KVCacheManager, RAGProcessor, RAGDataset 클래스는 그대로 유지

# LRU 캐시를 사용하고, 같은 document를 사용하는 것을 같은 배치에 들어가도록 처리 : load_kv time을 줄임.

class KVCacheManager:
    def __init__(self, kvcache_dir, cache_size = 16, llm_model="facebook/opt-1.3b"):
        self.kvcache_dir = kvcache_dir
        self.cache = {}     # Memory Cache. 크기 제한  = cache_sizeG
        self.model = llm_model
    
    # CPU 메모리에 있으면 바로 return. 없으면 disk애서 읽어옴.
    # def load_kvcache(self, idx):
    #     cache_type = 1  # From Memory
    #     if idx not in self.cache:
    #         kvcache_path = os.path.join(self.kvcache_dir, f'kvcache_{idx}.pt')
    #         if os.path.exists(kvcache_path):
    #             self.cache[idx] = torch.load(kvcache_path)
    #             cache_type = 0  # From disk
    #     return self.cache.get(idx), cache_type

    # 무조건 Disk에서 읽어 옴.
    def load_kvcache(self, idx):
        cache_type = 0  # From Memory
        kvcache_path = os.path.join(self.kvcache_dir, f'kvcache_{idx}.pt')
        return torch.load(kvcache_path), cache_type

    def clear_cache(self):
        self.cache.clear()
        gc.collect()

    def make_kvcache(self, idx):
        return
    
    def evic_memory_cache(self, idx):
        return


class RAGProcessor:
    def __init__(self, index_file, id_to_text_file, kvcache_manager, embedding_model_name='all-MiniLM-L6-v2', top_k=1):
        self.index = faiss.read_index(index_file)
        with open(id_to_text_file, 'r', encoding='utf-8') as f:
            self.id_to_text = json.load(f)
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.top_k = top_k
        self.kvcache_manager = kvcache_manager

    def search_contexts(self, query):
        # Context Search Time
        batch_search_time = time.time()
        query_vector = self.embed_model.encode([query])
        distances, indices = self.index.search(query_vector, self.top_k)
        results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]
        contexts = [self.id_to_text.get(str(idx), "") for idx, _ in results]
        batch_search_time = time.time() - batch_search_time
 
        ######################## KV 캐시를 Disk에서 CPU메모리로 읽어오는 시간 측정
        batch_kv_load_time = time.time()
        kv_caches, cache_types = zip(*[self.kvcache_manager.load_kvcache(idx) for idx, _ in results])
        batch_kv_load_time = time.time() - batch_kv_load_time
        kv_caches = list(kv_caches)
        cache_types = list(cache_types)
        return contexts, kv_caches, cache_types, batch_search_time, batch_kv_load_time


class RAGDataset(Dataset):
    def __init__(self, prompts, rag_processor):
        self.prompts = prompts
        self.rag_processor = rag_processor

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        contexts, kv_caches, cache_types, batch_search_time, batch_kv_load_time = self.rag_processor.search_contexts(prompt)
        full_prompt = f"Context: {' '.join(contexts)}\nQuestion: {prompt}\nAnswer:"
        que_prompt = f"\nQuestion: {prompt}\nAnswer:"  # Context를 제외한 prompt
        kv_seq_len = (kv_caches[0][0][0].shape)[2]
        return que_prompt, kv_caches[0], kv_seq_len, cache_types, full_prompt, batch_search_time, batch_kv_load_time if kv_caches else None


def custom_collate_fn(batch):
    batch_prompts = [item[0] for item in batch]  # prompts 추출
    batch_past_key_values = [item[1] for item in batch]  # past_key_values 추출
    batch_past_seq_lens = [item[2] for item in batch]  # 각 샘플의 sequence length 추출
    batch_cache_types = [item[3] for item in batch]  # 각 샘플의 sequence length 추출
    batch_full_prompts = [item[4] for item in batch]
    batch_search_time = sum([item[5] for item in batch])
    batch_kv_load_time = sum([item[6] for item in batch])

    combined_past_key_values = []
    kv_attention_masks = []

    num_layers = len(batch_past_key_values[0])
    
    for layer_idx in range(num_layers):
        keys = [sample[layer_idx][0] for sample in batch_past_key_values]  # 각 샘플의 key 텐서
        values = [sample[layer_idx][1] for sample in batch_past_key_values]  # 각 샘플의 value 텐서
        
        max_seq_len = max(key.size(2) for key in keys)
        
        padded_keys = [F.pad(key, (0, 0, 0, max_seq_len - key.size(2))) for key in keys]  # key 패딩
        padded_values = [F.pad(value, (0, 0, 0, max_seq_len - value.size(2))) for value in values]  # value 패딩
        
        combined_key = torch.cat(padded_keys, dim=0)  # (batch_size, num_heads, sequence_length, head_dim)
        combined_value = torch.cat(padded_values, dim=0)  # (batch_size, num_heads, sequence_length, head_dim)

        attention_masks = [torch.cat([torch.ones(key.size(2)), torch.zeros(max_seq_len - key.size(2))]) for key in keys]
        combined_attention_mask = torch.stack(attention_masks)  # (batch_size, sequence_length)
        
        combined_past_key_values.append((combined_key, combined_value))
        kv_attention_masks.append(combined_attention_mask)

    return batch_prompts, combined_past_key_values, kv_attention_masks[0], batch_past_seq_lens, batch_cache_types, batch_full_prompts, batch_search_time, batch_kv_load_time


ACCESS_TOKEN = "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"

def run_improved_rag(model_size):
    LLM_MODEL_NAME = f"facebook/{model_size}"
    
    # 데이터셋 로드 및 준비
    with open("./dataset/msquad_balance.json", 'r', encoding='utf-8') as f:
        squad_dataset = json.load(f, object_pairs_hook=OrderedDict)
    s_squad_dataset = sorted(squad_dataset, key=lambda x: x['id'])
    prompts = [entry['question'] for entry in s_squad_dataset][:2000]

    # KVCacheManager 및 RAGProcessor 초기화
    kvcache_manager = KVCacheManager(f"./kvcaches/{model_size}")
    rag_processor = RAGProcessor('context_index.faiss', 'id_to_text_mapping.json', kvcache_manager, top_k=1)

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_auth_token=ACCESS_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=ACCESS_TOKEN
    )
    model.eval()
    dataset = RAGDataset(prompts, rag_processor)
    
    for bsize in range(1, 11):
        results = []
        data_loader = DataLoader(dataset, batch_size=bsize, shuffle=False, collate_fn=custom_collate_fn)
        
        with torch.no_grad():
            # tqdm을 사용하여 진행 상황 표시
            pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Model:{model_size}, Batch size:{bsize}", unit="batch")
            for batch_idx, (batch_prompts, batch_kv_caches, batch_kv_attention_masks, batch_past_seq_lens, batch_cache_types, batch_full_prompts, batch_search_time, batch_kv_load_time) in pbar:
                device = next(model.parameters()).device

                ###############################################################################################
                ############################            KV Cache 사용 추론           ############################
                ###############################################################################################
                
                ###################### Prefill ######################
                # KV cache token length = batch_max_seq_len
                
                ################ KV 캐시를 CPU에서 GPU로 이동하는 시간 측정 ##################
                data_move_start_time = time.time()
                # KV 캐시를 GPU로 이동
                batch_kv_caches = [(key.to(device), value.to(device)) for key, value in batch_kv_caches]
                data_move_end_time = time.time()
                data_move_time = data_move_end_time - data_move_start_time

                # 이동 시간을 batch_kv_load_time에 더함
                batch_kv_load_time += data_move_time
                ######################################################################
                
                prefill_start_time = time.time()

                # kv cache가 있을 경우, Context를 제외한 input ids 및 attention mask 계산
                inputs = tokenizer(batch_prompts, return_tensors='pt', padding='longest', truncation=True, max_length=512)
                
                # input_ids는 context가 제외 되어 있으므로 그대로 사용
                input_ids = inputs['input_ids'].to(device)

                # Attention Mask는 KV cache에 대한 attetion mask 추가. 
                attention_mask = []
                input_ids_atn_mask = inputs['attention_mask']
                
                attention_mask = torch.cat((batch_kv_attention_masks, input_ids_atn_mask), dim=1)
                attention_mask = attention_mask.to(device)
                
                prefill_token_counts_kv = len(attention_mask[0]) * bsize

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=batch_kv_caches, use_cache=True)
                prefill_end_time = time.time()
                prefill_time_kv = prefill_end_time - prefill_start_time
                
                # # Prefiil 결과 처리                
                # # 다음 토큰 선택 (argmax 사용)
                # next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                # generated_ids = next_token_id
                # # 업데이트된 past_key_values
                # past_key_values = outputs.past_key_values
                # # attention_mask도 업데이트 (1을 추가)
                # attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=1)

                ###################### Decode ######################
                # max_new_tokens = 1
                
                # decode_start_time = time.time()
                # for _ in range(max_new_tokens):
                #     # 마지막 토큰만 사용해서 새로운 토큰을 생성
                #     last_token_id = next_token_id
                #     outputs = model(input_ids=last_token_id, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)                    
                #     # 다음 토큰 선택 (argmax 사용)
                #     next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                #     generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                #     # 업데이트된 past_key_values
                #     past_key_values = outputs.past_key_values
                #     # attention_mask도 업데이트 (1을 추가)
                #     attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=1)

                # decode_end_time = time.time()
                # decode_time = decode_end_time - decode_start_time

                # 최종 생성된 텍스트 디코딩 : 아래 코드를 실행할려면, generated_ids를 처리해줘야 함.
                # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                ###############################################################################################
                ############################             일반 추론 Prefill           ############################
                ###############################################################################################
                prefill_start_time = time.time()

                inputs = tokenizer(batch_full_prompts, return_tensors='pt', padding='longest', truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prefill_end_time = time.time()
                prefill_time = prefill_end_time - prefill_start_time

                prefill_token_counts = len(attention_mask[0]) * bsize
              
                results.append({
                        'Model':{model_size},
                        'batch_size': bsize,
                        'batch_prompts': batch_prompts,
                        'search_time' : batch_search_time,
                        '(Using KV)batch_prefill_tkn_cnt' : prefill_token_counts_kv,
                        '(Using KV)batch_prefill_time': prefill_time_kv,
                        '(Using KV)batch_kv_load_time': batch_kv_load_time,
                        '(Using KV)batch_cache_type': batch_cache_types,
                        '(No KV)batch_prefill_tkn_cnt' : prefill_token_counts,
                        '(No KV)batch_prefill_time': prefill_time,
                    })

                # tqdm을 통해 진행 상황 업데이트
                if prefill_time_kv + batch_kv_load_time > prefill_time :
                    fast_t = "NK"
                else :
                    fast_t = "KV"
                pbar.set_postfix({
                    'Fast' : fast_t,
                    '[KV] PF(s)': f'{prefill_time_kv:.4f}',
                    'Load(s)': f'{batch_kv_load_time:.4f}',
                    '[NK] PF(s)': f'{prefill_time:.4f}',
                })
        
        ######################## 결과를 파일로 저장 ########################
        csv_file = f'./result/withKVRAG_result_all.csv'
        csv_columns = ['Model', 'batch_size', 'batch_prompts', 'search_time',
                       '(Using KV)batch_prefill_tkn_cnt', '(Using KV)batch_prefill_time', '(Using KV)batch_kv_load_time', 
                       '(Using KV)batch_cache_type', 
                       '(No KV)batch_prefill_tkn_cnt', '(No KV)batch_prefill_time']

        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                if csvfile.tell() == 0:  # 파일이 비어있을 경우에만 헤더 작성
                    writer.writeheader()
                for data in results:
                    writer.writerow(data)
            print(f"Results saved to {csv_file}")
        except IOError:
            print("I/O error while writing CSV file")

        data_loader = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"############ Batch size {bsize} Complete ############")

    # kvcache_manager.clear_cache()
    return model, tokenizer

if __name__ == "__main__":
    model_sizes = ["opt-1.3b", 
                   "opt-2.7b", 
                   "opt-6.7b",
    ]
    for size in model_sizes:
        model, tokenizer = run_improved_rag(size)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()