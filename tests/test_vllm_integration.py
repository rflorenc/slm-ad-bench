#!/usr/bin/env python3
"""
Test vLLM integration with SLM-AD-BENCH models from quick_test config
Focus on embedding generation performance (95% of workload)
"""
import yaml
from vllm import LLM, SamplingParams, PoolingParams
import time
import numpy as np

def load_quick_test_models():
    """Load models from quick_test config"""
    with open('config/approaches.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    quick_test_models = []
    for approach in config['approaches']:
        if approach.get('quick_test', False):
            quick_test_models.append({
                'name': approach['name'],
                'model_path': approach['model_path']
            })
    
    return quick_test_models

def test_vllm_embeddings(model_info):
    """Test embedding generation with vLLM (95% of workload)"""
    
    try:
        # Initialize vLLM for embedding generation
        llm = LLM(
            model=model_info['model_path'],
            task="embed",  # Critical: Convert to embedding model
            max_model_len=2048,
            gpu_memory_utilization=0.8
        )
        
        test_texts = [
            "hdfs://10.0.0.1:9000/user/test/file.txt",
            "081109 203518 148 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906",
            "TCP connection established on port 80",
            "Suspicious network activity detected",
        ] * 50  # 200 samples to test batch processing
        
        pooling_params = PoolingParams()
        
        batch_sizes = [4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            if len(test_texts) < batch_size:
                continue
                
            batch_texts = test_texts[:batch_size]
            
            start_time = time.time()
            embeddings = llm.encode(batch_texts, pooling_params=pooling_params)
            inference_time = time.time() - start_time
            
            # Convert to numpy array for analysis
            embedding_array = np.array([emb.outputs.embedding for emb in embeddings])
            
        
        return True
        
    except Exception as e:
        return False

def test_vllm_text_generation(model_info):
    """Test text generation with vLLM (5% of workload)"""
    
    try:
        # Initialize vLLM for text generation
        llm = LLM(
            model=model_info['model_path'],
            max_model_len=2048,
            gpu_memory_utilization=0.8
        )
        
        test_prompts = [
            "Analyze this log entry for anomalies: hdfs://10.0.0.1:9000/user/test/file.txt",
            "Classify this network activity as NORMAL or ANOMALY: TCP connection established on port 666.777",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.95
        )
        
        start_time = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        inference_time = time.time() - start_time
        
        
        for i, output in enumerate(outputs[:1]):
            
        return True
        
    except Exception as e:
        return False

def main():
    
    models = load_quick_test_models()
    
    results = {}
    for model_info in models:
        
        embed_success = test_vllm_embeddings(model_info)
        
        text_success = test_vllm_text_generation(model_info)
        
        results[model_info['name']] = {
            'embeddings': embed_success,
            'text_generation': text_success
        }
    
    for model_name, result in results.items():
        embed_status = "PASS" if result['embeddings'] else "FAIL"
        text_status = "PASS" if result['text_generation'] else "FAIL"

if __name__ == "__main__":
    main()