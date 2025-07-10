#!/usr/bin/env python
"""
Basic usage example for the refactored SLM-AD-BENCH

This example shows how to use the new inference backend system
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference import InferenceBackendFactory, InferenceManager
from core.inference.local_transformers import LocalTransformersConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def basic_embedding_example():
    """Basic example of generating embeddings"""
    
    config = LocalTransformersConfig(
        model_name="ibm-granite/granite-3.2-2b-instruct",
        batch_size=2,
        max_length=128,
        use_4bit=True,
        use_nested_quant=True,
        use_cpu_offload=False
    )
    
    texts = [
        "User login successful from IP 192.168.1.100",
        "Failed authentication attempt from IP 10.0.0.5",
        "Database connection established",
        "Error: Unable to connect to external service",
        "System startup completed successfully"
    ]
    
    logger.info("Starting basic embedding generation example...")
    
    async with InferenceBackendFactory.create_backend("local_transformers", config) as backend:
        logger.info(f"Backend loaded: {backend.backend_name}")
        logger.info(f"Supports batch inference: {backend.supports_batch_inference}")
        logger.info(f"Max batch size: {backend.max_batch_size}")
        
        result = await backend.generate_embeddings(texts)
        
        logger.info(f"Generated embeddings shape: {result.embeddings.shape}")
        logger.info(f"Processing time: {result.processing_time:.2f} seconds")
        logger.info(f"Metadata: {result.metadata}")
        
        return result

async def fallback_example():
    """Example showing fallback mechanism"""
    
    primary_config = LocalTransformersConfig(
        model_name="non-existent-model",  # This will fail
        batch_size=2
    )
    
    fallback_config = LocalTransformersConfig(
        model_name="ibm-granite/granite-3.2-2b-instruct",
        batch_size=2,
        use_4bit=True
    )
    
    texts = ["Sample log entry for fallback test"]
    
    logger.info("Starting fallback mechanism example...")
    
    manager = InferenceManager()
    
    try:
        result = await manager.generate_embeddings_with_fallback(
            texts=texts,
            primary_backend="local_transformers",
            primary_config=primary_config,
            fallback_backends=[("local_transformers", fallback_config)]
        )
        
        logger.info(f"Fallback successful! Used backend attempt: {result.metadata['backend_attempt']}")
        logger.info(f"Fallback was used: {result.metadata['fallback_used']}")
        
    except Exception as e:
        logger.error(f"All backends failed: {e}")

async def text_generation_example():
    """Example of text generation"""
    
    config = LocalTransformersConfig(
        model_name="ibm-granite/granite-3.2-2b-instruct",
        batch_size=1
    )
    
    prompt = "Analyze this log entry for potential security issues: Failed login attempt from IP 192.168.1.999"
    
    logger.info("Starting text generation example...")
    
    async with InferenceBackendFactory.create_backend("local_transformers", config) as backend:
        result = await backend.generate_text(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        logger.info(f"Generated text: {result.text}")
        logger.info(f"Generation time: {result.processing_time:.2f} seconds")
        logger.info(f"Metadata: {result.metadata}")

def sync_example():
    """Example showing synchronous usage (for backward compatibility)"""
    
    config = LocalTransformersConfig(
        model_name="ibm-granite/granite-3.2-2b-instruct",
        batch_size=2,
        use_4bit=True
    )
    
    texts = ["Synchronous example log entry"]
    
    logger.info("Starting synchronous usage example...")
    
    with InferenceBackendFactory.create_backend("local_transformers", config) as backend:
        # Note: This runs async code in sync context
        import asyncio
        result = asyncio.run(backend.generate_embeddings(texts))
        
        logger.info(f"Sync result shape: {result.embeddings.shape}")

async def main():
    """Run all examples"""
    logger.info("=== SLM-AD-BENCH Refactored Usage Examples ===")
    
    try:
        # Example 1: Basic embedding generation
        logger.info("\n1. Basic Embedding Generation:")
        await basic_embedding_example()
        
        # Example 2: Text generation
        logger.info("\n2. Text Generation:")
        await text_generation_example()
        
        # Example 3: Fallback mechanism
        logger.info("\n3. Fallback Mechanism:")
        await fallback_example()
        
        # Example 4: Synchronous usage
        logger.info("\n4. Synchronous Usage:")
        sync_example()
        
        logger.info("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())