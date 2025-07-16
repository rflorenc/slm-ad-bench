"""
RAG using Milvus integration and explanation generation
"""

import os
import json
import time
import tempfile
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.json_utils import ensure_json_serializable

try:
    from langchain.docstore.document import Document
    from langchain_milvus import Milvus
    MILVUS_AVAILABLE = True
except ImportError:
    Document = None
    Milvus = None
    MILVUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealTimeEmbeddingFunction:
    """Embedding function that creates real-time embeddings for both documents and queries."""
    
    def __init__(self, tokenizer, model, vectorizer=None, reducer=None, embedding_type="llm"):
        self.tokenizer = tokenizer
        self.model = model
        self.vectorizer = vectorizer
        self.reducer = reducer
        self.embedding_type = embedding_type
        self.dim = None
        
    def __call__(self, texts):
        """Make the embedding function callable for Milvus"""
        if isinstance(texts, str):
            # Single query
            return self.embed_query(texts)
        else:
            # Multiple documents
            return self.embed_documents(texts)
    
    def embed_documents(self, documents):
        """Create embeddings for a list of documents."""
        if self.embedding_type == "llm":
            return self._embed_documents_llm(documents)
        elif self.embedding_type == "tfidf":
            return self._embed_documents_tfidf(documents)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    def embed_query(self, query):
        """Create embedding for a single query."""
        if self.embedding_type == "llm":
            return self._embed_query_llm(query)
        elif self.embedding_type == "tfidf":
            return self._embed_query_tfidf(query)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    def _embed_documents_llm(self, documents):
        """Create LLM embeddings for documents."""
        import torch
        
        embeddings = []
        
        # Check if this is a vLLM embedding model
        is_vllm = hasattr(self.model, '__class__') and 'LLM' in str(self.model.__class__)
        
        if is_vllm:
            # For vLLM backend, use the encode method
            try:
                from vllm import PoolingParams
                pooling_params = PoolingParams()
                
                # Process all documents at once
                outputs = self.model.encode(documents, pooling_params=pooling_params, use_tqdm=False)
                
                for output in outputs:
                    if hasattr(output, 'outputs') and hasattr(output.outputs, 'data'):
                        embedding_data = output.outputs.data
                        if hasattr(embedding_data, 'cpu'):
                            embedding_data = embedding_data.cpu().numpy()
                        elif isinstance(embedding_data, np.ndarray):
                            pass
                        else:
                            embedding_data = np.array(embedding_data)
                        
                        if embedding_data.ndim > 1:
                            embedding_data = embedding_data.squeeze()
                        
                        embeddings.append(embedding_data.tolist())
                    else:
                        logger.error("Failed to extract vLLM embedding")
                        dim = 768
                        embeddings.append([0.0] * dim)
                        
            except Exception as e:
                logger.error(f"vLLM embedding failed: {e}")
                # Fallback
                dim = 768
                for doc in documents:
                    embeddings.append([0.0] * dim)
        else:
            # Regular PyTorch model
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            for doc in documents:
                try:
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            doc, return_tensors="pt", truncation=True, 
                            max_length=128, padding=True
                        ).to(self.model.device)
                        
                        outputs = self.model(**inputs, output_hidden_states=True)
                        last_hidden = outputs.hidden_states[-1]
                        doc_emb = last_hidden.mean(dim=1).cpu().numpy()[0]
                        
                        # Apply dimensionality reduction if available
                        if self.reducer is not None:
                            doc_emb = self.reducer.transform(doc_emb.reshape(1, -1))[0]
                        
                        embeddings.append(doc_emb.tolist())
                        
                        # Cleanup
                        del inputs, outputs, last_hidden
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error embedding document: {e}")
                    # Fallback to zero embedding
                    dim = 768
                    embeddings.append([0.0] * dim)
        
        # Set dimension based on first embedding
        if embeddings and self.dim is None:
            self.dim = len(embeddings[0])
        
        return embeddings
    
    def _embed_query_llm(self, query):
        """Create LLM embedding for a single query."""
        import torch
        
        # Check if this is a vLLM backend
        is_vllm = hasattr(self.model, '__class__') and 'LLM' in str(self.model.__class__)
        
        if is_vllm:
            # For vLLM backend, use encode method
            try:
                from vllm import PoolingParams
                pooling_params = PoolingParams()
                
                outputs = self.model.encode([query], pooling_params=pooling_params, use_tqdm=False)
                
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    if hasattr(output, 'outputs') and hasattr(output.outputs, 'data'):
                        embedding_data = output.outputs.data
                        if hasattr(embedding_data, 'cpu'):
                            embedding_data = embedding_data.cpu().numpy()
                        elif isinstance(embedding_data, np.ndarray):
                            pass
                        else:
                            embedding_data = np.array(embedding_data)
                        
                        if embedding_data.ndim > 1:
                            embedding_data = embedding_data.squeeze()
                        
                        return embedding_data.tolist()
                
                # Fallback
                logger.error("Failed to extract vLLM query embedding")
                dim = self.dim or 768
                return [0.0] * dim
                
            except Exception as e:
                logger.error(f"vLLM query embedding failed: {e}")
                dim = self.dim or 768
                return [0.0] * dim
        
        try:
            # Regular PyTorch model
            if hasattr(self.model, 'eval'):
                self.model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(
                    query, return_tensors="pt", truncation=True,
                    max_length=128, padding=True
                ).to(self.model.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                query_emb = last_hidden.mean(dim=1).cpu().numpy()[0]
                
                # Apply dimensionality reduction if available
                if self.reducer is not None:
                    query_emb = self.reducer.transform(query_emb.reshape(1, -1))[0]
                
                # Cleanup
                del inputs, outputs, last_hidden
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return query_emb.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            dim = self.dim or 768
            return [0.0] * dim
    
    def _embed_documents_tfidf(self, documents):
        """Create TF-IDF embeddings for documents."""
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not provided")
        
        embeddings = []
        for doc in documents:
            doc_vec = self.vectorizer.transform([doc])
            embeddings.append(doc_vec.toarray()[0].tolist())
        
        return embeddings
    
    def _embed_query_tfidf(self, query):
        """Create TF-IDF embedding for a single query."""
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not provided")
        
        query_vec = self.vectorizer.transform([query])
        return query_vec.toarray()[0].tolist()

def store_embeddings_in_milvus(lines: List[str], embeddings: np.ndarray, 
                              tokenizer=None, model=None, vectorizer=None, 
                              reducer=None, embedding_type: str = "llm", 
                              precomputed_embeddings: Optional[np.ndarray] = None) -> Tuple[Any, str]:
    """
    Store embeddings in Milvus vector database for RAG functionality
    
    Args:
        lines: List of text lines to store
        embeddings: Precomputed embeddings (not used, computed real-time)
        tokenizer: Tokenizer for LLM embeddings
        model: Model for LLM embeddings
        vectorizer: TF-IDF vectorizer for TF-IDF embeddings
        reducer: Dimensionality reducer (PCA)
        embedding_type: Type of embeddings ("llm" or "tfidf")
    
    Returns:
        Tuple of (vector_db, db_file_path)
    """
    if not MILVUS_AVAILABLE:
        logger.warning("Milvus not available; skipping RAG storage.")
        return None, None
    
    db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
    logger.info(f"Creating Milvus DB at: {db_file}")
    
    embedding_func = RealTimeEmbeddingFunction(
        tokenizer=tokenizer,
        model=model,
        vectorizer=vectorizer,
        reducer=reducer,
        embedding_type=embedding_type
    )
    
    try:
        vector_db = Milvus(
            embedding_function=embedding_func,
            connection_args={"uri": db_file},
            auto_id=True,
            enable_dynamic_field=True,
            index_params={"index_type": "AUTOINDEX"},
        )
        
        docs_with_metadata = []
        for i, line in enumerate(lines):
            doc = Document(page_content=line, metadata={"idx": i})
            docs_with_metadata.append(doc)
        
        vector_db.add_documents(docs_with_metadata)
        logger.info(f"Added {len(docs_with_metadata)} documents to Milvus")
        
        return vector_db, db_file
    except Exception as e:
        logger.error(f"Error creating Milvus DB: {e}")
        if os.path.exists(db_file):
            try:
                os.unlink(db_file)
                logger.info(f"Removed temporary file: {db_file}")
            except:
                pass
        return None, None

def cleanup_milvus_db(db_file: str):
    """Clean up Milvus database file"""
    if db_file and os.path.exists(db_file):
        try:
            os.unlink(db_file)
            logger.info(f"Removed Milvus database file: {db_file}")
        except Exception as e:
            logger.error(f"Error cleaning up Milvus DB file: {e}")

def rag_explanation_monitored(vector_db, query_text: str, model, tokenizer,
                            labels=None, top_k: int = 3, structured: bool = True,
                            dataset_type: str = "eventtraces", results_dir: str = "outputs") -> Tuple[str, str, List[int], List[Tuple], List[Tuple], Dict]:
    """
    Enhanced RAG explanation with monitoring - matches original functionality exactly
    
    Args:
        vector_db: Milvus vector database
        query_text: Text to analyze
        model: Language model for generation
        tokenizer: Tokenizer for the model
        labels: Optional labels for context
        top_k: Number of similar examples to retrieve
        structured: Whether to use structured JSON prompts
        dataset_type: Type of dataset ("eventtraces" or "unsw-nb15")
        results_dir: Results directory
    
    Returns:
        Tuple of (prompt, response, retrieved_indices, predicted_spans, predicted_severity, summary)
    """
    if vector_db is None:
        logger.warning("No vector DB available, skipping RAG.")
        return "", "", [], [], [], {}
    
    # Retrieve similar examples
    results = vector_db.similarity_search_with_score(query_text, k=top_k)
    retrieved_logs = []
    retrieved_indices = []
    
    for doc, score in results:
        retrieved_logs.append(doc.page_content)
        if hasattr(doc, "metadata") and "idx" in doc.metadata:
            retrieved_indices.append(doc.metadata["idx"])
    
    # Build context text
    context_text = ""
    for i, (log, idx) in enumerate(zip(retrieved_logs, retrieved_indices)):
        if labels is not None and idx < len(labels):
            label_str = "Normal" if labels[idx] == 0 else "Anomaly"
            context_text += f"Context {i+1} [{label_str}]: {log}\\n"
        else:
            context_text += f"Context {i+1}: {log}\\n"
    
    # Generate prompt based on dataset type
    if structured:
        if dataset_type == "eventtraces":
            prompt = f"""You are a distributed file systems expert analyzing HDFS logs for anomaly detection.

New Log to Analyze: {query_text}

Retrieved Similar Logs:
{context_text}

Analyze the new log and provide your response in the following JSON format:
{{
    "is_anomalous": true or false,
    "confidence": 0.0 to 1.0,
    "explanation": "detailed explanation of your analysis",
    "critical_spans": [
        {{
            "start": <start_char_position>,
            "end": <end_char_position>,
            "severity": 1 to 3,
            "reason": "what makes this span anomalous"
        }}
    ]
}}

Rules:
- Focus on: ERROR/WARN levels, block IDs, exception messages, timeout values, replication issues
- Severity: 1=performance issue, 2=potential data issue, 3=critical failure or data loss risk
- Common issues: block corruption, DataNode failures, slow operations, replication failures"""
                
        elif dataset_type == "unsw-nb15":
            prompt = f"""You are a network security expert analyzing network flow data for intrusion detection.

New Network Flow to Analyze: {query_text}

Retrieved Similar Flows:
{context_text}

Analyze the flow and provide your response in the following JSON format:
{{
    "is_anomalous": true or false,
    "confidence": 0.0 to 1.0,
    "explanation": "detailed explanation of your analysis",
    "critical_spans": [
        {{
            "start": <start_char_position>,
            "end": <end_char_position>,
            "severity": 1 to 3,
            "reason": "what makes this span suspicious"
        }}
    ]
}}

Rules:
- Focus on: suspicious ports (445, 22, 3389), unusual protocols, abnormal packet sizes, attack services
- Severity: 1=unusual pattern, 2=suspicious activity, 3=clear attack signature
- Attack indicators: port scanning, brute force, exploitation attempts, DDoS patterns"""
        else:
            # Generic prompt
            prompt = f"""Analyze this data for anomalies.

Data to Analyze: {query_text}

Retrieved Similar Examples:
{context_text}

Provide a detailed analysis explaining whether this data appears anomalous and why."""
    
    # Generate response
    import torch
    
    generation_start = time.time()
    model.eval()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Cleanup
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        response = f"[Error generating response: {str(e)}]"
    
    generation_time = time.time() - generation_start
    
    # Parse structured response for spans and severity
    predicted_spans = []
    predicted_severity = []
    
    if structured and response.strip():
        try:
            # Try to parse JSON response
            import re
            json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                if "critical_spans" in parsed:
                    for span in parsed["critical_spans"]:
                        if "start" in span and "end" in span:
                            start, end = span["start"], span["end"]
                            severity = span.get("severity", 1)
                            predicted_spans.append((start, end))
                            predicted_severity.append((start, end, severity))
        except Exception as e:
            logger.debug(f"Could not parse JSON response: {e}")
    
    # Summary statistics
    summary = {
        "generation_time": generation_time,
        "retrieved_count": len(retrieved_indices),
        "response_length": len(response),
        "spans_detected": len(predicted_spans),
        "top_k": top_k,
        "dataset_type": dataset_type
    }
    
    return prompt, response, retrieved_indices, predicted_spans, predicted_severity, summary

class RAGEvaluationSystem:
    """
    Complete RAG evaluation system matching original functionality
    """
    
    def __init__(self):
        self.vector_db = None
        self.db_file = None
    
    def setup_rag_database(self, lines: List[str], model, tokenizer, 
                          reducer=None, embedding_type: str = "llm") -> bool:
        """Setup RAG database with embeddings"""
        try:
            self.vector_db, self.db_file = store_embeddings_in_milvus(
                lines, None, tokenizer, model, None, reducer, embedding_type
            )
            return self.vector_db is not None
        except Exception as e:
            logger.error(f"Failed to setup RAG database: {e}")
            return False
    
    def evaluate_with_rag(self, test_lines: List[str], test_indices: List[int],
                         model, tokenizer, labels=None, dataset_type: str = "eventtraces",
                         results_dir: str = "outputs") -> List[Dict[str, Any]]:
        """
        Evaluate multiple examples using RAG
        
        Args:
            test_lines: Lines to evaluate
            test_indices: Indices of test examples
            model: Language model
            tokenizer: Tokenizer
            labels: Optional labels for training data
            dataset_type: Dataset type
            results_dir: Results directory
        
        Returns:
            List of RAG evaluation results
        """
        if self.vector_db is None:
            logger.warning("RAG database not initialized")
            return []
        
        rag_results = []
        
        for idx in test_indices:
            if idx < len(test_lines):
                prompt, resp, ret_idx, pred_spans, pred_severity, rag_summary = rag_explanation_monitored(
                    self.vector_db, test_lines[idx], model, tokenizer,
                    labels=labels, top_k=3, structured=True,
                    dataset_type=dataset_type
                )
                
                rag_results.append({
                    "example_idx": idx,
                    "query": test_lines[idx],
                    "retrieved_indices": ret_idx,
                    "prompt": prompt,
                    "response": resp,
                    "pred_spans": pred_spans,
                    "pred_severity": pred_severity,
                    "generation_time": rag_summary.get("generation_time", 0.0)
                })
        
        return rag_results
    
    def save_rag_results(self, rag_results: List[Dict], approach_name: str, results_dir: str):
        """Save RAG results to JSON file"""
        if not rag_results:
            return
        
        results_dir = Path(results_dir)
        safe_name = approach_name.replace(" ", "_").replace("/", "_")
        rag_file = results_dir / f"rag_results_{safe_name}.json"
        
        # Ensure JSON serializable
        sanitized_results = ensure_json_serializable(rag_results)
        
        with open(rag_file, 'w') as f:
            json.dump(sanitized_results, f, indent=2)
        
        logger.info(f"Saved RAG results to {rag_file}")
    
    def cleanup(self):
        """Cleanup RAG database"""
        if self.db_file:
            cleanup_milvus_db(self.db_file)
            self.db_file = None
            self.vector_db = None
    
    def __del__(self):
        """Ensure cleanup on destruction"""
        self.cleanup()