"""
Patent Guard v2.0 - Intel XPU Optimized Embedder
=================================================
Embedding generation using Octen-Embedding-8B with IPEX-LLM INT4 quantization.

Supports:
- Intel XPU (IPEX-LLM) for local development
- NVIDIA CUDA (bitsandbytes) for RunPod training
- Weighted embedding for different content types

Author: Patent Guard Team
License: MIT
"""

from __future__ import annotations

import asyncio
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import config, EmbeddingConfig


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Detection
# =============================================================================

class ComputeBackend(Enum):
    INTEL_XPU = "xpu"
    NVIDIA_CUDA = "cuda"
    CPU = "cpu"


def detect_backend() -> ComputeBackend:
    """Detect available compute backend."""
    # Check for RunPod environment
    if os.environ.get("RUNPOD_POD_ID"):
        return ComputeBackend.NVIDIA_CUDA
    
    # Check for Intel XPU (with graceful fallback)
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return ComputeBackend.INTEL_XPU
    except Exception as e:
        # IPEX not properly installed or DLL issues on Windows
        logger.warning(f"Intel XPU not available: {e}")
    
    # Check for NVIDIA CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return ComputeBackend.NVIDIA_CUDA
    except Exception:
        pass
    
    return ComputeBackend.CPU


# Detect backend at module load
BACKEND = detect_backend()
logger.info(f"Detected compute backend: {BACKEND.value}")


# =============================================================================
# Lazy imports based on backend
# =============================================================================

import torch

if BACKEND == ComputeBackend.INTEL_XPU:
    try:
        import intel_extension_for_pytorch as ipex
        from ipex_llm.transformers import AutoModel as IPEXAutoModel
        from transformers import AutoTokenizer
        logger.info("Using Intel IPEX-LLM backend")
    except ImportError as e:
        logger.warning(f"IPEX-LLM import failed: {e}. Falling back to CPU.")
        BACKEND = ComputeBackend.CPU
        from transformers import AutoModel, AutoTokenizer

elif BACKEND == ComputeBackend.NVIDIA_CUDA:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    logger.info("Using NVIDIA CUDA backend")

else:
    from transformers import AutoModel, AutoTokenizer
    logger.info("Using CPU backend (no acceleration)")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text_id: str
    embedding: np.ndarray
    content_type: str  # "title", "abstract", "claim", "description"
    weight: float = 1.0
    metadata: Dict[str, Any] = None


# =============================================================================
# Model Loader
# =============================================================================

class ModelLoader:
    """Load embedding model with appropriate backend."""
    
    @staticmethod
    def load_intel_xpu(
        model_id: str,
        use_int4: bool = True,
    ) -> Tuple[Any, Any]:
        """Load model for Intel XPU with INT4 quantization."""
        logger.info(f"Loading {model_id} for Intel XPU (INT4={use_int4})...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        if use_int4:
            model = IPEXAutoModel.from_pretrained(
                model_id,
                load_in_4bit=True,
                trust_remote_code=True,
                optimize_model=True,
            )
        else:
            model = IPEXAutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        
        model = model.to("xpu")
        
        return model, tokenizer
    
    @staticmethod
    def load_nvidia_cuda(
        model_id: str,
        use_4bit: bool = True,
        device_map: str = "auto",
    ) -> Tuple[Any, Any]:
        """Load model for NVIDIA CUDA with bitsandbytes 4-bit."""
        logger.info(f"Loading {model_id} for NVIDIA CUDA (4bit={use_4bit})...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModel.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        return model, tokenizer
    
    @staticmethod
    def load_cpu(model_id: str) -> Tuple[Any, Any]:
        """Load model for CPU (no quantization)."""
        logger.warning(f"Loading {model_id} on CPU - this will be slow!")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        
        return model, tokenizer


# =============================================================================
# Patent Embedder
# =============================================================================

class PatentEmbedder:
    """
    Generate embeddings for patent content with weighted indexing.
    
    Supports different weights for:
    - Titles (higher weight for exact matching)
    - Claims (highest weight for legal precision)
    - Abstracts (medium weight for overview)
    - Descriptions (base weight for context)
    """
    
    def __init__(
        self,
        embedding_config: EmbeddingConfig = config.embedding,
        backend: ComputeBackend = BACKEND,
    ):
        self.config = embedding_config
        self.backend = backend
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
    
    async def load_model(self) -> None:
        """Load embedding model asynchronously."""
        if self._loaded:
            return
        
        loop = asyncio.get_event_loop()
        
        if self.backend == ComputeBackend.INTEL_XPU:
            self.model, self.tokenizer = await loop.run_in_executor(
                None,
                lambda: ModelLoader.load_intel_xpu(
                    self.config.model_id,
                    self.config.use_int4_quantization,
                )
            )
            self.device = torch.device("xpu")
            
        elif self.backend == ComputeBackend.NVIDIA_CUDA:
            self.model, self.tokenizer = await loop.run_in_executor(
                None,
                lambda: ModelLoader.load_nvidia_cuda(
                    self.config.model_id,
                    use_4bit=True,
                )
            )
            self.device = torch.device("cuda")
            
        else:
            self.model, self.tokenizer = await loop.run_in_executor(
                None,
                lambda: ModelLoader.load_cpu(self.config.model_id)
            )
            self.device = torch.device("cpu")
        
        self.model.eval()
        self._loaded = True
        logger.info(f"Model loaded on {self.device}")
    
    def _get_weight(self, content_type: str) -> float:
        """Get weight for content type."""
        weights = {
            "title": self.config.title_weight,
            "claim": self.config.claim_weight,
            "abstract": self.config.abstract_weight,
            "description": self.config.description_weight,
        }
        return weights.get(content_type, 1.0)
    
    @torch.no_grad()
    def _embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_length,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_length,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    async def embed_text(
        self,
        text: str,
        text_id: str = "",
        content_type: str = "description",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            text_id: Unique identifier
            content_type: Type of content for weight assignment
            metadata: Optional metadata to attach
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        await self.load_model()
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_single, text)
        
        return EmbeddingResult(
            text_id=text_id,
            embedding=embedding,
            content_type=content_type,
            weight=self._get_weight(content_type),
            metadata=metadata or {},
        )
    
    async def embed_batch(
        self,
        items: List[Dict[str, Any]],
        text_key: str = "text",
        id_key: str = "id",
        type_key: str = "type",
        show_progress: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of items.
        
        Args:
            items: List of dicts with text and metadata
            text_key: Key for text content in items
            id_key: Key for item ID
            type_key: Key for content type
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        await self.load_model()
        
        results = []
        batch_size = self.config.batch_size
        
        iterator = range(0, len(items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")
        
        loop = asyncio.get_event_loop()
        
        for i in iterator:
            batch_items = items[i:i + batch_size]
            batch_texts = [item[text_key] for item in batch_items]
            
            embeddings = await loop.run_in_executor(
                None, self._embed_batch, batch_texts
            )
            
            for j, item in enumerate(batch_items):
                content_type = item.get(type_key, "description")
                results.append(EmbeddingResult(
                    text_id=item.get(id_key, f"item_{i+j}"),
                    embedding=embeddings[j],
                    content_type=content_type,
                    weight=self._get_weight(content_type),
                    metadata={k: v for k, v in item.items() 
                              if k not in [text_key, id_key, type_key]},
                ))
        
        return results
    
    async def embed_patent_chunks(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for patent chunks with appropriate weights.
        
        Args:
            chunks: List of PatentChunk-like dicts
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        # Convert chunks to embedding input format
        items = []
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "description")
            
            # Map chunk types to content types
            content_type_map = {
                "parent": "abstract",
                "abstract": "abstract",
                "claim": "claim",
                "description_section": "description",
            }
            content_type = content_type_map.get(chunk_type, "description")
            
            items.append({
                "text": chunk["content"],
                "id": chunk["chunk_id"],
                "type": content_type,
                "patent_id": chunk.get("patent_id"),
                "chunk_type": chunk_type,
                "metadata": chunk.get("metadata", {}),
            })
        
        return await self.embed_batch(
            items,
            text_key="text",
            id_key="id",
            type_key="type",
            show_progress=show_progress,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Test embedding generation."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format=config.logging.log_format,
    )
    
    print("\n" + "=" * 70)
    print("🛡️  Patent Guard v2.0 - Patent Embedder Test")
    print(f"   Backend: {BACKEND.value.upper()}")
    print("=" * 70)
    
    # Test texts
    test_texts = [
        {
            "id": "test_title",
            "type": "title",
            "text": "Method for Retrieval-Augmented Generation in Patent Search",
        },
        {
            "id": "test_claim",
            "type": "claim",
            "text": """A computer-implemented method for semantic patent search comprising:
                receiving a query describing a technical concept;
                generating a dense vector embedding of the query;
                retrieving relevant patent documents from a vector database;
                synthesizing a response using a language model.""",
        },
        {
            "id": "test_abstract",
            "type": "abstract",
            "text": """This invention relates to an improved method for searching prior art 
                using retrieval-augmented generation techniques. The system combines 
                dense vector retrieval with large language model inference to provide 
                accurate and contextually relevant patent search results.""",
        },
    ]
    
    # Initialize embedder
    embedder = PatentEmbedder()
    
    print("\n📥 Loading model...")
    await embedder.load_model()
    
    print("\n📊 Generating embeddings...")
    results = await embedder.embed_batch(test_texts, show_progress=False)
    
    print("\n✅ Results:")
    for result in results:
        print(f"\n   ID: {result.text_id}")
        print(f"   Type: {result.content_type}")
        print(f"   Weight: {result.weight}")
        print(f"   Shape: {result.embedding.shape}")
        print(f"   L2 Norm: {np.linalg.norm(result.embedding):.4f}")
    
    # Compute similarities
    print("\n📐 Pairwise Similarities:")
    from scipy.spatial.distance import cosine
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i < j:
                sim = 1 - cosine(r1.embedding, r2.embedding)
                print(f"   {r1.text_id} <-> {r2.text_id}: {sim:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Embedding test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
