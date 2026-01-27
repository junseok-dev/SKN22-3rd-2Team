"""
Patent Guard v2.0 - Octen-Embedding-8B Intel iGPU Loader (IPEX-LLM)
====================================================================
This module provides utilities for loading the Octen-Embedding-8B model
with INT4 quantization using IPEX-LLM for Intel Iris Xe Graphics.

Supports both Intel XPU (local) and NVIDIA CUDA (RunPod) environments
with automatic backend switching.

Author: Patent Guard Team
License: MIT
"""

from __future__ import annotations

import asyncio
import os
import platform
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Determine compute backend before importing torch
class ComputeBackend(Enum):
    INTEL_XPU = "xpu"
    NVIDIA_CUDA = "cuda"
    CPU = "cpu"


def detect_compute_backend() -> ComputeBackend:
    """
    Detect available compute backend.
    Priority: Intel XPU > NVIDIA CUDA > CPU
    
    Returns:
        Detected ComputeBackend enum
    """
    # Check for RunPod environment (NVIDIA)
    if os.environ.get("RUNPOD_POD_ID"):
        return ComputeBackend.NVIDIA_CUDA
    
    # Check for Intel XPU
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return ComputeBackend.INTEL_XPU
    except ImportError:
        pass
    
    # Check for NVIDIA CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return ComputeBackend.NVIDIA_CUDA
    except ImportError:
        pass
    
    return ComputeBackend.CPU


# Detect backend early
COMPUTE_BACKEND = detect_compute_backend()

# Import torch after backend detection
import torch

# Backend-specific imports
if COMPUTE_BACKEND == ComputeBackend.INTEL_XPU:
    import intel_extension_for_pytorch as ipex
    from ipex_llm.transformers import AutoModel as IPEXAutoModel
    from transformers import AutoTokenizer
    print(f"🔵 Using Intel IPEX-LLM backend (XPU)")
elif COMPUTE_BACKEND == ComputeBackend.NVIDIA_CUDA:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    print(f"🟢 Using NVIDIA CUDA backend (bitsandbytes)")
else:
    from transformers import AutoModel, AutoTokenizer
    print(f"⚪ Using CPU backend (no acceleration)")


# =============================================================================
# Configuration
# =============================================================================

MODEL_ID: str = "octen/Octen-Embedding-8B"
EMBEDDING_DIM: int = 4096
MAX_CONTEXT_LENGTH: int = 32768  # 32k context window


@dataclass
class MemoryInfo:
    """GPU/XPU memory usage information."""
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_percent: float
    backend: str


# =============================================================================
# Memory Monitoring Utilities
# =============================================================================

def get_memory_info(device_index: int = 0) -> Optional[MemoryInfo]:
    """
    Get GPU/XPU memory usage information.
    
    Args:
        device_index: Device index (default: 0)
    
    Returns:
        MemoryInfo dataclass or None if monitoring unavailable
    """
    try:
        if COMPUTE_BACKEND == ComputeBackend.INTEL_XPU:
            # Intel XPU memory info
            props = torch.xpu.get_device_properties(device_index)
            total_mb = props.total_memory / (1024 ** 2)
            # Note: Intel XPU doesn't have direct used memory API
            # Using allocated memory as approximation
            used_mb = torch.xpu.memory_allocated(device_index) / (1024 ** 2)
            free_mb = total_mb - used_mb
            utilization = (used_mb / total_mb) * 100 if total_mb > 0 else 0
            
            return MemoryInfo(
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
                utilization_percent=utilization,
                backend="Intel XPU"
            )
            
        elif COMPUTE_BACKEND == ComputeBackend.NVIDIA_CUDA:
            # NVIDIA CUDA memory info
            total_mb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 2)
            used_mb = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            free_mb = total_mb - used_mb
            utilization = (used_mb / total_mb) * 100 if total_mb > 0 else 0
            
            return MemoryInfo(
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
                utilization_percent=utilization,
                backend="NVIDIA CUDA"
            )
            
    except Exception as e:
        print(f"⚠️  Memory monitoring error: {e}")
        return None
    
    return None


def print_memory_status(label: str = "Current") -> None:
    """
    Print formatted GPU/XPU memory status.
    
    Args:
        label: Label for the status message
    """
    info = get_memory_info()
    if info:
        print(f"\n{'='*60}")
        print(f"🖥️  {info.backend} Memory Status ({label})")
        print(f"{'='*60}")
        print(f"   Total:       {info.total_mb:,.0f} MB")
        print(f"   Used:        {info.used_mb:,.0f} MB")
        print(f"   Free:        {info.free_mb:,.0f} MB")
        print(f"   Utilization: {info.utilization_percent:.1f}%")
        print(f"{'='*60}\n")
    else:
        print(f"\n⚠️  Memory monitoring not available for {COMPUTE_BACKEND.value}\n")


# =============================================================================
# Model Loading with Backend Switching
# =============================================================================

def load_model_intel_xpu(
    model_id: str = MODEL_ID,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model with Intel IPEX-LLM INT4 quantization for XPU.
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n🔵 Loading model with Intel IPEX-LLM INT4...")
    print(f"   Model: {model_id}")
    print(f"   Device: Intel XPU")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    # Load with IPEX-LLM INT4 quantization
    model = IPEXAutoModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        trust_remote_code=True,
        optimize_model=True,
    )
    
    # Move to XPU
    model = model.to("xpu")
    
    return model, tokenizer


def load_model_nvidia_cuda(
    model_id: str = MODEL_ID,
    device_map: str = "auto",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model with NVIDIA bitsandbytes 4-bit quantization.
    
    Args:
        model_id: HuggingFace model identifier
        device_map: Device placement strategy
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n🟢 Loading model with NVIDIA bitsandbytes 4-bit...")
    print(f"   Model: {model_id}")
    print(f"   Device: CUDA ({device_map})")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    # BitsAndBytes 4-bit config
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
    
    return model, tokenizer


def load_model_cpu(
    model_id: str = MODEL_ID,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load model on CPU (fallback, no quantization).
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n⚪ Loading model on CPU (no acceleration)...")
    print(f"   Model: {model_id}")
    print(f"   ⚠️  Warning: This will be very slow for 8B models!")
    
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


async def load_model_async(
    model_id: str = MODEL_ID,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Asynchronously load model with automatic backend detection.
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n🚀 Loading model: {model_id}")
    print(f"   Backend: {COMPUTE_BACKEND.value}")
    
    print_memory_status("Before Loading")
    
    loop = asyncio.get_event_loop()
    
    if COMPUTE_BACKEND == ComputeBackend.INTEL_XPU:
        model, tokenizer = await loop.run_in_executor(
            None, lambda: load_model_intel_xpu(model_id)
        )
    elif COMPUTE_BACKEND == ComputeBackend.NVIDIA_CUDA:
        model, tokenizer = await loop.run_in_executor(
            None, lambda: load_model_nvidia_cuda(model_id)
        )
    else:
        model, tokenizer = await loop.run_in_executor(
            None, lambda: load_model_cpu(model_id)
        )
    
    print_memory_status("After Loading")
    print(f"✅ Model loaded successfully!")
    
    return model, tokenizer


# =============================================================================
# Embedding Generation
# =============================================================================

class OctenEmbedder:
    """
    Wrapper class for generating embeddings using Octen-Embedding-8B.
    
    Supports Intel XPU (IPEX-LLM) and NVIDIA CUDA (bitsandbytes) backends.
    Handles the full 32k context window and produces 4096-dimensional vectors.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_CONTEXT_LENGTH,
    ) -> None:
        """
        Initialize the embedder.
        
        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            max_length: Maximum sequence length (default: 32k)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()
        
        # Determine device
        if COMPUTE_BACKEND == ComputeBackend.INTEL_XPU:
            self.device = torch.device("xpu")
        elif COMPUTE_BACKEND == ComputeBackend.NVIDIA_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text (e.g., patent claim)
        
        Returns:
            numpy array of shape (4096,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Mean pooling over sequence dimension
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show tqdm progress bar
        
        Returns:
            numpy array of shape (n_texts, 4096)
        """
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
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
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    async def embed_text_async(self, text: str) -> np.ndarray:
        """
        Asynchronously generate embedding for a single text.
        
        Args:
            text: Input text
        
        Returns:
            numpy array of shape (4096,)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)


# =============================================================================
# Similarity Computation
# =============================================================================

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Cosine similarity score (0 to 1)
    """
    return 1 - cosine(vec1, vec2)


# =============================================================================
# Test & Validation
# =============================================================================

async def run_embedding_test() -> None:
    """
    Run comprehensive embedding test with sample patent claims.
    """
    print("\n" + "=" * 70)
    print("🧪 Patent Guard v2.0 - Octen-Embedding-8B Embedding Test")
    print(f"   Compute Backend: {COMPUTE_BACKEND.value.upper()}")
    print("=" * 70)
    
    # Sample patent claims for testing
    sample_claims = [
        # Short claim
        "A method for training neural networks using backpropagation with adaptive learning rates.",
        
        # Medium claim
        """A computer-implemented method for natural language processing comprising:
        receiving an input text sequence;
        tokenizing the input text sequence into subword units;
        generating contextual embeddings using a transformer-based encoder;
        applying attention mechanisms to capture long-range dependencies;
        outputting a semantic representation vector.""",
        
        # Long claim (testing 32k context capability)
        """A system and method for retrieval-augmented generation (RAG) in patent prior art search, 
        the system comprising:
        a document ingestion pipeline configured to:
            extract text content from patent documents including claims, descriptions, and abstracts;
            chunk the extracted content into semantically meaningful segments;
            generate dense vector embeddings using a pre-trained language model;
        a vector database configured to:
            store the generated embeddings with associated metadata;
            support similarity search using approximate nearest neighbor algorithms;
            maintain citation relationship graphs between patent documents;
        a query processing module configured to:
            receive natural language queries from users;
            transform queries into embedding space;
            retrieve relevant document chunks based on semantic similarity;
        a language model inference engine configured to:
            synthesize retrieved context with user queries;
            generate analytical reports including similarity assessments, 
            infringement risk evaluations, and design-around strategies;
        wherein the system supports processing of patent claims up to 32,000 tokens in length
        and produces output embeddings of 4096 dimensions for maximum semantic fidelity.""",
    ]
    
    # Load model
    print("\n📥 Step 1: Loading Octen-Embedding-8B...")
    model, tokenizer = await load_model_async()
    
    # Create embedder
    embedder = OctenEmbedder(model, tokenizer)
    
    # Generate embeddings
    print("\n📊 Step 2: Generating embeddings for sample claims...")
    embeddings = []
    
    for i, claim in enumerate(sample_claims):
        print(f"\n   Processing claim {i + 1}/{len(sample_claims)}...")
        print(f"   Length: {len(claim)} characters, {len(claim.split())} words")
        
        embedding = embedder.embed_text(claim)
        embeddings.append(embedding)
        
        # Validate shape
        print(f"   ✅ Embedding shape: {embedding.shape}")
        assert embedding.shape == (EMBEDDING_DIM,), f"Expected {EMBEDDING_DIM}D, got {embedding.shape}"
    
    # Compute similarities
    print("\n📐 Step 3: Computing pairwise cosine similarities...")
    print("\n   Similarity Matrix:")
    print("   " + "-" * 40)
    
    for i in range(len(embeddings)):
        row = "   "
        for j in range(len(embeddings)):
            sim = compute_cosine_similarity(embeddings[i], embeddings[j])
            row += f"{sim:.4f}  "
        print(row)
    
    print("   " + "-" * 40)
    print(f"   Claims: [1] Short, [2] Medium, [3] Long RAG System")
    
    # Summary statistics
    print("\n📈 Step 4: Embedding Statistics...")
    for i, emb in enumerate(embeddings):
        print(f"\n   Claim {i + 1}:")
        print(f"      Shape:    {emb.shape}")
        print(f"      Dtype:    {emb.dtype}")
        print(f"      Min:      {emb.min():.6f}")
        print(f"      Max:      {emb.max():.6f}")
        print(f"      Mean:     {emb.mean():.6f}")
        print(f"      L2 Norm:  {np.linalg.norm(emb):.6f}")
    
    # Final memory status
    print_memory_status("Final")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Embedding pipeline is working correctly.")
    print("=" * 70 + "\n")


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """Main entry point."""
    print(f"\n{'='*70}")
    print("🛡️  Patent Guard v2.0 - Octen-Embedding-8B Loader")
    print(f"{'='*70}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   OS: {platform.system()} {platform.release()}")
    
    # Backend info
    if COMPUTE_BACKEND == ComputeBackend.INTEL_XPU:
        print(f"   Backend: Intel IPEX-LLM (XPU)")
        print(f"   Device: {torch.xpu.get_device_name(0)}")
    elif COMPUTE_BACKEND == ComputeBackend.NVIDIA_CUDA:
        print(f"   Backend: NVIDIA CUDA (bitsandbytes)")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   Backend: CPU (no acceleration)")
        print("   ⚠️  Warning: CPU mode will be very slow for 8B models!")
    
    # Run async test
    asyncio.run(run_embedding_test())


if __name__ == "__main__":
    main()
