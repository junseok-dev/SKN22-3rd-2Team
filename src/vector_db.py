"""
Patent Guard v2.0 - Milvus Vector Database Operations
======================================================
Store and retrieve patent embeddings with metadata filtering.

Features:
- Hybrid indexing with importance scores
- IPC-based filtering
- Citation metadata for PAI-NET integration
- Async operations

Author: Patent Guard Team
License: MIT
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from tqdm import tqdm

from config import config, MilvusConfig, EMBEDDINGS_DIR


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Import Milvus
# =============================================================================

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus not installed. Install with: pip install pymilvus")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """Result from vector similarity search."""
    chunk_id: str
    patent_id: str
    score: float
    content: str
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsertResult:
    """Result from inserting vectors."""
    success: bool
    inserted_count: int
    collection_name: str
    error_message: Optional[str] = None


# =============================================================================
# Schema Definitions
# =============================================================================

def get_patent_chunks_schema(embedding_dim: int = 4096) -> CollectionSchema:
    """
    Create schema for patent chunks collection.
    
    Fields:
    - chunk_id: Primary key
    - patent_id: Parent patent publication number
    - content: Text content
    - content_type: Type (title, abstract, claim, description)
    - embedding: Dense vector
    - ipc_code: Classification for filtering
    - importance_score: Citation-based importance
    - weight: Content type weight
    """
    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        ),
        FieldSchema(
            name="patent_id",
            dtype=DataType.VARCHAR,
            max_length=50,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,  # Max for Milvus VARCHAR
        ),
        FieldSchema(
            name="content_type",
            dtype=DataType.VARCHAR,
            max_length=20,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        ),
        FieldSchema(
            name="ipc_code",
            dtype=DataType.VARCHAR,
            max_length=20,
        ),
        FieldSchema(
            name="importance_score",
            dtype=DataType.FLOAT,
        ),
        FieldSchema(
            name="weight",
            dtype=DataType.FLOAT,
        ),
        FieldSchema(
            name="claim_number",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="rag_components",
            dtype=DataType.VARCHAR,
            max_length=500,  # JSON array as string
        ),
    ]
    
    return CollectionSchema(
        fields=fields,
        description="Patent Guard v2.0 - Patent Chunks with Embeddings",
    )


def get_triplets_schema(embedding_dim: int = 4096) -> CollectionSchema:
    """
    Create schema for PAI-NET triplets collection.
    """
    fields = [
        FieldSchema(
            name="triplet_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        ),
        FieldSchema(
            name="anchor_id",
            dtype=DataType.VARCHAR,
            max_length=50,
        ),
        FieldSchema(
            name="positive_id",
            dtype=DataType.VARCHAR,
            max_length=50,
        ),
        FieldSchema(
            name="negative_id",
            dtype=DataType.VARCHAR,
            max_length=50,
        ),
        FieldSchema(
            name="anchor_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        ),
        FieldSchema(
            name="positive_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        ),
        FieldSchema(
            name="negative_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        ),
        FieldSchema(
            name="sampling_method",
            dtype=DataType.VARCHAR,
            max_length=10,  # "hard" or "random"
        ),
    ]
    
    return CollectionSchema(
        fields=fields,
        description="Patent Guard v2.0 - PAI-NET Triplets",
    )


# =============================================================================
# Milvus Client
# =============================================================================

class MilvusClient:
    """
    Client for Milvus vector database operations.
    
    Handles:
    - Collection creation with proper schemas
    - Index creation optimized for 4096-dim vectors
    - Insert and search operations
    - Metadata filtering (IPC, importance score)
    """
    
    def __init__(
        self,
        milvus_config: MilvusConfig = config.milvus,
        embedding_dim: int = config.embedding.embedding_dim,
    ):
        self.config = milvus_config
        self.embedding_dim = embedding_dim
        self._connected = False
        
        if not MILVUS_AVAILABLE:
            raise ImportError("pymilvus is required. Install with: pip install pymilvus")
    
    async def connect(self) -> None:
        """Connect to Milvus server."""
        if self._connected:
            return
        
        loop = asyncio.get_event_loop()
        
        await loop.run_in_executor(
            None,
            lambda: connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port,
            )
        )
        
        self._connected = True
        logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")
    
    async def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        if not self._connected:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: connections.disconnect("default"))
        
        self._connected = False
        logger.info("Disconnected from Milvus")
    
    async def create_patents_collection(
        self,
        collection_name: Optional[str] = None,
        drop_existing: bool = False,
    ) -> Collection:
        """
        Create collection for patent chunks.
        
        Args:
            collection_name: Name of collection
            drop_existing: Whether to drop existing collection
            
        Returns:
            Milvus Collection object
        """
        await self.connect()
        
        name = collection_name or self.config.patents_collection
        loop = asyncio.get_event_loop()
        
        # Check if exists
        exists = await loop.run_in_executor(
            None, lambda: utility.has_collection(name)
        )
        
        if exists:
            if drop_existing:
                logger.warning(f"Dropping existing collection: {name}")
                await loop.run_in_executor(
                    None, lambda: utility.drop_collection(name)
                )
            else:
                logger.info(f"Using existing collection: {name}")
                return Collection(name)
        
        # Create collection
        schema = get_patent_chunks_schema(self.embedding_dim)
        
        collection = await loop.run_in_executor(
            None,
            lambda: Collection(name=name, schema=schema)
        )
        
        logger.info(f"Created collection: {name}")
        
        # Create index
        await self._create_index(collection, "embedding")
        
        return collection
    
    async def create_triplets_collection(
        self,
        collection_name: Optional[str] = None,
        drop_existing: bool = False,
    ) -> Collection:
        """Create collection for PAI-NET triplets."""
        await self.connect()
        
        name = collection_name or self.config.triplets_collection
        loop = asyncio.get_event_loop()
        
        exists = await loop.run_in_executor(
            None, lambda: utility.has_collection(name)
        )
        
        if exists:
            if drop_existing:
                await loop.run_in_executor(
                    None, lambda: utility.drop_collection(name)
                )
            else:
                return Collection(name)
        
        schema = get_triplets_schema(self.embedding_dim)
        
        collection = await loop.run_in_executor(
            None,
            lambda: Collection(name=name, schema=schema)
        )
        
        # Create indices for all embedding fields
        for field_name in ["anchor_embedding", "positive_embedding", "negative_embedding"]:
            await self._create_index(collection, field_name)
        
        logger.info(f"Created triplets collection: {name}")
        
        return collection
    
    async def _create_index(
        self,
        collection: Collection,
        field_name: str,
    ) -> None:
        """Create index on embedding field."""
        loop = asyncio.get_event_loop()
        
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"nlist": self.config.nlist},
        }
        
        await loop.run_in_executor(
            None,
            lambda: collection.create_index(field_name, index_params)
        )
        
        logger.info(f"Created {self.config.index_type} index on {field_name}")
    
    async def insert_patent_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        collection_name: Optional[str] = None,
        batch_size: int = 100,
    ) -> InsertResult:
        """
        Insert patent chunks with embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Corresponding embeddings
            collection_name: Target collection
            batch_size: Batch size for insertion
            
        Returns:
            InsertResult with status
        """
        await self.connect()
        
        name = collection_name or self.config.patents_collection
        collection = Collection(name)
        
        loop = asyncio.get_event_loop()
        total_inserted = 0
        
        try:
            for i in tqdm(range(0, len(chunks), batch_size), desc="Inserting"):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                # Prepare data for insertion
                data = [
                    [c.get("chunk_id", f"chunk_{i+j}") for j, c in enumerate(batch_chunks)],
                    [c.get("patent_id", "") for c in batch_chunks],
                    [c.get("content", "")[:65535] for c in batch_chunks],  # Truncate if needed
                    [c.get("content_type", "description") for c in batch_chunks],
                    [e.tolist() for e in batch_embeddings],
                    [c.get("ipc_code", "")[:20] for c in batch_chunks],
                    [float(c.get("importance_score", 0.0)) for c in batch_chunks],
                    [float(c.get("weight", 1.0)) for c in batch_chunks],
                    [int(c.get("claim_number", 0)) for c in batch_chunks],
                    [str(c.get("rag_components", "[]"))[:500] for c in batch_chunks],
                ]
                
                await loop.run_in_executor(
                    None,
                    lambda d=data: collection.insert(d)
                )
                
                total_inserted += len(batch_chunks)
            
            # Flush to ensure data is persisted
            await loop.run_in_executor(None, collection.flush)
            
            return InsertResult(
                success=True,
                inserted_count=total_inserted,
                collection_name=name,
            )
            
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return InsertResult(
                success=False,
                inserted_count=total_inserted,
                collection_name=name,
                error_message=str(e),
            )
    
    async def search(
        self,
        query_embedding: np.ndarray,
        collection_name: Optional[str] = None,
        top_k: int = 10,
        ipc_filter: Optional[str] = None,
        min_importance: Optional[float] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar patent chunks.
        
        Args:
            query_embedding: Query vector
            collection_name: Collection to search
            top_k: Number of results
            ipc_filter: Filter by IPC code prefix
            min_importance: Minimum importance score
            content_type_filter: Filter by content type
            
        Returns:
            List of SearchResult objects
        """
        await self.connect()
        
        name = collection_name or self.config.patents_collection
        collection = Collection(name)
        
        loop = asyncio.get_event_loop()
        
        # Load collection
        await loop.run_in_executor(None, collection.load)
        
        # Build filter expression
        expr_parts = []
        if ipc_filter:
            expr_parts.append(f'ipc_code like "{ipc_filter}%"')
        if min_importance is not None:
            expr_parts.append(f'importance_score >= {min_importance}')
        if content_type_filter:
            expr_parts.append(f'content_type == "{content_type_filter}"')
        
        expr = " and ".join(expr_parts) if expr_parts else None
        
        # Search
        search_params = self.config.search_params
        
        results = await loop.run_in_executor(
            None,
            lambda: collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[
                    "chunk_id", "patent_id", "content", "content_type",
                    "ipc_code", "importance_score", "weight", "rag_components"
                ],
            )
        )
        
        # Convert to SearchResult
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    chunk_id=hit.entity.get("chunk_id"),
                    patent_id=hit.entity.get("patent_id"),
                    score=hit.score,
                    content=hit.entity.get("content"),
                    content_type=hit.entity.get("content_type"),
                    metadata={
                        "ipc_code": hit.entity.get("ipc_code"),
                        "importance_score": hit.entity.get("importance_score"),
                        "weight": hit.entity.get("weight"),
                        "rag_components": hit.entity.get("rag_components"),
                    },
                ))
        
        return search_results
    
    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get collection statistics."""
        await self.connect()
        
        name = collection_name or self.config.patents_collection
        loop = asyncio.get_event_loop()
        
        if not await loop.run_in_executor(None, lambda: utility.has_collection(name)):
            return {"exists": False}
        
        collection = Collection(name)
        
        stats = await loop.run_in_executor(None, lambda: collection.num_entities)
        
        return {
            "exists": True,
            "name": name,
            "num_entities": stats,
        }


# =============================================================================
# High-Level Operations
# =============================================================================

async def index_processed_patents(
    processed_patents: List[Dict[str, Any]],
    embedder,  # PatentEmbedder instance
    milvus_client: MilvusClient,
    collection_name: Optional[str] = None,
) -> InsertResult:
    """
    Index processed patents with embeddings.
    
    Args:
        processed_patents: List of processed patent dictionaries
        embedder: PatentEmbedder instance
        milvus_client: MilvusClient instance
        collection_name: Target collection
        
    Returns:
        InsertResult with status
    """
    import json
    
    logger.info(f"Indexing {len(processed_patents)} patents...")
    
    # Extract all chunks
    all_chunks = []
    
    for patent in tqdm(processed_patents, desc="Extracting chunks"):
        for chunk in patent.get("chunks", []):
            chunk_data = {
                "chunk_id": chunk.get("chunk_id"),
                "patent_id": chunk.get("patent_id") or patent.get("publication_number"),
                "content": chunk.get("content", ""),
                "content_type": chunk.get("chunk_type", "description"),
                "ipc_code": (patent.get("ipc_codes") or [""])[0][:20],
                "importance_score": patent.get("importance_score", 0.0),
                "weight": 1.0,
                "claim_number": chunk.get("metadata", {}).get("claim_number", 0),
                "rag_components": json.dumps(chunk.get("rag_components", [])),
            }
            all_chunks.append(chunk_data)
    
    logger.info(f"Total chunks to index: {len(all_chunks)}")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    
    embedding_items = [
        {
            "id": c["chunk_id"],
            "text": c["content"],
            "type": c["content_type"],
        }
        for c in all_chunks
    ]
    
    embedding_results = await embedder.embed_batch(embedding_items)
    
    embeddings = [r.embedding for r in embedding_results]
    
    # Update weights from embedder
    for i, result in enumerate(embedding_results):
        all_chunks[i]["weight"] = result.weight
    
    # Create collection if needed
    await milvus_client.create_patents_collection(collection_name)
    
    # Insert
    return await milvus_client.insert_patent_chunks(
        all_chunks,
        embeddings,
        collection_name,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Test Milvus operations."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format=config.logging.log_format,
    )
    
    print("\n" + "=" * 70)
    print("🛡️  Patent Guard v2.0 - Milvus Vector DB Test")
    print("=" * 70)
    
    if not MILVUS_AVAILABLE:
        print("❌ pymilvus not installed. Install with: pip install pymilvus")
        return
    
    # Initialize client
    client = MilvusClient()
    
    try:
        # Connect
        print("\n📡 Connecting to Milvus...")
        await client.connect()
        
        # Get stats
        stats = await client.get_collection_stats()
        print(f"\n📊 Collection Stats: {stats}")
        
        # Create test collection
        print("\n📦 Creating test collection...")
        collection = await client.create_patents_collection(
            "patent_guard_test",
            drop_existing=True,
        )
        
        # Insert test data
        print("\n📥 Inserting test data...")
        test_chunks = [
            {
                "chunk_id": "test_1",
                "patent_id": "US-1234567-A",
                "content": "A method for retrieval-augmented generation...",
                "content_type": "claim",
                "ipc_code": "G06N3",
                "importance_score": 10.0,
                "weight": 2.0,
                "claim_number": 1,
                "rag_components": '["retriever", "generator"]',
            },
        ]
        
        test_embeddings = [np.random.randn(4096).astype(np.float32)]
        
        result = await client.insert_patent_chunks(
            test_chunks,
            test_embeddings,
            "patent_guard_test",
        )
        
        print(f"   Inserted: {result.inserted_count}")
        
        # Search
        print("\n🔍 Testing search...")
        search_results = await client.search(
            query_embedding=np.random.randn(4096).astype(np.float32),
            collection_name="patent_guard_test",
            top_k=5,
        )
        
        print(f"   Found: {len(search_results)} results")
        
        # Cleanup
        print("\n🧹 Cleaning up test collection...")
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: utility.drop_collection("patent_guard_test")
        )
        
        print("\n✅ Milvus test complete!")
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
