"""
Patent Guard v2.0 - Main Pipeline Orchestrator
================================================
Orchestrates the complete patent data pipeline.

Pipeline stages:
1. BigQuery extraction
2. Preprocessing & chunking
3. PAI-NET triplet generation
4. Embedding generation
5. Vector DB indexing
6. Self-RAG training data generation

Author: Patent Guard Team
License: MIT
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import (
    config,
    print_config_summary,
    update_config_from_env,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRIPLETS_DIR,
)

# =============================================================================
# Logging Setup  
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format=config.logging.log_format,
        handlers=[
            logging.StreamHandler(),
        ],
    )
    
    if config.logging.log_file:
        file_handler = logging.FileHandler(config.logging.log_file)
        file_handler.setFormatter(logging.Formatter(config.logging.log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================================
# Pipeline Stages
# =============================================================================

async def stage_1_extraction(
    limit: Optional[int] = None,
    dry_run: bool = True,
) -> Optional[Path]:
    """
    Stage 1: Extract patent data from BigQuery.
    
    Returns:
        Path to extracted data file, or None if dry run
    """
    from bigquery_extractor import BigQueryExtractor
    
    print("\n" + "=" * 70)
    print("📥 Stage 1: BigQuery Data Extraction")
    print("=" * 70)
    
    # Update config for dry run
    config.bigquery.dry_run = dry_run
    
    extractor = BigQueryExtractor()
    result = await extractor.extract_patents(limit=limit)
    
    if result.success:
        print(f"✅ Extraction complete: {result.patents_count} patents")
        if result.cost_estimate:
            print(f"   {result.cost_estimate}")
        return result.output_path
    else:
        print(f"❌ Extraction failed: {result.error_message}")
        return None


async def stage_2_preprocessing(
    input_path: Path,
) -> Optional[Path]:
    """
    Stage 2: Preprocess patents with claim parsing and chunking.
    
    Returns:
        Path to processed data file
    """
    import json
    from preprocessor import PatentPreprocessor
    
    print("\n" + "=" * 70)
    print("🔧 Stage 2: Patent Preprocessing")
    print("=" * 70)
    
    # Load raw data
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_patents = json.load(f)
    
    print(f"📂 Loaded {len(raw_patents)} raw patents")
    
    preprocessor = PatentPreprocessor()
    
    output_path = PROCESSED_DATA_DIR / f"processed_{input_path.stem}.json"
    processed = await preprocessor.process_patents_batch(raw_patents, output_path)
    
    total_claims = sum(len(p.claims) for p in processed)
    total_chunks = sum(len(p.chunks) for p in processed)
    
    print(f"✅ Preprocessing complete:")
    print(f"   Patents: {len(processed)}")
    print(f"   Claims: {total_claims}")
    print(f"   Chunks: {total_chunks}")
    
    return output_path


async def stage_3_triplet_generation(
    input_path: Path,
) -> Optional[Path]:
    """
    Stage 3: Generate PAI-NET triplets from citation relationships.
    
    Returns:
        Path to triplets file
    """
    import json
    from triplet_generator import PAINETTripletGenerator
    
    print("\n" + "=" * 70)
    print("🔗 Stage 3: PAI-NET Triplet Generation")
    print("=" * 70)
    
    # Load processed data
    with open(input_path, 'r', encoding='utf-8') as f:
        processed_patents = json.load(f)
    
    print(f"📂 Loaded {len(processed_patents)} processed patents")
    
    generator = PAINETTripletGenerator()
    generator.build_graph(processed_patents, text_field="abstract")
    
    output_path = TRIPLETS_DIR / f"triplets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    dataset = await generator.generate_triplets(output_path)
    
    print(f"✅ Triplet generation complete:")
    print(f"   Triplets: {dataset.total_triplets}")
    print(f"   Unique anchors: {dataset.unique_anchors}")
    print(f"   Hard negative ratio: {dataset.hard_negative_ratio:.2%}")
    
    return output_path


async def stage_4_embedding(
    input_path: Path,
) -> Optional[Path]:
    """
    Stage 4: Generate embeddings for patent chunks.
    
    Returns:
        Path to embeddings file
    """
    import json
    import numpy as np
    from embedder import PatentEmbedder
    from config import EMBEDDINGS_DIR
    
    print("\n" + "=" * 70)
    print("🧠 Stage 4: Embedding Generation")
    print("=" * 70)
    
    # Load processed data
    with open(input_path, 'r', encoding='utf-8') as f:
        processed_patents = json.load(f)
    
    # Extract all chunks
    all_chunks = []
    for patent in processed_patents:
        for chunk in patent.get("chunks", []):
            all_chunks.append(chunk)
    
    print(f"📂 Total chunks to embed: {len(all_chunks)}")
    
    # Initialize embedder
    embedder = PatentEmbedder()
    
    # Generate embeddings
    results = await embedder.embed_patent_chunks(all_chunks)
    
    # Save embeddings
    output_path = EMBEDDINGS_DIR / f"embeddings_{input_path.stem}.npz"
    
    embeddings = np.array([r.embedding for r in results])
    chunk_ids = [r.text_id for r in results]
    
    np.savez(
        output_path,
        embeddings=embeddings,
        chunk_ids=chunk_ids,
    )
    
    print(f"✅ Embedding generation complete:")
    print(f"   Embeddings: {len(results)}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Output: {output_path}")
    
    return output_path


async def stage_5_vector_indexing(
    processed_path: Path,
    embeddings_path: Path,
) -> bool:
    """
    Stage 5: Index embeddings in Milvus vector database.
    
    Returns:
        True if successful
    """
    import json
    import numpy as np
    from vector_db import MilvusClient
    
    print("\n" + "=" * 70)
    print("🗄️  Stage 5: Vector Database Indexing")
    print("=" * 70)
    
    # Load data
    with open(processed_path, 'r', encoding='utf-8') as f:
        processed_patents = json.load(f)
    
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    chunk_ids = data['chunk_ids'].tolist()
    
    print(f"📂 Loaded {len(embeddings)} embeddings")
    
    # Build chunk lookup
    chunk_lookup = {}
    for patent in processed_patents:
        for chunk in patent.get("chunks", []):
            chunk_lookup[chunk["chunk_id"]] = {
                **chunk,
                "ipc_code": (patent.get("ipc_codes") or [""])[0][:20],
                "importance_score": patent.get("importance_score", 0.0),
            }
    
    # Prepare data for insertion
    chunks = [chunk_lookup.get(cid, {"chunk_id": cid}) for cid in chunk_ids]
    
    # Initialize Milvus client
    client = MilvusClient()
    
    try:
        await client.create_patents_collection(drop_existing=True)
        
        result = await client.insert_patent_chunks(
            chunks,
            embeddings.tolist(),
        )
        
        if result.success:
            print(f"✅ Vector indexing complete:")
            print(f"   Inserted: {result.inserted_count}")
            return True
        else:
            print(f"❌ Indexing failed: {result.error_message}")
            return False
            
    finally:
        await client.disconnect()


async def stage_6_selfrag_generation(
    input_path: Path,
) -> Optional[Path]:
    """
    Stage 6: Generate Self-RAG training data using Gemini.
    
    Returns:
        Path to training data file
    """
    import json
    from self_rag_generator import SelfRAGDataGenerator, GENAI_AVAILABLE
    
    print("\n" + "=" * 70)
    print("📝 Stage 6: Self-RAG Training Data Generation")
    print("=" * 70)
    
    if not GENAI_AVAILABLE:
        print("⚠️  google-generativeai not available. Skipping...")
        return None
    
    if not config.self_rag.gemini_api_key:
        print("⚠️  GOOGLE_API_KEY not set. Skipping...")
        return None
    
    # Load processed data
    with open(input_path, 'r', encoding='utf-8') as f:
        processed_patents = json.load(f)
    
    print(f"📂 Loaded {len(processed_patents)} processed patents")
    
    generator = SelfRAGDataGenerator()
    
    output_path = PROCESSED_DATA_DIR / f"selfrag_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    samples = await generator.generate_training_samples(
        processed_patents,
        output_path,
    )
    
    print(f"✅ Self-RAG data generation complete:")
    print(f"   Samples: {len(samples)}")
    
    return output_path


# =============================================================================
# Full Pipeline
# =============================================================================

async def run_full_pipeline(
    extraction_limit: Optional[int] = 100,
    dry_run: bool = True,
    skip_stages: Optional[list] = None,
) -> None:
    """
    Run the complete patent data pipeline.
    
    Args:
        extraction_limit: Limit patents for testing
        dry_run: If True, only estimate BigQuery cost
        skip_stages: List of stage numbers to skip (1-6)
    """
    skip_stages = skip_stages or []
    
    print("\n" + "=" * 70)
    print("🛡️  Patent Guard v2.0 - Full Pipeline Execution")
    print("=" * 70)
    
    # Update config from environment
    update_config_from_env()
    print_config_summary()
    
    # Track outputs
    raw_data_path = None
    processed_path = None
    triplets_path = None
    embeddings_path = None
    selfrag_path = None
    
    try:
        # Stage 1: Extraction
        if 1 not in skip_stages:
            raw_data_path = await stage_1_extraction(
                limit=extraction_limit,
                dry_run=dry_run,
            )
            
            if not raw_data_path and not dry_run:
                print("❌ Pipeline stopped: Extraction failed")
                return
        else:
            # Look for existing raw data
            raw_files = list(RAW_DATA_DIR.glob("patents_*.json"))
            if raw_files:
                raw_data_path = max(raw_files, key=lambda p: p.stat().st_mtime)
                print(f"📂 Using existing raw data: {raw_data_path}")
        
        if dry_run:
            print("\n📊 Dry run complete. Set dry_run=False to execute.")
            return
        
        # Stage 2: Preprocessing
        if 2 not in skip_stages and raw_data_path:
            processed_path = await stage_2_preprocessing(raw_data_path)
        else:
            processed_files = list(PROCESSED_DATA_DIR.glob("processed_*.json"))
            if processed_files:
                processed_path = max(processed_files, key=lambda p: p.stat().st_mtime)
                print(f"📂 Using existing processed data: {processed_path}")
        
        # Stage 3: Triplet Generation
        if 3 not in skip_stages and processed_path:
            triplets_path = await stage_3_triplet_generation(processed_path)
        
        # Stage 4: Embedding Generation
        if 4 not in skip_stages and processed_path:
            embeddings_path = await stage_4_embedding(processed_path)
        
        # Stage 5: Vector Indexing
        if 5 not in skip_stages and processed_path and embeddings_path:
            await stage_5_vector_indexing(processed_path, embeddings_path)
        
        # Stage 6: Self-RAG Training Data
        if 6 not in skip_stages and processed_path:
            selfrag_path = await stage_6_selfrag_generation(processed_path)
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Pipeline Execution Complete!")
        print("=" * 70)
        print("\n📁 Output Files:")
        if raw_data_path:
            print(f"   Raw Data: {raw_data_path}")
        if processed_path:
            print(f"   Processed: {processed_path}")
        if triplets_path:
            print(f"   Triplets: {triplets_path}")
        if embeddings_path:
            print(f"   Embeddings: {embeddings_path}")
        if selfrag_path:
            print(f"   Self-RAG: {selfrag_path}")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")


# =============================================================================
# CLI Entry Points
# =============================================================================

async def main():
    """Main entry point."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Patent Guard v2.0 - Patent Data Pipeline"
    )
    
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run specific stage only",
    )
    
    parser.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="Stages to skip (e.g., --skip 5 6)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit patents for extraction (default: 100)",
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute BigQuery (not dry run)",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file for specific stage",
    )
    
    args = parser.parse_args()
    
    if args.stage:
        # Run specific stage
        input_path = Path(args.input) if args.input else None
        
        if args.stage == 1:
            await stage_1_extraction(limit=args.limit, dry_run=not args.execute)
        elif args.stage == 2:
            if not input_path:
                raw_files = list(RAW_DATA_DIR.glob("patents_*.json"))
                input_path = max(raw_files, key=lambda p: p.stat().st_mtime) if raw_files else None
            if input_path:
                await stage_2_preprocessing(input_path)
        elif args.stage == 3:
            if not input_path:
                proc_files = list(PROCESSED_DATA_DIR.glob("processed_*.json"))
                input_path = max(proc_files, key=lambda p: p.stat().st_mtime) if proc_files else None
            if input_path:
                await stage_3_triplet_generation(input_path)
        elif args.stage == 4:
            if not input_path:
                proc_files = list(PROCESSED_DATA_DIR.glob("processed_*.json"))
                input_path = max(proc_files, key=lambda p: p.stat().st_mtime) if proc_files else None
            if input_path:
                await stage_4_embedding(input_path)
        elif args.stage == 5:
            print("Stage 5 requires both processed data and embeddings paths.")
            print("Use full pipeline instead.")
        elif args.stage == 6:
            if not input_path:
                proc_files = list(PROCESSED_DATA_DIR.glob("processed_*.json"))
                input_path = max(proc_files, key=lambda p: p.stat().st_mtime) if proc_files else None
            if input_path:
                await stage_6_selfrag_generation(input_path)
    else:
        # Run full pipeline
        await run_full_pipeline(
            extraction_limit=args.limit,
            dry_run=not args.execute,
            skip_stages=args.skip,
        )


if __name__ == "__main__":
    asyncio.run(main())
