"""
Patent Guard v2.0 - Patent Data Preprocessor
=============================================
Processes raw patent data with hierarchical chunking and claim parsing.

Features:
- Individual claim extraction (Claim 1, 2, 3...)
- Parent-Child chunking strategy
- RAG component keyword tagging
- Async batch processing

Author: Patent Guard Team
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from tqdm import tqdm

from config import config, DomainConfig, PROCESSED_DATA_DIR

# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParsedClaim:
    """Represents a single parsed claim."""
    claim_number: int
    claim_text: str
    claim_type: str  # "independent" or "dependent"
    parent_claim: Optional[int]  # Reference to parent claim if dependent
    rag_components: List[str]  # Detected RAG component keywords
    char_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        self.char_count = len(self.claim_text)
        self.word_count = len(self.claim_text.split())


@dataclass
class PatentChunk:
    """Represents a chunk of patent content."""
    chunk_id: str
    patent_id: str
    chunk_type: str  # "parent", "claim", "abstract", "description_section"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rag_components: List[str] = field(default_factory=list)
    
    # Hierarchy
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class ProcessedPatent:
    """Fully processed patent document."""
    publication_number: str
    title: str
    abstract: str
    filing_date: Optional[str]
    
    # Parsed claims
    claims: List[ParsedClaim]
    
    # Chunked content
    chunks: List[PatentChunk]
    
    # Classification
    ipc_codes: List[str]
    cpc_codes: List[str]
    
    # Citations
    cited_publications: List[str]
    citation_count: int
    
    # RAG relevance
    rag_component_tags: List[str]
    importance_score: float
    
    # Metadata
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Claim Parser
# =============================================================================

class ClaimParser:
    """Parse patent claims into individual numbered claims."""
    
    # Regex patterns for claim extraction
    CLAIM_PATTERNS = [
        # Pattern 1: "1. A method comprising..."
        r'(?P<num>\d+)\.\s*(?P<text>(?:(?!\n\d+\.).)+)',
        
        # Pattern 2: "Claim 1: A method..."
        r'[Cc]laim\s*(?P<num>\d+)[:\.]?\s*(?P<text>(?:(?!\n[Cc]laim\s*\d+).)+)',
        
        # Pattern 3: Numbered with parentheses "(1) A method..."
        r'\((?P<num>\d+)\)\s*(?P<text>(?:(?!\n\(\d+\)).)+)',
    ]
    
    # Patterns indicating dependent claims
    DEPENDENT_PATTERNS = [
        r'according to claim\s*(\d+)',
        r'as claimed in claim\s*(\d+)',
        r'of claim\s*(\d+)',
        r'claim\s*(\d+),?\s*wherein',
        r'the (?:method|system|apparatus|device) of claim\s*(\d+)',
    ]
    
    def __init__(self, domain_config: DomainConfig = config.domain):
        self.rag_keywords = [kw.lower() for kw in domain_config.rag_component_keywords]
    
    def parse_claims_text(self, claims_text: str) -> List[ParsedClaim]:
        """
        Parse claims text into individual claims.
        
        Args:
            claims_text: Full claims section text
            
        Returns:
            List of ParsedClaim objects
        """
        if not claims_text or not claims_text.strip():
            return []
        
        # Try each pattern
        claims = []
        
        for pattern in self.CLAIM_PATTERNS:
            matches = list(re.finditer(pattern, claims_text, re.DOTALL | re.MULTILINE))
            if matches:
                for match in matches:
                    claim_num = int(match.group('num'))
                    claim_text = self._clean_claim_text(match.group('text'))
                    
                    if claim_text:  # Skip empty claims
                        claim_type, parent_claim = self._determine_claim_type(claim_text)
                        rag_components = self._detect_rag_components(claim_text)
                        
                        claims.append(ParsedClaim(
                            claim_number=claim_num,
                            claim_text=claim_text,
                            claim_type=claim_type,
                            parent_claim=parent_claim,
                            rag_components=rag_components,
                        ))
                break  # Use first matching pattern
        
        # Fallback: split by numbered lines if no pattern matched
        if not claims:
            claims = self._fallback_parse(claims_text)
        
        # Sort by claim number
        claims.sort(key=lambda c: c.claim_number)
        
        return claims
    
    def _clean_claim_text(self, text: str) -> str:
        """Clean and normalize claim text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        # Remove trailing claim numbers from next claim
        text = re.sub(r'\s*\d+\.\s*$', '', text)
        return text
    
    def _determine_claim_type(self, claim_text: str) -> Tuple[str, Optional[int]]:
        """
        Determine if claim is independent or dependent.
        
        Returns:
            Tuple of (claim_type, parent_claim_number)
        """
        claim_lower = claim_text.lower()
        
        for pattern in self.DEPENDENT_PATTERNS:
            match = re.search(pattern, claim_lower)
            if match:
                parent_num = int(match.group(1))
                return "dependent", parent_num
        
        return "independent", None
    
    def _detect_rag_components(self, claim_text: str) -> List[str]:
        """Detect RAG-related component keywords in claim."""
        claim_lower = claim_text.lower()
        detected = []
        
        for keyword in self.rag_keywords:
            if keyword in claim_lower:
                detected.append(keyword)
        
        return detected
    
    def _fallback_parse(self, claims_text: str) -> List[ParsedClaim]:
        """Fallback parsing when no pattern matches."""
        claims = []
        
        # Split by lines starting with numbers
        lines = claims_text.split('\n')
        current_claim = []
        current_num = None
        
        for line in lines:
            # Check if line starts a new claim
            match = re.match(r'^(\d+)[.\)]\s*(.*)$', line.strip())
            if match:
                # Save previous claim if exists
                if current_num is not None and current_claim:
                    claim_text = ' '.join(current_claim)
                    claim_type, parent = self._determine_claim_type(claim_text)
                    claims.append(ParsedClaim(
                        claim_number=current_num,
                        claim_text=claim_text,
                        claim_type=claim_type,
                        parent_claim=parent,
                        rag_components=self._detect_rag_components(claim_text),
                    ))
                
                current_num = int(match.group(1))
                current_claim = [match.group(2)]
            else:
                if current_num is not None:
                    current_claim.append(line.strip())
        
        # Don't forget last claim
        if current_num is not None and current_claim:
            claim_text = ' '.join(current_claim)
            claim_type, parent = self._determine_claim_type(claim_text)
            claims.append(ParsedClaim(
                claim_number=current_num,
                claim_text=claim_text,
                claim_type=claim_type,
                parent_claim=parent,
                rag_components=self._detect_rag_components(claim_text),
            ))
        
        return claims


# =============================================================================
# Hierarchical Chunker
# =============================================================================

class HierarchicalChunker:
    """
    Create hierarchical chunks with Parent-Child strategy.
    
    Parent chunks: Full patent context
    Child chunks: Individual claims, description sections
    """
    
    def __init__(
        self,
        max_chunk_size: int = 8000,  # Characters
        overlap_size: int = 200,
        domain_config: DomainConfig = config.domain,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.rag_keywords = [kw.lower() for kw in domain_config.rag_component_keywords]
    
    def create_chunks(
        self,
        patent_id: str,
        title: str,
        abstract: str,
        claims: List[ParsedClaim],
        description: str = "",
    ) -> List[PatentChunk]:
        """
        Create hierarchical chunks for a patent.
        
        Args:
            patent_id: Publication number
            title: Patent title
            abstract: Patent abstract
            claims: Parsed claims
            description: Full description text
            
        Returns:
            List of PatentChunk objects
        """
        chunks = []
        
        # 1. Parent chunk (full context summary)
        parent_chunk = self._create_parent_chunk(
            patent_id, title, abstract, claims
        )
        chunks.append(parent_chunk)
        
        # 2. Abstract chunk
        if abstract:
            abstract_chunk = PatentChunk(
                chunk_id=f"{patent_id}_abstract",
                patent_id=patent_id,
                chunk_type="abstract",
                content=abstract,
                parent_chunk_id=parent_chunk.chunk_id,
                rag_components=self._detect_rag_components(abstract),
                metadata={"section": "abstract"},
            )
            chunks.append(abstract_chunk)
            parent_chunk.child_chunk_ids.append(abstract_chunk.chunk_id)
        
        # 3. Individual claim chunks
        for claim in claims:
            claim_chunk = PatentChunk(
                chunk_id=f"{patent_id}_claim_{claim.claim_number}",
                patent_id=patent_id,
                chunk_type="claim",
                content=claim.claim_text,
                parent_chunk_id=parent_chunk.chunk_id,
                rag_components=claim.rag_components,
                metadata={
                    "claim_number": claim.claim_number,
                    "claim_type": claim.claim_type,
                    "parent_claim": claim.parent_claim,
                },
            )
            chunks.append(claim_chunk)
            parent_chunk.child_chunk_ids.append(claim_chunk.chunk_id)
        
        # 4. Description section chunks (if provided)
        if description:
            desc_chunks = self._chunk_description(
                patent_id, description, parent_chunk.chunk_id
            )
            chunks.extend(desc_chunks)
            parent_chunk.child_chunk_ids.extend([c.chunk_id for c in desc_chunks])
        
        return chunks
    
    def _create_parent_chunk(
        self,
        patent_id: str,
        title: str,
        abstract: str,
        claims: List[ParsedClaim],
    ) -> PatentChunk:
        """Create parent chunk with full patent context."""
        
        # Combine key information
        content_parts = [
            f"Title: {title}",
            f"\nAbstract: {abstract}",
            f"\nNumber of Claims: {len(claims)}",
        ]
        
        # Add independent claims summary
        independent_claims = [c for c in claims if c.claim_type == "independent"]
        if independent_claims:
            content_parts.append("\nIndependent Claims Summary:")
            for claim in independent_claims[:3]:  # Top 3 independent claims
                content_parts.append(f"  Claim {claim.claim_number}: {claim.claim_text[:500]}...")
        
        content = "\n".join(content_parts)
        
        # Aggregate RAG components
        all_rag_components = set()
        for claim in claims:
            all_rag_components.update(claim.rag_components)
        all_rag_components.update(self._detect_rag_components(title))
        all_rag_components.update(self._detect_rag_components(abstract))
        
        return PatentChunk(
            chunk_id=f"{patent_id}_parent",
            patent_id=patent_id,
            chunk_type="parent",
            content=content,
            rag_components=list(all_rag_components),
            metadata={
                "title": title,
                "total_claims": len(claims),
                "independent_claims": len(independent_claims),
            },
        )
    
    def _chunk_description(
        self,
        patent_id: str,
        description: str,
        parent_chunk_id: str,
    ) -> List[PatentChunk]:
        """Chunk description into sections."""
        chunks = []
        
        # Split by common section headers
        section_pattern = r'(?:DETAILED DESCRIPTION|BACKGROUND|SUMMARY|BRIEF DESCRIPTION|CLAIMS|FIELD OF THE INVENTION)'
        
        sections = re.split(f'({section_pattern})', description, flags=re.IGNORECASE)
        
        section_name = "introduction"
        chunk_idx = 0
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Check if this is a section header
            if re.match(section_pattern, section, re.IGNORECASE):
                section_name = section.strip().lower().replace(' ', '_')
                continue
            
            # Split large sections
            section_chunks = self._split_text(section)
            
            for j, chunk_text in enumerate(section_chunks):
                if not chunk_text.strip():
                    continue
                
                chunk = PatentChunk(
                    chunk_id=f"{patent_id}_desc_{chunk_idx}",
                    patent_id=patent_id,
                    chunk_type="description_section",
                    content=chunk_text,
                    parent_chunk_id=parent_chunk_id,
                    rag_components=self._detect_rag_components(chunk_text),
                    metadata={
                        "section": section_name,
                        "chunk_index": chunk_idx,
                    },
                )
                chunks.append(chunk)
                chunk_idx += 1
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting max size with overlap."""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for sep in ['. ', '.\n', '! ', '? ']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.max_chunk_size // 2:
                        end = last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap_size
        
        return chunks
    
    def _detect_rag_components(self, text: str) -> List[str]:
        """Detect RAG component keywords in text."""
        if not text:
            return []
        
        text_lower = text.lower()
        detected = []
        
        for keyword in self.rag_keywords:
            if keyword in text_lower:
                detected.append(keyword)
        
        return detected


# =============================================================================
# Main Preprocessor
# =============================================================================

class PatentPreprocessor:
    """
    Main preprocessing pipeline for patent data.
    
    Combines claim parsing and hierarchical chunking.
    """
    
    def __init__(
        self,
        domain_config: DomainConfig = config.domain,
        max_chunk_size: int = 8000,
    ):
        self.claim_parser = ClaimParser(domain_config)
        self.chunker = HierarchicalChunker(max_chunk_size, domain_config=domain_config)
        self.domain_config = domain_config
    
    def process_patent(self, raw_patent: Dict[str, Any]) -> ProcessedPatent:
        """
        Process a single raw patent record.
        
        Args:
            raw_patent: Raw patent data from BigQuery
            
        Returns:
            ProcessedPatent object
        """
        publication_number = raw_patent.get('publication_number', 'UNKNOWN')
        
        # Extract text content
        title = self._extract_localized_text(raw_patent.get('title_localized', []))
        abstract = self._extract_localized_text(raw_patent.get('abstract_localized', []))
        claims_text = self._extract_localized_text(raw_patent.get('claims_localized', []))
        description = self._extract_localized_text(raw_patent.get('description_localized', []))
        
        # Parse claims
        parsed_claims = self.claim_parser.parse_claims_text(claims_text)
        
        # Create chunks
        chunks = self.chunker.create_chunks(
            patent_id=publication_number,
            title=title,
            abstract=abstract,
            claims=parsed_claims,
            description=description,
        )
        
        # Extract classification codes
        ipc_codes = self._extract_codes(raw_patent.get('ipc', []))
        cpc_codes = self._extract_codes(raw_patent.get('cpc', []))
        
        # Extract citations
        cited_publications = raw_patent.get('cited_publications', [])
        if not cited_publications:
            # Fallback: extract from citation array
            citations = raw_patent.get('citation', [])
            if citations:
                cited_publications = [
                    c.get('publication_number') or c.get('npl_text', '')
                    for c in citations
                    if c
                ]
        
        # Aggregate RAG components
        all_rag_components = set()
        for claim in parsed_claims:
            all_rag_components.update(claim.rag_components)
        for chunk in chunks:
            all_rag_components.update(chunk.rag_components)
        
        return ProcessedPatent(
            publication_number=publication_number,
            title=title,
            abstract=abstract,
            filing_date=raw_patent.get('filing_date_parsed'),
            claims=parsed_claims,
            chunks=chunks,
            ipc_codes=ipc_codes,
            cpc_codes=cpc_codes,
            cited_publications=cited_publications,
            citation_count=raw_patent.get('citation_count', len(cited_publications)),
            rag_component_tags=list(all_rag_components),
            importance_score=raw_patent.get('importance_score', 0.0),
        )
    
    async def process_patents_batch(
        self,
        raw_patents: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> List[ProcessedPatent]:
        """
        Process a batch of patents asynchronously.
        
        Args:
            raw_patents: List of raw patent records
            output_path: Optional path to save processed data
            
        Returns:
            List of ProcessedPatent objects
        """
        loop = asyncio.get_event_loop()
        
        processed = []
        
        for raw in tqdm(raw_patents, desc="Processing patents"):
            patent = await loop.run_in_executor(
                None, self.process_patent, raw
            )
            processed.append(patent)
        
        # Save if path provided
        if output_path:
            self._save_processed_patents(processed, output_path)
        
        return processed
    
    def _extract_localized_text(
        self,
        localized_texts: List[Dict[str, str]],
        preferred_lang: str = "en",
    ) -> str:
        """Extract text from localized array, preferring English."""
        if not localized_texts:
            return ""
        
        # Try preferred language first
        for item in localized_texts:
            if isinstance(item, dict) and item.get('language') == preferred_lang:
                return item.get('text', '')
        
        # Fallback to first available
        if isinstance(localized_texts[0], dict):
            return localized_texts[0].get('text', '')
        elif isinstance(localized_texts[0], str):
            return localized_texts[0]
        
        return ""
    
    def _extract_codes(self, code_array: List[Any]) -> List[str]:
        """Extract classification codes from array."""
        codes = []
        
        for item in code_array:
            if isinstance(item, dict):
                code = item.get('code', '')
                if code:
                    codes.append(code)
            elif isinstance(item, str):
                codes.append(item)
        
        return codes
    
    def _save_processed_patents(
        self,
        patents: List[ProcessedPatent],
        output_path: Path,
    ) -> None:
        """Save processed patents to JSON file."""
        from dataclasses import asdict
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = []
        for patent in patents:
            patent_dict = asdict(patent)
            # Convert ParsedClaim and PatentChunk to dicts
            patent_dict['claims'] = [asdict(c) for c in patent.claims]
            patent_dict['chunks'] = [asdict(c) for c in patent.chunks]
            data.append(patent_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(patents)} processed patents to: {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point for standalone execution."""
    import sys
    from config import RAW_DATA_DIR
    
    logging.basicConfig(
        level=logging.INFO,
        format=config.logging.log_format,
    )
    
    print("\n" + "=" * 70)
    print("🛡️  Patent Guard v2.0 - Patent Preprocessor")
    print("=" * 70)
    
    # Check for input file argument
    if len(sys.argv) < 2:
        # Look for most recent raw data file
        raw_files = list(RAW_DATA_DIR.glob("patents_*.json"))
        if not raw_files:
            print("❌ No raw patent data found. Run bigquery_extractor.py first.")
            return
        input_path = max(raw_files, key=lambda p: p.stat().st_mtime)
    else:
        input_path = Path(sys.argv[1])
    
    print(f"📂 Input: {input_path}")
    
    # Load raw data
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_patents = json.load(f)
    
    print(f"📊 Loaded {len(raw_patents)} raw patents")
    
    # Process
    preprocessor = PatentPreprocessor()
    
    output_path = PROCESSED_DATA_DIR / f"processed_{input_path.stem}.json"
    processed = await preprocessor.process_patents_batch(raw_patents, output_path)
    
    # Summary
    total_claims = sum(len(p.claims) for p in processed)
    total_chunks = sum(len(p.chunks) for p in processed)
    rag_tagged = sum(1 for p in processed if p.rag_component_tags)
    
    print(f"\n✅ Processing complete!")
    print(f"   Patents: {len(processed)}")
    print(f"   Claims: {total_claims}")
    print(f"   Chunks: {total_chunks}")
    print(f"   RAG-tagged: {rag_tagged}")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
