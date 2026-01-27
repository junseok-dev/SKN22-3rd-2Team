"""
Patent Guard v2.0 - Self-RAG Patent Agent
==========================================
Advanced RAG pipeline with HyDE, Grading Loop, and Critical CoT Analysis.

Features:
1. HyDE (Hypothetical Document Embedding) - Generate virtual claims for better retrieval
2. Grading & Rewrite Loop - Score results and optimize queries
3. Critical CoT Analysis - Detailed similarity/infringement/avoidance analysis

Author: Patent Guard Team
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

load_dotenv()

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (Environment Variables)
# =============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Models - configurable via environment variables
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
GRADING_MODEL = os.environ.get("GRADING_MODEL", "gpt-4o-mini")  # Cost-effective
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "gpt-4o")  # High quality
HYDE_MODEL = os.environ.get("HYDE_MODEL", "gpt-4o-mini")

# Thresholds - configurable via environment variables
GRADING_THRESHOLD = float(os.environ.get("GRADING_THRESHOLD", "0.6"))
MAX_REWRITE_ATTEMPTS = int(os.environ.get("MAX_REWRITE_ATTEMPTS", "1"))
TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", "5"))

# Data paths - relative to this file
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================

class GradingResult(BaseModel):
    """Structured grading result from GPT."""
    patent_id: str = Field(description="Patent publication number")
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation for the score")


class GradingResponse(BaseModel):
    """Response containing all grading results."""
    results: List[GradingResult] = Field(description="List of grading results")
    average_score: float = Field(description="Average score across all results")


class QueryRewriteResponse(BaseModel):
    """Optimized search query from GPT."""
    optimized_query: str = Field(description="Improved search query")
    keywords: List[str] = Field(description="Key technical terms to search")
    reasoning: str = Field(description="Why this query should work better")


class SimilarityAnalysis(BaseModel):
    """유사도 평가 section."""
    score: int = Field(description="Technical similarity score 0-100")
    common_elements: List[str] = Field(description="Shared technical elements")
    summary: str = Field(description="Overall similarity assessment")
    evidence_patents: List[str] = Field(description="Patent IDs supporting this analysis")


class InfringementAnalysis(BaseModel):
    """침해 리스크 section."""
    risk_level: str = Field(description="high, medium, or low")
    risk_factors: List[str] = Field(description="Specific infringement concerns")
    summary: str = Field(description="Overall risk assessment")
    evidence_patents: List[str] = Field(description="Patent IDs supporting this analysis")


class AvoidanceStrategy(BaseModel):
    """회피 전략 section."""
    strategies: List[str] = Field(description="Design-around approaches")
    alternative_technologies: List[str] = Field(description="Alternative implementations")
    summary: str = Field(description="Recommended avoidance approach")
    evidence_patents: List[str] = Field(description="Patent IDs informing these strategies")


class CriticalAnalysisResponse(BaseModel):
    """Complete critical analysis response."""
    similarity: SimilarityAnalysis
    infringement: InfringementAnalysis
    avoidance: AvoidanceStrategy
    conclusion: str = Field(description="Final recommendation")


# =============================================================================
# Patent Search Result
# =============================================================================

@dataclass
class PatentSearchResult:
    """A single patent search result."""
    publication_number: str
    title: str
    abstract: str
    claims: str
    ipc_codes: List[str]
    similarity_score: float = 0.0  # Vector similarity
    grading_score: float = 0.0  # LLM grading score
    grading_reason: str = ""


# =============================================================================
# Mock Milvus Client (For Demo without Milvus)
# =============================================================================

class MockMilvusClient:
    """
    Mock Milvus client for demo purposes.
    In production, replace with actual pymilvus connection.
    """
    
    def __init__(self, data_path: str = None):
        self.patents = []
        
        # Use absolute path based on this file's location
        if data_path is None:
            from pathlib import Path
            data_path = Path(__file__).resolve().parent / "data" / "processed"
        
        self._load_patents(str(data_path))
    
    def _load_patents(self, data_path: str) -> None:
        """Load patents from processed JSON."""
        from pathlib import Path
        
        data_dir = Path(data_path)
        
        # Try the simplified filename first
        simple_file = data_dir / "processed_patents_10k.json"
        if simple_file.exists():
            with open(simple_file, 'r', encoding='utf-8') as f:
                self.patents = json.load(f)
            logger.info(f"Loaded {len(self.patents)} patents from {simple_file.name}")
            return
        
        # Fallback to glob pattern for any processed_patents file
        files = list(data_dir.glob("processed_patents_*.json"))
        
        if files:
            # Select the largest file (most patents)
            latest_file = max(files, key=lambda f: f.stat().st_size)
            with open(latest_file, 'r', encoding='utf-8') as f:
                self.patents = json.load(f)
            logger.info(f"Loaded {len(self.patents)} patents from {latest_file.name}")
        else:
            logger.warning(f"No patent files found in {data_path}")
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[PatentSearchResult]:
        """
        Mock vector search using cosine similarity.
        In production, this calls Milvus.
        """
        import random
        
        # For demo: return random patents
        if not self.patents:
            return []
        
        sample_size = min(top_k, len(self.patents))
        sampled = random.sample(self.patents, sample_size)
        
        results = []
        for i, patent in enumerate(sampled):
            # Get text content
            abstract = patent.get("abstract", "")
            claims_list = patent.get("claims", [])
            claims_text = ""
            if claims_list and isinstance(claims_list[0], dict):
                claims_text = claims_list[0].get("claim_text", "")
            
            results.append(PatentSearchResult(
                publication_number=patent.get("publication_number", f"UNKNOWN-{i}"),
                title=patent.get("title", ""),
                abstract=abstract[:500] if abstract else "",
                claims=claims_text[:1000] if claims_text else "",
                ipc_codes=patent.get("ipc_codes", []),
                similarity_score=0.8 - (i * 0.1),  # Mock scores
            ))
        
        return results


# =============================================================================
# Patent Agent - Main Class
# =============================================================================

class PatentAgent:
    """
    Self-RAG Patent Analysis Agent.
    
    Implements:
    1. HyDE - Hypothetical Document Embedding
    2. Grading & Rewrite Loop
    3. Critical CoT Analysis
    """
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check .env file.")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.vector_db = MockMilvusClient()
    
    # =========================================================================
    # 1. HyDE - Hypothetical Document Embedding
    # =========================================================================
    
    async def generate_hypothetical_claim(self, user_idea: str) -> str:
        """
        Generate a hypothetical patent claim from user's idea.
        
        This claim will be embedded and used for vector search,
        improving retrieval quality by matching the document format.
        """
        prompt = f"""당신은 특허 청구항 작성 전문가입니다.

사용자의 아이디어를 바탕으로 **가상의 특허 청구항(Claim)**을 1~2문장으로 작성해주세요.
실제 특허 청구항과 유사한 형식으로 작성하되, 핵심 기술 요소를 포함해야 합니다.

[사용자 아이디어]
{user_idea}

[가상 특허 청구항]"""

        response = await self.client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        
        hypothetical_claim = response.choices[0].message.content.strip()
        logger.info(f"Generated hypothetical claim: {hypothetical_claim[:100]}...")
        
        return hypothetical_claim
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI text-embedding-3-small."""
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    
    async def hyde_search(
        self,
        user_idea: str,
        top_k: int = TOP_K_RESULTS,
    ) -> Tuple[str, List[PatentSearchResult]]:
        """
        HyDE-enhanced patent search.
        
        1. Generate hypothetical claim from user idea
        2. Embed the hypothetical claim
        3. Search vector DB for similar patents
        
        Returns:
            Tuple of (hypothetical_claim, search_results)
        """
        # Generate hypothetical claim
        hypothetical_claim = await self.generate_hypothetical_claim(user_idea)
        
        # Embed the claim
        query_embedding = await self.embed_text(hypothetical_claim)
        
        # Search vector DB
        results = await self.vector_db.search(query_embedding, top_k=top_k)
        
        return hypothetical_claim, results
    
    # =========================================================================
    # 2. Grading & Rewrite Loop
    # =========================================================================
    
    async def grade_results(
        self,
        user_idea: str,
        results: List[PatentSearchResult],
    ) -> GradingResponse:
        """
        Grade each search result for relevance to user's idea.
        
        Uses structured output (JSON mode) for reliable parsing.
        """
        if not results:
            return GradingResponse(results=[], average_score=0.0)
        
        # Format results for grading
        results_text = "\n\n".join([
            f"[특허 {i+1}: {r.publication_number}]\n"
            f"제목: {r.title}\n"
            f"초록: {r.abstract[:300]}...\n"
            f"청구항: {r.claims[:300]}..."
            for i, r in enumerate(results)
        ])
        
        prompt = f"""당신은 특허 관련성 평가 전문가입니다.

사용자의 아이디어와 검색된 특허들의 관련성을 0.0~1.0 점수로 평가해주세요.

[사용자 아이디어]
{user_idea}

[검색된 특허 목록]
{results_text}

각 특허에 대해 다음 JSON 형식으로 응답해주세요:
{{
  "results": [
    {{"patent_id": "특허번호", "score": 0.0-1.0, "reason": "점수 이유"}}
  ],
  "average_score": 평균점수
}}"""

        response = await self.client.chat.completions.create(
            model=GRADING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        
        try:
            grading_data = json.loads(response.choices[0].message.content)
            grading_response = GradingResponse(**grading_data)
            
            # Update results with grades
            for grade in grading_response.results:
                for result in results:
                    if result.publication_number == grade.patent_id:
                        result.grading_score = grade.score
                        result.grading_reason = grade.reason
            
            return grading_response
            
        except Exception as e:
            logger.error(f"Failed to parse grading response: {e}")
            return GradingResponse(results=[], average_score=0.0)
    
    async def rewrite_query(
        self,
        user_idea: str,
        previous_results: List[PatentSearchResult],
    ) -> QueryRewriteResponse:
        """
        Optimize search query based on poor results.
        """
        results_summary = "\n".join([
            f"- {r.publication_number}: score={r.grading_score:.2f}, {r.grading_reason}"
            for r in previous_results
        ])
        
        prompt = f"""검색 결과가 관련성이 낮습니다. 검색 쿼리를 최적화해주세요.

[원래 아이디어]
{user_idea}

[이전 검색 결과 (낮은 점수)]
{results_summary}

더 관련성 높은 특허를 찾기 위해 검색 쿼리를 개선해주세요.
JSON 형식으로 응답:
{{
  "optimized_query": "개선된 검색 쿼리",
  "keywords": ["핵심", "기술", "키워드"],
  "reasoning": "개선 이유"
}}"""

        response = await self.client.chat.completions.create(
            model=GRADING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        try:
            data = json.loads(response.choices[0].message.content)
            return QueryRewriteResponse(**data)
        except Exception as e:
            logger.error(f"Failed to parse rewrite response: {e}")
            return QueryRewriteResponse(
                optimized_query=user_idea,
                keywords=[],
                reasoning="Failed to optimize"
            )
    
    async def search_with_grading(
        self,
        user_idea: str,
    ) -> List[PatentSearchResult]:
        """
        Complete search pipeline with grading and optional rewrite.
        """
        # Initial HyDE search
        hypothetical_claim, results = await self.hyde_search(user_idea)
        
        if not results:
            logger.warning("No search results found")
            return []
        
        # Grade results
        grading = await self.grade_results(user_idea, results)
        logger.info(f"Initial grading - Average score: {grading.average_score:.2f}")
        
        # Check if rewrite is needed
        if grading.average_score < GRADING_THRESHOLD:
            logger.info(f"Score below threshold ({GRADING_THRESHOLD}), attempting query rewrite...")
            
            # Rewrite query
            rewrite = await self.rewrite_query(user_idea, results)
            logger.info(f"Rewritten query: {rewrite.optimized_query}")
            
            # Search again with optimized query
            _, new_results = await self.hyde_search(rewrite.optimized_query)
            
            # Re-grade
            new_grading = await self.grade_results(user_idea, new_results)
            logger.info(f"After rewrite - Average score: {new_grading.average_score:.2f}")
            
            # Use better results
            if new_grading.average_score > grading.average_score:
                results = new_results
                grading = new_grading
        
        # Sort by grading score
        results.sort(key=lambda x: x.grading_score, reverse=True)
        
        return results
    
    # =========================================================================
    # 3. Critical CoT Analysis
    # =========================================================================
    
    async def critical_analysis(
        self,
        user_idea: str,
        results: List[PatentSearchResult],
    ) -> CriticalAnalysisResponse:
        """
        Perform critical Chain-of-Thought analysis.
        
        Generates detailed analysis with:
        - [유사도 평가] Similarity assessment
        - [침해 리스크] Infringement risk
        - [회피 전략] Avoidance strategy
        
        Each claim is backed by specific patent citations.
        """
        if not results:
            return self._empty_analysis()
        
        # Format patents for analysis
        patents_text = "\n\n".join([
            f"=== 특허 {r.publication_number} ===\n"
            f"제목: {r.title}\n"
            f"IPC: {', '.join(r.ipc_codes[:3])}\n"
            f"초록: {r.abstract}\n"
            f"청구항: {r.claims}\n"
            f"관련성 점수: {r.grading_score:.2f} ({r.grading_reason})"
            for r in results[:5]
        ])
        
        prompt = f"""당신은 특허 분석 전문 변리사입니다. Chain-of-Thought 방식으로 심층 분석해주세요.

[분석 대상: 사용자 아이디어]
{user_idea}

[참조 특허 목록]
{patents_text}

아래 JSON 형식으로 상세 분석을 제공해주세요. 
**중요**: 모든 분석 내용에는 반드시 근거가 된 특허 번호를 evidence_patents에 명시하세요.

{{
  "similarity": {{
    "score": 0-100 사이 점수,
    "common_elements": ["공통 기술 요소 목록"],
    "summary": "유사도에 대한 종합 평가",
    "evidence_patents": ["근거 특허 번호들"]
  }},
  "infringement": {{
    "risk_level": "high/medium/low",
    "risk_factors": ["구체적 침해 위험 요소"],
    "summary": "침해 리스크 종합 평가",
    "evidence_patents": ["근거 특허 번호들"]
  }},
  "avoidance": {{
    "strategies": ["회피 설계 방안들"],
    "alternative_technologies": ["대안 기술 접근법"],
    "summary": "권장 회피 전략",
    "evidence_patents": ["참고한 특허 번호들"]
  }},
  "conclusion": "최종 권고 사항 (특허 출원 가능성, 주의점 등)"
}}"""

        response = await self.client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 특허 분석 전문가입니다. 항상 근거를 명시하며 분석합니다."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2000,
        )
        
        try:
            data = json.loads(response.choices[0].message.content)
            return CriticalAnalysisResponse(**data)
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return self._empty_analysis()
    
    def _empty_analysis(self) -> CriticalAnalysisResponse:
        """Return empty analysis when no results."""
        return CriticalAnalysisResponse(
            similarity=SimilarityAnalysis(
                score=0,
                common_elements=[],
                summary="분석할 특허가 없습니다.",
                evidence_patents=[]
            ),
            infringement=InfringementAnalysis(
                risk_level="unknown",
                risk_factors=[],
                summary="분석할 특허가 없습니다.",
                evidence_patents=[]
            ),
            avoidance=AvoidanceStrategy(
                strategies=[],
                alternative_technologies=[],
                summary="분석할 특허가 없습니다.",
                evidence_patents=[]
            ),
            conclusion="검색 결과가 없어 분석을 수행할 수 없습니다."
        )
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    async def analyze(self, user_idea: str) -> Dict[str, Any]:
        """
        Complete Self-RAG pipeline.
        
        1. HyDE search with hypothetical claim
        2. Grade results and rewrite if needed
        3. Critical CoT analysis
        
        Returns complete analysis result.
        """
        print("\n" + "=" * 70)
        print("🛡️  Patent Guard - Self-RAG Analysis")
        print("=" * 70)
        
        print(f"\n📝 User Idea: {user_idea[:100]}...")
        
        # Step 1 & 2: Search with grading
        print("\n🔍 Step 1-2: HyDE Search & Grading...")
        results = await self.search_with_grading(user_idea)
        
        if not results:
            return {"error": "No relevant patents found"}
        
        print(f"   Found {len(results)} relevant patents")
        for r in results[:3]:
            print(f"   - {r.publication_number}: {r.grading_score:.2f}")
        
        # Step 3: Critical analysis
        print("\n🧠 Step 3: Critical CoT Analysis...")
        analysis = await self.critical_analysis(user_idea, results)
        
        # Format output
        output = {
            "user_idea": user_idea,
            "search_results": [
                {
                    "patent_id": r.publication_number,
                    "title": r.title,
                    "grading_score": r.grading_score,
                    "grading_reason": r.grading_reason,
                }
                for r in results
            ],
            "analysis": {
                "similarity": {
                    "score": analysis.similarity.score,
                    "common_elements": analysis.similarity.common_elements,
                    "summary": analysis.similarity.summary,
                    "evidence": analysis.similarity.evidence_patents,
                },
                "infringement": {
                    "risk_level": analysis.infringement.risk_level,
                    "risk_factors": analysis.infringement.risk_factors,
                    "summary": analysis.infringement.summary,
                    "evidence": analysis.infringement.evidence_patents,
                },
                "avoidance": {
                    "strategies": analysis.avoidance.strategies,
                    "alternatives": analysis.avoidance.alternative_technologies,
                    "summary": analysis.avoidance.summary,
                    "evidence": analysis.avoidance.evidence_patents,
                },
                "conclusion": analysis.conclusion,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 Analysis Complete!")
        print("=" * 70)
        print(f"\n[유사도 평가] Score: {analysis.similarity.score}/100")
        print(f"   {analysis.similarity.summary[:100]}...")
        print(f"\n[침해 리스크] Level: {analysis.infringement.risk_level.upper()}")
        print(f"   {analysis.infringement.summary[:100]}...")
        print(f"\n[회피 전략]")
        for s in analysis.avoidance.strategies[:2]:
            print(f"   - {s}")
        print(f"\n📌 Conclusion: {analysis.conclusion[:150]}...")
        
        return output


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Interactive CLI for patent analysis."""
    print("\n" + "=" * 70)
    print("🛡️  Patent Guard v2.0 - Self-RAG Patent Agent")
    print("=" * 70)
    print("\n특허 분석을 위한 아이디어를 입력하세요.")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")
    
    agent = PatentAgent()
    
    while True:
        try:
            user_input = input("\n💡 Your idea: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("❌ Please enter an idea.")
                continue
            
            # Run analysis
            result = await agent.analyze(user_input)
            
            # Save result to outputs directory
            output_path = OUTPUT_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Result saved to: {output_path}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
