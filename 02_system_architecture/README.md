# 🏗️ 시스템 아키텍처

> **⚡ 쇼특허 (Short-Cut) v3.0 - AI 특허 선행 기술 조사 시스템**  
> Team: 뀨💕 | 작성일: 2026-01-28

---

## 1. 시스템 개요

### 1.1 목적
사용자가 특허 출원을 고려하는 아이디어를 입력하면, AI가 기존 특허 데이터베이스를 검색하여 **유사 특허**, **침해 리스크**, **회피 전략**을 제공하는 시스템

### 1.2 핵심 기술 (v3.0)

| 기술 | 설명 |
|------|------|
| **Self-RAG** | 검색 결과를 비판적으로 평가하고 재검색하는 지능형 RAG |
| **HyDE** | 가상 문서 생성으로 검색 품질 향상 |
| **Hybrid Search** | FAISS (Dense) + BM25 (Sparse) + RRF 융합 검색 |
| **LLM Streaming** | 실시간 분석 결과 출력 (체감 대기시간 0초) |
| **4-Level Parser** | 다국어 청구항 파싱 (US/EP/KR 지원) |
| **Chain-of-Thought** | 단계별 추론으로 정확한 분석 제공 |

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ⚡ 쇼특허 (Short-Cut) v3.0                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐ │
│  │   사용자    │───▶│   Streamlit App      │───▶│     OpenAI API          │ │
│  │   입력      │    │   (app.py)           │    │  ┌─────────────────────┐│ │
│  │             │    │                      │    │  │ text-embedding-3    ││ │
│  │ "아이디어"  │    │  ┌────────────────┐  │    │  │ gpt-4o-mini         ││ │
│  └─────────────┘    │  │  PatentAgent   │  │    │  │ gpt-4o              ││ │
│                     │  │  ┌──────────┐  │  │    │  └─────────────────────┘│ │
│                     │  │  │  HyDE    │  │  │    └─────────────────────────┘ │
│                     │  │  └──────────┘  │  │                                │
│                     │  │  ┌──────────┐  │  │    ┌─────────────────────────┐ │
│                     │  │  │ Hybrid   │  │  │    │    Vector DB (In-Mem)   │ │
│                     │  │  │ Search   │──┼──┼───▶│  ┌─────────────────────┐│ │
│                     │  │  └──────────┘  │  │    │  │ FAISS (Dense)       ││ │
│                     │  │  ┌──────────┐  │  │    │  │ BM25 (Sparse)        ││ │
│                     │  │  │ Grading  │  │  │    │  │ RRF Fusion           ││ │
│                     │  │  └──────────┘  │  │    │  └─────────────────────┘│ │
│                     │  │  ┌──────────┐  │  │    │  20,664 chunks          │ │
│                     │  │  │Streaming │  │  │    └─────────────────────────┘ │
│                     │  │  │ Analysis │  │  │                                │
│                     │  │  └──────────┘  │  │                                │
│                     │  └────────────────┘  │                                │
│                     └──────────────────────┘                                │
│                            │                                                │
│                            ▼                                                │
│                     ┌─────────────────────────┐                             │
│                     │  분석 결과 (Streaming)   │                             │
│                     │ ┌─────────────────────┐ │                             │
│                     │ │ 유사도 평가         │ │                             │
│                     │ │ 침해 리스크         │ │                             │
│                     │ │ 구성요소 대비표     │ │                             │
│                     │ │ 회피 전략           │ │                             │
│                     │ └─────────────────────┘ │                             │
│                     └─────────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 컴포넌트

### 3.1 Patent Agent (`patent_agent.py`)

Self-RAG 파이프라인 + Streaming 분석 엔진

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PatentAgent v3.0                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Stage 1] HyDE (Hypothetical Document Embedding)                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 사용자 아이디어 → GPT-4o-mini → 가상 특허 청구항               │  │
│  │ → text-embedding-3-small → 1536차원 벡터                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           ↓                                          │
│  [Stage 2] Hybrid Search (FAISS + BM25 + RRF)                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Dense: FAISS IndexFlatIP → Top-K vectors                       │  │
│  │ Sparse: BM25 keyword matching → Top-K docs                     │  │
│  │ RRF Fusion: k=60, weight=0.5:0.5                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           ↓                                          │
│  [Stage 3] Grading & Rewrite Loop                                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 검색 결과 5개 → GPT-4o-mini → 관련성 점수 (0~1)                │  │
│  │ 평균 < 0.6 → 쿼리 재작성 → 재검색 (최대 1회)                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           ↓                                          │
│  [Stage 4] Streaming Critical Analysis                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 최종 선정 특허 → GPT-4o (Streaming) → 실시간 분석              │  │
│  │ [유사도] + [침해 리스크] + [구성요소 대비표] + [회피 전략]     │  │
│  │ 각 분석마다 근거 특허 번호 명시                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Vector Database (`vector_db.py`)

하이브리드 검색 엔진 (FAISS + BM25 + RRF)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FaissClient v3.0                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐   │
│  │       FAISS (Dense)         │  │       BM25 (Sparse)         │   │
│  │  ┌───────────────────────┐  │  │  ┌───────────────────────┐  │   │
│  │  │ IndexFlatIP           │  │  │  │ rank_bm25 (Okapi)     │  │   │
│  │  │ Inner Product Search  │  │  │  │ Keyword Matching      │  │   │
│  │  │ Cosine Similarity     │  │  │  │ TF-IDF based          │  │   │
│  │  │ 20,664 vectors        │  │  │  │ 20,664 documents      │  │   │
│  │  └───────────────────────┘  │  │  └───────────────────────┘  │   │
│  │            ↓                │  │            ↓                │   │
│  │      Top-K (by score)       │  │      Top-K (by BM25)        │   │
│  └─────────────────────────────┘  └─────────────────────────────┘   │
│                    ↘                    ↙                            │
│                 ┌───────────────────────────┐                        │
│                 │    RRF Fusion (k=60)      │                        │
│                 │ ────────────────────────  │                        │
│                 │ score = Σ 1/(k + rank)    │                        │
│                 │ dense_weight: 0.5         │                        │
│                 │ sparse_weight: 0.5        │                        │
│                 └───────────────────────────┘                        │
│                             ↓                                        │
│                      Final Top-K Results                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 Claim Parser (`preprocessor.py`)

4-Level Fallback 청구항 파서

```
┌──────────────────────────────────────────────────────────────────────┐
│                      ClaimParser v3.0                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Level 1] Regex Pattern Matching                                    │
│  ├── US/EP: "1. A method comprising..."                              │
│  ├── Claim: "Claim 1: A method..."                                   │
│  ├── Bracket: "(1) A method...", "[1] A method..."                   │
│  └── Korean: "제1항:", "청구항 1:"                                   │
│                           ↓ (매칭 실패 시)                            │
│  [Level 2] Structure-Based Parsing                                   │
│  ├── 들여쓰기 분석                                                   │
│  └── 번호 체계 탐지                                                  │
│                           ↓ (매칭 실패 시)                            │
│  [Level 3] NLP Sentence Segmentation (Spacy)                         │
│  └── 문장 경계 탐지                                                  │
│                           ↓ (매칭 실패 시)                            │
│  [Level 4] Minimal Fallback                                          │
│  ├── 문단 분리 (\n\n)                                                │
│  └── 전체 1개 청구항 (최후 수단)                                     │
│                                                                      │
│  ✅ 파싱 성공률: ~95%                                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 데이터 파이프라인 (`pipeline.py`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Pipeline v3.0                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Stage 1] BigQuery Extraction                                       │
│      Google Patents → SQL 쿼리 → patents_10k.json                   │
│                                                                     │
│  [Stage 2] Preprocessing                                             │
│      원본 JSON → 정규화 → 메타데이터 추가                           │
│                                                                     │
│  [Stage 3] Chunking                                                  │
│      전처리 JSON → 4-Level 파싱 → 청킹 (1024 tokens)                │
│                                                                     │
│  [Stage 4] Embedding Generation                                      │
│      청킹 데이터 → OpenAI API → embeddings_*.npz                    │
│                                                                     │
│  [Stage 5] Index Building                                            │
│      임베딩 → FAISS 인덱스 + BM25 인덱스                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 기술 스택

### 4.1 Backend

| 구분 | 기술 |
|------|------|
| **언어** | Python 3.11+ |
| **Web UI** | Streamlit |
| **비동기** | asyncio, nest-asyncio |
| **데이터 검증** | Pydantic v2 |
| **환경 변수** | python-dotenv |

### 4.2 AI/ML

| 구분 | 기술 |
|------|------|
| **LLM** | OpenAI GPT-4o, GPT-4o-mini |
| **Embedding** | OpenAI text-embedding-3-small (1536차원) |
| **Dense Search** | FAISS IndexFlatIP (Inner Product) |
| **Sparse Search** | rank-bm25 (Okapi BM25) |
| **Fusion** | RRF (Reciprocal Rank Fusion) |
| **NLP** | Spacy en_core_web_sm (선택) |

### 4.3 Data

| 구분 | 기술 |
|------|------|
| **Data Source** | Google Patents (BigQuery) |
| **Storage** | JSON, NPZ (NumPy), PKL (Pickle) |
| **Indexing** | FAISS .bin, BM25 .pkl |

---

## 5. API 설계

### 5.1 OpenAI API 사용

| 모델 | 용도 | 비용 |
|------|------|------|
| `text-embedding-3-small` | 텍스트 임베딩 | $0.02/1M tokens |
| `gpt-4o-mini` | HyDE, Grading, Query Rewrite | $0.15/1M input |
| `gpt-4o` | Critical Analysis (Streaming) | $5.00/1M input |

### 5.2 Pydantic Models

```python
# Patent Search Result
@dataclass
class PatentSearchResult:
    publication_number: str
    title: str
    abstract: str
    claims: str
    ipc_codes: List[str]
    similarity_score: float = 0.0
    dense_score: float = 0.0      # FAISS score
    sparse_score: float = 0.0     # BM25 score
    rrf_score: float = 0.0        # RRF fusion score
    grading_score: float = 0.0
    grading_reason: str = ""

# Critical Analysis Response
class CriticalAnalysisResponse(BaseModel):
    similarity: SimilarityAnalysis
    infringement: InfringementAnalysis
    avoidance: AvoidanceStrategy
    component_comparison: ComponentComparison
    conclusion: str
```

### 5.3 응답 형식 (JSON Mode)

```json
{
  "similarity": {
    "score": 45,
    "common_elements": ["vector search", "neural network"],
    "summary": "중간 수준 유사도",
    "evidence_patents": ["US-123", "CN-456"]
  },
  "infringement": {
    "risk_level": "medium",
    "risk_factors": ["벡터 검색 방법 유사"],
    "summary": "잠재적 침해 위험",
    "evidence_patents": ["US-123"]
  },
  "component_comparison": {
    "idea_components": ["RAG", "hybrid search"],
    "matched_components": ["vector search"],
    "unmatched_components": ["RRF fusion"],
    "risk_components": ["embedding method"]
  },
  "avoidance": {
    "strategies": ["차별화된 융합 알고리즘 사용"],
    "alternative_technologies": ["BM25-only approach"],
    "summary": "회피 가능",
    "evidence_patents": ["EP-789"]
  },
  "conclusion": "특허 출원 가능, 차별화 권장"
}
```

---

## 6. 파일 구조

```
SKN22-3rd-2Team/
├── app.py                    # 🎯 Streamlit 웹 앱 (루트)
├── src/
│   ├── patent_agent.py       # 🤖 Self-RAG 에이전트
│   │   ├── PatentAgent       # 메인 분석 클래스
│   │   ├── PatentSearchResult
│   │   └── Pydantic Models   # 응답 구조 정의
│   │
│   ├── vector_db.py          # 🗄️ FAISS + BM25 하이브리드
│   │   ├── FaissClient       # 하이브리드 검색 클라이언트
│   │   ├── BM25SearchEngine  # BM25 검색 엔진
│   │   └── SearchResult      # 검색 결과 데이터클래스
│   │
│   ├── preprocessor.py       # 🔧 4-Level 청구항 파서
│   │   ├── ClaimParser       # 4-Level Fallback 파서
│   │   ├── ParsedClaim       # 파싱된 청구항 데이터클래스
│   │   └── PatentPreprocessor
│   │
│   ├── embedder.py           # 🧠 OpenAI 임베딩
│   ├── pipeline.py           # ⚙️ Stage 1-5 파이프라인
│   ├── config.py             # ⚙️ 설정 관리
│   │
│   └── data/
│       ├── raw/              # 원본 JSON
│       ├── processed/        # 전처리 JSON
│       ├── embeddings/       # NPZ 벡터 파일
│       └── index/            # FAISS + BM25 인덱스
│
├── tests/
│   ├── test_hybrid_search.py # RRF 알고리즘 테스트
│   ├── test_parser.py        # 4-Level 파서 테스트
│   └── conftest.py           # pytest 설정
│
└── report/
    ├── v3_technical_proposal.md
    └── test_report*.html
```

---

## 7. 데이터 흐름

```
[사용자 입력]
    │
    ▼
[HyDE] ─────────────────────────────────────────────┐
    │ 가상 청구항 생성 (GPT-4o-mini)                │
    ▼                                               │
[Embedding] ────────────────────────────────────────┤
    │ text-embedding-3-small → 1536 dim             │
    ▼                                               │
[Hybrid Search] ────────────────────────────────────┤
    │ ┌─ FAISS (Dense) ─┐                           │
    │ │  Inner Product   │                          │
    │ └─────────────────┘                           │
    │ ┌─ BM25 (Sparse) ─┐                           │
    │ │  Keyword Match   │                          │
    │ └─────────────────┘                           │
    │ ┌─ RRF Fusion ────┐                           │
    │ │  k=60, 0.5:0.5  │                           │
    │ └─────────────────┘                           │
    ▼                                               │
[Grading] ──────────────────────────────────────────┤
    │ 관련성 평가 (GPT-4o-mini)                     │
    │ 점수 < 0.6 → Query Rewrite → 재검색           │
    ▼                                               │
[Streaming Analysis] ───────────────────────────────┤
    │ GPT-4o Streaming                              │
    │ 실시간 토큰 출력                              │
    ▼                                               │
[Output]                                            │
    ├─ 유사도 평가 (0-100)                          │
    ├─ 침해 리스크 (high/medium/low)                │
    ├─ 구성요소 대비표                              │
    └─ 회피 전략                                    │
                                                    │
    [API 호출: ~3-5회/분석]─────────────────────────┘
```

---

## 8. 하이브리드 검색 상세

### 8.1 RRF (Reciprocal Rank Fusion)

```
RRF_score(d) = Σ weight_i / (k + rank_i(d))

where:
  k = 60 (smoothing constant)
  weight_i = 0.5 (dense), 0.5 (sparse)
  rank_i(d) = rank of document d in result list i
```

### 8.2 검색 품질 비교

| 방식 | 장점 | 단점 | 품질 |
|------|------|------|------|
| BM25 Only | 빠름, 무료 | 의미 유사성 X | 70-75% |
| FAISS Only | 의미 유사성 O | 키워드 누락 시 약함 | 85-90% |
| **Hybrid (RRF)** | **둘의 장점 결합** | 복잡도 증가 | **92-95%** |

---

## 9. 환경 변수

```env
# 필수
OPENAI_API_KEY=your-api-key

# BigQuery (데이터 추출 시)
GCP_PROJECT_ID=your-project-id

# 선택 (기본값 있음)
EMBEDDING_MODEL=text-embedding-3-small
GRADING_MODEL=gpt-4o-mini
ANALYSIS_MODEL=gpt-4o
GRADING_THRESHOLD=0.6
TOP_K_RESULTS=5

# Hybrid Search 설정
DENSE_WEIGHT=0.5
SPARSE_WEIGHT=0.5
RRF_K=60
```

---

## 10. 성능 지표

| 항목 | v2.0 | v3.0 |
|------|------|------|
| 검색 품질 | 70-75% (BM25) | **92-95%** (Hybrid RRF) |
| 파싱 성공률 | ~90% | **~95%** (4-Level) |
| 응답 체감 시간 | 10-15초 | **0초** (Streaming) |

> 📋 테스트 계획 및 결과 보고서는 [03_test_report/](../03_test_report/)를 참고하세요.

---

*작성: ⚡ 쇼특허 (Short-Cut) Team - 뀨💕*

