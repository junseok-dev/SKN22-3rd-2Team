# ⚡ 쇼특허 (Short-Cut) v3.0

> **AI 기반 특허 침해 리스크 분석 시스템**  
> Self-RAG + Hybrid Search + Streaming Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple.svg)](https://www.pinecone.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [시스템 아키텍처](#-시스템-아키텍처)
3. [설치 및 환경 설정](#-설치-및-환경-설정)
4. [파이프라인 실행](#-파이프라인-실행)
5. [웹 애플리케이션 실행](#-웹-애플리케이션-실행)
6. [테스트 실행](#-테스트-실행)
7. [프로젝트 구조](#-프로젝트-구조)
8. [라이선스](#-라이선스)

---

## 👥 팀 구성 (Team Composition)

## 팀 정보: Team 뀨💕

| 역할 (Role) | 이름 (Name) | 깃허브 (GitHub) | 담당 업무 (Responsibilities) |
| :--- | :---: | :---: | :--- |
| **Team Leader** | **한승혁** | [![GitHub](https://img.shields.io/badge/GitHub-gksshing-181717?style=flat-square&logo=github)](https://github.com/gksshing) | • 프로젝트 기획 및 전체 아키텍처(Architecture) 설계 <br> • 주요 마일스톤 관리 및 기술 스택 최종 의사결정 <br> • 팀 프로젝트 개발 워크플로우 및 프로세스 최적화 |
| **Core Dev** | **박준석** | [![GitHub](https://img.shields.io/badge/GitHub-junseok--dev-181717?style=flat-square&logo=github)](https://github.com/junseok-dev) | • 아이디어의 실제 코드 구현 및 핵심 로직(RAG/LLM) 개발 <br> • 기술적 병목 현상 발생 시 문제 해결 및 빠른 프로토타이핑 실행 <br> • 시스템 성능 최적화 및 인프라 구축 주도 |
| **Communication**| **이규빈** | [![GitHub](https://img.shields.io/badge/GitHub-giant0212-181717?style=flat-square&logo=github)](https://github.com/giant0212) | • 복잡한 기술 구조의 스토리텔링 및 외부 발표 자료 구성 <br> • 프로젝트 핵심 가치 전달 및 사용자 경험(UX) 시나리오 설계 <br> • 팀 프로젝트 시연 및 성과 공유 담당 |
| **Quality Assurance**| **최정환** | [![GitHub](https://img.shields.io/badge/GitHub-hwany--ai-181717?style=flat-square&logo=github)](https://github.com/hwany-ai) | • 전체 기술 문서화(README.md) 및 세부 완성도 관리 <br> • 단위 테스트 수행 및 사용자 인터페이스(UI) 오류 검수 <br> • 프로젝트 일정 추적 및 누락된 세부 사항 보완 |

---

## 💬 팀 프로젝트 후기 (Team Retrospective)

| 이름 | 프로젝트 후기 및 소감 |
| :---: | :--- |
| **한승혁** | 특허 도메인의 기술적 난제를 리랭커와 자가 검증 기반의 RAG로 도전하며, AI 에이전트의 실무적 신뢰성을 입증하는 뜻깊은 도전이었습니다. |
| **박준석** | LLM에 대해서 배우고 처음으로 RAG를 활용한 LLM 챗봇 만드는 프로젝트였는데 생각보다 쉽지 않았고 부족한 점이 아직 매우 많다고 느낀 프로젝트였습니다. 그래도 팀장과 팀원들의 도움을 받아 해낼 수 있었던 프로젝트라고 생각합니다. |
| **이규빈** | 특허가 워낙 복잡해서, 아이디어만으로 비슷한 특허를 제대로 찾고 판단하는 게 생각보다 어렵다는 걸 많이 느꼈습니다. |
| **최정환** | 아직 부족한 부분이 많아 팀장리드에 잘 맞춰 팀장과 팀원들이 요청하는 부분에 대해 최선을 다했던 시간 이었습니다. |

---




## 🎯 프로젝트 개요

**쇼특허(Short-Cut)** 는 사용자의 아이디어를 입력받아 유사한 선행 특허를 검색하고, AI 기반으로 **침해 리스크 분석** 및 **회피 전략**을 제공하는 시스템입니다.

### 주요 기능

| 기능 | 설명 |
|------|------|
| **HyDE** | 사용자 아이디어로부터 '가상 청구항'을 생성하여 검색 재현율(Recall)을 획기적으로 향상 |
| **Multi-Query RAG** | 기술적/법적/문제해결 관점으로 쿼리를 자동 확장하여 검색 누락 최소화 |
| **Hybrid Search** | Pinecone Serverless (Dense + Sparse) 통합 인덱스 검색으로 정확도 극대화 |
| **RAG Evaluation** | **DeepEval** 프레임워크를 활용한 답변의 충실도(Faithfulness) 및 관련성(Relevancy) 자동 검증 |
| **Reranker** | Cross-Encoder 모델을 활용하여 검색 결과의 관련성 정밀 재정렬 |
| **Intelligent Parsing** | Regex → 구조 분석 → NLP → Fallback의 **4-Level 청구항 파싱**으로 비정형 데이터 정복 |
| **Claim-Level Analysis** | '모든 구성요소 법칙(All Elements Rule)'을 적용한 특허 침해 리스크 정밀 진단 |
| **Feedback Loop** | 분석 결과에 대한 사용자 피드백(👍/👎) 수집 및 검색 품질 개선 엔진 |
| **Guardian Map** | 아이디어를 성(Castle)으로, 위협 특허를 침입자로 시각화한 **직관적 방어 전략 지도** |
| **PDF Report** | 분석된 모든 내용을 깔끔한 PDF 보고서로 자동 생성 및 다운로드 |
| **지식재산권 용어 사전 제공** | 특허 분야가 생소한 사용자를 위해 필수 지식재산권 용어를 정리한 PDF 가이드를 사이드바에서 상시 제공 |

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 아이디어 입력                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Step 1: HyDE (가상 청구항 생성)                 │
│                         GPT-4o-mini                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Step 2: Hybrid Search                        │
│              Pinecone (Dense) + BM25 (Sparse) + RRF             │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Step 3: Self-Grading                         │
│                 관련성 점수 평가 + Query Rewrite                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Step 4: Critical Analysis                    │
│              유사도 / 침해 리스크 / 회피 전략 도출                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         결과 출력 (Streaming)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 설치 및 환경 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 값을 입력합니다:

```bash
cp .env.example .env
```

필수 환경 변수:

| 변수명 | 설명 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) |
| `PINECONE_API_KEY` | Pinecone API 키 (필수) |
| `GCP_PROJECT_ID` | Google Cloud Project ID (BigQuery 사용 시) |

---

## 🚀 파이프라인 실행

데이터 파이프라인은 6단계로 구성됩니다:

### 전체 파이프라인 실행

```bash
python -m src.pipeline --full --limit 1000
```

### 개별 스테이지 실행

| 스테이지 | 명령어 | 설명 |
|---------|--------|------|
| **1. 데이터 추출** | `python -m src.pipeline --stage 1` | BigQuery에서 특허 데이터 추출 |
| **2. 전처리** | `python -m src.pipeline --stage 2` | 청구항 파싱 및 청킹 |
| **3. Triplet 생성** | `python -m src.pipeline --stage 3` | PAI-NET 학습용 triplet 생성 |
| **4. 임베딩** | `python -m src.pipeline --stage 4` | OpenAI API로 벡터 임베딩 생성 |
| **5. 인덱싱** | `python -m src.pipeline --stage 5` | Pinecone에 벡터 업로드 |
| **6. Self-RAG 데이터** | `python -m src.pipeline --stage 6` | Ground Truth 데이터 생성 |

### 파이프라인 스테이지 상세

```
┌──────────────────────────────────────────────────────────────┐
│       Stage 1: BigQuery Extraction                           │
│       ├── patents-public-data.patents.publications          │
│       └── AI/NLP 도메인 필터링 (IPC: G06F, G06N)              │
└──────────────────────────────────┬───────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│       Stage 2: Preprocessing                                 │
│       ├── 청구항 파싱 (독립항/종속항)                           │
│       ├── 텍스트 청킹 (512 tokens)                            │
│       └── RAG 컴포넌트 태깅                                   │
└──────────────────────────────────┬───────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│       Stage 3: Triplet Generation                            │
│       ├── Anchor: 피인용 특허                                 │
│       ├── Positive: 인용 특허                                 │
│       └── Negative: 무관한 특허 (Hard Negative 포함)           │
└──────────────────────────────────┬───────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│       Stage 4: Embedding                                     │
│       └── OpenAI text-embedding-3-small (1536 dim)           │
└──────────────────────────────────┬───────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│       Stage 5: Vector Indexing                               │
│       └── Pinecone Serverless (us-east-1)                    │
└──────────────────────────────────┬───────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│       Stage 6: Self-RAG Training Data                        │
│       └── GPT-4o-mini로 Ground Truth 생성                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 🌐 웹 애플리케이션 실행

Streamlit 기반 웹 UI를 실행합니다:

```bash
streamlit run app.py
```

### 웹 UI 기능

- 💡 **아이디어 입력**: 텍스트 영역에 특허 아이디어 입력
- 🔍 **분석 시작**: 유사 특허 검색 및 침해 리스크 분석
- 📊 **결과 확인**: 유사도 점수, 리스크 수준, 회피 전략 확인
- 📜 **히스토리**: 이전 분석 결과 조회

---

## 🧪 테스트 실행

### RAG 품질 평가 테스트

DeepEval 기반의 Faithfulness 및 Relevancy 평가를 수행합니다:

```bash
# 기본 테스트 (순차 실행)
pytest tests/test_evaluation_golden.py -v

# 병렬 실행 + 리포트 생성
pytest tests/test_evaluation_golden.py -v -n 2 \
    --junitxml=tests/report_golden.xml \
    --html=tests/report_golden.html \
    --self-contained-html
```

### 평가 메트릭

| 메트릭 | 설명 | 임계값 |
|--------|------|--------|
| **Faithfulness** | 응답이 컨텍스트에 기반하는지 평가 | ≥ 0.6 |
| **Answer Relevancy** | 응답이 질문에 관련성이 있는지 평가 | ≥ 0.6 |

### 기타 테스트

```bash
# 하이브리드 검색 테스트
pytest tests/test_hybrid_search.py -v

# 파서 테스트
pytest tests/test_parser.py -v

# ID 기반 검색 테스트
pytest tests/test_id_retrieval.py -v
```

---

## 📁 프로젝트 구조

```
SKN22-3rd-2Team/
├── 📄 app.py                    # Streamlit 웹 애플리케이션
├── 📄 main.py                   # 임베딩 모델 로더 (Intel XPU/NVIDIA CUDA)
├── 📄 requirements.txt          # Python 의존성
├── 📄 pytest.ini               # Pytest 설정
│
├── 📁 src/                     # 소스 코드
│   ├── 📄 config.py            # 설정 관리
│   ├── 📄 pipeline.py          # 파이프라인 오케스트레이터
│   ├── 📄 patent_agent.py      # Self-RAG 특허 분석 에이전트
│   ├── 📄 vector_db.py         # Pinecone 클라이언트 + BM25
│   ├── 📄 bigquery_extractor.py # BigQuery 데이터 추출
│   ├── 📄 preprocessor.py      # 전처리 및 청킹
│   ├── 📄 triplet_generator.py # PAI-NET triplet 생성
│   ├── 📄 embedder.py          # OpenAI 임베딩 생성
│   ├── 📄 self_rag_generator.py # Self-RAG 학습 데이터 생성
│   ├── 📄 analysis_logic.py    # 분석 로직
│   ├── 📄 session_manager.py   # 세션 관리
│   │
│   ├── 📁 data/               # 데이터 디렉토리
│   │   ├── 📁 raw/            # 원본 데이터
│   │   ├── 📁 processed/      # 전처리된 데이터
│   │   ├── 📁 triplets/       # Triplet 데이터
│   │   ├── 📁 embeddings/     # 임베딩 벡터
│   │   └── 📁 index/          # 인덱스 파일
│   │
│   └── 📁 ui/                 # UI 컴포넌트
│       ├── 📄 components.py   # UI 컴포넌트
│       └── 📄 styles.py       # CSS 스타일
│
├── 📁 tests/                  # 테스트 코드
│   ├── 📄 test_evaluation_golden.py  # Golden Dataset 평가
│   ├── 📄 test_evaluation.py         # RAG 품질 테스트
│   ├── 📄 test_hybrid_search.py      # 하이브리드 검색 테스트
│   └── 📄 test_parser.py             # 파서 테스트
│
├── 📁 scripts/                # 유틸리티 스크립트
│   └── 📄 filter_outliers.py  # 아웃라이어 필터링
│
└── 📁 docs/                   # 문서
    ├── 📁 01_data_preprocessing/
    ├── 📁 02_system_architecture/
    └── 📁 03_test_report/
```

---

## 🔧 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Python 3.10+ |
| **LLM** | OpenAI GPT-4o-mini |
| **임베딩** | OpenAI text-embedding-3-small (1536 dim) |
| **벡터 DB** | Pinecone Serverless |
| **스파스 검색** | Pinecone BM25Encoder (pinecone-text) |
| **프론트엔드** | Streamlit |
| **테스트** | Pytest + DeepEval |
| **데이터 소스** | Google BigQuery (patents-public-data) |

---

## 👥 팀 정보

**Team 뀨💕**

---

## 📄 라이선스

MIT License

---

## 👥 Team 뀨💕
**쇼특허 (Short-Cut)** - "특허 분석의 지름길을 제시합니다." 
