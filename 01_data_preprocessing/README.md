# 🛠 01. 데이터 전처리 (Data Preprocessing)

본 디렉토리는 특허 데이터의 **추출(Extraction)**, **정제(Cleaning)**, **청킹(Chunking)** 과정을 설명합니다.

---

## 📋 개요

쇼특허(Short-Cut) 시스템은 구글 BigQuery의 퍼블릭 데이터셋을 원천으로 하여, AI/NLP 도메인에 특화된 특허 데이터를 구축합니다. RAG(Retrieval-Augmented Generation) 성능 극대화를 위해 **계층적 청킹(Hierarchical Chunking)** 과 **정밀한 청구항 파싱(Claim Parsing)** 기술이 적용되었습니다.

---

## 🚀 파이프라인 스테이지

### 1단계: BigQuery 데이터 추출 (Stage 1)

Google Patents Public Dataset(`patents-public-data.patents.publications`)에서 조건에 맞는 특허를 추출합니다.

- **관련 코드**: `src/bigquery_extractor.py`
- **주요 필터링 조건**:
    - **도메인**: AI, NLP, Search, Embedding, Neural Network 등 도메인 키워드 기반 필터링
    - **IPC/CPC 분류**: `G06F`(정보 검색), `G06N`(인공지능) 등 관련 분류 코드
    - **국가**: 주요 6개국 (US, EP, WO, CN, JP, KR)
    - **기간**: 최근 기술 동향 반영 (기본값: 2018년 ~ 2024년)
- **비용 최적화**:
    - `dry_run` 모드를 통해 쿼리 실행 전 예상 비용($) 산출
    - 불필요한 필드(전문 Description 등) 제외로 스캔 용량 최소화

**실행 명령어:**
```bash
# 데이터 추출 (기본 100건 제한)
python -m src.pipeline --stage 1 --limit 100
```

---

### 2단계: 전처리 및 청킹 (Stage 2)

추출된 원본 데이터(JSON)를 RAG 검색에 최적화된 형태로 가공합니다.

- **관련 코드**: `src/preprocessor.py`

#### 1. 정밀 청구항 파싱 (Claim Parsing)
단순 텍스트 분할이 아닌, 특허 청구항의 구조를 분석하여 개별 청구항 단위로 분리합니다. 4단계 Fallback 전략을 사용하여 파싱 정확도를 높였습니다.

| 단계 | 방식 | 설명 |
|------|------|------|
| **Level 1** | **Regex Pattern** | 표준 포맷("1. ", "[1]") 정규식 매칭 (가장 빠르고 정확) |
| **Level 2** | **Structure** | 들여쓰기 및 번호 체계 분석 |
| **Level 3** | **NLP Fallback** | Spacy 문장 분리 모델 활용 |
| **Level 4** | **Minimal Split** | 단락 단위 분할 (최후의 수단) |

#### 2. 계층적 청킹 (Hierarchical Chunking)
검색 정확도와 답변 생성 품질을 동시에 잡기 위해 **Parent-Child** 전략을 사용합니다.

- **Parent Chunk (부모)**: 특허 전체 요약, 주요 독립항 모음 (LLM 답변 생성용 문맥)
- **Child Chunk (자식)**: 개별 청구항, 초록, 상세 설명의 섹션 (정밀 검색용)

#### 3. RAG 컴포넌트 태깅
각 청크에 RAG 관련 키워드(예: `retriever`, `embedding`, `vector store`)가 포함되어 있는지 태깅하여, 향후 메타데이터 필터링에 활용합니다.

**실행 명령어:**
```bash
# 전처리 수행 (Stage 1 결과물 필요)
python -m src.pipeline --stage 2
```

---

## 📂 데이터 구조

전처리가 완료된 데이터는 다음과 같은 구조를 가집니다:

```json
{
  "publication_number": "US-12345678-B2",
  "title": "Method for ...",
  "chunks": [
    {
      "chunk_id": "US-12345678-B2_parent",
      "chunk_type": "parent",
      "content": "Title: ...\nAbstract: ...\nIndependent Claims: ...",
      "child_chunk_ids": ["US-12345678-B2_claim_1", ...]
    },
    {
      "chunk_id": "US-12345678-B2_claim_1",
      "chunk_type": "claim",
      "content": "1. A method comprising...",
      "metadata": {
        "claim_number": 1,
        "claim_type": "independent"
      }
    }
  ]
}
```
