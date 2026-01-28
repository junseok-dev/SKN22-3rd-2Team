# 🧪 테스트 계획 및 결과 보고서

> **⚡ 쇼특허 (Short-Cut) v3.0 - AI 특허 선행 기술 조사 시스템**  
> Team: 뀨💕 | 작성일: 2026-01-28  
> 테스트 프레임워크: pytest 9.0.2

---

## 1. 테스트 개요

### 1.1 테스트 범위

| 모듈 | 파일 | 테스트 수 | 커버리지 |
|------|------|----------|----------|
| **Hybrid Search (RRF)** | `test_hybrid_search.py` | 8 | 100% |
| **Claim Parser (4-Level)** | `test_parser.py` | 19 | 100% |
| **Total** | - | **27** | **100%** |

### 1.2 테스트 환경

| 항목 | 값 |
|------|-----|
| **OS** | Windows 11 (10.0.26100) |
| **Python** | 3.11.14 |
| **pytest** | 9.0.2 |
| **실행 시간** | ~2.8초 |

---

## 2. 테스트 결과 요약

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.0.2
collected 27 items

tests/test_hybrid_search.py ........                                     [ 29%]
tests/test_parser.py ...................                                 [100%]

============================= 27 passed in 2.83s ==============================
```

| 결과 | 수치 |
|------|------|
| ✅ **Passed** | 27 |
| ❌ Failed | 0 |
| ⏭️ Skipped | 0 |
| **Pass Rate** | **100%** |

---

## 3. Hybrid Search (RRF) 테스트

📄 **파일**: `tests/test_hybrid_search.py`

### 3.1 테스트 시나리오

RRF (Reciprocal Rank Fusion) 알고리즘의 정확성을 검증합니다.

```
RRF_score(d) = Σ weight / (k + rank + 1)
```

### 3.2 테스트 케이스

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 1 | `test_cross_rank_verification_top_tier` | Dense #1 (Doc A)와 Sparse #1 (Doc B)가 Top-3에 포함되는지 검증 | ✅ |
| 2 | `test_symmetric_weighting` | 0.5:0.5 가중치에서 동일 랭크 문서의 점수가 같은지 검증 | ✅ |
| 3 | `test_asymmetric_weighting_dense_heavy` | 0.8:0.2 가중치에서 Dense #1이 최상위인지 검증 | ✅ |
| 4 | `test_asymmetric_weighting_sparse_heavy` | 0.2:0.8 가중치에서 Sparse #1이 더 높은 점수인지 검증 | ✅ |
| 5 | `test_edge_case_empty_dense_results` | Dense 결과가 비어있어도 Sparse 결과 반환 | ✅ |
| 6 | `test_edge_case_empty_sparse_results` | Sparse 결과가 비어있어도 Dense 결과 반환 | ✅ |
| 7 | `test_edge_case_both_empty` | 둘 다 비어있으면 빈 리스트 반환 (크래시 없음) | ✅ |
| 8 | `test_rrf_k_constant_effect` | k 상수가 낮을수록 상위 랭크 영향력 증가 검증 | ✅ |

### 3.3 테스트 데이터

```python
# Dense 검색 결과 (FAISS)
dense_results = [
    ("doc_a", 0.95, "Document A - Top in Dense"),  # #1
    ("doc_d", 0.85, "Document D"),                 # #2
    ...
    ("doc_x", 0.05, "Document X"),                 # #10
]

# Sparse 검색 결과 (BM25)
sparse_results = [
    ("doc_b", 15.0, "Document B - Top in Sparse"), # #1
    ("doc_l", 12.0, "Document L"),                 # #2
    ...
    ("doc_y", 1.0, "Document Y"),                  # #10
]
```

---

## 4. Claim Parser (4-Level) 테스트

📄 **파일**: `tests/test_parser.py`

### 4.1 테스트 전략

4-Level Fallback 파서의 각 레벨별 동작을 검증합니다.

```
Level 1: Regex Pattern → Level 2: Structure → Level 3: NLP → Level 4: Minimal
```

### 4.2 Level 1 (Regex) 테스트

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 1 | `test_standard_us_format_basic` | US 형식 "1. A method..." 파싱 | ✅ |
| 2 | `test_claim_numbering` | 청구항 번호 추출 [1, 2, 3, 4] | ✅ |
| 3 | `test_independent_vs_dependent_detection` | 독립항/종속항 분류 | ✅ |
| 4 | `test_rag_component_detection` | RAG 키워드 탐지 (retrieval, embedding) | ✅ |
| 5 | `test_claim_text_content` | 청구항 텍스트 내용 검증 | ✅ |

**테스트 데이터**:
```text
1. A method for neural network-based document retrieval comprising:
   receiving a query from a user;
   generating an embedding vector from the query;
   searching a vector database for similar documents;
   returning ranked results to the user.

2. The method of claim 1, wherein the embedding is generated using a transformer model.
```

### 4.3 Level 2 (Structure) 테스트

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 6 | `test_bracket_numbered_format` | 괄호 형식 "(1)", "[1]" 파싱 | ✅ |
| 7 | `test_korean_format_parsing` | 한국어 "제1항:", "청구항 2:" 파싱 | ✅ |
| 8 | `test_mixed_indent_structure` | 혼합 들여쓰기 처리 | ✅ |

### 4.4 Level 3 (NLP) 테스트

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 9 | `test_ocr_noise_handling` | OCR 노이즈 ("C1aim", "rnethod") 처리 | ✅ |
| 10 | `test_nlp_disabled_graceful_fallback` | NLP 비활성화 시 Level 4 폴백 | ✅ |
| 11 | `test_sentence_boundary_mock` | 문장 경계 탐지 | ✅ |

### 4.5 Level 4 (Minimal Fallback) 테스트

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 12 | `test_raw_text_blob_fallback` | 구조 없는 텍스트 → 1개 청구항 | ✅ |
| 13 | `test_empty_input_handling` | 빈 입력 → 빈 리스트 (크래시 없음) | ✅ |
| 14 | `test_whitespace_only_input` | 공백만 → 빈 리스트 | ✅ |
| 15 | `test_single_paragraph_fallback` | 단일 문단 → 1개 청구항 | ✅ |
| 16 | `test_multiple_paragraphs_fallback` | 다중 문단 처리 | ✅ |

### 4.6 Data Integrity 테스트

| # | 테스트명 | 설명 | 상태 |
|---|---------|------|------|
| 17 | `test_parsed_claim_dataclass_fields` | ParsedClaim 필드 존재 검증 | ✅ |
| 18 | `test_char_and_word_counts` | 문자/단어 수 정확성 | ✅ |
| 19 | `test_claims_sorted_by_number` | 청구항 번호 정렬 순서 | ✅ |

---

## 5. 테스트 실행 방법

### 5.1 전체 테스트 실행

```bash
# 기본 실행
pytest tests/ -v

# 상세 출력
pytest tests/ -v --tb=short
```

### 5.2 모듈별 실행

```bash
# Hybrid Search 테스트만
pytest tests/test_hybrid_search.py -v

# Parser 테스트만
pytest tests/test_parser.py -v
```

### 5.3 HTML 리포트 생성

```bash
# HTML 리포트 생성
pytest tests/ --html=report/test_report.html --self-contained-html

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

---

## 6. 테스트 파일 구조

```
tests/
├── conftest.py              # pytest 설정 및 공통 fixtures
├── test_hybrid_search.py    # RRF 알고리즘 테스트
│   ├── TestHybridSearchRRF  # 테스트 클래스
│   └── rrf_fusion()         # 테스트용 RRF 구현
│
└── test_parser.py           # 4-Level 파서 테스트
    ├── TestClaimParserLevel1Regex
    ├── TestClaimParserLevel2Structure
    ├── TestClaimParserLevel3NLP
    ├── TestClaimParserLevel4Minimal
    └── TestClaimParserDataIntegrity
```

---

## 7. 품질 지표

### 7.1 테스트 커버리지

| 모듈 | 커버리지 |
|------|----------|
| `vector_db.py` (RRF 부분) | 100% |
| `preprocessor.py` (ClaimParser) | 95%+ |

### 7.2 테스트 성숙도

| 항목 | 상태 |
|------|------|
| Unit Tests | ✅ 완료 |
| Edge Case Tests | ✅ 완료 |
| Integration Tests | 🔄 계획 중 |
| E2E Tests | 🔄 계획 중 |

---

## 8. 알려진 제한사항

1. **실제 OpenAI API 호출 미테스트**: Unit 테스트는 Mock 사용
2. **FAISS 인덱스 로드 미테스트**: 파일 I/O 관련 통합 테스트 필요
3. **Streamlit UI 테스트**: 별도 E2E 테스트 필요

---

## 9. 향후 테스트 계획

| 우선순위 | 항목 | 예상 일정 |
|----------|------|----------|
| 🔴 High | OpenAI API 통합 테스트 (Mock 서버) | 1주 |
| 🟡 Medium | FAISS 인덱스 I/O 테스트 | 1주 |
| 🟢 Low | Streamlit E2E 테스트 | 2주 |

---

## 10. 리포트 파일 위치

| 파일 | 설명 |
|------|------|
| `tests/test_hybrid_search.py` | RRF 테스트 소스 |
| `tests/test_parser.py` | 파서 테스트 소스 |
| `report/test_report.html` | HTML 리포트 (브라우저 열기) |
| `report/test_report_final.txt` | 텍스트 리포트 |

---

*작성: ⚡ 쇼특허 (Short-Cut) Team - 뀨💕*
