# Patent Guard v2.0 - Project Rules

## 1. 프로젝트 비전 & 도메인
- **목적**: RAG 및 sLLM 기술 도메인 특화 선행 기술 조사 시스템.
- **타겟 도메인**: AI, NLP, Information Retrieval, Neural Networks 관련 특허.

## 2. 핵심 원칙 ⭐ (2026-01-27 업데이트)

### OpenAI API Only
- 모든 LLM/임베딩 작업은 **OpenAI API** 사용
- 로컬 모델(Octen-Embedding-8B 등) **사용 안 함**
- Intel iGPU/IPEX-LLM 관련 코드 **불필요**

### No Fine-tuning
- 모델 학습/튜닝 **없음**
- Self-RAG 학습 데이터 생성 **불필요**
- **프롬프트 엔지니어링**으로 해결

### Focus on Controllable Parts
- 프롬프트 템플릿 최적화
- 시스템 메시지 설계
- RAG 검색 로직
- 출력 포맷팅

## 3. 기술 스택

| 영역 | 사용 | 미사용 |
|------|------|--------|
| 분석/생성 | OpenAI GPT API | 로컬 LLM, Gemini |
| 임베딩 | OpenAI Embeddings | Octen-Embedding-8B |
| 검색 | 프롬프트 기반 | PAI-NET 트리플렛 |
| 데이터 | BigQuery 특허 데이터 | 학습용 데이터셋 |

## 4. 파이프라인

```
Stage 1: BigQuery 추출 ✅
Stage 2: 전처리 ✅
Stage 3: 트리플렛 ❌ (불필요)
Stage 4: 임베딩 → OpenAI Embeddings
Stage 5: 벡터 DB (선택)
Stage 6: OpenAI 분석 ✅
```

## 5. 데이터
- **Data Source**: Google Patents Public Dataset (BigQuery)
- **프로젝트 ID**: patent-485605
- **검색 기간**: 2018년 이후

## 6. 출력 형식
모든 특허 분석 결과는 다음 섹션 포함:
- `[유사도 평가]` - 기술적 유사성 점수 (0-100)
- `[침해 리스크]` - high/medium/low
- `[회피 전략]` - 설계 변경 방안

## 7. Architecture Decision: Demo vs Production

### Demo (Current) ✅
- **10K pre-loaded patents** from BigQuery
- **Cost**: ~$2 one-time
- **Use case**: Prototype, presentation, testing
- **Limitation**: Only searches within 10K patents (not comprehensive)

### Production (Future)
- **On-demand BigQuery search** for each customer query
- **Cost**: ~$0.50-2 per query
- **Use case**: Real business, accurate prior art search
- **Benefit**: Searches entire patent database (millions)

### Migration Path
```
Demo (10K) → Validate UX/Prompts → Production (On-demand)
```

**Decision**: Focus on demo first. Upgrade to on-demand when productionizing.
