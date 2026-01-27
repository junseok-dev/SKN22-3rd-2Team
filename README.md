# 🛡️ Patent Guard v2.0

**AI 기반 특허 선행 기술 조사 시스템**

사용자의 아이디어를 입력하면 기존 특허와 비교하여 **유사도**, **침해 리스크**, **회피 전략**을 분석해주는 Self-RAG 기반 특허 분석 도구입니다.

---

## 🎯 주요 기능

| 기능 | 설명 |
|------|------|
| **HyDE (Hypothetical Document Embedding)** | 사용자 아이디어를 가상 특허 청구항으로 변환하여 검색 품질 향상 |
| **Grading & Rewrite Loop** | 검색 결과 관련성 평가 (0~1점), 점수 낮으면 자동 재검색 |
| **Critical CoT Analysis** | 유사도/침해/회피 분석 + 근거 특허 명시 |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n patent-guard python=3.11 -y
conda activate patent-guard

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일 편집:
```env
GCP_PROJECT_ID=your-gcp-project-id
OPENAI_API_KEY=your-openai-api-key
```

### 3. 실행

```bash
# 특허 분석 에이전트 실행
python src/patent_agent.py
```

입력 예시:
```
💡 Your idea: 딥러닝 기반 문서 자동 요약 시스템
```

---

## 📁 프로젝트 구조

```
patent-guard/
├── src/
│   ├── patent_agent.py      # 🎯 메인 분석 에이전트 (HyDE + Grading + CoT)
│   ├── bigquery_extractor.py  # BigQuery 데이터 추출
│   ├── preprocessor.py      # 특허 데이터 전처리
│   ├── pipeline.py          # 파이프라인 오케스트레이터
│   ├── config.py            # 설정 관리
│   ├── embedder.py          # 임베딩 생성 (OpenAI)
│   ├── vector_db.py         # Milvus 벡터 DB 연동
│   └── data/
│       ├── raw/             # 원본 특허 데이터
│       ├── processed/       # 전처리된 데이터
│       └── outputs/         # 분석 결과 저장
├── .env.example             # 환경 변수 템플릿
├── requirements.txt         # Python 의존성
└── README.md
```

---

## 🔧 설정 옵션

`.env` 파일에서 설정 가능:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | - | OpenAI API 키 (필수) |
| `GCP_PROJECT_ID` | - | Google Cloud 프로젝트 ID |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 임베딩 모델 |
| `GRADING_MODEL` | `gpt-4o-mini` | 관련성 평가 모델 |
| `ANALYSIS_MODEL` | `gpt-4o` | 최종 분석 모델 |
| `GRADING_THRESHOLD` | `0.6` | 재검색 기준 점수 |
| `TOP_K_RESULTS` | `5` | 검색 결과 개수 |

---

## 📊 파이프라인 단계

```
[사용자 아이디어]
        ↓
[Stage 1] BigQuery에서 특허 데이터 추출
        ↓
[Stage 2] 전처리 (청구항 파싱, 청킹)
        ↓
[Stage 3] HyDE - 가상 청구항 생성
        ↓
[Stage 4] 벡터 검색 (유사 특허 찾기)
        ↓
[Stage 5] Grading - 관련성 평가 (필요시 재검색)
        ↓
[Stage 6] Critical Analysis - 상세 분석
        ↓
[분석 결과]
├── 유사도 평가 (0-100점)
├── 침해 리스크 (high/medium/low)
└── 회피 전략
```

---

## 📝 분석 결과 예시

```json
{
  "similarity": {
    "score": 45,
    "summary": "비선형 회귀 기술과 유사한 요소가 일부 존재",
    "evidence": ["CN-119864160-A", "CN-119358658-B"]
  },
  "infringement": {
    "risk_level": "medium",
    "risk_factors": ["데이터 예측 방법의 일반성"],
    "summary": "직접적 침해 위험은 중간 수준"
  },
  "avoidance": {
    "strategies": [
      "비선형 회귀의 구체적 알고리즘 차별화",
      "특정 산업 분야에 특화된 적용"
    ]
  }
}
```

---

## 💰 비용 정보

| 작업 | 예상 비용 |
|------|----------|
| BigQuery 쿼리 (10K 특허) | ~$2 (1회) |
| OpenAI 분석 (1건) | ~$0.01-0.05 |

---

## 📄 라이선스

MIT License

---

## 👥 팀

Patent Guard Team - SKN22-3rd-2Team
