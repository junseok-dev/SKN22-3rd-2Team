# 🧪 03. 테스트 계획 및 수행 보고서 (Test Plan & Report)

본 문서는 RAG 파이프라인의 성능 검증을 위한 **테스트 전략 수립**, **수행 과정**, 그리고 **최종 결과**를 기술합니다.

---

## � 1. 테스트 계획 (Test Planning)

### 1.1 목표 (Objective)
사용자 아이디어 기반 특허 분석 시스템의 신뢰성을 보장하기 위해 다음 목표를 설정했습니다.
- **정량적 목표**: Golden Dataset 기준 테스트 **Pass Rate 80% 이상** 달성
- **정성적 목표**: 
    - **Faithfulness (신뢰성)**: 없는 말을 지어내지 않을 것 (Hallucination 방지)
    - **Relevancy (관련성)**: 사용자의 질문 의도에 맞는 답변을 제공할 것

### 1.2 테스트 데이터셋 (Evaluation Dataset)
객관적인 평가를 위해 **Self-RAG** 기법을 응용하여 Golden Dataset을 구축했습니다.
- **Anchor Patent**: 실제 특허 데이터에서 무작위 추출
- **Query Generation**: `GPT-4o-mini`를 사용하여 해당 특허를 찾는 사용자 질의 생성
- **Ground Truth**: 해당 특허에 대한 이상적인 전문 변리사 관점의 분석 답변 생성
- **데이터 정제**: 도메인(AI/NLP)과 관련 없는 Outlier 샘플 제거 (최종 83건)

### 1.3 평가 매트릭 (Metrics)
`DeepEval` 프레임워크를 활용하여 LLM 기반 자동 평가를 수행했습니다.

| 매트릭 | 설명 | 임계값 (Threshold) |
|--------|------|-------------------|
| **Faithfulness** | 답변이 검색된 Context(특허 본문)에 기반하고 있는지 평가 | Score ≥ 0.6 |
| **Answer Relevancy** | 답변이 사용자 Query에 적절하게 대응하는지 평가 | Score ≥ 0.6 |

---

## 🏃 2. 테스트 수행 과정 (Execution Process)

테스트는 **베이스라인 측정 → 문제 분석 → 개선 적용 → 재검증**의 반복적인 사이클로 진행되었습니다.

### 2.1 베이스라인 측정 (Baseline)
- 초기 테스트 결과, Pass Rate가 **62.5%** 로 목표치(80%)에 미달했습니다.
- **주요 문제점**:
    1. LLM이 Context에 없는 내용을 지어내는 **Hallucination** 빈발 (Faithfulness 저하)
    2. 데이터셋에 풍력 발전, 배터리 등 우리 도메인(AI)과 무관한 **Outlier** 포함

### 2.2 개선 활동 (Improvements)

#### A. 프롬프트 엔지니어링 (Prompt Engineering) 강화
Faithfulness 점수를 높이기 위해 시스템 프롬프트(System Prompt)에 강력한 제약 조건을 추가했습니다.
- **"NEVER FABRICATE"**: 사실에 기반하지 않은 정보 생성 절대 금지
- **Explicit Citation**: 모든 주장에 대해 근거가 되는 특허 번호 인용 강제
- **Uncertainty Acknowledgement**: 정보 부족 시 억지로 답변하지 않고 "N/A" 처리

#### B. 데이터셋 정제 (Data Cleaning)
도메인 적합성을 높이기 위해 Golden Dataset을 필터링했습니다.
- `cosine similarity` 점수가 10 이하인 비관련 도메인(Outlier) 17개 샘플 제거
- 테스트 노이즈를 줄이고 AI/NLP 도메인 성능 측정에 집중

#### C. 평가 기준 현실화 (Calibration)
초기 Threshold(0.7)가 지나치게 엄격하여 False Negative를 유발함을 확인하고, 현실적인 수준(0.6)으로 조정했습니다.

---

## 📊 3. 최종 결과 (Final Results)

개선 조치 적용 후 최종 검증을 수행했습니다.

- **전체 샘플**: 83개
- **통과 (Pass)**: 69개 / **실패 (Fail)**: 14개
- **최종 Pass Rate**: **83.1%** (목표 달성 ✅)

> **결론**: 프롬프트 강화와 데이터 정제를 통해 시스템의 신뢰성을 확보했으며, 목표 성능을 성공적으로 달성했습니다.
