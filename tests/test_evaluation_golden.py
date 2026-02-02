"""
Short-Cut v3.0 - Golden Dataset Evaluation
==========================================
Evaluation using the automatically generated Ground Truth dataset.
Loads the latest `src/data/processed/selfrag_training_*.json`.

Metrics:
1. FaithfulnessMetric
2. AnswerRelevancyMetric
"""

import pytest
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR

# Load Env
from dotenv import load_dotenv
load_dotenv()

# Check for required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set.", allow_module_level=True)

try:
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
except ImportError:
    pytest.skip("deepeval not installed.", allow_module_level=True)

from src.patent_agent import PatentAgent
from tests.test_evaluation import extract_retrieval_context, extract_actual_output, run_agent_analysis

def get_latest_golden_dataset() -> List[Dict[str, Any]]:
    """Load the latest Self-RAG training data."""
    files = list(PROCESSED_DATA_DIR.glob("selfrag_training_*.json"))
    if not files:
        return []
    
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    print(f"\n📂 Loading Golden Dataset from: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load dataset at module level (or emtpy list if none)
GOLDEN_DATASET = get_latest_golden_dataset()

@pytest.fixture(scope="module")
def patent_agent():
    return PatentAgent()

@pytest.fixture(scope="module")
def metrics():
    return {
        "faithfulness": FaithfulnessMetric(threshold=0.6, model="gpt-4o-mini", include_reason=True),
        "relevancy": AnswerRelevancyMetric(threshold=0.6, model="gpt-4o-mini", include_reason=True)
    }

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("sample", GOLDEN_DATASET)
async def test_golden_dataset_quality(
    sample: Dict[str, Any],
    patent_agent: PatentAgent,
    metrics: Dict[str, Any],
    record_property,
):
    """
    Test RAG quality using Golden Dataset samples.
    """
    query = sample["query"]
    ground_truth = sample["ground_truth_critique"]
    
    print(f"\n{'='*60}")
    print(f"🧪 Sample ID: {sample['sample_id']}")
    print(f"{'='*60}")
    
    # 1. Run Analysis
    try:
        result = await run_agent_analysis(patent_agent, query)
    except Exception as e:
        pytest.fail(f"Agent failed: {e}")

    if "error" in result:
        pytest.skip(f"No patents found: {result['error']}")

    # 2. Extract Data
    actual_output = extract_actual_output(result.get("analysis", {}))
    retrieval_context = extract_retrieval_context(result.get("search_results", []))
    
    print(f"\n📥 Input: {query[:100]}...")
    print(f"📤 Output: {actual_output[:100]}...")
    print(f"📚 Context: {len(retrieval_context)} chunks")

    # 3. Create Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=ground_truth  # Included for reference/future metrics
    )
    
    # 4. Measure Faithfulness
    fm = metrics["faithfulness"]
    fm.measure(test_case)
    print(f"   📊 Faithfulness: {fm.score:.2f} ({fm.reason[:100]}...)")
    record_property("faithfulness_score", fm.score)
    
    # 5. Measure Relevancy
    rm = metrics["relevancy"]
    rm.measure(test_case)
    print(f"   📊 Relevancy: {rm.score:.2f} ({rm.reason[:100]}...)")
    record_property("relevancy_score", rm.score)

    # Assertions
    assert fm.score >= 0.6, f"Faithfulness low: {fm.score}"
    assert rm.score >= 0.6, f"Relevancy low: {rm.score}"
