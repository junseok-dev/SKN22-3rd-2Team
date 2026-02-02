"""
UI Components for the application - Fixed ImportError
"""
import streamlit as st
from datetime import datetime
# 아래 utils 임포트 경로는 사용자 환경에 맞춰 유지합니다.
from src.utils import get_risk_color, get_score_color, get_patent_link, display_patent_with_link, format_analysis_markdown
from src.ui.styles import apply_theme_css

def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="main-header">
        <h1>⚡ 쇼특허 (Short-Cut)</h1>
        <p style="font-size: 1.2rem; color: #888;">특허 검색부터 분석까지, 가장 빠른 지름길</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(openai_api_key, db_client):
    """Render the sidebar top part (Title to History)."""
    with st.sidebar:
        # 1. 앱 제목
        st.markdown("# ⚡ 쇼특허")
        st.markdown("### Short-Cut")
        st.divider()
        
        apply_theme_css()
        
        # 2. 검색 옵션 (🔧) - 최상단 배치
        st.markdown("### 🔧 검색 옵션")
        IPC_CATEGORIES = {
            "G06 (컴퓨터/AI)": "G06",
            "H04 (통신/네트워크)": "H04",
            "A61 (의료/헬스케어)": "A61",
            "H01 (반도체/전자)": "H01",
            "B60 (차량/운송)": "B60",
            "C12 (바이오/생명)": "C12",
            "F02 (기계/엔진)": "F02",
        }
        selected_categories = st.multiselect(
            "관심 기술 분야 (선택 시 필터링)",
            options=list(IPC_CATEGORIES.keys()),
            default=[],
            key="ipc_multiselect_unique",
            help="특정 기술 분야(IPC)로 검색 범위를 제한하여 정확도를 높입니다."
        )
        selected_ipc_codes = [IPC_CATEGORIES[cat] for cat in selected_categories]
        st.divider()

        # 3. 특허 가이드 (📖)
        st.markdown("### 📖 특허 가이드")
        st.caption("처음 사용하시나요? 가이드 영상을 확인하세요.")
        
        @st.dialog("📖 특허 출원 가이드", width="large")
        def show_patent_guide_popup():
            st.write("**특허 출원 전 알아야 할 핵심 정보:**")
            video_url = "https://www.youtube.com/watch?v=HSWXcMSneB4"
            st.video(video_url)
            st.write("---")
            st.caption("닫기 버튼이나 배경을 클릭하면 팝업이 닫힙니다.")
        
        if st.button("🎥 가이드 영상 보기", key="sidebar_guide_btn_unique", use_container_width=True):
            show_patent_guide_popup()
        st.divider()
        
        # 4. 분석 히스토리
        st.markdown("### 📜 분석 히스토리")
        if st.session_state.get("analysis_history"):
            for i, hist in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"#{len(st.session_state.analysis_history)-i}: {hist['user_idea'][:20]}..."):
                    risk = hist.get('analysis', {}).get('infringement', {}).get('risk_level', 'unknown')
                    score = hist.get('analysis', {}).get('similarity', {}).get('score', 0)
                    st.write(f"🎯 유사도: {score}/100")
                    st.write(f"⚠️ 리스크: {risk.upper()}")
        else:
            st.caption("아직 분석 기록이 없습니다.")
            
        if st.button("🗑️ 기록 삭제", key="clear_history_btn_unique", use_container_width=True):
            st.session_state.analysis_history = []
            from src.session_manager import clear_user_history
            clear_user_history()
        
        return True, selected_ipc_codes

def render_search_results(result):
    """Render search result metrics and details."""
    analysis = result.get("analysis", {})
    st.divider()
    st.markdown("## 📊 분석 결과")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        score = analysis.get("similarity", {}).get("score", 0)
        st.metric(label="🎯 유사도 점수", value=f"{score}/100")
    with col2:
        risk_level = analysis.get("infringement", {}).get("risk_level", "unknown")
        color, emoji, _ = get_risk_color(risk_level)
        st.metric(label="⚠️ 침해 리스크", value=f"{emoji} {risk_level.upper()}")
    with col3:
        patent_count = len(result.get("search_results", []))
        st.metric(label="📚 참조 특허", value=f"{patent_count}건")
    
    # 탭 구성 (기존 로직 유지)
    tab1, tab2, tab3 = st.tabs(["📝 종합 리포트", "🎯 유사도 분석", "⚠️ 침해 리스크"])
    with tab1:
        st.info(analysis.get("conclusion", "분석 결과가 없습니다."))
    with tab2:
        st.write(analysis.get("similarity", {}).get("summary", "N/A"))
    with tab3:
        st.write(analysis.get("infringement", {}).get("summary", "N/A"))

def render_footer():
    """Render the application footer."""
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem; padding-bottom: 2rem;">
        <p>⚠️ <b>면책 조항 (Disclaimer)</b></p>
        <p>본 시스템이 제공하는 모든 분석 결과는 RAG(Retrieval-Augmented Generation) 기술 및 고도화된 AI 알고리즘에 의해 도출된 선행 기술 조사 참고 데이터입니다. 본 정보는 데이터 기반의 통계적 예측치일 뿐, 어떠한 경우에도 국가 기관의 공식적인 판정이나 법적 효력을 가진 증빙 자료로 활용될 수 없음을 명시합니다.

실제 특허권의 유효성, 침해 여부 및 등록 가능성에 대한 최종적인 판단은 고도의 전문성을 요하는 영역이므로, 반드시 공인된 전문 변리사의 정밀한 법률 검토 및 자문을 거치시기를 강력히 권고드립니다.

쇼특허(Short-Cut) 팀은 제공되는 정보의 정밀도 향상을 위해 최선을 다하고 있으나, 데이터의 완전성이나 최신성, 혹은 이용자의 특정 목적 부합 여부에 대해 어떠한 명시적·묵시적 보증도 하지 않습니다. 따라서 본 서비스의 분석 내용을 신뢰하여 행해진 이용자의 개별적 판단이나 투자, 법적 대응 등 제반 활동으로 인해 발생하는 직·간접적인 손실에 대하여 당사는 **일체의 법적 책임(Liability)**을 부담하지 않음을 알려드립니다.</p>
        <p>© 2026 Short-Cut Team. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)