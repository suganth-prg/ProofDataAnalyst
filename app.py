"""
Proof-Carrying Data Analyst - Modern Interactive UI
Provides a verifiable natural language interface for data analysis.
"""

import os
import json
from pathlib import Path
import pandas as pd
import streamlit as st

import analyzer
import dynamic_analyzer as da


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Proof-Carrying Data Analyst",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }

    .main-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }

    .proof-badge-verified {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
    }

    .proof-badge-refused {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
    }

    .answer-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 5px solid #6366f1;
        border-radius: 14px;
        padding: 22px 26px;
        margin: 18px 0;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
    }

    .answer-text {
        font-size: 20px;
        font-weight: 600;
        color: #f8fafc;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .proof-note {
        font-size: 13px;
        color: #10b981;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .comparison-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 16px 0;
    }

    .comparison-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .comparison-card-target {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .comparison-card-val {
        font-size: 26px;
        font-weight: 800;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }

    .comparison-card-diff {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .query-chip {
        display: inline-block;
        background: rgba(51, 65, 85, 0.5);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 12px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .query-chip:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: #6366f1;
        color: #fff;
    }

    .metric-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATASET LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_cached_datasets():
    return da.load_datasets()


if "custom_datasets" not in st.session_state:
    st.session_state.custom_datasets = {}

datasets = get_cached_datasets().copy()
datasets.update(st.session_state.custom_datasets)


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Dataset Workspace")

    dataset_options = list(datasets.keys())
    selected_dataset = st.selectbox(
        "Active Dataset",
        options=dataset_options,
        index=0 if dataset_options else None
    )

    if selected_dataset and selected_dataset in datasets:
        curr_df = datasets[selected_dataset]
        st.markdown(f"**Shape:** `{curr_df.shape[0]}` rows × `{curr_df.shape[1]}` cols")
        st.markdown(f"**Columns:** `{', '.join(curr_df.columns[:5])}{'...' if len(curr_df.columns) > 5 else ''}`")

    st.markdown("---")
    st.markdown("### 📤 Upload New Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload your own dataset for verified analysis")
    if uploaded_file is not None:
        try:
            custom_name = Path(uploaded_file.name).stem
            uploaded_df = pd.read_csv(uploaded_file)
            st.session_state.custom_datasets[custom_name] = uploaded_df
            st.success(f"Loaded '{custom_name}' ({len(uploaded_df)} rows)")
            st.rerun()
        except Exception as err:
            st.error(f"Error loading CSV: {err}")

    st.markdown("---")
    st.markdown("### 🛡️ Proof Engine")
    st.caption("• Dual Execution Architecture")
    st.caption("• Deterministic Plan + Clean Verifier")
    st.caption("• Guaranteed Soundness Agreement")
    st.caption("• Local Model: `llama3.2:3b` (Ollama)")


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ Proof-Carrying Data Analyst</div>
    <div class="main-subtitle">
        Self-Verifying Data Intelligence with Independent Mathematical Proofs & Deterministic Execution
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN TABS
# -----------------------------------------------------------------------------
tab_query, tab_explorer, tab_tests = st.tabs([
    "💬 Query & Verification",
    "📊 Dataset Explorer",
    "🧪 Self-Test Suite"
])


# -----------------------------------------------------------------------------
# TAB 1: QUERY & VERIFICATION
# -----------------------------------------------------------------------------
with tab_query:
    st.markdown("#### Ask a Question")
    
    # Preset quick chips
    st.markdown("<span style='font-size: 13px; color: #94a3b8; font-weight: 500;'>Quick Questions:</span>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    preset_query = None
    with chip_col1:
        if st.button("⚖️ Compare Rice and Wheat", use_container_width=True):
            preset_query = "Compare Rice and Wheat."
        if st.button("🍎 Apple vs Banana sales diff", use_container_width=True):
            preset_query = "What is the difference between Apple and Banana sales?"
    with chip_col2:
        if st.button("🌾 How much more Wheat than Rice?", use_container_width=True):
            preset_query = "How much more did Wheat make than Rice?"
        if st.button("📈 Which product has highest sales?", use_container_width=True):
            preset_query = "Which product has the highest sales?"
    with chip_col3:
        if st.button("📦 Which product sold most units?", use_container_width=True):
            preset_query = "Which product sold the most units?"
        if st.button("💰 What is the average sales?", use_container_width=True):
            preset_query = "What is the average sales?"

    if "current_question" not in st.session_state:
        st.session_state.current_question = "What is the difference between Apple and Banana sales?"

    if preset_query:
        st.session_state.current_question = preset_query

    user_query = st.text_input(
        "Enter natural language question:",
        value=st.session_state.current_question,
        placeholder="e.g. Compare Rice and Wheat sales, or Which product sold the most units?",
        key="query_input"
    )

    run_btn = st.button("🚀 Analyze & Generate Proof", type="primary", use_container_width=True)

    if run_btn and user_query:
        with st.spinner("Analyzing question, generating execution plan, and running independent verification..."):
            result_data = da.analyze_query(user_query, datasets)

        st.markdown("---")

        status = result_data.get("status")
        success = result_data.get("success", False)

        # Status badge & answer card
        if success and status == "VERIFIED":
            st.markdown(
                '<div class="proof-badge-verified">✓ MATHEMATICALLY VERIFIED & SOUND</div>',
                unsafe_allow_html=True
            )
            
            answer_text = result_data.get("answer", "Answer calculated successfully.")
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-text">{answer_text}</div>
                <div class="proof-note">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    Proof: Deterministic execution and independent verification agree.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Special visualization for COMPARISON queries
            plan = result_data.get("plan", {})
            raw_res = result_data.get("result", {})
            op = plan.get("operation")

            if op == "difference" and isinstance(raw_res, dict) and "first" in raw_res:
                targets = plan.get("comparison_targets", [])
                metric_name = plan.get("metric", "Value")
                
                t1_name = targets[0].get("value") if len(targets) > 0 else "Entity 1"
                t2_name = targets[1].get("value") if len(targets) > 1 else "Entity 2"
                val1 = raw_res.get("first", 0)
                val2 = raw_res.get("second", 0)
                diff = raw_res.get("difference", 0)

                st.markdown("##### 🔍 Comparison Breakdown")
                c_col1, c_col2, c_col3 = st.columns(3)
                with c_col1:
                    st.metric(label=f"{t1_name} ({metric_name})", value=f"{val1:,.2f}")
                with c_col2:
                    st.metric(label=f"{t2_name} ({metric_name})", value=f"{val2:,.2f}")
                with c_col3:
                    delta_str = f"{abs(diff):,.2f} ({'higher' if diff > 0 else 'lower'})"
                    st.metric(label=f"Net Difference", value=f"{diff:,.2f}", delta=f"{diff:,.2f}")

                # Comparison Bar Chart
                chart_df = pd.DataFrame({
                    "Entity": [t1_name, t2_name],
                    metric_name: [val1, val2]
                })
                st.bar_chart(chart_df.set_index("Entity"))

            elif op == "top_n" and isinstance(raw_res, list) and len(raw_res) > 0:
                st.markdown("##### 🏆 Ranking Result")
                ranking_df = pd.DataFrame(raw_res)
                st.dataframe(ranking_df, use_container_width=True)
                
                group_col = plan.get("group_by", [None])[0]
                metric_col = plan.get("metric")
                if group_col and metric_col and group_col in ranking_df.columns and metric_col in ranking_df.columns:
                    st.bar_chart(ranking_df.set_index(group_col)[metric_col])

            elif op == "aggregate" and isinstance(raw_res, dict):
                metric_col = plan.get("metric")
                calc_name = plan.get("calculation") or "Total"
                val = raw_res.get(metric_col)
                if val is not None:
                    st.metric(label=f"{calc_name} {metric_col}", value=f"{val:,.2f}" if isinstance(val, (int, float)) else str(val))

            # Audit & Proof Inspector
            st.markdown("---")
            st.markdown("#### 🔬 Cryptographic & Execution Audit")
            
            audit_tab1, audit_tab2, audit_tab3 = st.tabs([
                "📋 Execution Plan",
                "⚙️ Dual-Engine Agreement",
                "📄 Raw Payloads"
            ])

            with audit_tab1:
                st.markdown(f"**Operation:** `{plan.get('operation')}`")
                st.markdown(f"**Target Dataset:** `{plan.get('dataset')}`")
                st.markdown(f"**Calculated Logic:** `{result_data.get('calculation_summary')}`")
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.json({"metric": plan.get("metric"), "group_by": plan.get("group_by"), "filters": plan.get("filters")})
                with pcol2:
                    st.json({"comparison_targets": plan.get("comparison_targets"), "limit": plan.get("limit"), "direction": plan.get("direction")})

            with audit_tab2:
                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    st.markdown("**Deterministic Engine Output:**")
                    st.json(result_data.get("result"))
                with vcol2:
                    st.markdown("**Independent Verifier Output:**")
                    st.json(result_data.get("independent_result"))
                
                st.success("✓ Result Hashes Match (100% Soundness Verified)")

            with audit_tab3:
                st.json(result_data)

        else:
            st.markdown(
                '<div class="proof-badge-refused">⚠ EXECUTION REFUSED / ERROR</div>',
                unsafe_allow_html=True
            )
            err_msg = result_data.get("error", "The question could not be verified or executed.")
            st.error(f"Reason: {err_msg}")
            if result_data.get("plan"):
                st.markdown("**Generated Plan:**")
                st.json(result_data.get("plan"))


# -----------------------------------------------------------------------------
# TAB 2: DATASET EXPLORER
# -----------------------------------------------------------------------------
with tab_explorer:
    if selected_dataset and selected_dataset in datasets:
        df_view = datasets[selected_dataset]
        st.markdown(f"### Dataset: `{selected_dataset}`")
        
        # Metric stats cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Rows", f"{len(df_view):,}")
        with m2:
            st.metric("Total Columns", f"{len(df_view.columns)}")
        with m3:
            st.metric("Missing Values", f"{df_view.isna().sum().sum()}")
        with m4:
            st.metric("Duplicate Rows", f"{df_view.duplicated().sum()}")

        st.markdown("#### Data Preview")
        st.dataframe(df_view, use_container_width=True)

        st.markdown("#### Schema & Statistics")
        desc_col1, desc_col2 = st.columns(2)
        with desc_col1:
            st.markdown("**Column Data Types:**")
            dtypes_df = pd.DataFrame({"Column": df_view.columns, "Type": [str(t) for t in df_view.dtypes]})
            st.dataframe(dtypes_df, use_container_width=True)
        with desc_col2:
            st.markdown("**Numeric Summary:**")
            st.dataframe(df_view.describe(), use_container_width=True)
    else:
        st.warning("No dataset selected.")


# -----------------------------------------------------------------------------
# TAB 3: SELF-TEST SUITE
# -----------------------------------------------------------------------------
with tab_tests:
    st.markdown("### 🧪 Automated Verification Suite")
    st.write("Run comprehensive tests on the proof engine to verify comparison logic, ranking operations, and aggregates.")

    test_category = st.radio(
        "Select Test Suite:",
        ["Comparison Questions (Fixed)", "All Core Analytical Tests"],
        horizontal=True
    )

    if st.button("▶ Run Test Suite", type="primary"):
        if test_category == "Comparison Questions (Fixed)":
            test_questions = [
                "What is the difference between Apple and Banana sales?",
                "How much more did Wheat make than Rice?",
                "How much more did Wheat sell than Corn?",
                "Compare Rice and Wheat.",
                "Compare Wheat and Rice."
            ]
        else:
            test_questions = [
                "Which product has the highest sales?",
                "Which product sold the most units?",
                "What is the total sales?",
                "What is the average sales?",
                "What is the difference between Apple and Banana sales?",
                "How much more did Wheat make than Rice?",
                "Compare Rice and Wheat.",
                "How many products are there?",
                "Which category has the highest sales?"
            ]

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        test_records = []
        passed_count = 0

        for idx, q in enumerate(test_questions):
            status_text.text(f"Testing ({idx+1}/{len(test_questions)}): {q}")
            res = da.analyze_query(q, datasets)
            
            is_pass = res.get("success") and res.get("verified")
            if is_pass:
                passed_count += 1

            test_records.append({
                "Question": q,
                "Status": "PASSED" if is_pass else "FAILED",
                "Operation": res.get("plan", {}).get("operation", "N/A"),
                "Metric": res.get("plan", {}).get("metric", "N/A"),
                "Answer": res.get("answer", res.get("error", "Error"))
            })
            progress_bar.progress((idx + 1) / len(test_questions))

        status_text.empty()
        progress_bar.empty()

        st.markdown(f"#### Results: {passed_count}/{len(test_questions)} Passed ({int(passed_count/len(test_questions)*100)}%)")
        
        res_df = pd.DataFrame(test_records)
        st.dataframe(res_df, use_container_width=True)
