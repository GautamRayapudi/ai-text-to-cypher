import streamlit as st
import pandas as pd
from datetime import datetime
from utils import save_feedback

def clear_feedback_state():
    for k in ["feedback_choice", "feedback_reason", "feedback_submitted", "feedback_saved_key"]:
        if k in st.session_state:
            del st.session_state[k]

def set_results_state(user_query, cypher_query, records):
    st.session_state["has_results"] = True
    st.session_state["last_user_query"] = user_query
    st.session_state["last_cypher_query"] = cypher_query
    st.session_state["last_records"] = records
    st.session_state["last_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # reset feedback UI for the new result set
    clear_feedback_state()

def render_results_and_feedback(debug_on: bool):
    if not st.session_state.get("has_results"):
        return

    user_query = st.session_state.get("last_user_query", "")
    cypher_query = st.session_state.get("last_cypher_query", "")
    records = st.session_state.get("last_records", []) or []

    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    st.subheader("⚡ Generated Cypher Query")
    st.code(cypher_query, language='cypher')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader(f"📋 Query Results ({len(records)} records)")
    if records:
        df = pd.DataFrame(records)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).replace('None', None)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Records", len(records))
        with c2: st.metric("Columns", len(df.columns))
        with c3: st.metric("Non-null Values", df.count().sum())
        st.dataframe(df, use_container_width=True)
        csv_download = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_download,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        with st.expander("View Sample Records (JSON)"):
            for i, record in enumerate(records[:3]):
                st.json(record)
                if i < 2 and i < len(records) - 1:
                    st.divider()
    else:
        st.info("No records found matching your query.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- FEEDBACK SECTION (persistent across reruns) ----------
    st.markdown("---")
    st.subheader("📝 Was this result helpful?")

    feedback = st.radio("Please select:", ["Yes", "No"], horizontal=True, index=None, key="feedback_choice")
    
    if feedback == "Yes":
        if st.button("Submit Feedback", key="yes_feedback_submit"):
            data_csv = save_feedback(user_query, cypher_query, "")
            st.success("✅ Thank you for the feedback!")
            if not data_csv.empty:  # Updated condition
                st.download_button(
                    label="📥 Download CSV",
                    data=data_csv.to_csv(index=False),  # Convert to CSV string
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="csv_after_yes"
                )

    elif feedback == "No":
        reason = st.text_area(
            "Please tell us why you are not satisfied:",
            placeholder="Enter your feedback here...",
            key="feedback_reason"
        )
        if st.button("Submit Feedback", key="no_feedback_submit"):
            if reason.strip():
                data_csv = save_feedback(user_query, cypher_query, reason.strip())
                st.success("🙏 Thank you for the feedback! We'll use this to improve.")
                if not data_csv.empty:  # Updated condition
                    st.download_button(
                        label="📥 Download CSV",
                        data=data_csv.to_csv(index=False),  # Convert to CSV string
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="csv_after_no"
                    )
            else:
                st.warning("⚠️ Please provide a reason before submitting.")

    # ---------- Debug view ----------
    if debug_on:
        with st.expander("🐞 Feedback Debug"):
            st.write({
                "has_results": st.session_state.get("has_results"),
                "last_generated_at": st.session_state.get("last_generated_at"),
                "feedback_choice": st.session_state.get("feedback_choice"),
                "feedback_reason_len": len(st.session_state.get("feedback_reason", "")),
                "feedback_submitted": st.session_state.get("feedback_submitted"),
                "feedback_saved_key": st.session_state.get("feedback_saved_key"),

            })


