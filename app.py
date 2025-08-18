# app.py
import streamlit as st
from dotenv import load_dotenv
from config import FIELD_ALIASES, SIMILARITY_THRESHOLD, DEBUG_MATCHING, SUPPORTED_FIELDS, REVERSE_FIELD_LOOKUP, CROP_FIELDS, DATE_FIELDS, TEMPLATE_PROMPT
from utils import load_sentence_transformer, initialize_chroma_client, initialize_vector_store, match_field, extract_field_references, extract_constraints, save_feedback
from ai import gemini_api_call, process_user_query
from database import run_cypher, validate_cypher_query
from components import clear_feedback_state, set_results_state, render_results_and_feedback
import os
from datetime import datetime

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Neo4j Natural Language Query Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.query-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.result-box {
    background-color: #e8f4f8;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.error-box {
    background-color: #ffe6e6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔍 Neo4j Natural Language Query Interface</h1>', unsafe_allow_html=True)

    # Initialize vector store
    field_collection = initialize_vector_store()
    api_key = os.getenv("GEMINI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # Main UI
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("💬 Natural Language Query")
        user_query = st.text_area(
            "Enter your query:",
            placeholder="e.g., Show me farms with average body weight above 15",
            height=100
        )
        query_button = st.button("🔍 Execute Query", type="primary")

    with c2:
        st.subheader("📊 Query Statistics")
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        if 'last_query_time' not in st.session_state:
            st.session_state.last_query_time = None
        st.metric("Total Queries", st.session_state.query_count)
        if st.session_state.last_query_time:
            st.write(f"Last Query: {st.session_state.last_query_time}")

    # Execute new query
    if query_button and user_query.strip():
        try:
            with st.spinner("Processing your query..."):
                cypher_query, constraints = process_user_query(user_query, field_collection, api_key)

                # Display extracted constraints (immediately)
                if constraints:
                    st.markdown('<div class="query-box">', unsafe_allow_html=True)
                    st.subheader("🔍 Extracted Constraints")
                    for field, op, val in constraints:
                        st.write(f"- **{field}** {op} {val}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.spinner("Executing query against Neo4j..."):
                    records = run_cypher(cypher_query, neo4j_uri, neo4j_user, neo4j_password)

                # Update stats + persist results for stable reruns
                st.session_state.query_count += 1
                st.session_state.last_query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                set_results_state(user_query, cypher_query, records)

        except Exception as e:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error(f"Error processing query: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Always render last results + feedback if available (persists across reruns)
    render_results_and_feedback(debug_on=st.session_state.get("debug_feedback_toggle", False))

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with Streamlit • Powered by Neo4j, ChromaDB & Gemini AI
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":

    main()
