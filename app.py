# app.py
import streamlit as st
from dotenv import load_dotenv
from utils import initialize_chroma_client, initialize_vector_store, extract_field_references, extract_constraints, save_feedback
from ai import process_user_query_optimized
from database import run_cypher, validate_cypher_query
from components import clear_feedback_state, set_results_state, render_results_and_feedback
import os
from datetime import datetime
import time

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Neo4j Natural Language Query Interface",
    page_icon="🔍",
    layout="wide"
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
.optimization-box {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔍 Neo4j Natural Language Query Interface</h1>', unsafe_allow_html=True)

    # Initialize vector store
    if 'vector_store_initialized' not in st.session_state:
        t_vs_start = time.perf_counter()
        field_collection = initialize_vector_store()
        t_vs_end = time.perf_counter()
        st.session_state['vector_store_initialized'] = True
        st.session_state['vector_init_time_ms'] = round((t_vs_end - t_vs_start) * 1000)
        # print(f"[TIMING] Vector store init: {st.session_state['vector_init_time_ms']} ms")
    else:
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
        if 'total_processing_time_ms' not in st.session_state:
            st.session_state.total_processing_time_ms = 0
            
        st.metric("Total Queries", st.session_state.query_count)
        
        if st.session_state.total_processing_time_ms > 0:
            avg_time_ms = st.session_state.total_processing_time_ms / max(st.session_state.query_count, 1)
            st.metric("Avg Generation Time", f"{avg_time_ms:.0f}ms", help="Average constraint generation time")
        
        if st.session_state.last_query_time:
            st.write(f"Last Query: {st.session_state.last_query_time}")

    # Execute new query
    if query_button and user_query.strip():
        try:
            with st.spinner("Generating optimized constraints..."):
                t_total_start = time.perf_counter()
                
                # Use AI-only constraint generation
                cypher_query, ai_constraints = process_user_query_optimized(user_query, field_collection, api_key)
                
                t_processing_end = time.perf_counter()
                processing_ms = round((t_processing_end - t_total_start) * 1000)
                
                # Update processing time tracking
                st.session_state.total_processing_time_ms += processing_ms

                # Display optimization info
                st.markdown('<div class="optimization-box">', unsafe_allow_html=True)
                st.success(f"🤖 **Query generation completed** in {processing_ms}ms")
                st.markdown('</div>', unsafe_allow_html=True)

                # Display generated constraints
                # if show_constraints:
                st.markdown('<div class="query-box">', unsafe_allow_html=True)
                st.subheader("🔍 Generated Constraints")
                
                # if ai_constraints:
                #     st.code(ai_constraints, language="cypher")
                #     st.write("*These constraints were plugged into the base Cypher query template*")
                # else:
                #     st.write("No specific constraints needed for this query")
                #     st.info("Query will return all available records")
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Execute query against Neo4j
                with st.spinner("Executing query against Neo4j..."):
                    t_db_start = time.perf_counter()
                    records = run_cypher(cypher_query, neo4j_uri, neo4j_user, neo4j_password)
                    t_db_end = time.perf_counter()
                    db_ms = round((t_db_end - t_db_start) * 1000)

                # Update stats + persist results for stable reruns
                st.session_state.query_count += 1
                st.session_state.last_query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                set_results_state(user_query, cypher_query, records)
                
                # Show detailed timings if enabled
                # if show_timings:
                total_ms = processing_ms + db_ms
                st.markdown('<div class="query-box">', unsafe_allow_html=True)
                st.subheader("⏱️ Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Constraint Generation", f"{processing_ms}ms")
                with col2:
                    st.metric("Database Execution", f"{db_ms}ms")
                with col3:
                    st.metric("Total Time", f"{total_ms}ms")
                
                # Show token efficiency
                st.info("🤖 Constraint-only approach: ~50% fewer tokens than full query generation")
                
                # Show comparison with traditional approach
                estimated_traditional_time = processing_ms * 2  # Rough estimate
                # st.write(f"**Optimization Impact:** ~{estimated_traditional_time - processing_ms}ms faster than full query generation")
                
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error(f"Error processing query: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Always render last results + feedback if available (persists across reruns)
    render_results_and_feedback(debug_on=st.session_state.get("debug_feedback_toggle", False))

    # Footer with optimization info
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666;'>
            Built with Streamlit • Powered by Neo4j, ChromaDB<br>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

