# AI-Text-to-Cypher

This is a Streamlit application that allows users to query a Neo4j graph database using natural language. The app converts natural language queries into Cypher queries using Gemini AI, matches fields using semantic embeddings (SentenceTransformer and ChromaDB), executes the Cypher queries on Neo4j, and displays results with feedback mechanisms.

## File Structure

- **app.py**: Main entry point for the Streamlit app. Handles UI, query execution, and orchestration.
- **config.py**: Contains constants like field aliases, similarity thresholds, schema mappings, and the Cypher prompt template.
- **utils.py**: Utility functions for embedding management (ChromaDB), field matching, constraint extraction, and feedback saving.
- **ai.py**: AI-related functions for Gemini API calls and processing user queries into Cypher.
- **database.py**: Database interaction functions for executing Cypher queries on Neo4j and validating queries.
- **components.py**: UI component functions for rendering results, handling state, and feedback.
- **.env**: Environment variables (e.g., GEMINI_API_KEY). Create this file and add your keys.
- **user_feedback.csv**: Generated file for storing user feedback (query, cypher, reason).
- **chroma_embeddings/**: Directory for persistent ChromaDB embeddings.
- **requirements.txt**: List of dependencies (create based on imports: streamlit, sentence-transformers, numpy, re, neo4j, chromadb, google-generativeai, pandas, json, datetime, dotenv, ratelimit, csv).

## Setup Instructions

1. **Install Dependencies**:

pip install -r requirements.txt

3. **Run the App**:
   
streamlit run app.py

5. **Usage**:
   
- Enter a natural language query (e.g., "Show me farms with average body weight above 15").
- Configure Neo4j connection in the sidebar if needed.
- View generated Cypher, results, and provide feedback.
- Feedback is saved to `user_feedback.csv`.
- Embeddings are persisted in `./chroma_embeddings`.

## Notes
- The app assumes specific Neo4j schema and field types (e.g., dates as Neo4j Date types).
- Rate limiting is applied to Gemini API calls (20/min).
- Debug mode in sidebar for feedback inspection.
- For production, secure sensitive info and handle errors robustly.

## Contributing
Feel free to open issues or PRs for improvements!
