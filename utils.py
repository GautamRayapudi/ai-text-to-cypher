# utils.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import sys
"""
Ensure chromadb sees a modern sqlite3 by monkeypatching with pysqlite3 if the
system sqlite3 is too old (e.g., on some hosted environments).
"""
try:
    import pysqlite3 as _pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # If pysqlite3 isn't available, fall back; chromadb may still work if sqlite3 is new enough
    pass

import chromadb
import os
import csv
from config import FIELD_ALIASES, SIMILARITY_THRESHOLD, CROP_FIELDS
import pandas as pd

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def initialize_chroma_client():
    CHROMA_DB_PATH = "./chroma_embeddings"
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

@st.cache_resource
def initialize_vector_store():
    model = load_sentence_transformer()
    chroma_client = initialize_chroma_client()
    try:
        collection = chroma_client.get_collection("field_embeddings")
        st.success("Loaded existing field embeddings from ChromaDB")
        return collection
    except:
        st.info("Creating new field embeddings collection...")
        collection = chroma_client.create_collection(
            name="field_embeddings",
            metadata={"description": "Field synonyms and aliases for NL to Cypher conversion"}
        )
        all_synonyms = []
        canonical_fields = []
        for field, synonyms in FIELD_ALIASES.items():
            all_synonyms.append(field)
            canonical_fields.append(field)
            for syn in synonyms:
                all_synonyms.append(syn)
                canonical_fields.append(field)
        embeddings = model.encode(all_synonyms, normalize_embeddings=True)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=all_synonyms,
            metadatas=[{"canonical_field": cf} for cf in canonical_fields],
            ids=[f"field_embedding_{i}" for i in range(len(all_synonyms))]
        )
        st.success(f"Stored {len(all_synonyms)} field embeddings in ChromaDB")
        return collection

def match_field(user_term, field_collection, threshold=SIMILARITY_THRESHOLD):
    model = load_sentence_transformer()
    try:
        query_embedding = model.encode([user_term], normalize_embeddings=True)
        results = field_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3,
            include=['metadatas', 'documents', 'distances']
        )
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            canonical_field = results['metadatas'][0][0]['canonical_field']
            matched_document = results['documents'][0][0]
            similarity_score = 1 - results['distances'][0][0]
            if similarity_score >= threshold:
                return canonical_field, matched_document, similarity_score
            else:
                return None, None, similarity_score
        else:
            return None, None, 0
    except Exception as e:
        st.error(f"Error matching field '{user_term}': {e}")
        return None, None, 0

def extract_field_references(user_query, field_collection):
    potential_fields = set()
    underscore_pattern = r'\b\w+_\w+(?:_\w+)*\b'
    underscore_matches = re.findall(underscore_pattern, user_query.lower())
    for match in underscore_matches:
        space_version = match.replace('_', ' ')
        potential_fields.add(match)
        potential_fields.add(space_version)
    multiword_patterns = [
        r'\b(?:average|avg)\s+(?:body\s+)?weight(?:\s+gain)?\b',
        r'\b(?:days?\s+of\s+)?culture(?:\s+days?)?\b', 
        r'\b(?:disease\s+)?health\s+score\b',
        r'\b(?:pond\s+)?(?:carrying\s+)?capacity\b',
        r'\b(?:farm(?:er)?\s+)?name\b',
        r'\b(?:feed\s+)?biomass\b',
        r'\b(?:seed\s+)?quantity\b',
        r'\b(?:total\s+)?feed\b',
        r'\b(?:smart\s+)?scale\b',
        r'\bactive\b',
        r'\bacres?\b',
        r'\bsize\b',
        r'\bstocking\s+date\b',
        r'\bbrooder\b',
        r'\bhatchery\s+name\b',
        r'\bharvest\s+reason\b',
        r'\brisk\s+status\b',
        r'\bharvest\s+date\b',
        r'\bharvested\s+date\b',
        r'\bharvest\b'
    ]
    for pattern in multiword_patterns:
        matches = re.findall(pattern, user_query.lower())
        potential_fields.update(matches)
    stop_words = {'show', 'me', 'with', 'the', 'and', 'or', 'is', 'are', 'have', 'has', 'than', 'less', 'more', 'greater', 'above', 'below', 'farms', 'farm', 'crops', 'crop', 'pond', 'ponds', 'who', 'what', 'when', 'where', 'how', 'many', 'most', 'number'}
    words = re.findall(r'\b\w+\b', user_query.lower())
    significant_words = [w for w in words if len(w) > 2 and w not in stop_words]
    field_indicators = ['weight', 'score', 'days', 'culture', 'feed', 'scale', 'active', 'biomass', 'capacity', 'quantity', 'name', 'acres', 'size', 'stocking', 'brooder', 'hatchery', 'harvest', 'risk', 'date']
    for word in significant_words:
        if any(indicator in word for indicator in field_indicators):
            potential_fields.add(word)
    return list(potential_fields)

def extract_constraints(user_query, field_collection):
    constraints = []
    uq = user_query.lower()
    potential_fields = extract_field_references(user_query, field_collection)
    for field_term in potential_fields:
        canonical_field, matched_doc, similarity = match_field(field_term, field_collection)
        if field_term.lower() in ["harvest", "harvest date"] and canonical_field == "harvest":
            date_patterns = [
                r"harvest\s+(before|after|on)\s+[\d-]+\b",
                r"harvest\s+date\s+(before|after|<=|>=|<|>|=)\s+[\d-]+\b",
                r"harvest\s+(<=|>=|<|>|=)\s+[\d-]+\b"
            ]
            if any(re.search(pattern, uq) for pattern in date_patterns):
                canonical_field = "harvesteddate"
        if canonical_field and similarity >= SIMILARITY_THRESHOLD:
            field_term_escaped = re.escape(field_term.lower())
            for op in [">=", "<=", ">", "<", "="]:
                pattern = rf"{field_term_escaped}\s*{re.escape(op)}\s*([\w\.]+)"
                matches = re.findall(pattern, uq)
                for val in matches:
                    constraints.append((canonical_field, op, val))
            above_pattern = rf"{field_term_escaped}\s+(above|greater\s+than|over|more\s+than)\s+(\d+(?:\.\d+)?)"
            matches = re.findall(above_pattern, uq)
            for _, val in matches:
                constraints.append((canonical_field, ">", val))
            below_pattern = rf"{field_term_escaped}\s+(below|less\s+than|under)\s+(\d+(?:\.\d+)?)"
            matches = re.findall(below_pattern, uq)
            for _, val in matches:
                constraints.append((canonical_field, "<", val))
            reverse_above_pattern = rf"(\d+(?:\.\d+)?)\s+(or\s+)?(above|greater|more|over)\s+{field_term_escaped}"
            matches = re.findall(reverse_above_pattern, uq)
            for val, _, _ in matches:
                constraints.append((canonical_field, ">", val))
            if canonical_field in CROP_FIELDS:
                duration_pattern = rf"{field_term_escaped}\s+(?:last|older\s+than|more\s+than)\s+(\d+)\s+days\b"
                matches = re.findall(duration_pattern, uq)
                for val in matches:
                    constraints.append((canonical_field, "duration", val))
    if re.search(r'\bactive\b', uq):
        field, _, similarity = match_field("active", field_collection)
        if field and similarity >= SIMILARITY_THRESHOLD:
            constraints.append((field, "=", "true"))
    farmer_name_patterns = [
        r"(?:farmer|person|user)\s+(?:named|called)?\s*([A-Z][a-zA-Z]+)",
        r"([A-Z][a-zA-Z]+)(?:\s+(?:has|have|owns?))",
        r"(?:does|how\s+many)\s+([A-Z][a-zA-Z]+)\s+(?:have|own)",
        r"([A-Z][a-zA-Z]+)(?:'s|'s)\s+(?:farm|pond|crop)",
        r"show.*?([A-Z][a-zA-Z]+)(?:\s+(?:farm|pond|crop))?",
    ]
    for pattern in farmer_name_patterns:
        matches = re.findall(pattern, user_query)
        for match in matches:
            if len(match.strip()) > 2:
                farmer_field, _, similarity = match_field("farmer name", field_collection)
                if farmer_field and similarity >= SIMILARITY_THRESHOLD:
                    if match.strip().lower() not in ["who", "what", "when", "where", "how"]:
                        constraints.append((farmer_field, "=", match.strip()))
    return constraints

def save_feedback(user_query, cypher_query, reason=None, append_mode=False):
    """Convert feedback into a pandas DataFrame with query, cypher, reason columns"""
    # Create new feedback entry
    new_data = {
        "user_query": [user_query],
        "cypher_query": [cypher_query],
        "reason": [reason if reason else ""]
    }
    new_df = pd.DataFrame(new_data)
    
    # Clean the new dataframe
    for col in new_df.columns:
        if new_df[col].dtype == 'object':
            new_df[col] = new_df[col].astype(str).replace('None', None)
    
    if append_mode and 'feedback_dataframe' in st.session_state:
        # Append to existing dataframe
        existing_df = st.session_state['feedback_dataframe']
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        st.session_state['feedback_dataframe'] = combined_df
        print(f"[DEBUG] Feedback appended to existing DataFrame: query={user_query}, cypher={cypher_query}, reason={reason}")
        print(f"[DEBUG] Total feedback entries: {len(combined_df)}")
        return combined_df
    else:
        # Create new dataframe or replace existing one
        st.session_state['feedback_dataframe'] = new_df
        print(f"[DEBUG] New Feedback DataFrame created: query={user_query}, cypher={cypher_query}, reason={reason}")
        return new_df



