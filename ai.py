# ai.py
import re
from ratelimit import limits, sleep_and_retry
import google.generativeai as genai
from config import CONSTRAINT_ONLY_TEMPLATE, BASE_CYPHER_QUERY, CROP_FIELDS, DATE_FIELDS, SIMILARITY_THRESHOLD
from utils import extract_field_references, match_field, extract_constraints
import streamlit as st
import time

# Rate limit configuration for Gemini API (20 requests per minute)
CALLS = 20
PERIOD = 60  # 60 seconds = 1 minute

@st.cache_resource(show_spinner=False)
def get_gemini_model(api_key: str):
    """Cache and return a configured Gemini model to avoid per-call setup overhead."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def gemini_api_call(prompt: str, api_key: str) -> str:
    try:
        model = get_gemini_model(api_key)
        t0 = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=256,  # Reduced since we only need constraints
                top_p=1.0,
            )
        )
        t1 = time.perf_counter()
        # print(f"[TIMING] Gemini API: {round((t1 - t0) * 1000)} ms")
        
        constraints = response.text.strip()
        
        # Clean up the response - remove any markdown code blocks
        if constraints.startswith('```'):
            lines = constraints.split('\n')
            if len(lines) > 2:
                constraints = '\n'.join(lines[1:-1])
            elif len(lines) == 2:
                constraints = lines[0] if lines[0] != '```' else lines[1]
        
        return constraints.strip()
    except Exception as e:
        raise e

def generate_constraints_only(user_query: str, api_key: str) -> str:
    """Generate only the WHERE clause constraints using Gemini."""
    prompt = CONSTRAINT_ONLY_TEMPLATE.format(user_query=user_query)
    constraints = gemini_api_call(prompt, api_key)
    
    # If the model returns just conditions without WHERE, add it
    if constraints and not constraints.upper().startswith('WHERE'):
        constraints = f"WHERE {constraints}"
    
    return constraints

def process_user_query_optimized(user_query: str, field_collection, api_key: str):
    """Optimized version that only generates constraints via AI."""
    
    # Check for duration queries to show warning
    duration_query_detected = False
    if "days" in user_query.lower() or "older" in user_query.lower() or "since" in user_query.lower():
        duration_query_detected = True
        # st.warning(
        #     "⚠️ Duration-based queries require Neo4j date types. If your dates are strings, duration calculations may fail."
        # )
    
    # Generate only the constraints using AI
    t0 = time.perf_counter()
    ai_constraints = generate_constraints_only(user_query, api_key)
    t1 = time.perf_counter()
    # print(f"[TIMING] Constraint generation: {round((t1 - t0) * 1000)} ms")
    
    # Combine with base query
    if ai_constraints:
        final_query = BASE_CYPHER_QUERY.format(constraints=ai_constraints)
    else:
        final_query = BASE_CYPHER_QUERY.format(constraints="")
    
    return final_query, ai_constraints

def process_user_query_hybrid(user_query: str, field_collection, api_key: str):
    """Hybrid approach: use rule-based for simple queries, AI for complex ones."""
    
    # Try rule-based constraint generation first
    constraints = extract_constraints(user_query, field_collection)
    
    # If we can generate constraints using rules, use them
    if constraints:
        # print("[INFO] Using rule-based constraint generation")
        constraint_strs = []
        
        # Check for crop fields
        crop_field_detected = False
        potential_fields = extract_field_references(user_query, field_collection)
        for field_term in potential_fields:
            canonical_field, _, similarity = match_field(field_term, field_collection)
            if canonical_field and similarity >= SIMILARITY_THRESHOLD and canonical_field in CROP_FIELDS:
                crop_field_detected = True
                break
        
        if crop_field_detected:
            constraint_strs.append("pond.currentCrop = crop.id")
        
        field_to_cypher_property = {
            "abw": "cropSummary.abw",
            "awg": "cropSummary.awg",
            "fcr": "cropSummary.fcr",
            "dhscore": "dhs.totalScore",
            "aascore": "aascore.totalScore",
            "doc": "crop.doc",
            "farmname": "farm.farm_name",
            "farmername": "farmer.name",
            "tcfbiomass": "cropSummary.tcfBiomass",
            "yesterdayfeedbiomass": "cropSummary.yesterdayFeedBiomass",
            "pcc": "pond.bearing_capacity",
            "acres": "pond.acres",
            "seed": "crop.seedQuantity",
            "totalfeed": "cropSummary.totalFeed",
            "smartscalekgs": "cropSummary.smartscaleTCF",
            "isactive": "crop.isActive",
            "stockingdate": "crop.stockedOn",
            "brooder": "crop.brooder",
            "hatcheryname": "crop.hatchery",
            "harvestreason": "crop.harvestReason",
            "riskstatus": "crop.riskType",
            "harvest": "cropSummary.totalHarvest",
            "harvesteddate": "cropSummary.harvestedDate"
        }
        
        for f, op, val in constraints:
            real_name = field_to_cypher_property.get(f, f)
            if f == "farmername":
                constraint_strs.append(
                    f"(farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL AND "
                    f"(farmer.firstname = \"{val}\" OR farmer.lastname = \"{val}\"))"
                )
            else:
                if isinstance(val, str) and val.lower() in ["true", "false"]:
                    val_str = val.lower()
                else:
                    try:
                        float(val)
                        val_str = val
                    except ValueError:
                        val_str = f'\"{val}\"'
                
                if f in ["acres", "seed", "pcc"] and op in [">", "<", ">=", "<="]:
                    constraint_strs.append(f"toFloat({real_name}) IS NOT NULL AND toFloat({real_name}) {op} {val_str}")
                elif f in DATE_FIELDS and op == "duration":
                    constraint_strs.append(
                        f"{real_name} IS NOT NULL AND duration.between({real_name}, date()).days > {val}"
                    )
                elif f in DATE_FIELDS and op in [">", "<", ">=", "<=", "="]:
                    constraint_strs.append(f"{real_name} IS NOT NULL AND {real_name} {op} {val_str}")
                else:
                    constraint_strs.append(f"{real_name} IS NOT NULL AND {real_name} {op} {val_str}")
        
        if "most number of farms" in user_query.lower() or "count" in user_query.lower():
            constraint_strs.append("farmer.firstname IS NOT NULL")
            constraint_strs.append("farmer.lastname IS NOT NULL")
        
        constraints_cypher = "WHERE " + " AND ".join(constraint_strs) if constraint_strs else ""
        final_query = BASE_CYPHER_QUERY.format(constraints=constraints_cypher)
        
        return final_query, constraints
    
    else:
        # Fall back to AI-based generation for complex queries
        # print("[INFO] Falling back to AI-based constraint generation")
        return process_user_query_optimized(user_query, field_collection, api_key)

