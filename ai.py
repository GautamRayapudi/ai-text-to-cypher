# ai.py
import re
from ratelimit import limits, sleep_and_retry
import google.generativeai as genai
from config import TEMPLATE_PROMPT, CROP_FIELDS, DATE_FIELDS, SIMILARITY_THRESHOLD
from utils import extract_field_references, match_field, extract_constraints
import streamlit as st

# Rate limit configuration for Gemini API (20 requests per minute)
CALLS = 20
PERIOD = 60  # 60 seconds = 1 minute

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def gemini_api_call(prompt: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1024,
                top_p=1.0,
            )
        )
        cypher_query = response.text.strip()
        if cypher_query.startswith('```'):
            lines = cypher_query.split('\n')
            if len(lines) > 2:
                cypher_query = '\n'.join(lines[1:-1])
            elif len(lines) == 2:
                cypher_query = lines[0] if lines[0] != '```' else lines[1]
        return cypher_query
    except Exception as e:
        raise e

def process_user_query(user_query: str, field_collection, api_key: str):
    constraints = extract_constraints(user_query, field_collection)
    field_to_cypher_property = {
        "abw": "cropSummary.abw",
        "awg": "cropSummary.awg", 
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

    constraint_strs = []
    crop_field_detected = False
    potential_fields = extract_field_references(user_query, field_collection)
    for field_term in potential_fields:
        canonical_field, _, similarity = match_field(field_term, field_collection)
        if canonical_field and similarity >= SIMILARITY_THRESHOLD and canonical_field in CROP_FIELDS:
            crop_field_detected = True
            break
    if crop_field_detected:
        constraint_strs.append("pond.currentCrop = crop.id")
    duration_query_detected = False

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
                duration_query_detected = True
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

    if duration_query_detected:
        st.warning(
            "⚠️ Duration-based queries require Neo4j date types. If your dates are strings, duration calculations may fail."
        )

    constraints_cypher = "WHERE " + " AND ".join(constraint_strs) if constraint_strs else ""
    prompt = TEMPLATE_PROMPT.format(constraints=constraints_cypher, user_query=user_query)
    cypher_query = gemini_api_call(prompt, api_key)

    lines = cypher_query.split('\n')
    cleaned_lines = []
    found_with = False
    skip_until_with = False
    for line in lines:
        if line.strip().startswith('WITH'):
            found_with = True
            skip_until_with = False
        if not found_with and line.strip().startswith('WHERE'):
            skip_until_with = True
            continue
        if skip_until_with:
            continue
        cleaned_lines.append(line)
    cleaned_cypher = '\n'.join(cleaned_lines)

    if constraints_cypher and constraints_cypher not in cleaned_cypher:
        for i, line in enumerate(cleaned_lines):
            if line.strip().startswith('WITH'):
                cleaned_lines.insert(i + 1, constraints_cypher)
                break
        cleaned_cypher = '\n'.join(cleaned_lines)


    return cleaned_cypher, constraints
