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
    first_with_idx = None
    pre_with_wheres = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('WITH') and first_with_idx is None:
            first_with_idx = len(cleaned_lines)
        # Collect WHEREs before the first WITH and skip them for now
        if first_with_idx is None and stripped.startswith('WHERE'):
            pre_with_wheres.append(line)
            continue
        cleaned_lines.append(line)

    # Determine if the first WITH is aggregating
    def is_aggregating_with(ln: str) -> bool:
        s = ln.lower()
        return ' distinct ' in f" {s} " or 'avg(' in s or 'count(' in s or 'sum(' in s or 'min(' in s or 'max(' in s

    if first_with_idx is not None:
        # If LLM placed a WHERE immediately after an aggregating WITH referencing base vars, move it before the WITH
        if first_with_idx + 1 < len(cleaned_lines):
            with_line = cleaned_lines[first_with_idx]
            next_line = cleaned_lines[first_with_idx + 1]
            if next_line.strip().startswith('WHERE') and is_aggregating_with(with_line):
                # Move this WHERE to before the WITH
                moved_where = cleaned_lines.pop(first_with_idx + 1)
                pre_with_wheres.append(moved_where)

        # Insert constraints and any collected WHEREs before the first WITH to keep variables in scope
        insertion_block = []
        if constraints_cypher:
            insertion_block.append(constraints_cypher)
        insertion_block.extend(pre_with_wheres)
        if insertion_block:
            cleaned_lines[first_with_idx:first_with_idx] = insertion_block

    cleaned_cypher = '\n'.join(cleaned_lines)

    return cleaned_cypher, constraints
