# ai.py - Extended version with aggregation support
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

# New template for constraint + aggregation generation
CONSTRAINT_AND_AGGREGATION_TEMPLATE = """
You are an expert Neo4j Cypher query component generator.

Field mappings:
- abw: cropSummary.abw
- awg: cropSummary.awg
- fcr: cropSummary.fcr
- dhscore: dhs.totalScore
- aascore: aascore.totalScore
- doc: crop.doc
- farmname: farm.farm_name
- farmername: farmer.firstname OR farmer.lastname (check both fields)
- tcfbiomass: cropSummary.tcfBiomass
- yesterdayfeedbiomass: cropSummary.yesterdayFeedBiomass
- pcc: pond.bearing_capacity
- acres: pond.acres
- seed: crop.seedQuantity
- totalfeed: cropSummary.totalFeed
- smartscalekgs: cropSummary.smartscaleTCF
- isactive: crop.isActive
- stockingdate: crop.stockedOn
- brooder: crop.brooder
- hatcheryname: crop.hatchery
- harvestreason: crop.harvestReason
- riskstatus: crop.riskType
- harvest: cropSummary.totalHarvest
- harvesteddate: cropSummary.harvestedDate

Generate TWO components for this query: "{user_query}"

Return in this EXACT format:
CONSTRAINTS: [WHERE clause constraints or empty]
AGGREGATION: [ORDER BY/LIMIT clauses using aggregated field aliases or empty]

Rules for CONSTRAINTS:
1. Return ONLY "WHERE condition1 AND condition2..." format
2. If no constraints needed, return empty
3. Always include NOT NULL checks: "field IS NOT NULL AND field > value"
4. For farmer names: "(farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL AND (farmer.firstname = 'name' OR farmer.lastname = 'name'))"
5. For numeric string fields (acres, seed, pcc): "toFloat(field) IS NOT NULL AND toFloat(field) > value"
6. For dates: "field IS NOT NULL AND duration.between(field, date()).days > days" for duration queries
7. For crop fields, include: "pond.currentCrop = crop.id"

Rules for AGGREGATION:
1. Use ONLY aggregated field aliases in ORDER BY, not raw aggregation functions
2. Available aggregated aliases: farmCount, pondCount, avgABW, avgFCR, totalHarvest, avgAAScore, avgDHScore
3. For "highest abw" → ORDER BY avgABW DESC
4. For "most farms" → ORDER BY farmCount DESC  
5. For "highest aascore" → ORDER BY avgAAScore DESC
6. For "lowest fcr" → ORDER BY avgFCR ASC
7. Always include LIMIT for top/best/worst queries
8. If no aggregation needed, return empty

Examples:
Query: "top 5 farmers with highest average abw"
CONSTRAINTS: WHERE cropSummary.abw IS NOT NULL AND farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL
AGGREGATION: ORDER BY avgABW DESC LIMIT 5

Query: "farmers with fcr below 1.3"
CONSTRAINTS: WHERE cropSummary.fcr IS NOT NULL AND cropSummary.fcr < 1.3 AND farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL
AGGREGATION: 

Query: "which farmer has most farms"
CONSTRAINTS: WHERE farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL
AGGREGATION: ORDER BY farmCount DESC LIMIT 1

Query: "top 3 farmers by aa score"
CONSTRAINTS: WHERE aascore.totalScore IS NOT NULL AND farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL
AGGREGATION: ORDER BY avgAAScore DESC LIMIT 3

User query: "{user_query}"
"""

# Detection patterns for aggregation queries
AGGREGATION_PATTERNS = [
    r'\btop\s+\d+\b',
    r'\bhighest\b', r'\blowest\b', r'\bmost\b', r'\bleast\b',
    r'\bbest\b', r'\bworst\b', r'\blargest\b', r'\bsmallest\b',
    r'\bfirst\s+\d+\b', r'\blast\s+\d+\b',
    r'\bcount\b', r'\baverage\b', r'\bsum\b', r'\bmax\b', r'\bmin\b',
    r'\bgroup\s+by\b', r'\bsort\b', r'\border\b',
    r'\bwho\s+has\s+(most|least|highest|lowest)\b',
    r'\bwhich\s+\w+\s+has\s+(most|least|highest|lowest)\b'
]

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
                max_output_tokens=512,  # Increased for aggregation support
                top_p=1.0,
            )
        )
        t1 = time.perf_counter()
        # print(f"[TIMING] Gemini API: {round((t1 - t0) * 1000)} ms")
        
        result = response.text.strip()
        
        # Clean up the response - remove any markdown code blocks
        if result.startswith('```'):
            lines = result.split('\n')
            if len(lines) > 2:
                result = '\n'.join(lines[1:-1])
            elif len(lines) == 2:
                result = lines[0] if lines[0] != '```' else lines[1]
        
        return result.strip()
    except Exception as e:
        raise e

def detect_aggregation_query(user_query: str) -> bool:
    """Detect if query requires aggregation operations."""
    query_lower = user_query.lower()
    return any(re.search(pattern, query_lower) for pattern in AGGREGATION_PATTERNS)

def parse_constraint_and_aggregation_response(response: str) -> tuple[str, str]:
    """Parse the dual-component response from the LLM."""
    constraints = ""
    aggregation = ""
    
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('CONSTRAINTS:'):
            current_section = 'constraints'
            constraint_part = line.replace('CONSTRAINTS:', '').strip()
            if constraint_part and constraint_part != 'empty':
                constraints = constraint_part
        elif line.startswith('AGGREGATION:'):
            current_section = 'aggregation'
            agg_part = line.replace('AGGREGATION:', '').strip()
            if agg_part and agg_part != 'empty':
                aggregation = agg_part
        elif current_section == 'constraints' and line:
            if line != 'empty':
                constraints += (' ' + line) if constraints else line
        elif current_section == 'aggregation' and line:
            if line != 'empty':
                aggregation += (' ' + line) if aggregation else line
    
    # Clean up constraints
    if constraints and not constraints.upper().startswith('WHERE'):
        constraints = f"WHERE {constraints}"
    
    return constraints, aggregation

def generate_constraints_only(user_query: str, api_key: str) -> str:
    """Generate only the WHERE clause constraints using Gemini."""
    prompt = CONSTRAINT_ONLY_TEMPLATE.format(user_query=user_query)
    constraints = gemini_api_call(prompt, api_key)
    
    # If the model returns just conditions without WHERE, add it
    if constraints and not constraints.upper().startswith('WHERE'):
        constraints = f"WHERE {constraints}"
    
    return constraints

def generate_constraints_and_aggregation(user_query: str, api_key: str) -> tuple[str, str]:
    """Generate both constraints and aggregation components using Gemini."""
    prompt = CONSTRAINT_AND_AGGREGATION_TEMPLATE.format(user_query=user_query)
    response = gemini_api_call(prompt, api_key)
    return parse_constraint_and_aggregation_response(response)

def build_query_with_aggregation(constraints: str, aggregation: str, user_query: str) -> str:
    """Build the final Cypher query with both constraints and aggregation."""
    
    # Check if we need grouping for farmer-level aggregations
    query_lower = user_query.lower()
    needs_farmer_grouping = any(phrase in query_lower for phrase in [
        'which farmer', 'farmer has most', 'farmer with highest', 'farmer with lowest',
        'farmers by', 'farmers with most', 'farmers with highest', 'farmers with lowest',
        'top farmers', 'best farmers', 'worst farmers'
    ])
    
    # Check if we need farm-level aggregations
    needs_farm_grouping = any(phrase in query_lower for phrase in [
        'farms by', 'which farm', 'farm has most', 'farm with highest',
        'top farms', 'best farms', 'worst farms'
    ]) and not needs_farmer_grouping  # Farmer takes precedence
    
    if needs_farmer_grouping:
        # Group by farmer and aggregate farms/ponds
        base_query = """MATCH (farm:Farm)-[:HAS_POND]->(pond:Pond)-[:HAS_CROP]->(crop:Crop)
OPTIONAL MATCH (crop)-[:HAS_SUMMARY]->(cropSummary:CropSummary)
WITH farm, pond, crop, cropSummary
OPTIONAL MATCH (farmer:Farmer)-[:HAS_FRAM]->(farm)
WHERE farmer.id = farm.farmer_id
WITH farm, pond, crop, cropSummary, farmer
OPTIONAL MATCH (crop)-[:HAS_INSURANCE]-(ins:Insurance)
WITH farm, pond, crop, cropSummary, farmer, COUNT(ins.id) > 0 AS hasInsurance
OPTIONAL MATCH (crop)-[:HAS_AASCORE]-(aascore:AAScore)
WITH farm, pond, crop, cropSummary, farmer, hasInsurance, aascore
OPTIONAL MATCH (crop)-[:HAS_DHSCORE]->(dhs:DHScore)
WITH farm, pond, crop, cropSummary, farmer, hasInsurance, aascore, dhs
OPTIONAL MATCH (farm)-[:HAS_SECTION]-(cycle:Cycle)
OPTIONAL MATCH (cycle)-[:STOCKED]-(cycleSummary:CycleSummary)
WITH farm, pond, crop, cropSummary, farmer, hasInsurance, aascore, dhs, cycle, cycleSummary
{constraints}
WITH farmer, 
     COUNT(DISTINCT farm.id) as farmCount,
     COUNT(DISTINCT pond.id) as pondCount,
     AVG(cropSummary.abw) as avgABW,
     AVG(cropSummary.fcr) as avgFCR,
     SUM(cropSummary.totalHarvest) as totalHarvest,
     AVG(aascore.totalScore) as avgAAScore,
     AVG(dhs.totalScore) as avgDHScore
RETURN farmer.firstname + ' ' + farmer.lastname AS farmerName,
       farmCount, pondCount, avgABW, avgFCR, totalHarvest, avgAAScore, avgDHScore
{aggregation}"""
    
    elif needs_farm_grouping:
        # Group by farm and aggregate ponds/crops
        base_query = """MATCH (farm:Farm)-[:HAS_POND]->(pond:Pond)-[:HAS_CROP]->(crop:Crop)
OPTIONAL MATCH (crop)-[:HAS_SUMMARY]->(cropSummary:CropSummary)
WITH farm, pond, crop, cropSummary
OPTIONAL MATCH (farmer:Farmer)-[:HAS_FRAM]->(farm)
WHERE farmer.id = farm.farmer_id
WITH farm, pond, crop, cropSummary, farmer
OPTIONAL MATCH (crop)-[:HAS_INSURANCE]-(ins:Insurance)
WITH farm, pond, crop, cropSummary, farmer, COUNT(ins.id) > 0 AS hasInsurance
OPTIONAL MATCH (crop)-[:HAS_AASCORE]-(aascore:AAScore)
WITH farm, pond, crop, cropSummary, farmer, hasInsurance, aascore
OPTIONAL MATCH (crop)-[:HAS_DHSCORE]->(dhs:DHScore)
WITH farm, pond, crop, cropSummary, farmer, hasInsurance, aascore, dhs
OPTIONAL MATCH (farm)-[:HAS_SECTION]-(cycle:Cycle)
OPTIONAL MATCH (cycle)-[:STOCKED]-(cycleSummary:CycleSummary)
WITH farm, pond, crop, cropSummary, farmer, hasInsurance, aascore, dhs, cycle, cycleSummary
{constraints}
WITH farm, farmer,
     COUNT(DISTINCT pond.id) as pondCount,
     AVG(cropSummary.abw) as avgABW,
     AVG(cropSummary.fcr) as avgFCR,
     SUM(cropSummary.totalHarvest) as totalHarvest,
     AVG(aascore.totalScore) as avgAAScore,
     AVG(dhs.totalScore) as avgDHScore
RETURN farm.farm_name AS farmName,
       farmer.firstname + ' ' + farmer.lastname AS farmerName,
       pondCount, avgABW, avgFCR, totalHarvest, avgAAScore, avgDHScore
{aggregation}"""
    
    else:
        # Use original base query with aggregation
        base_query = BASE_CYPHER_QUERY + "\n{aggregation}"
    
    return base_query.format(constraints=constraints, aggregation=aggregation)

def process_user_query_optimized(user_query: str, field_collection, api_key: str):
    """Optimized version that handles both simple and aggregation queries efficiently."""
    
    # Check for duration queries to show warning
    duration_query_detected = False
    if "days" in user_query.lower() or "older" in user_query.lower() or "since" in user_query.lower():
        duration_query_detected = True
    
    # Detect if aggregation is needed
    needs_aggregation = detect_aggregation_query(user_query)
    
    t0 = time.perf_counter()
    
    if needs_aggregation:
        # Generate both constraints and aggregation components
        constraints, aggregation = generate_constraints_and_aggregation(user_query, api_key)
        final_query = build_query_with_aggregation(constraints, aggregation, user_query)
        ai_constraints = f"{constraints}\n{aggregation}" if aggregation else constraints
    else:
        # Generate only constraints for simple queries
        constraints = generate_constraints_only(user_query, api_key)
        final_query = BASE_CYPHER_QUERY.format(constraints=constraints) if constraints else BASE_CYPHER_QUERY.format(constraints="")
        ai_constraints = constraints
    
    t1 = time.perf_counter()
    # print(f"[TIMING] {'Constraint+Aggregation' if needs_aggregation else 'Constraint-only'} generation: {round((t1 - t0) * 1000)} ms")
    
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
