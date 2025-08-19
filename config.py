# config.py
FIELD_ALIASES = {
    "abw": ["average body weight", "avg body weight", "body weight"],
    "awg": ["average weight gain", "avg weight gain", "weight gain"],
    "fcr": ["feed conversion ratio", "fcr", "feed ratio"],
    "dhscore": ["disease health score", "dh score", "disease score", "health score"],
    "aascore": ["aa score", "aquaculture score"],
    "doc": ["days of culture", "culture days", "doc"],
    "farmname": ["farm name", "name of farm"],
    "farmername": ["farmer name", "name of farmer"],
    "tcfbiomass": ["tcf biomass", "biomass"],
    "yesterdayfeedbiomass": ["yesterday feed biomass", "feed biomass yesterday"],
    "pcc": ["pond carrying capacity", "carrying capacity"],
    "acres": ["pond size", "size in acres", "acres", "area"],
    "seed": ["seed quantity", "seed count", "seed", "stocking"],
    "totalfeed": ["total feed", "feed used", "feed quantity"],
    "smartscalekgs": ["smart scale", "scale weight", "scale kgs"],
    "isactive": ["active", "crop active", "currently active", "status"],
    "stockingdate": ["stocking date", "date of stocking"],
    "brooder": ["brooder"],
    "hatcheryname": ["hatchery name", "hatchery"],
    "harvestreason": ["harvest reason", "reason for harvest"],
    "riskstatus": ["risk status", "crop risk"],
    "harvest": ["harvest"],
    "harvesteddate": ["harvested date", "date of harvest"],
}

SIMILARITY_THRESHOLD = 0.6
DEBUG_MATCHING = False
SUPPORTED_FIELDS = set(FIELD_ALIASES.keys())

REVERSE_FIELD_LOOKUP = {}
for key, values in FIELD_ALIASES.items():
    REVERSE_FIELD_LOOKUP[key] = key
    for v in values:
        REVERSE_FIELD_LOOKUP[v.lower()] = key

CROP_FIELDS = {
    "doc", "seed", "isactive", "stockingdate", "brooder",
    "hatcheryname", "harvestreason", "riskstatus", "harvest", "harvesteddate"
}
DATE_FIELDS = {"stockingdate", "harvesteddate"}

# Base Cypher query template - static part
BASE_CYPHER_QUERY = """MATCH (farm:Farm)-[:HAS_POND]->(pond:Pond)-[:HAS_CROP]->(crop:Crop)
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
RETURN DISTINCT farm.id AS farmId, farmer.firstname as farmerFirstName, farmer.lastname as farmerLastName, farm.farm_name AS farmName,
cycle.id AS sectionId, pond.id AS pondId, pond.name AS pondName, pond.acres AS acres,
pond.bearing_capacity AS PCC, crop.stockedOn AS stockingDate, crop.brooder AS brooder,
crop.hatchery AS hatcheryName, crop.harvestReason AS harvestReason,
hasInsurance, cropSummary.fcr AS FCR, crop.seedQuantity AS seed, crop.pondLevel AS pondLevel,
cropSummary.tcf AS totalFeed, cropSummary.smartscaleTCF AS smartScaleKgs,
crop.isActive AS isActive, crop.doc AS doc, farm.lastModified AS dataLastUpdated,
cropSummary.feedLastUpdatedAt AS feedLastUpdated, cropSummary.nettingLastUpdatedAt AS nettingLastUpdated,
cropSummary.abw AS abw, cropSummary.awg AS awg, crop.riskType AS riskStatus,
cropSummary.totalHarvest AS harvest, aascore.totalScore AS AAScore, dhs.totalScore AS DHScore,
cropSummary.totalOrderedFeed AS totalOrderedFeed, cropSummary.yesterdayFeedBiomass AS yesterdayFeedBiomass,
cropSummary.tcfBiomass AS tcfBiomass, cropSummary.harvestedDate AS harvestedDate"""

# Optimized constraint-only template for Gemini
CONSTRAINT_ONLY_TEMPLATE = """
You are an expert Neo4j Cypher constraint generator.

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

Generate ONLY the WHERE clause constraints for this query: "{user_query}"

Rules:
1. Return ONLY "WHERE condition1 AND condition2..." format
2. If no constraints needed, return empty string
3. Always include NOT NULL checks: "field IS NOT NULL AND field > value"
4. For farmer names: "(farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL AND (farmer.firstname = 'name' OR farmer.lastname = 'name'))"
5. For numeric string fields (acres, seed, pcc): "toFloat(field) IS NOT NULL AND toFloat(field) > value"
6. For dates: "field IS NOT NULL AND duration.between(field, date()).days > days" for duration queries
7. For aggregations, include: "farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL"
8. Don't interpret question words (who, what, when) as values
9. Quantities: 1k=1000, 1l/1m=1000000, 1b=1000000000
10. For 'fcr' or 'feed conversion ratio', ALWAYS use cropSummary.fcr and NEVER compute a ratio from totalFeed and totalHarvest
11. For crop fields, include: "pond.currentCrop = crop.id"

Examples:
Query: "farms with abw > 10"
Output: WHERE cropSummary.abw IS NOT NULL AND cropSummary.abw > 10 AND pond.currentCrop = crop.id

Query: "active crops"
Output: WHERE crop.isActive IS NOT NULL AND crop.isActive = true AND pond.currentCrop = crop.id

Query: "ponds with fcr below 1.3"
Output: WHERE cropSummary.fcr IS NOT NULL AND cropSummary.fcr < 1.3 AND pond.currentCrop = crop.id

Query: "harvested ponds with abw below 20 and fcr below 1.3"
Output: WHERE cropSummary.abw IS NOT NULL AND cropSummary.abw < 20 AND cropSummary.fcr IS NOT NULL AND cropSummary.fcr < 1.3 AND crop.harvestReason IS NOT NULL

Query: "all farms"
Output: 

User query: "{user_query}"
"""



