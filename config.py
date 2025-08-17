# config.py
FIELD_ALIASES = {
    "abw": ["average body weight", "avg body weight", "body weight"],
    "awg": ["average weight gain", "avg weight gain", "weight gain"], 
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

TEMPLATE_PROMPT = """
You are an expert Neo4j Cypher query generator.
Schema:
(Farm)-[:HAS_POND]->(Pond)-[:HAS_CROP]->(Crop)
(Crop)-[:HAS_SUMMARY]->(CropSummary)
(Farmer)-[:HAS_FRAM]->(Farm)
(Crop)-[:HAS_AASCORE]->(AAScore)
(Crop)-[:HAS_DHSCORE]->(DHScore)
(Farm)-[:HAS_SECTION]-(Cycle)
(Cycle)-[:STOCKED]-(CycleSummary)
(Crop)-[:HAS_INSURANCE]-(Insurance)

Important fields mapping:
- abw: cropSummary.abw
- awg: cropSummary.awg
- DHScore.totalScore: dhs.totalScore
- AAScore.totalScore: aascore.totalScore
- doc: crop.doc
- farmName: farm.farm_name
- farmerName: farmer.firstname OR farmer.lastname (check both fields)
- tcfBiomass: cropSummary.tcfBiomass
- yesterdayFeedBiomass: cropSummary.yesterdayFeedBiomass
- PCC: pond.bearing_capacity
- acres: pond.acres
- seed: crop.seedQuantity
- totalFeed: cropSummary.totalFeed
- smartScaleKgs: cropSummary.smartscaleTCF
- isActive: crop.isActive
- stockingDate: crop.stockedOn
- brooder: crop.brooder
- hatcheryName: crop.hatchery
- harvestReason: crop.harvestReason
- riskStatus: crop.riskType
- harvest: cropSummary.totalHarvest
- harvestedDate: cropSummary.harvestedDate

Base Cypher (use only when needed based on the query):
MATCH (farm:Farm)-[:HAS_POND]->(pond:Pond)-[:HAS_CROP]->(crop:Crop)
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
cropSummary.tcfBiomass AS tcfBiomass, cropSummary.harvestedDate AS harvestedDate

User query: "{user_query}"

IMPORTANT:
1. Place ALL constraints EXCLUSIVELY in the {constraints} section. Do NOT add any WHERE clauses in the MATCH statements or anywhere else in the query unless explicitly part of the {constraints} section.
2. Ensure the generated query does not repeat constraints in multiple WHERE clauses.
3. ALWAYS ensure that any field used in filtering (WHERE clause) or RETURN statement is checked for NOT NULL. For example, if filtering abw > 10, use "cropSummary.abw IS NOT NULL AND cropSummary.abw > 10". For RETURN, ensure fields like farmer.firstname are checked with IS NOT NULL in the WHERE clause.
4. When filtering by farmer names, always check BOTH farmer.firstname AND farmer.lastname using OR condition, and ensure they are NOT NULL. For example: (farmer.firstname = "Manikanta" OR farmer.lastname = "Manikanta") AND farmer.firstname IS NOT NULL AND farmer.lastname IS NOT NULL.
5. For fields stored as strings that require numerical comparisons (>, <, >=, <=), such as pond.acres, crop.seedQuantity, pond.bearing_capacity, use toFloat() to convert them to float. For example: toFloat(pond.acres) IS NOT NULL AND toFloat(pond.acres) > 2.0.
6. For date fields like crop.stockedOn and cropSummary.harvestedDate, assume they are stored as Neo4j date types (not strings). Use them directly in comparisons or duration calculations. For duration-based constraints (e.g., "last X days" or "older than X days"), use duration.between(date_field, date()).days. For example: crop.stockedOn IS NOT NULL AND duration.between(crop.stockedOn, date()).days > 150.
7. Strictly add the constraints in the {constraints} section and don't add anywhere else, and ensure each constraint includes a NOT NULL check.
8. For aggregation queries (e.g., COUNT, MAX, MIN), ensure the aggregated fields and fields in RETURN are not null in the WHERE clause.
9. Do NOT interpret question words like "who", "what", "when", "where", or "how" as field values or filter conditions. These are query indicators, not data values.
10. Use the base Cypher only when the query involves farm, pond, or crop relationships. For simple queries like counting farms per farmer, use a minimal MATCH pattern like MATCH (farmer:Farmer)-[:HAS_FRAM]->(farm:Farm).
11. If the query references crop-related fields (doc, seed, isActive, stockingDate, brooder, hatcheryName, harvestReason, riskStatus, harvest, harvestedDate), include "pond.currentCrop = crop.id" in the WHERE clause alongside other constraints.
12. Quantities: 1k = 1000, 1l = 1000000, 1m = 1000000, 1b = 1000000000
"""