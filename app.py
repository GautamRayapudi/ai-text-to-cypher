import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from neo4j import GraphDatabase
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import difflib
import logging
import time
import asyncio
import uvloop
import torch
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@dataclass
class CacheEntry:
    """Enhanced cache entry without pagination"""
    cypher_query: str
    total_count: int
    timestamp: datetime
    results: List[Dict]

    def is_expired(self, ttl_minutes: int = 5) -> bool:
        return datetime.now() - self.timestamp > timedelta(minutes=ttl_minutes)

class EnhancedQueryCache:
    """Enhanced caching system without pagination support"""
    def __init__(self, max_entries: int = 100, ttl_minutes: int = 5):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_entries = max_entries
        self.ttl_minutes = ttl_minutes

    def _generate_cache_key(self, query: str, similarity_threshold: float) -> str:
        """Generate cache key for the base query"""
        normalized_query = query.strip().lower()
        key_string = f"{normalized_query}:{similarity_threshold}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cypher_query(self, query: str, similarity_threshold: float) -> Optional[str]:
        """Get cached Cypher query if available and not expired"""
        cache_key = self._generate_cache_key(query, similarity_threshold)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired(self.ttl_minutes):
                logger.info("Retrieved Cypher query from cache")
                return entry.cypher_query
            else:
                del self.cache[cache_key]
                logger.info("Removed expired cache entry")
        return None

    def get_cached_results(self, query: str, similarity_threshold: float) -> Optional[Tuple[List[Dict], int]]:
        """Get cached full results"""
        cache_key = self._generate_cache_key(query, similarity_threshold)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired(self.ttl_minutes):
                logger.info("Retrieved cached results")
                return entry.results, entry.total_count
            else:
                del self.cache[cache_key]
        return None

    def cache_query_and_results(self, query: str, similarity_threshold: float, cypher_query: str, total_count: int, results: List[Dict]):
        """Cache both query and full results"""
        cache_key = self._generate_cache_key(query, similarity_threshold)
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        entry = CacheEntry(
            cypher_query=cypher_query,
            total_count=total_count,
            timestamp=datetime.now(),
            results=results
        )
        self.cache[cache_key] = entry
        logger.info("Cached query and results")

    def clear_expired(self):
        """Remove all expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items() if entry.is_expired(self.ttl_minutes)
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

class CustomSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB using SentenceTransformer"""
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.device = device
        self._name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, device=self.device).tolist()

    def name(self) -> str:
        return self._name

class FieldMapping:
    """Handles field mappings, aliases, and improved fuzzy matching"""
    def __init__(self):
        start_time = time.time()
        self.field_mappings = {
            # Farmer fields
            "farmer_name": {
                "cypher_field": "frm.firstname, frm.lastname, frm.farmer_name, frm.fullName",
                "aliases": ["farmer name", "farmer", "grower", "owner", "farmer_name", "full_name", "first_name", "last_name"],
                "type": "text",
                "description": "Name of the farmer/grower"
            },
            "farmer_id": {
                "cypher_field": "frm.id",
                "aliases": ["farmer id", "farmer_id", "grower id", "owner id"],
                "type": "text",
                "description": "Unique identifier for farmer"
            },
            # Farm fields
            "farm_name": {
                "cypher_field": "f.farm_name",
                "aliases": ["farm name", "farm", "farm_name", "property", "location", "farmer farm", "farmer property"],
                "type": "text",
                "description": "Name of the farm"
            },
            "farm_id": {
                "cypher_field": "f.id",
                "aliases": ["farm id", "farm_id", "property id", "farmer farm id"],
                "type": "text",
                "description": "Unique identifier for farm"
            },
            "section": {
                "cypher_field": "f.section",
                "aliases": ["section", "section id", "area", "zone"],
                "type": "text",
                "description": "Farm section identifier"
            },
            # Pond fields
            "pond_name": {
                "cypher_field": "p.name",
                "aliases": ["pond name", "pond", "tank", "pond_name"],
                "type": "text",
                "description": "Name of the pond"
            },
            "pond_id": {
                "cypher_field": "p.id",
                "aliases": ["pond id", "pond_id", "tank id"],
                "type": "text",
                "description": "Unique identifier for pond"
            },
            "pond_type": {
                "cypher_field": "p.pondType",
                "aliases": ["pond type", "tank type", "pond_type", "type"],
                "type": "text",
                "description": "Type of pond (earthen, concrete, etc.)"
            },
            "acres": {
                "cypher_field": "toFloat(p.acres)",
                "aliases": ["acres", "area", "size", "acreage", "acre", "acr"],
                "type": "numeric",
                "description": "Size of pond in acres"
            },
            "pcc": {
                "cypher_field": "toFloat(p.bearing_capacity)",
                "aliases": ["pcc", "bearing capacity", "capacity", "pond carrying capacity"],
                "type": "numeric",
                "description": "Pond carrying capacity"
            },
            # Crop/Cycle fields
            "stocking_date": {
                "cypher_field": "c.stockingDate",
                "aliases": ["stocking date", "stock date", "stocked on", "stocking_date", "date stocked"],
                "type": "date",
                "description": "Date when crop was stocked"
            },
            "seed_quantity": {
                "cypher_field": "toFloat(c.seedQuantity)",
                "aliases": ["seed", "seeds", "seed quantity", "stocking density", "seed density"],
                "type": "numeric",
                "description": "Number of seeds stocked"
            },
            "doc": {
                "cypher_field": "c.doc",
                "aliases": ["doc", "days of culture", "culture days", "age"],
                "type": "numeric",
                "description": "Days of culture"
            },
            "pond_level": {
                "cypher_field": "c.pondLevel",
                "aliases": ["pond level", "water level", "level"],
                "type": "text",
                "description": "Current pond water level"
            },
            "risk_status": {
                "cypher_field": "c.riskType",
                "aliases": ["risk", "risk status", "risk type", "status"],
                "type": "text",
                "description": "Risk classification of the crop"
            },
            "is_active": {
                "cypher_field": "c.isActive",
                "aliases": ["active", "is active", "status", "live"],
                "type": "boolean",
                "description": "Whether the crop is currently active"
            },
            # Feed and Growth fields
            "abw": {
                "cypher_field": "cs.abw",
                "aliases": ["abw", "average body weight", "body weight", "weight"],
                "type": "numeric",
                "description": "Average body weight in grams"
            },
            "awg": {
                "cypher_field": "cs.awg",
                "aliases": ["awg", "average weight gain", "weight gain", "growth rate"],
                "type": "numeric",
                "description": "Average weight gain"
            },
            "pc": {
                "cypher_field": "cs.count",
                "aliases": ["pc", "shrimp count", "count"],
                "type": "numeric",
                "description": "Current shrimp count"
            },
            "total_feed": {
                "cypher_field": "totalFeed",
                "aliases": ["total feed", "feed", "tcf", "total consumed feed"],
                "type": "numeric",
                "description": "Total feed consumed"
            },
            "harvest_date": {
                "cypher_field": "cs.harvestDate",
                "aliases": ["harvest date", "harvest_date", "harvested on", "harvested_date"],
                "type": "date",
                "description": "Date when crop was harvested"
            },
            "biomass": {
                "cypher_field": "cs.tcfBiomass",
                "aliases": ["biomass", "biomass_kg", "biomass_kilograms"],
                "type": "numeric",
                "description": "Biomass of the crop"
            },
            "fcr": {
                "cypher_field": "n.est_survival",
                "aliases": ["fcr", "feed conversion ratio", "conversion ratio"],
                "type": "numeric",
                "description": "Feed conversion ratio"
            },
            # Scores
            "aa_score": {
                "cypher_field": "aascore.totalScore",
                "aliases": ["aa score", "shs score", "shrimp health score", "aa", "score"],
                "type": "numeric",
                "description": "Shrimp Health score"
            },
            "dh_score": {
                "cypher_field": "dhs.totalScore",
                "aliases": ["dh score", "data health score", "health score"],
                "type": "numeric",
                "description": "Digital health score"
            },
            # Insurance and other fields
            "has_insurance": {
                "cypher_field": "hasInsurance",
                "aliases": ["insurance", "has insurance", "insured", "covered"],
                "type": "boolean",
                "description": "Whether the crop has insurance coverage"
            },
            "brooder": {
                "cypher_field": "s.brooder",
                "aliases": ["brooder", "hatchery", "nursery"],
                "type": "text",
                "description": "Brooder/hatchery information"
            }
        }
        self.field_terms = self._build_field_terms()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("GPU detected and will be used for embeddings.")
        else:
            self.device = "cpu"
            logger.info("No GPU detected; falling back to CPU.")
        self.embedding_fn = CustomSentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=self.device
        )
        try:
            self.collection = self.chroma_client.get_collection(
                name="field_mappings",
                embedding_function=self.embedding_fn
            )
            logger.info("Found existing field_mappings collection with compatible embedding function")
        except (chromadb.errors.NotFoundError, ValueError) as e:
            logger.info(f"Collection issue detected: {str(e)}")
            try:
                existing_collection = self.chroma_client.get_collection(name="field_mappings")
                logger.warning("Existing collection found with potentially incompatible embedding function. Deleting and recreating.")
                self.chroma_client.delete_collection(name="field_mappings")
            except chromadb.errors.NotFoundError:
                logger.info("No existing collection found, proceeding to create a new one.")
            self.collection = self.chroma_client.create_collection(
                name="field_mappings",
                embedding_function=self.embedding_fn
            )
            logger.info("Created new field_mappings collection")
            self._populate_vector_db()
        end_time = time.time()
        logger.info(f"FieldMapping initialization completed in {(end_time - start_time) * 1000:.2f} ms")
        self.field_patterns = [
            r'\b(?:farmer[_\s]?name|farmer|grower|owner)\b',
            r'\b(?:farm[_\s]?name|farm|property)\b',
            r'\b(?:pond[_\s]?name|pond|tank)\b',
            r'\b(?:acres?|acreage|area|size|acr)\b',
            r'\b(?:seed[_\s]?(?:quantity|density)|seeds?|stocking[_\s]?density)\b',
            r'\b(?:abw|average[_\s]?body[_\s]?weight|body[_\s]?weight|weight)\b',
            r'\b(?:awg|average[_\s]?weight[_\s]?gain|weight[_\s]?gain)\b',
            r'\b(?:doc|days[_\s]?on[_\s]?culture|culture[_\s]?days|age)\b',
            r'\b(?:pc|population[_\s]?count|count|survival)\b',
            r'\b(?:fcr|feed[_\s]?conversion[_\s]?ratio)\b',
            r'\b(?:pcc|pond[_\s]?carrying[_\s]?capacity|capacity)\b',
            r'\b(?:risk[_\s]?status|risk|status)\b',
            r'\b(?:pond[_\s]?level|water[_\s]?level|level)\b',
            r'\b(?:stocking[_\s]?date|stock[_\s]?date|date[_\s]?stocked)\b',
            r'\b(?:total[_\s]?feed|feed)\b',
            r'\b(?:aa[_\s]?score|aqua[_\s]?agriculture[_\s]?score|score)\b',
            r'\b(?:dh[_\s]?score|digital[_\s]?health[_\s]?score|health[_\s]?score)\b',
            r'\b(?:insurance|has[_\s]?insurance|insured|covered)\b',
            r'\b(?:biomass|total[_\s]?biomass|tcf[_\s]?biomass|total[_\s]?weight)\b',
            r'\b(?:harvest[_\s]?date|harvested[_\s]?date|harvested[_\s]?on|date[_\s]?harvested)\b',
        ]

    def _build_field_terms(self):
        terms = {}
        for field_name, mapping in self.field_mappings.items():
            terms[field_name.lower()] = field_name
            for alias in mapping["aliases"]:
                terms[alias.lower()] = field_name
        return terms

    def _populate_vector_db(self):
        documents = []
        metadatas = []
        ids = []
        seen_ids = set()
        for field_name, mapping in self.field_mappings.items():
            main_id = f"{field_name}_main"
            documents.append(f"{field_name}: {mapping['description']}")
            metadatas.append({
                "field_name": field_name,
                "cypher_field": mapping["cypher_field"],
                "type": mapping["type"],
                "is_alias": False
            })
            ids.append(main_id)
            seen_ids.add(main_id)
            for alias in mapping["aliases"]:
                alias_id = f"{field_name}_{alias.replace(' ', '_').replace('-', '_')}"
                if alias_id in seen_ids or alias.replace(' ', '_') == field_name:
                    continue
                documents.append(f"{alias}: {mapping['description']}")
                metadatas.append({
                    "field_name": field_name,
                    "cypher_field": mapping["cypher_field"],
                    "type": mapping["type"],
                    "is_alias": True,
                    "alias": alias
                })
                ids.append(alias_id)
                seen_ids.add(alias_id)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

    def extract_field_mentions(self, query: str) -> List[Dict]:
        field_mentions = []
        query_lower = query.lower()
        for pattern in self.field_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                matched_text = match.group().strip()
                start_pos = match.start()
                end_pos = match.end()
                best_field = self._match_to_field(matched_text)
                if best_field:
                    field_mentions.append({
                        'original_text': query[start_pos:end_pos],
                        'matched_field': best_field,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'confidence': 0.9
                    })
        words = re.findall(r'\b\w+(?:\s+\w+){0,2}\b', query_lower)
        for word_group in words:
            word_group = word_group.strip()
            if len(word_group) < 3:
                continue
            if word_group in self.field_terms:
                field_name = self.field_terms[word_group]
                pattern = re.compile(r'\b' + re.escape(word_group) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(query):
                    if not any(fm['start_pos'] == match.start() for fm in field_mentions):
                        field_mentions.append({
                            'original_text': match.group(),
                            'matched_field': field_name,
                            'start_pos': match.start(),
                            'end_pos': match.end(),
                            'confidence': 1.0
                        })
        field_mentions.sort(key=lambda x: (x['start_pos'], -x['confidence']))
        filtered_mentions = []
        for mention in field_mentions:
            overlap = False
            for existing in filtered_mentions:
                if (mention['start_pos'] < existing['end_pos'] and mention['end_pos'] > existing['start_pos']):
                    if mention['confidence'] > existing['confidence']:
                        filtered_mentions.remove(existing)
                    else:
                        overlap = True
                    break
            if not overlap:
                filtered_mentions.append(mention)
        return filtered_mentions

    def _match_to_field(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.field_terms:
            return self.field_terms[text_lower]
        best_match = None
        best_ratio = 0.7
        for term, field_name in self.field_terms.items():
            ratio = difflib.SequenceMatcher(None, text_lower, term).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = field_name
        return best_match

    def find_similar_fields(self, query_text: str, threshold: float = 0.7, top_k: int = 5) -> List[Dict]:
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            similar_fields = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1 - distance
                    if similarity >= threshold:
                        similar_fields.append({
                            'field_name': metadata['field_name'],
                            'cypher_field': metadata['cypher_field'],
                            'type': metadata['type'],
                            'similarity': similarity,
                            'matched_text': doc.split(':')[0],
                            'is_alias': metadata.get('is_alias', False)
                        })
            return similar_fields
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def fuzzy_match_field(self, user_input: str, threshold: float = 0.6) -> Optional[Dict]:
        best_match = None
        best_ratio = 0
        for field_name, mapping in self.field_mappings.items():
            ratio = difflib.SequenceMatcher(None, user_input.lower(), field_name.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = {
                    'field_name': field_name,
                    'cypher_field': mapping['cypher_field'],
                    'type': mapping['type'],
                    'similarity': ratio,
                    'matched_text': field_name
                }
            for alias in mapping["aliases"]:
                ratio = difflib.SequenceMatcher(None, user_input.lower(), alias.lower()).ratio()
                if ratio > best_ratio and ratio >= threshold:
                    best_ratio = ratio
                    best_match = {
                        'field_name': field_name,
                        'cypher_field': mapping['cypher_field'],
                        'type': mapping['type'],
                        'similarity': ratio,
                        'matched_text': alias
                    }
        return best_match

    def extract_and_correct_fields(self, user_query: str, threshold: float = 0.8) -> Tuple[str, Dict[str, str]]:
        normalized_query = user_query
        farmer_to_farm_mappings = {
            'farmers with': 'farms with',
            'farmer with': 'farm with',
            'farmers having': 'farms having',
            'farmer having': 'farm having',
            'farmers that': 'farms that',
            'farmer that': 'farm that',
            'farmers where': 'farms where',
            'farmer where': 'farm where',
            'show farmers': 'show farms',
            'get farmers': 'get farms',
            'find farmers': 'find farms',
            'list farmers': 'list farms'
        }
        query_lower = normalized_query.lower()
        for farmer_phrase, farm_phrase in farmer_to_farm_mappings.items():
            if farmer_phrase in query_lower:
                pattern = re.compile(re.escape(farmer_phrase), re.IGNORECASE)
                normalized_query = pattern.sub(farm_phrase, normalized_query)
                break
        field_mentions = self.extract_field_mentions(normalized_query)
        corrections = {}
        corrected_query = normalized_query
        if normalized_query != user_query:
            corrections['farmer_to_farm_normalization'] = 'Normalized farmer query to farm query'
        for mention in reversed(field_mentions):
            original_text = mention['original_text']
            matched_field = mention['matched_field']
            if (original_text.lower() != matched_field.lower() and
                mention['confidence'] >= threshold and
                len(original_text) > 2):
                if not re.match(r'^\d+(\.\d+)?[a-zA-Z]*$', original_text.strip()):
                    corrections[original_text] = matched_field
                    pattern = r'\b' + re.escape(original_text) + r'\b'
                    corrected_query = re.sub(
                        pattern,
                        matched_field,
                        corrected_query,
                        flags=re.IGNORECASE
                    )
        return corrected_query, corrections

class Neo4jConnection:
    def __init__(self):
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
            )
            with self.driver.session() as session:
                session.run("RETURN 1 AS ok")
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise e

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def execute_query(self, query: str) -> List[Dict]:
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = []
                for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, 'isoformat'):
                            record_dict[key] = value.isoformat()
                        elif isinstance(value, dict):
                            record_dict[key] = dict(value)
                        elif isinstance(value, list):
                            record_dict[key] = [dict(item) if hasattr(item, 'keys') else item for item in value]
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
            return records
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise e

class CypherQueryGenerator:
    def __init__(self, field_mapping: FieldMapping):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.field_mapping = field_mapping
        self.base_query_template = """MATCH(f:Farm)-[:HAS_POND]->(p:Pond) WITH f,p OPTIONAL MATCH(p)-[:HAS_CROP]->(c:Crop) OPTIONAL MATCH(c)-[:HAS_SUMMARY]->(cs:CropSummary) WITH p,c,f,cs where p.currentCrop=c.id AND c.stockedOn IS NOT NULL WITH f,p,c,cs OPTIONAL MATCH(frm:Farmer)-[:HAS_FRAM]->(f) WITH f, p, c, frm, cs WHERE frm.id = f.farmer_id OPTIONAL MATCH(c)-[:HAS_INSURANCE]-(ins:Insurance) WITH f, p, c, frm,cs, COUNT(ins.id) > 0 AS hasInsurance, CASE WHEN c.shiftedDate IS NOT NULL THEN true ELSE false END AS shiftingDone OPTIONAL MATCH(c)-[:HAS_AASCORE]-(aascore:AAScore)
WITH f, p, c, frm,cs, hasInsurance,aascore, shiftingDone OPTIONAL MATCH (c)-[:HAS_DHSCORE]->(dhs:DHScore)
WITH f, p, c, frm,cs, hasInsurance,aascore,dhs, shiftingDone
WITH f, p, c, frm,cs,hasInsurance,aascore,dhs, shiftingDone
OPTIONAL MATCH(c)-[:STOCKED]-(s:Stocking) WITH f,p,c,frm,cs,hasInsurance,CASE WHEN SIZE(collect(s)) > 0 THEN apoc.coll.sortNodes(collect(s), 'time')[0] ELSE NULL END AS s,aascore,dhs, shiftingDone
WITH f,p,c,frm,cs,hasInsurance,s,aascore,dhs, shiftingDone OPTIONAL MATCH(c)-[:DONE_NETTING]->(n:Netting) WITH f,p,c,frm,cs,hasInsurance,s,CASE WHEN SIZE(collect(n)) > 0 THEN apoc.coll.sortNodes(collect(n), 'time')[0] ELSE NULL END AS n,aascore,dhs, shiftingDone WITH f,p,c,frm,cs,hasInsurance,s,n,coalesce(cs.tcf, 0) AS totalFeed,aascore,dhs, shiftingDone
WITH p,c,f,frm,cs,hasInsurance,s,n,totalFeed,aascore,dhs, shiftingDone
WITH p,c,f,frm,cs,hasInsurance,s,n,totalFeed,aascore,dhs, shiftingDone
WITH p,c,f,frm,cs,hasInsurance,s,n,totalFeed,aascore,dhs, shiftingDone
OPTIONAL MATCH(c)-[:DONE_FEEDING]->(feeding:Feeding) WITH f,p,c,frm,s,n,cs,totalFeed,aascore,dhs, toInteger(coalesce(cs.smartscaleTCF,0)) AS smartScaleKgs,hasInsurance, shiftingDone OPTIONAL MATCH(c)-[:DONE_HARVEST]->(hr:Harvest) WITH f, p, c,n,totalFeed,aascore,dhs, frm,s,cs, smartScaleKgs,hasInsurance, shiftingDone, COLLECT(DISTINCT CASE WHEN hr IS NOT NULL THEN { id: hr.id, biomass: hr.biomass, kgs: hr.kgs, presentStocking: hr.presentStocking, count: hr.count, harvestType: hr.harvestType } ELSE NULL END) AS harvest OPTIONAL MATCH(f)-[:HAS_SECTION]-(cyl:Cycle) WITH f,p,c,frm,n,totalFeed,smartScaleKgs,harvest,s,cs,aascore,dhs,hasInsurance, shiftingDone
WITH p,c,n,totalFeed,f,smartScaleKgs,frm,cs, harvest,hasInsurance,s, shiftingDone, aascore,dhs
WITH p,c,n,totalFeed,f,smartScaleKgs,frm,cs, harvest,s,hasInsurance,aascore,dhs, shiftingDone order by n.time desc WITH head(collect(frm)) as frm,p,c,f,cs,smartScaleKgs, harvest,n,s,hasInsurance,aascore,dhs,totalFeed, shiftingDone
{constraints}
WITH DISTINCT s.brooder AS brooder, s.hatchery AS hatcheryName, s.mortalityReason AS harvestReason, hasInsurance, shiftingDone, n.est_survival AS FCR, CASE WHEN frm.firstname IS NOT NULL AND frm.lastname IS NOT NULL AND frm.firstname <> "" AND frm.lastname <> "" THEN frm.firstname + ' ' + frm.lastname WHEN frm.farmer_name IS NOT NULL AND frm.farmer_name <> "" THEN frm.farmer_name WHEN frm.fullName IS NOT NULL AND frm.fullName <> "" THEN frm.fullName ELSE '' END as farmerName, f.id as farmId, f.farm_name as farmName, f.section as sectionId, p.id as pondId, p.name as pondName, p.pondType as pondType, p.acres as acres, coalesce(toFloat(p.bearing_capacity), 0) AS PCC, c.stockingDate as stockingDate, toFloat(c.seedQuantity) as seed, coalesce(c.pondLevel, "") AS pondLevel, totalFeed, smartScaleKgs, c.isActive as isActive, coalesce(c.doc,0) AS DOC, f.lastModified as dataLastUpdated, f.feedingLastModified as feedLastUpdated, f.nettingLastModified as neetingLastUpdated, cs.count as PC, cs.abw as abw, cs.awg as awg, cs.yesterdayFeedBiomass AS yesterdayFeedBiomass, cs.tcfBiomass AS tcfBiomass, cs.maxFeedBiomass as maxFeedBiomass, coalesce(cs.totalPurchasedFeed, 0) AS totalOrderedFeed, c.riskType as riskStatus, harvest, { pondFlag:aascore.pondFlag, score:aascore.totalScore, flag:apoc.convert.fromJsonMap(aascore.textColor) } AS AAScore, { weeklyHpScore:dhs.weeklyHpScore, weeklyWaterTestScore:dhs.weeklyWaterTestScore, weeklyGutScore:dhs.weeklyGutScore, checkTrayCoverageScore:dhs.checkTrayCoverageScore, weeklyNettingScore:dhs.weeklyNettingScore, dailyFeedScore:dhs.dailyFeedScore, stockingScore:dhs.stockingScore, smartScaleScore:dhs.smartScaleScore, totalScore:dhs.totalScore, category:dhs.category } AS DHScore
RETURN DISTINCT brooder,hatcheryName,harvestReason,hasInsurance, shiftingDone,FCR, farmerName,farmId,farmName,sectionId,pondId,pondName,acres,PCC,stockingDate, seed,pondLevel,totalFeed,smartScaleKgs,isActive,DOC,dataLastUpdated,feedLastUpdated,neetingLastUpdated, PC,abw,awg,riskStatus,harvest,AAScore,DHScore,totalOrderedFeed,yesterdayFeedBiomass,tcfBiomass,pondType"""

        self.constraints_prompt = """ You are an expert Neo4j Cypher constraints generator with access to field mappings and aliases. 
Available field mappings: {field_mappings} 
Schema: 
(Farm)-[:HAS_POND]->(Pond)-[:HAS_CROP]->(Crop) 
(Crop)-[:HAS_SUMMARY]->(CropSummary) 
(Farmer)-[:HAS_FRAM]->(Farm) 
(Crop)-[:HAS_AASCORE]->(AAScore) 
(Crop)-[:HAS_DHSCORE]->(DHScore) 
(Farm)-[:HAS_SECTION]-(Cycle) 
(Cycle)-[:STOCKED]-(CycleSummary) 
(Crop)-[:HAS_INSURANCE]-(Insurance) 
(Crop)-[:STOCKED]-(Stocking) 
(Crop)-[:DONE_NETTING]->(Netting) 
(Crop)-[:DONE_FEEDING]->(Feeding) 
(Crop)-[:DONE_HARVEST]->(Harvest) 
(Crop)-[:DAY_FEED]->(DayFeed) 
IMPORTANT FIELD CORRECTIONS: 
- For harvest date queries: Use cs.harvestedDate (from CropSummary) instead of harvest collection 
- For biomass queries: Use cs.tcfBiomass (from CropSummary) instead of harvest collection 
- The harvest field in the query is a COLLECTION, not a single object - do not use harvest.property syntax 
- When user asks about "farmers", treat it as "farms" since we're querying farm data that includes farmer information 
- Farmer data is included in results via the farmerName field, but filtering/grouping should be done on farm entities 
User query: "{user_query}" 
CRITICAL CYPHER SYNTAX RULES: 
1. Generate ONLY WHERE clause conditions, separated by AND 
2. ALWAYS include NOT NULL checks for fields used in filtering 
3. Use the exact cypher_field from the field mappings above 
4. For numerical string fields, use appropriate toFloat() conversions 
5. For date fields, use duration.between() for relative dates 
6. Quantities: 1k = 1000, 1l = 100000, 1m = 1000000, 1b = 1000000000 
7. Map user terms to the correct field names using the aliases provided 
8. Interpret arithmetic operations in the query (less than, <, greater than, >, less than or equal to/atmost, <=, greater than or equal to/atleast, >= etc.) 
9. Identify decimal values and quantities correctly 
10. For harvest date/harvested ponds, biomass related queries, use cs.harvestedDate and cs.tcfBiomass, NOT the harvest collection properties
11. If the user asks about pond type, use p.pondType and there are 3 kinds of pondTypes: "Mother", "Nursery" & "Growout".
   For ex: User asks about mother ponds → p.pondType IS NOT NULL AND p.pondType = "Mother"
   For ex: User asks about nursery ponds → p.pondType IS NOT NULL AND p.pondType = "Nursery"
   For ex: User asks about growout ponds → p.pondType IS NOT NULL AND p.pondType = "Growout"
12. The user may specify the following in the user query: RMS, Regular, WSSV,Others, EHP, Vibriosis, Survival Loss, DO. These are the harvestReasons. Use s.mortalityReason to filter harvestReason.
   For ex: show wssv harvested ponds → s.mortalityReason IS NOT NULL AND cs.harvestedDate IS NOT NULL AND s.mortalityReason = "WSSV"
Generate ONLY the WHERE conditions (without the WHERE keyword). If no filtering needed, return empty string. 
Examples: 
- "farms with more than 2 acres" → toFloat(p.acres) IS NOT NULL AND toFloat(p.acres) > 2.0 
- "ABW greater than 30" → cs.abw IS NOT NULL AND cs.abw > 30 
- "acres less than 1" → toFloat(p.acres) IS NOT NULL AND toFloat(p.acres) < 1.0 
- "seed more than 1.5L" → toFloat(c.seedQuantity) IS NOT NULL AND toFloat(c.seedQuantity) > 150000 
- "crops stocked in last 30 days" → c.stockingDate IS NOT NULL AND duration.between(c.stockingDate, date()).days <= 30 
- "biomass greater than 1000 kg" → cs.tcfBiomass IS NOT NULL AND cs.tcfBiomass > 1000.0 """

    def get_field_mappings_context(self) -> str:
        context = ""
        for field_name, mapping in self.field_mapping.field_mappings.items():
            aliases_str = ", ".join(mapping["aliases"])
            context += f"- {field_name}: {mapping['cypher_field']} (aliases: {aliases_str})\n"
        return context

    def generate_constraints(self, user_query: str) -> str:
        try:
            field_mappings_context = self.get_field_mappings_context()
            prompt = self.constraints_prompt.replace("{user_query}", user_query)
            prompt = prompt.replace("{field_mappings}", field_mappings_context)
            response = self.model.generate_content(prompt)
            constraints = response.text.strip()
            if "```" in constraints:
                constraints = constraints.split("```")[1].split("```")[0].strip()
            if constraints.upper().startswith("WHERE "):
                constraints = constraints[6:]
            return constraints
        except Exception as e:
            logger.error(f"Error generating constraints: {e}")
            raise ValueError(f"Failed to generate constraints: {str(e)}")

    def generate_cypher(self, user_query: str) -> str:
        try:
            constraints = self.generate_constraints(user_query)
            if constraints.strip():
                constraints_section = f"WHERE {constraints}"
            else:
                constraints_section = ""
            cypher_query = self.base_query_template.replace("{constraints}", constraints_section)
            return cypher_query
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            raise ValueError(f"Failed to generate query: {str(e)}")

class DataFormatter:
    @staticmethod
    def format_date(date_value):
        if date_value is None:
            return None
        if hasattr(date_value, 'iso_format'):
            return date_value.iso_format()
        elif isinstance(date_value, str):
            try:
                dt = datetime.fromisoformat(date_value.replace('T', ' ').replace('Z', ''))
                return dt.strftime("%d-%m-%Y %H:%M:%S")
            except:
                return date_value
        return str(date_value)

    @staticmethod
    def format_response(raw_data: List[Dict], total_count: int) -> Dict[str, Any]:
        formatted_data = []
        for record in raw_data:
            formatted_record = {
                "abw": record.get("abw"),
                "acres": record.get("acres"),
                "awg": record.get("awg"),
                "cycleId": None,
                "cycleName": None,
                "dataLastUpdated": DataFormatter.format_date(record.get("dataLastUpdated")),
                "farmerName": record.get("farmerName", ""),
                "farmId": record.get("farmId"),
                "farmName": record.get("farmName"),
                "sectionId": record.get("sectionId"),
                "pondId": record.get("pondId"),
                "pondName": record.get("pondName"),
                "stockingDate": DataFormatter.format_date(record.get("stockingDate")),
                "seed": record.get("seed"),
                "totalFeed": record.get("totalFeed"),
                "totalOrderedFeed": record.get("totalOrderedFeed"),
                "PCC": record.get("PCC"),
                "smartScaleKgs": record.get("smartScaleKgs"),
                "isActive": record.get("isActive"),
                "hasInsurance": record.get("hasInsurance"),
                "DOC": record.get("DOC"),
                "pondLevel": record.get("pondLevel", ""),
                "feedLastUpdated": DataFormatter.format_date(record.get("feedLastUpdated")),
                "neetingLastUpdated": DataFormatter.format_date(record.get("neetingLastUpdated")),
                "PC": record.get("PC"),
                "riskStatus": record.get("riskStatus", ""),
                "AAScore": record.get("AAScore", {}),
                "harvestReason": record.get("harvestReason", ""),
                "harvest": record.get("harvest", []),
                "DHScore": record.get("DHScore", {}),
                "brooder": record.get("brooder", ""),
                "hatcheryName": record.get("hatcheryName", ""),
                "pondType": record.get("pondType"),
                "yesterdayFeedBiomass": record.get("yesterdayFeedBiomass"),
                "tcfBiomass": record.get("tcfBiomass"),
                "FCR": record.get("FCR"),
                "shiftingDone": record.get("shiftingDone")
            }
            formatted_data.append(formatted_record)
        return {
            "data": {
                "getFarmPondData": {
                    "totalCount": total_count,
                    "data": formatted_data
                }
            }
        }

@st.cache_resource
def init_app():
    neo4j_conn = Neo4jConnection()
    neo4j_conn.connect()
    field_mapping = FieldMapping()
    cypher_generator = CypherQueryGenerator(field_mapping)
    enhanced_cache = EnhancedQueryCache(max_entries=100, ttl_minutes=5)
    return neo4j_conn, field_mapping, cypher_generator, enhanced_cache

# Initialize Streamlit app
st.set_page_config(page_title="Farm Data Query", page_icon="🌾", layout="wide")

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
    }
    .query-info {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize components
neo4j_conn, field_mapping, cypher_generator, enhanced_cache = init_app()

# Sidebar with additional options
with st.sidebar:
    st.header("Query Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Adjust the similarity threshold for field matching"
    )
    st.markdown("---")
    st.subheader("Example Queries")
    st.markdown("""
    - Show farms with more than 2 acres
    - List ponds with ABW greater than 30
    - Find crops stocked in last 30 days
    - Get mother ponds with biomass > 1000 kg
    - Show WSSV harvested ponds
    """)

# Main content
st.title("Text to Cypher Query System")
st.markdown("Enter a natural language query to explore farm and pond data.")

# Query input
query = st.text_input(
    "Enter your query:",
    placeholder="e.g., farms with more than 2 acres",
    key="query_input"
)

# Display query processing status
query_status = st.empty()

if st.button("Search", key="search_button"):
    if not query.strip():
        st.error("Please enter a valid query.", icon="🚫")
    else:
        start_time = time.time()
        try:
            query_status.info("Processing query...", icon="⏳")
            enhanced_cache.clear_expired()
            field_start = time.time()
            corrected_query, corrections = field_mapping.extract_and_correct_fields(
                query, similarity_threshold
            )
            field_end = time.time()
            logger.info(f"Field extraction and correction completed in {(field_end - field_start) * 1000:.2f} ms")

            # Display corrections if any
            if corrections:
                st.markdown("### Query Corrections")
                st.markdown(f"Original Query: `{query}`")
                st.markdown(f"Corrected Query: `{corrected_query}`")
                for original, corrected in corrections.items():
                    st.markdown(f"- `{original}` → `{corrected}`")

            # Check cache
            cached_results = enhanced_cache.get_cached_results(query, similarity_threshold)
            if cached_results:
                results, total_count = cached_results
                formatted_response = DataFormatter.format_response(results, total_count)
                logger.info("Returned cached results")
            else:
                cypher_query = enhanced_cache.get_cypher_query(query, similarity_threshold)
                if not cypher_query:
                    cypher_start = time.time()
                    cypher_query = cypher_generator.generate_cypher(corrected_query)
                    cypher_end = time.time()
                    logger.info(f"Cypher query generation completed in {(cypher_end - cypher_start) * 1000:.2f} ms")

                with st.expander("View Generated Cypher Query"):
                    st.code(cypher_query, language="cypher")

                exec_start = time.time()
                raw_results = neo4j_conn.execute_query(cypher_query)
                exec_end = time.time()
                logger.info(f"Query execution completed in {(exec_end - exec_start) * 1000:.2f} ms")

                total_count = len(raw_results)
                enhanced_cache.cache_query_and_results(
                    query, similarity_threshold, cypher_query, total_count, raw_results
                )

                format_start = time.time()
                formatted_response = DataFormatter.format_response(raw_results, total_count)
                format_end = time.time()
                logger.info(f"Response formatting completed in {(format_end - format_start) * 1000:.2f} ms")

            df = pd.DataFrame(formatted_response["data"]["getFarmPondData"]["data"])
            st.markdown(f"### Results (Total: {total_count})")
            st.dataframe(df, use_container_width=True)

            end_time = time.time()
            query_status.success(f"Query processed in {(end_time - start_time):.2f} seconds!", icon="✅")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            query_status.error(f"Error: {str(e)}", icon="❌")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by Neo4j and Gemini AI")
