# database.py
from neo4j import GraphDatabase
from neo4j.time import DateTime as Neo4jDateTime, Date as Neo4jDate
from datetime import datetime, date
import re
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, password: str):
    """Create and cache a Neo4j driver to reuse across requests."""
    # Use modest pool + timeouts to avoid long hangs and reduce reconnect overhead
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_pool_size=10,
        connection_acquisition_timeout=15.0,
        connection_timeout=15.0,
        max_connection_lifetime=3600.0,
    )

def run_cypher(cypher_query: str, uri: str, user: str, password: str, timeout_seconds: float | None = None):
    driver = get_driver(uri, user, password)
    with driver.session() as session:
        run_kwargs = {"timeout": timeout_seconds} if timeout_seconds else {}
        result = session.run(cypher_query, **run_kwargs)
        records = []
        for r in result:
            record_dict = {}
            for key, value in dict(r).items():
                if isinstance(value, Neo4jDateTime):
                    record_dict[key] = datetime(
                        value.year, value.month, value.day,
                        value.hour, value.minute, value.second,
                        value.nanosecond // 1000
                    )
                elif isinstance(value, Neo4jDate):
                    record_dict[key] = date(value.year, value.month, value.day)
                elif value is None:
                    record_dict[key] = None
                else:
                    if isinstance(value, str) and value.strip():
                        try:
                            if '.' not in value and value.replace('-', '').isdigit():
                                record_dict[key] = int(value)
                            else:
                                float_val = float(value)
                                record_dict[key] = float_val
                        except (ValueError, TypeError):
                            record_dict[key] = str(value)
                    else:
                        record_dict[key] = value
            records.append(record_dict)
    return records

def validate_cypher_query(cypher_query, valid_fields):
    invalid_fields = []
    for field in re.findall(r'\b\w+\.\w+\b', cypher_query):
        if field not in valid_fields and not field.startswith('toFloat'):
            invalid_fields.append(field)
    if invalid_fields:
        return False
    return True
