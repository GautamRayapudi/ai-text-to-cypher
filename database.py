# database.py
from neo4j import GraphDatabase
from neo4j.time import DateTime as Neo4jDateTime, Date as Neo4jDate
from datetime import datetime, date
import re

def run_cypher(cypher_query: str, uri: str, user: str, password: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
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
    finally:
        driver.close()
    return records

def validate_cypher_query(cypher_query, valid_fields):
    invalid_fields = []
    for field in re.findall(r'\b\w+\.\w+\b', cypher_query):
        if field not in valid_fields and not field.startswith('toFloat'):
            invalid_fields.append(field)
    if invalid_fields:
        return False
    return True