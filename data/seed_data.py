"""
Programmatic seed-data generator for three heterogeneous source systems.

Each call to generate_batch() produces *new* documents with unique entity_ids
so collection counts grow on every ingestion run.
"""

import random
import uuid

DOMAINS = [
    "Customer Management", "Account Management", "Payments",
    "Marketing", "Merchant Network", "Risk & Fraud",
    "Compliance", "Loyalty & Rewards", "Lending",
    "Insurance", "Collections", "Servicing",
    "Digital Channels", "Analytics", "Data Governance",
]
ENTITY_PREFIXES = [
    "Customer", "Account", "Transaction", "Offer", "Merchant",
    "Card", "Payment", "Reward", "Claim", "Policy",
    "Application", "Credit_Line", "Statement", "Dispute",
    "Campaign", "Segment", "Score", "Alert", "Case",
    "Fee", "Interest", "Balance", "Limit", "Threshold",
    "Partner", "Vendor", "Channel", "Device", "Session",
    "Event", "Notification", "Preference", "Address", "Contact",
    "Beneficiary", "Mandate", "Standing_Order", "Direct_Debit",
    "FX_Rate", "Currency", "Country", "Region", "Branch",
]
ATTR_NAMES = [
    "id", "name", "type", "status", "code", "description", "amount",
    "date", "timestamp", "flag", "count", "rate", "score", "tier",
    "level", "category", "channel", "source", "target", "version",
    "priority", "weight", "duration", "frequency", "limit",
    "threshold", "min_value", "max_value", "avg_value", "total",
    "email", "phone", "address", "city", "state", "zip", "country",
]
ATTR_TYPES = ["string", "number", "date", "datetime", "boolean", "array"]
DB_NAMES = ["CARD_DW", "PAYMENTS_DW", "MKTG_DW", "RISK_DW", "LOYALTY_DW",
            "LENDING_DW", "INSURANCE_DW", "DIGITAL_DW", "ANALYTICS_DW"]
SCHEMA_NAMES = ["card_platform", "payments", "marketing", "risk_engine",
                "loyalty", "lending", "insurance", "digital", "analytics"]
STORAGE_FORMATS = ["Parquet", "Delta", "Iceberg", "Avro", "ORC"]
SQL_TYPES = ["VARCHAR(36)", "VARCHAR(200)", "VARCHAR(100)", "INTEGER",
             "BIGINT", "DECIMAL(15,2)", "DATE", "TIMESTAMP", "BOOLEAN"]
TAG_TYPES = ["PII", "SID", "Financial", "Business", "Technical", "Derived"]
SENSITIVITIES = ["High", "Medium", "Low"]
CLASSIFICATIONS = ["Confidential", "Restricted", "Internal", "Public"]
REGULATIONS = ["GDPR", "CCPA", "PCI-DSS", "SOX", "AML", "KYC", "GLBA",
               "FCRA", "CAN-SPAM", "HIPAA", "DORA"]
STEWARDS = [
    "Jane Rivera", "Michael Chen", "Sarah Patel", "David Kim", "Lisa Wong",
    "Robert Martinez", "Emily Johnson", "James Wilson", "Priya Sharma",
    "Carlos Rodriguez", "Anna Kowalski", "Mohammed Al-Rashid",
    "Sophie Dubois", "Thomas Anderson", "Yuki Tanaka",
]
PTB_STATUSES = ["Approved", "Pending", "Under Review", "Conditionally Approved"]


def _new_id() -> str:
    return f"ENT-{uuid.uuid4().hex[:8].upper()}"


def _gen_logical_model(entity_id: str | None = None) -> dict:
    eid = entity_id or _new_id()
    prefix = random.choice(ENTITY_PREFIXES)
    suffix = uuid.uuid4().hex[:4].upper()
    n = random.randint(3, 10)
    attrs, used = [], set()
    for _ in range(n):
        a = random.choice(ATTR_NAMES)
        while a in used:
            a = random.choice(ATTR_NAMES)
        used.add(a)
        attrs.append({"name": f"{prefix.lower()}_{a}",
                      "type": random.choice(ATTR_TYPES),
                      "description": f"{a.replace('_',' ').title()} for {prefix}"})
    return {"entity_id": eid, "entity_name": f"{prefix}_{suffix}",
            "domain": random.choice(DOMAINS),
            "description": f"{prefix} entity for {random.choice(DOMAINS).lower()}.",
            "attributes": attrs,
            "relationships": random.sample(ENTITY_PREFIXES, k=random.randint(1, 4)),
            "source_system": "enterprise_logical_model"}


def _gen_physical_schema(entity_id: str) -> dict:
    n = random.randint(3, 8)
    cols = [{"name": f"col_{uuid.uuid4().hex[:6]}", "data_type": random.choice(SQL_TYPES),
             "nullable": random.choice([True, False]),
             "primary_key": i == 0} for i in range(n)]
    return {"entity_id": entity_id,
            "schema_name": random.choice(SCHEMA_NAMES),
            "table_name": f"tbl_{uuid.uuid4().hex[:8]}",
            "database": random.choice(DB_NAMES),
            "columns": cols,
            "storage_format": random.choice(STORAGE_FORMATS),
            "partition_key": cols[0]["name"] if random.random() > 0.3 else None,
            "row_count_approx": random.randint(1000, 5_000_000_000),
            "source_system": "physical_schema_catalog"}


def _gen_governance_tags(entity_id: str, attrs: list[dict]) -> dict:
    tags = []
    for a in attrs:
        if random.random() > 0.4:
            tags.append({"field": a["name"], "tag": random.choice(TAG_TYPES),
                         "sensitivity": random.choice(SENSITIVITIES)})
    if not tags:
        tags.append({"field": attrs[0]["name"], "tag": "SID", "sensitivity": "Medium"})
    return {"entity_id": entity_id,
            "classification": random.choice(CLASSIFICATIONS),
            "data_steward": random.choice(STEWARDS),
            "retention_policy": f"{random.choice([3,5,7,10])} years",
            "tags": tags,
            "regulatory_frameworks": random.sample(REGULATIONS, k=random.randint(1, 4)),
            "ptb_status": random.choice(PTB_STATUSES),
            "source_system": "governance_catalog"}


def generate_batch(count: int = 1000):
    """Generate *count* new entities across all three source types.

    Returns (logical_models, physical_schemas, governance_tags) — each a list
    of dicts ready for MongoDB insertion.
    """
    lm_list, ps_list, gt_list = [], [], []
    for _ in range(count):
        lm = _gen_logical_model()
        lm_list.append(lm)
        ps_list.append(_gen_physical_schema(lm["entity_id"]))
        gt_list.append(_gen_governance_tags(lm["entity_id"], lm["attributes"]))
    return lm_list, ps_list, gt_list


def generate_updates(existing_ids: list[str], count: int = 50):
    """Generate updated docs for *count* randomly chosen existing entity_ids.

    Simulates schema evolution, new tags, changed stewards, etc.
    Returns (logical_models, physical_schemas, governance_tags).
    """
    if not existing_ids:
        return [], [], []
    ids = random.sample(existing_ids, k=min(count, len(existing_ids)))
    lm_list, ps_list, gt_list = [], [], []
    for eid in ids:
        lm = _gen_logical_model(entity_id=eid)
        lm_list.append(lm)
        ps_list.append(_gen_physical_schema(eid))
        gt_list.append(_gen_governance_tags(eid, lm["attributes"]))
    return lm_list, ps_list, gt_list