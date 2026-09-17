from __future__ import annotations
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
POLICIES_CSV = DATA_DIR / "policies.csv"
CLAIMS_CSV = DATA_DIR / "claims.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEMO_MODE = len(OPENAI_API_KEY) == 0

DEMO_CUSTOMER_NAME = os.getenv("DEMO_CUSTOMER_NAME", "Customer")
DEMO_CUSTOMER_ID = os.getenv("DEMO_CUSTOMER_ID", "")

POLICY_COLUMNS = {
    "policy_id": "policy_id", "customer_id": "customer_id", "policy_type": "policy_type",
    "premium": "premium", "sum_assured": "sum_assured", "tenure_months": "tenure_months",
    "start_date": "start_date", "end_date": "end_date", "channel": "channel",
    "product_name": "product_name",
}
CLAIM_COLUMNS = {
    "claim_id": "claim_id", "policy_id": "policy_id", "customer_id": "customer_id",
    "claim_date": "claim_date", "claim_type": "claim_type", "claim_amount": "claim_amount",
    "settlement_amount": "settlement_amount", "status": "status", "fraud_flag": "fraud_flag",
    "days_to_settle": "days_to_settle",
}
REQUIRED_POLICY_COLUMNS = list(POLICY_COLUMNS.values())
REQUIRED_CLAIM_COLUMNS = list(CLAIM_COLUMNS.values())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
