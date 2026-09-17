from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import config

logger = config.get_logger(__name__)


class DataLoadError(Exception):
    pass


@dataclass
class DataStatus:
    policies_loaded: bool
    claims_loaded: bool
    policies_rows: int
    claims_rows: int
    missing_columns: list[str]


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("%s is missing columns: %s", label, missing)
    return missing


def load_policies() -> Optional[pd.DataFrame]:
    if not config.POLICIES_CSV.exists():
        logger.error("policies.csv not found at %s", config.POLICIES_CSV)
        return None
    try:
        df = pd.read_csv(config.POLICIES_CSV, parse_dates=["start_date", "end_date"])
    except Exception as exc:
        logger.error("Failed to read policies.csv: %s", exc)
        return None
    if df.empty:
        logger.warning("policies.csv loaded but is empty.")
    _validate_columns(df, config.REQUIRED_POLICY_COLUMNS, "policies.csv")
    return df


def load_claims() -> Optional[pd.DataFrame]:
    if not config.CLAIMS_CSV.exists():
        logger.error("claims.csv not found at %s", config.CLAIMS_CSV)
        return None
    try:
        df = pd.read_csv(config.CLAIMS_CSV, parse_dates=["claim_date"])
    except Exception as exc:
        logger.error("Failed to read claims.csv: %s", exc)
        return None
    if df.empty:
        logger.warning("claims.csv loaded but is empty.")
    _validate_columns(df, config.REQUIRED_CLAIM_COLUMNS, "claims.csv")
    return df


def get_data_status(policies: Optional[pd.DataFrame], claims: Optional[pd.DataFrame]) -> DataStatus:
    missing = []
    if policies is not None:
        missing += _validate_columns(policies, config.REQUIRED_POLICY_COLUMNS, "policies.csv")
    if claims is not None:
        missing += _validate_columns(claims, config.REQUIRED_CLAIM_COLUMNS, "claims.csv")
    return DataStatus(
        policies_loaded=policies is not None,
        claims_loaded=claims is not None,
        policies_rows=0 if policies is None else len(policies),
        claims_rows=0 if claims is None else len(claims),
        missing_columns=missing,
    )


def get_policy(policies: pd.DataFrame, policy_id: str) -> Optional[dict]:
    match = policies[policies["policy_id"] == policy_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_policies_for_customer(policies: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    return policies[policies["customer_id"] == customer_id]


def get_claims_for_policy(claims: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    return claims[claims["policy_id"] == policy_id]


def get_claim(claims: pd.DataFrame, claim_id: str) -> Optional[dict]:
    match = claims[claims["claim_id"] == claim_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def search(policies: pd.DataFrame, claims: pd.DataFrame, query: str) -> dict:
    q = query.strip().lower()
    if not q:
        return {"policies": policies.iloc[0:0], "claims": claims.iloc[0:0]}

    policy_hits = policies[
        policies["policy_id"].str.lower().str.contains(q, na=False)
        | policies["customer_id"].str.lower().str.contains(q, na=False)
        | policies["policy_type"].str.lower().str.contains(q, na=False)
    ]
    claim_hits = claims[
        claims["claim_id"].str.lower().str.contains(q, na=False)
        | claims["policy_id"].str.lower().str.contains(q, na=False)
        | claims["status"].str.lower().str.contains(q, na=False)
    ]
    return {"policies": policy_hits, "claims": claim_hits}


def mask_policy_number(policy_id: str) -> str:
    if not policy_id or len(policy_id) < 4:
        return "****"
    return f"{policy_id[:3]}-****-{policy_id[-4:]}"
