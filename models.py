from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    grounded: bool = False
    escalate: bool = False


class SafetyCategory(str, Enum):
    FRAUD = "Fraud Prevention"
    HALLUCINATION = "Hallucination Prevention"
    OUTCOME_GUARANTEE = "Claim Integrity"
    LEGAL_MEDICAL = "Legal/Medical Safety"
    PRIVACY = "Privacy Protection"


class SafetyTestResult(BaseModel):
    test_id: str
    prompt: str
    expected_behavior: str
    actual_response: str
    passed: bool
    category: SafetyCategory


class IncidentType(str, Enum):
    ACCIDENT = "Accident"
    THEFT = "Theft"
    FIRE = "Fire"
    NATURAL_CALAMITY = "Natural Calamity"
    MEDICAL = "Medical"
    OTHER = "Other"


class FNOLClaim(BaseModel):
    policy_number: str = Field(min_length=3, max_length=30)
    incident_date: date
    incident_type: IncidentType
    location: str = Field(min_length=2, max_length=200)
    description: str = Field(min_length=10, max_length=2000)
    estimated_loss: float = Field(gt=0, le=100_000_000)
    contact_name: str = Field(min_length=2, max_length=100)
    contact_phone: str = Field(min_length=7, max_length=20)
    contact_email: str = Field(min_length=5, max_length=120)
    supporting_notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("incident_date")
    @classmethod
    def not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError("Incident date cannot be in the future.")
        return v

    @field_validator("contact_email")
    @classmethod
    def looks_like_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Contact email does not look valid.")
        return v


class FNOLSubmissionResult(BaseModel):
    claim_reference: str
    submitted_at: datetime
    status: str = "REPORTED"
