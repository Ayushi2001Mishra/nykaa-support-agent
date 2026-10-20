from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    """Task 9 - every crew response must conform to this."""
    answer: str = Field(..., min_length=1)
    grounded: bool
    sources: List[str] = Field(default_factory=list)
    order: Optional[Dict[str, Any]] = None
    escalate: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False


class Verdict(BaseModel):
    """Task 14 - Final-Editor's structured verdict."""
    approved: bool
    final_answer: str
    reason: str
