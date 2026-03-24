from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class EntryCreate(BaseModel):
    title: Optional[str] = None
    content: str
    entry_type: str


class EntryUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    entry_type: Optional[str] = None


class EntryResponse(BaseModel):
    id: int
    title: Optional[str]
    content: str
    entry_type: str
    created_at: str
    updated_at: str


class ReportSectionSuggestion(BaseModel):
    text: str
    source_entry_id: int
    confidence: float


class ReportSection(BaseModel):
    suggestions: List[ReportSectionSuggestion]
    selected: List[int] = []


class ReferenceMention(BaseModel):
    text: str
    entry_id: int


class Reference(BaseModel):
    type: str
    value: str
    mentions: List[ReferenceMention]


class ReportGenerateRequest(BaseModel):
    entry_ids: List[int]
    report_type: str


class ReportGenerateResponse(BaseModel):
    sections: Dict[str, Any]
    references: List[Reference]


class ReportCreate(BaseModel):
    title: str
    report_type: str
    content: Dict[str, Any]
    entry_ids: List[int]


class ReportResponse(BaseModel):
    id: int
    title: str
    report_type: str
    content: Dict[str, Any]
    entry_ids: List[int]
    created_at: str
