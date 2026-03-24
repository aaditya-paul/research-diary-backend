from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from ..models.database import get_db, Entry, Report
from ..models.schemas import (
    EntryCreate, EntryUpdate, EntryResponse,
    ReportGenerateRequest, ReportGenerateResponse,
    ReportCreate, ReportResponse
)
from ..services.embedding_service import embedding_service
from ..services.report_service import report_service
from ..services.export_service import export_service

router = APIRouter()


@router.post("/entries", response_model=EntryResponse)
def create_entry(entry: EntryCreate, db: Session = Depends(get_db)):
    db_entry = Entry(
        title=entry.title,
        content=entry.content,
        entry_type=entry.entry_type
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    try:
        embedding_service.generate_embeddings(
            entry_id=db_entry.id,
            html_content=db_entry.content,
            entry_type=db_entry.entry_type,
            date=db_entry.created_at.strftime("%Y-%m-%d") if db_entry.created_at else ""
        )
    except Exception as e:
        print(f"Embedding generation error: {e}")
    
    return db_entry.to_dict()


@router.get("/entries", response_model=List[EntryResponse])
def list_entries(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    entries = db.query(Entry).order_by(Entry.created_at.desc()).offset(skip).limit(limit).all()
    return [e.to_dict() for e in entries]


@router.get("/entries/{entry_id}", response_model=EntryResponse)
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(Entry).filter(Entry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry.to_dict()


@router.put("/entries/{entry_id}", response_model=EntryResponse)
def update_entry(entry_id: int, entry_update: EntryUpdate, db: Session = Depends(get_db)):
    entry = db.query(Entry).filter(Entry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    if entry_update.title is not None:
        entry.title = entry_update.title
    if entry_update.content is not None:
        entry.content = entry_update.content
    if entry_update.entry_type is not None:
        entry.entry_type = entry_update.entry_type
    
    entry.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(entry)
    
    try:
        embedding_service.generate_embeddings(
            entry_id=entry.id,
            html_content=entry.content,
            entry_type=entry.entry_type,
            date=entry.created_at.strftime("%Y-%m-%d") if entry.created_at else ""
        )
    except Exception as e:
        print(f"Embedding generation error: {e}")
    
    return entry.to_dict()


@router.delete("/entries/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(Entry).filter(Entry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    db.delete(entry)
    db.commit()
    
    return {"message": "Entry deleted successfully"}


@router.post("/report/generate", response_model=ReportGenerateResponse)
def generate_report_suggestions(request: ReportGenerateRequest):
    result = report_service.generate_report_suggestions(
        entry_ids=request.entry_ids,
        report_type=request.report_type
    )
    return result


@router.post("/report", response_model=ReportResponse)
def create_report(report: ReportCreate, db: Session = Depends(get_db)):
    created_report = report_service.create_report(
        title=report.title,
        report_type=report.report_type,
        content=report.content,
        entry_ids=report.entry_ids
    )
    return created_report


@router.get("/reports", response_model=List[ReportResponse])
def list_reports(db: Session = Depends(get_db)):
    reports = report_service.get_all_reports()
    return reports


@router.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: int):
    report = report_service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.delete("/reports/{report_id}")
def delete_report(report_id: int):
    success = report_service.delete_report(report_id)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report deleted successfully"}


@router.get("/export/markdown/{report_id}")
def export_markdown(report_id: int):
    report = report_service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    markdown_content = export_service.to_markdown(report)
    
    return {
        "content": markdown_content,
        "filename": f"{report['title'].replace(' ', '_')}.md"
    }


@router.get("/export/html/{report_id}")
def export_html(report_id: int):
    report = report_service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    html_content = export_service.to_html(report)
    
    return {
        "content": html_content,
        "filename": f"{report['title'].replace(' ', '_')}.html"
    }


@router.get("/export/pdf/{report_id}")
def export_pdf(report_id: int):
    report = report_service.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    try:
        pdf_bytes = export_service.to_pdf(report)
        import base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return {
            "content": pdf_base64,
            "filename": f"{report['title'].replace(' ', '_')}.pdf"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
