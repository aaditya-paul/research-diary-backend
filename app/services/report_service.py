from typing import List, Dict, Any
from ..models.database import Entry, SessionLocal
from .embedding_service import embedding_service
from .reference_service import reference_extractor


PROJECT_SEARCH_QUERIES = {
    "hypothesis": """
        goals intentions hypothesis what I wanted to do purpose aim
        planned wanted to build wanted to create wanted to make
        intended objective dream hope
    """.strip(),
    "methodology": """
        how I approached method methodology research steps process
        procedure technique approach buying purchased acquired
        simulation testing experiment trial built created made
        learned studied researched read watched
    """.strip(),
    "findings": """
        discovered results observations data analysis findings
        found that learned that noticed observed saw that
        outcome conclusion outcome result
    """.strip(),
    "conclusions": """
        conclusion summary takeaway lesson learned what I got
        outcome final result reflection thoughts about
        overall overall thoughts overall conclusion
    """.strip()
}

TIMELINE_SEARCH_QUERIES = {
    "timeline": """
        what happened events activities daily day today
        then later after that next before that
        morning afternoon evening night
    """.strip()
}


class ReportService:
    def generate_report_suggestions(self, entry_ids: List[int], report_type: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            entries = db.query(Entry).filter(Entry.id.in_(entry_ids)).all()
            
            if not entries:
                return {"sections": {}, "references": []}
            
            entries_data = [
                {
                    "id": e.id,
                    "title": e.title,
                    "content": e.content,
                    "entry_type": e.entry_type,
                    "created_at": e.created_at.isoformat() if e.created_at else ""
                }
                for e in entries
            ]
            
            search_queries = PROJECT_SEARCH_QUERIES if report_type == "project" else TIMELINE_SEARCH_QUERIES
            
            sections = {}
            for section_name, query in search_queries.items():
                results = embedding_service.semantic_search(query, entry_ids, n_results=5)
                
                suggestions = []
                for result in results:
                    confidence = 1.0 - result.get("distance", 0.5)
                    suggestions.append({
                        "text": result["text"],
                        "source_entry_id": result["entry_id"],
                        "confidence": round(confidence, 3)
                    })
                
                sections[section_name] = {
                    "suggestions": suggestions,
                    "selected": []
                }
            
            references = reference_extractor.extract_references(entries_data)
            
            return {
                "sections": sections,
                "references": references
            }
        finally:
            db.close()

    def create_report(self, title: str, report_type: str, content: Dict[str, Any], entry_ids: List[int]) -> Dict[str, Any]:
        from ..models.database import Report
        from datetime import datetime
        import json
        
        db = SessionLocal()
        try:
            report = Report(
                title=title,
                report_type=report_type,
                content=content,
                entry_ids=json.dumps(entry_ids),
                created_at=datetime.utcnow()
            )
            db.add(report)
            db.commit()
            db.refresh(report)
            
            return report.to_dict()
        finally:
            db.close()

    def get_report(self, report_id: int) -> Dict[str, Any]:
        from ..models.database import Report
        
        db = SessionLocal()
        try:
            report = db.query(Report).filter(Report.id == report_id).first()
            if not report:
                return None
            return report.to_dict()
        finally:
            db.close()

    def get_all_reports(self) -> List[Dict[str, Any]]:
        from ..models.database import Report
        
        db = SessionLocal()
        try:
            reports = db.query(Report).order_by(Report.created_at.desc()).all()
            return [r.to_dict() for r in reports]
        finally:
            db.close()

    def delete_report(self, report_id: int) -> bool:
        from ..models.database import Report
        
        db = SessionLocal()
        try:
            report = db.query(Report).filter(Report.id == report_id).first()
            if report:
                db.delete(report)
                db.commit()
                return True
            return False
        finally:
            db.close()


report_service = ReportService()
