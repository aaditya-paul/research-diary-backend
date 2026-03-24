from typing import Dict, Any
import json
from datetime import datetime


class ExportService:
    def to_markdown(self, report_data: Dict[str, Any]) -> str:
        content = report_data.get("content", {})
        title = report_data.get("title", "Untitled Report")
        report_type = report_data.get("report_type", "project")
        
        md = f"# {title}\n\n"
        
        if report_type == "project":
            sections_order = ["hypothesis", "methodology", "findings", "conclusions"]
            
            for section in sections_order:
                section_data = content.get(section, {})
                selected_texts = section_data.get("selected_texts", [])
                
                if section == "hypothesis":
                    md += "## Hypothesis\n\n"
                elif section == "methodology":
                    md += "## Methodology\n\n"
                elif section == "findings":
                    md += "## Findings\n\n"
                elif section == "conclusions":
                    md += "## Conclusions\n\n"
                
                if selected_texts:
                    for text in selected_texts:
                        md += f"{text}\n\n"
                else:
                    md += "\n\n"
        
        elif report_type == "timeline":
            timeline_data = content.get("timeline", {})
            selected_texts = timeline_data.get("selected_texts", [])
            
            md += "## Timeline\n\n"
            
            if selected_texts:
                for i, entry in enumerate(selected_texts, 1):
                    md += f"{i}. {entry}\n\n"
            else:
                md += "\n\n"
        
        references = content.get("references", [])
        if references:
            md += "---\n\n## References\n\n"
            
            for ref in references:
                ref_type = ref.get("type", "Unknown")
                ref_value = ref.get("value", "Unknown Source")
                citations = ref.get("mentions", [])
                
                md += f"- **{ref_type}**: {ref_value}\n"
                
                for citation in citations:
                    citation_text = citation.get("text", "")
                    if citation_text:
                        md += f"  - \"{citation_text[:100]}...\"\n" if len(citation_text) > 100 else f"  - \"{citation_text}\"\n"
            
            md += "\n"
        
        return md

    def to_html(self, report_data: Dict[str, Any]) -> str:
        content = report_data.get("content", {})
        title = report_data.get("title", "Untitled Report")
        report_type = report_data.get("report_type", "project")
        created_at = report_data.get("created_at", "")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2a2a2a;
            margin-top: 30px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 25px;
        }}
        .references {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        .reference-item {{
            margin-bottom: 15px;
        }}
        .citation {{
            color: #555;
            font-style: italic;
            margin-left: 20px;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="meta">Report Type: {report_type.title()} | Created: {created_at}</p>
"""
        
        if report_type == "project":
            sections_order = ["hypothesis", "methodology", "findings", "conclusions"]
            section_titles = {
                "hypothesis": "Hypothesis",
                "methodology": "Methodology", 
                "findings": "Findings",
                "conclusions": "Conclusions"
            }
            
            for section in sections_order:
                section_data = content.get(section, {})
                selected_texts = section_data.get("selected_texts", [])
                
                if selected_texts:
                    html += f"    <h2>{section_titles.get(section, section.title())}</h2>\n"
                    html += "    <div class='section'>\n"
                    for text in selected_texts:
                        html += f"        <p>{text}</p>\n"
                    html += "    </div>\n"
        
        elif report_type == "timeline":
            timeline_data = content.get("timeline", {})
            selected_texts = timeline_data.get("selected_texts", [])
            
            if selected_texts:
                html += "    <h2>Timeline</h2>\n"
                html += "    <div class='section'>\n        <ul>\n"
                for entry in selected_texts:
                    html += f"            <li>{entry}</li>\n"
                html += "        </ul>\n    </div>\n"
        
        references = content.get("references", [])
        if references:
            html += "    <div class='references'>\n        <h2>References</h2>\n"
            
            for ref in references:
                ref_type = ref.get("type", "Unknown")
                ref_value = ref.get("value", "Unknown Source")
                citations = ref.get("mentions", [])
                
                html += f"        <div class='reference-item'>\n"
                html += f"            <strong>{ref_type}</strong>: {ref_value}<br>\n"
                
                for citation in citations:
                    citation_text = citation.get("text", "")
                    if citation_text:
                        display_text = citation_text[:100] + "..." if len(citation_text) > 100 else citation_text
                        html += f"            <span class='citation'>\"{display_text}\"</span><br>\n"
                
                html += "        </div>\n"
            
            html += "    </div>\n"
        
        html += """</body>
</html>"""
        
        return html

    def to_pdf(self, report_data: Dict[str, Any]) -> bytes:
        html_content = self.to_html(report_data)
        
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=html_content).write_pdf()
            return pdf_bytes
        except Exception as e:
            raise Exception(f"PDF generation failed: {str(e)}")


export_service = ExportService()
