import re
from typing import List, Dict, Any
from collections import defaultdict
import html2text


class ReferenceExtractor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
        
        self.ai_keywords = [
            'chatgpt', 'gpt-', 'openai', 'claude', 'gemini', 'bard', 'palm',
            'llama', 'gemma', 'mistral', 'ai assistant', 'ai model', 'language model',
            'i asked ai', 'asked chatgpt', 'asked claude', 'the ai said', 'ai suggested'
        ]
        
        self.site_platforms = [
            'wikipedia', 'youtube', 'reddit', 'github', 'stackoverflow',
            'medium', 'blog', 'forum', 'documentation', 'docs',
            'site', 'website', 'webpage', 'page'
        ]
        
        self.purchase_patterns = [
            r'(bought|purchased|ordered|got)\s+(.*?)\s+(from|at)\s+(\w+)',
            r'(from|at)\s+(amazon|ebay|aliexpress|walmart|target|best buy|flipkart)',
        ]
        
        self.source_attribution = [
            'i read', 'i saw', 'i found', 'according to', 'from the', 'research',
            'the paper', 'the book', 'the article', 'the source', 'mentioned',
            'stated that', 'claims that', 'says that'
        ]

    def _html_to_text(self, html_content: str) -> str:
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        text = converter.handle(html_content)
        return text.strip()

    def extract_references(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        references = defaultdict(lambda: {"type": "", "value": "", "mentions": []})
        
        ai_refs = defaultdict(list)
        url_refs = defaultdict(list)
        site_refs = defaultdict(list)
        purchase_refs = defaultdict(list)
        
        for entry in entries:
            text = self._html_to_text(entry.get("content", ""))
            entry_id = entry.get("id")
            
            url_matches = self.url_pattern.findall(text)
            for url in url_matches:
                domain = self._extract_domain(url)
                if domain:
                    url_refs[domain].append({
                        "text": url,
                        "entry_id": entry_id
                    })
            
            lower_text = text.lower()
            for ai_keyword in self.ai_keywords:
                if ai_keyword in lower_text:
                    ai_name = self._extract_ai_name(ai_keyword, text)
                    if ai_name:
                        context = self._get_context_around(text, ai_keyword)
                        ai_refs[ai_name].append({
                            "text": context,
                            "entry_id": entry_id
                        })
            
            for platform in self.site_platforms:
                pattern = rf'\b([a-zA-Z0-9-]+\s+{platform}|{platform}\s+[a-zA-Z0-9-]+|\b{platform}\b)'
                matches = re.findall(pattern, lower_text)
                for match in matches:
                    site_refs[match if isinstance(match, str) else match[0]].append({
                        "text": match if isinstance(match, str) else match[0],
                        "entry_id": entry_id
                    })
            
            for pattern in self.purchase_patterns:
                matches = re.findall(pattern, lower_text)
                for match in matches:
                    if len(match) >= 4:
                        purchase_refs[match[3]].append({
                            "text": f"{match[0]} {match[1]} {match[2]} {match[3]}",
                            "entry_id": entry_id
                        })
            
            for attr_phrase in self.source_attribution:
                if attr_phrase in lower_text:
                    context = self._get_context_around(text, attr_phrase)
                    if len(context) > 10:
                        site_refs["general_sources"].append({
                            "text": context[:200],
                            "entry_id": entry_id
                        })
            
            quoted = re.findall(r'"([^"]+)"|"([^"]+)"|' + r"‘([^’]+)’", text)
            for quote in quoted:
                quote_text = quote[0] or quote[1] or quote[2]
                if len(quote_text) > 10:
                    site_refs["quoted_materials"].append({
                        "text": quote_text,
                        "entry_id": entry_id
                    })

        result = []
        
        for ai_name, mentions in ai_refs.items():
            result.append({
                "type": "ai",
                "value": ai_name.title(),
                "mentions": mentions
            })
        
        for domain, mentions in url_refs.items():
            result.append({
                "type": "website",
                "value": domain,
                "mentions": mentions
            })
        
        for site, mentions in site_refs.items():
            if site not in ["general_sources", "quoted_materials"] and site != "site":
                result.append({
                    "type": "source",
                    "value": site.title() if len(site) > 3 else site,
                    "mentions": mentions
                })
        
        for vendor, mentions in purchase_refs.items():
            result.append({
                "type": "purchase",
                "value": vendor.title(),
                "mentions": mentions
            })

        return result

    def _extract_domain(self, url: str) -> str:
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            domain = match.group(1)
            return domain.split('.')[0] if '.' in domain else domain
        return ""

    def _extract_ai_name(self, keyword: str, text: str) -> str:
        lower_text = text.lower()
        
        ai_names = {
            'chatgpt': 'ChatGPT',
            'gpt-': 'GPT',
            'openai': 'OpenAI',
            'claude': 'Claude',
            'gemini': 'Gemini',
            'bard': 'Bard',
            'palm': 'PaLM',
            'llama': 'Llama',
            'gemma': 'Gemma',
            'mistral': 'Mistral'
        }
        
        for name_key, name_value in ai_names.items():
            if name_key in keyword:
                return name_value
            
        for name_key, name_value in ai_names.items():
            if name_key in lower_text:
                return name_value
        
        return "AI"

    def _get_context_around(self, text: str, keyword: str, context_length: int = 100) -> str:
        lower_text = text.lower()
        keyword_lower = keyword.lower()
        
        pos = lower_text.find(keyword_lower)
        if pos == -1:
            return text[:200]
        
        start = max(0, pos - context_length)
        end = min(len(text), pos + len(keyword) + context_length)
        
        context = text[start:end].strip()
        
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context


reference_extractor = ReferenceExtractor()
