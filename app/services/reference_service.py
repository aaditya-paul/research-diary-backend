import re
import os
import asyncio
from typing import List, Dict, Any
from collections import defaultdict
import html2text
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from groq import Groq
    import httpx
    LLMS_AVAILABLE = True
except ImportError:
    LLMS_AVAILABLE = False


class ReferenceExtractor:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        
        self.gemini_client = None
        if self.gemini_api_key and LLMS_AVAILABLE:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        self.use_llm = LLMS_AVAILABLE and (self.gemini_api_key or self.groq_api_key)
        
        # Fallback patterns (used only if LLM is unavailable)
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
        if self.use_llm:
            return self._extract_references_with_llm(entries)
        else:
            return self._extract_references_with_regex(entries)

    def _extract_references_with_llm(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract references using LLM-powered batch processing for flexibility with natural language."""
        if not entries:
            return []
        
        # Batch entries into chunks to manage token usage
        batch_size = 5
        all_references: Dict[str, Dict[str, Any]] = {}
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            batch_refs = self._batch_extract_references(batch)
            
            # Merge batch results
            for ref_key, ref_value in batch_refs.items():
                if ref_key not in all_references:
                    all_references[ref_key] = ref_value
                else:
                    # Merge mentions
                    all_references[ref_key]["mentions"].extend(ref_value.get("mentions", []))
        
        return list(all_references.values())

    def _batch_extract_references(self, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract references from a batch of entries in a single LLM call."""
        texts_with_ids = []
        for entry in batch:
            text = self._html_to_text(entry.get("content", ""))
            entry_id = entry.get("id")
            texts_with_ids.append({"id": entry_id, "text": text[:2000]})
        
        if not texts_with_ids:
            return {}
        
        prompt = self._build_extraction_prompt(texts_with_ids)
        
        extraction_result = self._call_llm_for_extraction(prompt)
        if not extraction_result:
            return {}
        
        return self._parse_llm_extraction(extraction_result, texts_with_ids)

    def _build_extraction_prompt(self, texts_with_ids: List[Dict[str, Any]]) -> str:
        entries_str = ""
        for item in texts_with_ids:
            entries_str += f"\n[Entry ID: {item['id']}]\n{item['text']}\n"
        
        return (
            "Extract all citations, references, sources, and academic/technical mentions from the following diary entries. "
            "Look for:\n"
            "1. Author-year citations (e.g., 'Smith, 2021' or 'Smith & Jones 2020')\n"
            "2. Paper/article titles (especially in quotes or after 'titled', 'on', 'about')\n"
            "3. Dataset names (e.g., 'Allen Brain Observatory', 'CRCNS datasets')\n"
            "4. URLs and their context\n"
            "5. Code repositories and project names (e.g., GitHub repos)\n"
            "6. Tools, frameworks, libraries mentioned (e.g., 'PyTorch', 'TensorFlow')\n"
            "7. AI systems mentioned (ChatGPT, Claude, etc.)\n"
            "8. Any book or publication names\n\n"
            "For each reference found, output ONLY in this exact format (one per line):\n"
            "[ENTRY_ID|TYPE|VALUE|QUOTE]\n\n"
            "Where:\n"
            "- ENTRY_ID: the ID of the entry\n"
            "- TYPE: one of: paper, dataset, url, repository, tool, ai, publication, conference\n"
            "- VALUE: the reference name or citation (e.g., 'Lottem & Azouz 2011', 'Allen Brain Observatory')\n"
            "- QUOTE: a short quote or context where this reference appears (40-100 chars)\n\n"
            "Do not include made-up references. Only extract what is explicitly mentioned.\n"
            "Output ONLY the [ENTRY_ID|TYPE|VALUE|QUOTE] lines, nothing else.\n\n"
            "Entries to process:" + entries_str
        )

    def _call_llm_for_extraction(self, prompt: str) -> str:
        """Call LLM with fallback support."""
        # Try Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(temperature=0.3),
                )
                if response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"Gemini reference extraction failed: {e}")
        
        # Try Groq
        if self.groq_api_key:
            try:
                client = Groq(api_key=self.groq_api_key)
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                    temperature=0.3,
                )
                if response.choices:
                    return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Groq reference extraction failed: {e}")
        
        # Try Ollama (without async since this is called from sync context)
        try:
            response = httpx.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=30.0,
            )
            if response.json().get("response"):
                return response.json()["response"].strip()
        except Exception as e:
            print(f"Ollama reference extraction failed: {e}")
        
        return ""

    def _parse_llm_extraction(
        self, extraction_result: str, texts_with_ids: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse structured LLM output into reference objects."""
        references: Dict[str, Dict[str, Any]] = {}
        
        lines = extraction_result.split("\n")
        for line in lines:
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            
            try:
                # Parse [ENTRY_ID|TYPE|VALUE|QUOTE]
                if not line.endswith("]"):
                    continue
                parts = line[1:-1].split("|")
                if len(parts) < 3:
                    continue
                
                entry_id = int(parts[0])
                ref_type = parts[1].strip().lower()
                ref_value = parts[2].strip()
                ref_quote = parts[3].strip() if len(parts) > 3 else ref_value[:100]
                
                # Validate type
                if ref_type not in ["paper", "dataset", "url", "repository", "tool", "ai", "publication", "conference"]:
                    ref_type = "source"
                
                if not ref_value:
                    continue
                
                # Create stable key for merging duplicates
                ref_key = f"{ref_type}:{ref_value.lower()}"
                
                if ref_key not in references:
                    references[ref_key] = {
                        "type": ref_type,
                        "value": ref_value,
                        "mentions": [],
                    }
                
                references[ref_key]["mentions"].append(
                    {"text": ref_quote, "entry_id": entry_id}
                )
            except (ValueError, IndexError):
                continue
        
        return references

    def _extract_references_with_regex(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback regex-based extraction when LLM is unavailable."""
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


def build_numeric_citation_registry(references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a stable numeric citation registry for inline [n] references."""
    registry: List[Dict[str, Any]] = []

    for idx, ref in enumerate(references, start=1):
        mentions = ref.get("mentions", []) or []
        entry_ids = sorted(
            {
                mention.get("entry_id")
                for mention in mentions
                if isinstance(mention, dict) and mention.get("entry_id") is not None
            }
        )
        registry.append(
            {
                "id": idx,
                "type": ref.get("type", "source"),
                "value": ref.get("value", "Unknown Source"),
                "entry_ids": entry_ids,
            }
        )

    return registry
