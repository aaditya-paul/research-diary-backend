import os
import re
from google import genai
from groq import Groq
import httpx
from typing import List, Dict, Any, Tuple

# Attempt to load from .env if present
from dotenv import load_dotenv
from .reference_service import reference_extractor, build_numeric_citation_registry
load_dotenv()

class LLMService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.min_words = int(os.getenv("REPORT_MIN_SECTION_WORDS", "250"))
        # Total attempts = 1 initial pass + max rewrites
        self.max_rewrites = int(os.getenv("REPORT_MAX_REWRITES", "1"))

        self.gemini_client = None
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

        self.section_profiles = {
            "hypothesis": {
                "purpose": "State the research aim, assumptions, scope constraints, and expected outcomes for the current iteration.",
                "must_include": ["aim", "objective", "scope", "expected"],
            },
            "methodology": {
                "purpose": "Describe procedures, architecture choices, debugging actions, implementation steps, and rationale.",
                "must_include": ["method", "approach", "process", "step"],
            },
            "findings": {
                "purpose": "Report evidence-backed observations, technical results, errors encountered, fixes applied, and resulting behavior.",
                "must_include": ["result", "observed", "found", "evidence"],
            },
            "conclusions": {
                "purpose": "Synthesize outcomes, remaining limitations, and concrete next actions without repeating methodology details.",
                "must_include": ["conclusion", "overall", "limitation", "next"],
            },
            "timeline": {
                "purpose": "Present an ordered chronology of events and decisions with concise transitions.",
                "must_include": ["morning", "afternoon", "evening", "next"],
            },
        }

    async def rephrase_section(self, section_name: str, raw_texts: List[str]) -> Dict[str, Any]:
        if not raw_texts:
            return self._empty_result("No source snippets were provided.")

        citation_registry = self._build_citation_registry(raw_texts)
        min_citations_required = self._min_citations_required(citation_registry)
        attempt_limit = max(1, self.max_rewrites + 1)

        best_result: Dict[str, Any] = self._empty_result("Generation not attempted.")
        best_score = -1

        critique_issues: List[str] = []
        for attempt in range(1, attempt_limit + 1):
            prompt = self._build_prompt(
                section_name=section_name,
                raw_texts=raw_texts,
                citation_registry=citation_registry,
                min_citations_required=min_citations_required,
                attempt=attempt,
                critique_issues=critique_issues,
                previous_draft=best_result.get("draft", ""),
            )

            temperature = 0.55 if attempt == 1 else 0.35
            generated_text, provider_used = await self._generate_with_fallback(prompt, temperature)

            if not generated_text:
                critique_issues = ["Model returned empty output."]
                continue

            quality, issues = self._evaluate_quality(
                section_name=section_name,
                text=generated_text,
                citation_registry=citation_registry,
                min_citations_required=min_citations_required,
            )
            score = self._score_quality(quality)

            candidate = {
                "draft": generated_text,
                "inline_citation_ids": self._extract_inline_citation_ids(generated_text),
                "citation_registry": citation_registry,
                "quality": quality,
                "diagnostics": {
                    "status": "passed" if not issues else "quality-gate-failed",
                    "attempts_used": attempt,
                    "provider_used": provider_used,
                    "issues": issues,
                },
            }

            if score > best_score:
                best_score = score
                best_result = candidate

            if not issues:
                return candidate

            critique_issues = issues

        best_result["diagnostics"]["status"] = "degraded"
        return best_result

    def _empty_result(self, issue: str) -> Dict[str, Any]:
        return {
            "draft": "",
            "inline_citation_ids": [],
            "citation_registry": [],
            "quality": {
                "word_count": 0,
                "min_words_required": self.min_words,
                "min_words_passed": False,
                "citation_count": 0,
                "min_citations_required": 0,
                "citations_passed": True,
                "structure_passed": False,
            },
            "diagnostics": {
                "status": "failed",
                "attempts_used": 0,
                "provider_used": "none",
                "issues": [issue],
            },
        }

    def _build_citation_registry(self, raw_texts: List[str]) -> List[Dict[str, Any]]:
        entries = [
            {"id": idx + 1, "title": None, "content": text, "entry_type": "project", "created_at": ""}
            for idx, text in enumerate(raw_texts)
        ]
        extracted = reference_extractor.extract_references(entries)
        registry = build_numeric_citation_registry(extracted)

        if registry:
            return registry

        # Fallback registry keeps inline citations possible even when explicit references are sparse.
        fallback_registry: List[Dict[str, Any]] = []
        for idx, text in enumerate(raw_texts[:4], start=1):
            snippet = re.sub(r"\s+", " ", text).strip()
            if not snippet:
                continue
            fallback_registry.append(
                {
                    "id": idx,
                    "type": "entry",
                    "value": f"Diary excerpt {idx}: {snippet[:80]}",
                    "entry_ids": [idx],
                }
            )
        return fallback_registry

    def _min_citations_required(self, citation_registry: List[Dict[str, Any]]) -> int:
        if len(citation_registry) >= 2:
            return 2
        if len(citation_registry) == 1:
            return 1
        return 0

    def _build_prompt(
        self,
        section_name: str,
        raw_texts: List[str],
        citation_registry: List[Dict[str, Any]],
        min_citations_required: int,
        attempt: int,
        critique_issues: List[str],
        previous_draft: str,
    ) -> str:
        profile = self.section_profiles.get(section_name.lower(), self.section_profiles["findings"])
        combined_text = "\n\n".join(raw_texts)
        references_block = "\n".join(
            [f"[{ref['id']}] {ref['type']}: {ref['value']}" for ref in citation_registry]
        )
        must_include = ", ".join(profile["must_include"])

        rewrite_constraints = ""
        if attempt > 1:
            rewrite_constraints = (
                f"\nPrevious draft issues to fix: {', '.join(critique_issues) if critique_issues else 'Improve clarity and compliance.'}\n"
                f"Previous draft:\n{previous_draft}\n"
            )

        return (
            "You are a meticulous technical research report writer. "
            "Write in neutral professional language, prioritize technical specifics over emotional commentary, and avoid repetition.\n"
            f"Section type: {section_name.upper()}\n"
            f"Section purpose: {profile['purpose']}\n"
            f"Must include concepts related to: {must_include}.\n"
            f"Minimum length: {self.min_words} words.\n"
            f"Inline citation style: numeric brackets like [1], [2]. Use only ids from the provided citation registry.\n"
            f"Minimum inline citations required: {min_citations_required}.\n"
            "Do not invent references; if unsure, cite the closest relevant provided source id.\n"
            "Do not output bullet points, headings, or prefatory lines. Output only the final section text.\n"
            f"Attempt number: {attempt}.\n"
            f"Citation registry:\n{references_block if references_block else 'No explicit references available.'}\n"
            f"Raw notes:\n{combined_text}\n"
            f"{rewrite_constraints}"
        )

    async def _generate_with_fallback(self, prompt: str, temperature: float) -> Tuple[str, str]:
        # 1. Try Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                    )
                )
                if response.text:
                    return response.text.strip(), "gemini"
            except Exception as e:
                print(f"Gemini failed: {e}")

        # 2. Try Groq
        if self.groq_api_key:
            try:
                client = Groq(api_key=self.groq_api_key)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama3-8b-8192",
                    temperature=temperature,
                )
                if chat_completion.choices and len(chat_completion.choices) > 0:
                    content = chat_completion.choices[0].message.content
                    if content:
                        return content.strip(), "groq"
            except Exception as e:
                print(f"Groq failed: {e}")

        # 3. Try local Ollama
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                if "response" in data:
                    return data["response"].strip(), "ollama"
        except Exception as e:
            print(f"Ollama failed: {e}")

        return "", "none"

    def _extract_inline_citation_ids(self, text: str) -> List[int]:
        return sorted({int(match) for match in re.findall(r"\[(\d+)\]", text)})

    def _validate_citation_ids(self, citation_ids: List[int], citation_registry: List[Dict[str, Any]]) -> bool:
        allowed_ids = {item["id"] for item in citation_registry}
        return all(citation_id in allowed_ids for citation_id in citation_ids)

    def _structure_passed(self, section_name: str, text: str) -> bool:
        profile = self.section_profiles.get(section_name.lower(), self.section_profiles["findings"])
        lower_text = text.lower()
        return any(keyword in lower_text for keyword in profile["must_include"])

    def _evaluate_quality(
        self,
        section_name: str,
        text: str,
        citation_registry: List[Dict[str, Any]],
        min_citations_required: int,
    ) -> Tuple[Dict[str, Any], List[str]]:
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        citation_ids = self._extract_inline_citation_ids(text)
        citation_count = len(citation_ids)
        citations_valid = self._validate_citation_ids(citation_ids, citation_registry)
        structure_passed = self._structure_passed(section_name, text)

        quality = {
            "word_count": word_count,
            "min_words_required": self.min_words,
            "min_words_passed": word_count >= self.min_words,
            "citation_count": citation_count,
            "min_citations_required": min_citations_required,
            "citations_passed": citation_count >= min_citations_required and citations_valid,
            "structure_passed": structure_passed,
        }

        issues: List[str] = []
        if not quality["min_words_passed"]:
            issues.append(
                f"Section is too short ({word_count} words). Minimum required is {self.min_words}."
            )
        if min_citations_required > 0 and citation_count < min_citations_required:
            issues.append(
                f"Insufficient inline citations ({citation_count}). Minimum required is {min_citations_required}."
            )
        if citation_count > 0 and not citations_valid:
            issues.append("Contains citation ids not present in the provided citation registry.")
        if not structure_passed:
            issues.append(
                f"Section does not reflect expected {section_name} structure strongly enough."
            )

        return quality, issues

    def _score_quality(self, quality: Dict[str, Any]) -> int:
        score = 0
        if quality.get("min_words_passed"):
            score += 3
        if quality.get("citations_passed"):
            score += 3
        if quality.get("structure_passed"):
            score += 2
        score += min(2, quality.get("citation_count", 0))
        return score

llm_service = LLMService()
