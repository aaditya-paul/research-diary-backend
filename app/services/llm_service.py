import os
import re
import yaml
import logging
import time
from google import genai
from groq import Groq
import httpx
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Attempt to load from .env if present
from dotenv import load_dotenv
from .reference_service import reference_extractor, build_numeric_citation_registry
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LLMService:
    def __init__(self):
        logger.info("=" * 70)
        logger.info("Initializing LLM Service")
        logger.info("=" * 70)
        
        # ========== API Key Configuration ==========
        # Primary: NVIDIA NIM for Qwen 3.5-122B
        self.nvidia_nim_api_key = os.getenv("NVIDIA_NIM_API_KEY")
        self.nvidia_nim_base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.nvidia_nim_model = os.getenv("NVIDIA_NIM_MODEL", "qwen/qwen3.5-122b-a10b")
        self.nvidia_nim_enable_thinking = os.getenv("NVIDIA_NIM_ENABLE_THINKING", "true").lower() == "true"
        
        # Fallback: Google Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = None
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        # Fallback: Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Fallback: Local Ollama
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        
        # ========== Report Generation Settings ==========
        self.min_words = int(os.getenv("REPORT_MIN_SECTION_WORDS", "250"))
        self.max_rewrites = int(os.getenv("REPORT_MAX_REWRITES", "1"))
        
        # ========== Load Prompts Configuration ==========
        self._load_prompts_config()
        
        # ========== Log Provider Configuration ==========
        self._log_provider_configuration()
    
    def _log_provider_configuration(self):
        """Log which LLM providers are available and configured."""
        logger.info("\nLLM Provider Configuration:")
        logger.info("-" * 70)
        
        # Check NVIDIA NIM
        if self.nvidia_nim_api_key:
            logger.info(f"✓ [PRIMARY] NVIDIA NIM (Qwen 3.5-122B)")
            logger.info(f"  Model: {self.nvidia_nim_model}")
            logger.info(f"  URL: {self.nvidia_nim_base_url}")
        else:
            logger.warning("✗ NVIDIA NIM: Not configured (API key missing)")
        
        # Check Gemini
        if self.gemini_client:
            logger.info(f"✓ [FALLBACK 1] Google Gemini")
            logger.info(f"  Model: gemini-2.5-flash")
        else:
            logger.warning("✗ Gemini: Not configured (API key missing)")
        
        # Check Groq
        if self.groq_api_key:
            logger.info(f"✓ [FALLBACK 2] Groq")
            logger.info(f"  Model: llama-3.3-70b-versatile")
        else:
            logger.warning("✗ Groq: Not configured (API key missing)")
        
        # Check Ollama
        logger.info(f"✓ [FALLBACK 3] Local Ollama")
        logger.info(f"  Model: {self.ollama_model}")
        logger.info(f"  URL: {self.ollama_url}")
        
        logger.info("-" * 70)
        logger.info("Provider fallback chain: NVIDIA Qwen → Gemini → Groq → Ollama")
        logger.info("=" * 70 + "\n")
    
    def _load_prompts_config(self):
        """Load prompts and section profiles from external YAML configuration."""
        config_path = Path(__file__).parent.parent / "config" / "prompts_config.yaml"
        
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Load section profiles from config
            self.section_profiles = config.get("section_profiles", self._get_default_section_profiles())
            
            # Load prompt templates from config
            self.draft_system_prompt = config.get("draft_generation_system_prompt", "").strip()
            self.draft_user_prompt_template = config.get("draft_generation_user_prompt", "").strip()
            self.rewrite_constraints_template = config.get("rewrite_constraints_template", "").strip()
            
            print(f"✓ Loaded prompts configuration from {config_path}")
        except FileNotFoundError:
            print(f"⚠ Prompts config not found at {config_path}, using defaults")
            self.section_profiles = self._get_default_section_profiles()
            self._set_default_prompts()
        except Exception as e:
            print(f"⚠ Error loading prompts config: {e}, using defaults")
            self.section_profiles = self._get_default_section_profiles()
            self._set_default_prompts()
    
    def _get_default_section_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Return default section profiles if config not available."""
        return {
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
    
    def _set_default_prompts(self):
        """Set default prompts if config not available."""
        self.draft_system_prompt = (
            "You are a meticulous technical research report writer. "
            "Write in neutral professional language, prioritize technical specifics over emotional commentary, and avoid repetition."
        )
        self.draft_user_prompt_template = (
            "Section type: {section_name}\n"
            "Section purpose: {section_purpose}\n"
            "Must include concepts related to: {must_include}.\n"
            "Minimum length: {min_words} words.\n"
            "Inline citation style: numeric brackets like [1], [2]. Use only ids from the provided citation registry.\n"
            "Minimum inline citations required: {min_citations_required}.\n"
            "Do not invent references; if unsure, cite the closest relevant provided source id.\n"
            "Do not output bullet points, headings, or prefatory lines. Output only the final section text.\n"
            "Attempt number: {attempt}.\n"
            "Citation registry:\n{citation_registry}\n"
            "Raw notes:\n{raw_texts}{rewrite_constraints}"
        )
        self.rewrite_constraints_template = (
            "\n\nPrevious draft issues to fix: {critique_issues}\n"
            "Previous draft:\n{previous_draft}"
        )

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
        must_include_values = profile.get("must_include", [])
        if isinstance(must_include_values, list):
            must_include = ", ".join(must_include_values)
        else:
            must_include = str(must_include_values)
        must_exclude_values = profile.get("must_exclude", [])
        if isinstance(must_exclude_values, list):
            must_exclude = ", ".join(must_exclude_values)
        else:
            must_exclude = str(must_exclude_values)
        scope_note = profile.get("scope_note", "")

        rewrite_constraints = ""
        if attempt > 1:
            critique_text = ", ".join(critique_issues) if critique_issues else "Improve clarity and compliance."
            rewrite_constraints = self.rewrite_constraints_template.format(
                critique_issues=critique_text,
                previous_draft=previous_draft,
            )

        # Build user prompt from template
        user_prompt = self.draft_user_prompt_template.format(
            section_name=section_name.upper(),
            section_purpose=profile.get("purpose", "Write a technically accurate section based on the provided notes."),
            must_include=must_include,
            must_exclude=must_exclude,
            scope_note=scope_note,
            min_words=self.min_words,
            min_citations_required=min_citations_required,
            attempt=attempt,
            citation_registry=references_block if references_block else "No explicit references available.",
            raw_texts=combined_text,
            rewrite_constraints=rewrite_constraints,
        )
        
        # Return combined system + user prompt for compatibility with fallback providers
        return f"{self.draft_system_prompt}\n\n{user_prompt}"

    async def _generate_with_fallback(self, prompt: str, temperature: float) -> Tuple[str, str]:
        # ========== PROVIDER FALLBACK CHAIN ==========
        logger.info(f"Starting LLM generation (temperature={temperature})")
        
        # 1. NVIDIA Qwen 3.5-122B (Primary)
        if self.nvidia_nim_api_key:
            logger.info("→ Attempting NVIDIA NIM (Qwen 3.5-122B) [PRIMARY]")
            result = await self._call_nvidia_nim(prompt, temperature)
            if result:
                logger.info("✓ SUCCESS: NVIDIA NIM (Qwen 3.5-122B) generated content")
                return result, "nvidia-qwen-3.5"
            logger.warning("✗ NVIDIA NIM failed, trying next provider...")
        
        # 2. Google Gemini (Fallback)
        if self.gemini_client:
            logger.info("→ Attempting Google Gemini (gemini-2.5-flash) [FALLBACK 1]")
            result = await self._call_gemini(prompt, temperature)
            if result:
                logger.info("✓ SUCCESS: Google Gemini (gemini-2.5-flash) generated content")
                return result, "gemini"
            logger.warning("✗ Gemini failed, trying next provider...")
        
        # 3. Groq Llama (Fallback)
        if self.groq_api_key:
            logger.info("→ Attempting Groq (llama-3.3-70b-versatile) [FALLBACK 2]")
            result = await self._call_groq(prompt, temperature)
            if result:
                logger.info("✓ SUCCESS: Groq (llama-3.3-70b-versatile) generated content")
                return result, "groq"
            logger.warning("✗ Groq failed, trying next provider...")
        
        # 4. Local Ollama (Final Fallback)
        logger.info(f"→ Attempting Local Ollama ({self.ollama_model}) [FALLBACK 3]")
        result = await self._call_ollama(prompt, temperature)
        if result:
            logger.info(f"✓ SUCCESS: Local Ollama ({self.ollama_model}) generated content")
            return result, "ollama"
        
        logger.error("✗ ALL PROVIDERS FAILED - No LLM available")
        return "", "none"
    
    async def _call_nvidia_nim(self, prompt: str, temperature: float) -> str:
        """Call NVIDIA NIM API for Qwen 3.5-122B with extended parameters."""
        try:
            start_time = time.time()
            payload = {
                "model": self.nvidia_nim_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16384,
                "temperature": temperature,
                "top_p": 0.95,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": self.nvidia_nim_enable_thinking},
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.nvidia_nim_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.nvidia_nim_api_key[:20]}...",  # Log partial key for security
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("choices") and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "").strip()
                    if content:
                        elapsed = time.time() - start_time
                        logger.debug(f"NVIDIA NIM (Qwen 3.5) returned {len(content)} chars in {elapsed:.2f}s (enable_thinking={self.nvidia_nim_enable_thinking})")
                        return content
        except Exception as e:
            logger.debug(f"NVIDIA NIM (Qwen 3.5) error: {type(e).__name__}: {str(e)[:100]}")
        return ""
    
    async def _call_gemini(self, prompt: str, temperature: float) -> str:
        """Call Google Gemini API."""
        try:
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                )
            )
            if response.text:
                content = response.text.strip()
                elapsed = time.time() - start_time
                logger.debug(f"Gemini (gemini-2.5-flash) returned {len(content)} chars in {elapsed:.2f}s")
                return content
        except Exception as e:
            logger.debug(f"Gemini (gemini-2.5-flash) error: {type(e).__name__}: {str(e)[:100]}")
        return ""
    
    async def _call_groq(self, prompt: str, temperature: float) -> str:
        """Call Groq API."""
        try:
            start_time = time.time()
            client = Groq(api_key=self.groq_api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=temperature,
            )
            if chat_completion.choices and len(chat_completion.choices) > 0:
                content = chat_completion.choices[0].message.content
                if content:
                    content = content.strip()
                    elapsed = time.time() - start_time
                    logger.debug(f"Groq (llama-3.3-70b-versatile) returned {len(content)} chars in {elapsed:.2f}s")
                    return content
        except Exception as e:
            logger.debug(f"Groq (llama-3.3-70b-versatile) error: {type(e).__name__}: {str(e)[:100]}")
        return ""
    
    async def _call_ollama(self, prompt: str, temperature: float) -> str:
        """Call local Ollama API."""
        try:
            start_time = time.time()
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
                    content = data["response"].strip()
                    elapsed = time.time() - start_time
                    logger.debug(f"Ollama ({self.ollama_model}) returned {len(content)} chars in {elapsed:.2f}s")
                    return content
        except Exception as e:
            logger.debug(f"Ollama ({self.ollama_model}) error: {type(e).__name__}: {str(e)[:100]}")
        return ""

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
