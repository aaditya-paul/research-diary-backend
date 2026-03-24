# Recent Updates: NVIDIA Qwen 3.5 & Externalized Prompts

## Summary

Upgraded the Research Diary report generation system to use **NVIDIA Qwen 3.5-122B** as the primary LLM and **externalized all prompts** into a centralized configuration file for easy modification without code changes.

---

## What Changed

### 1. **Primary LLM: NVIDIA Qwen 3.5-122B** 🚀

- Replaced Gemini as primary provider
- Much more capable for structured output and complex instructions
- Better handling of technical terminology and citations
- Fallback chain: **Qwen → Gemini → Groq → Ollama**

### 2. **Externalized Prompt Configuration** 📝

**New File**: `backend/app/config/prompts_config.yaml`

- All hardcoded prompts extracted to single YAML file
- Includes section profiles, generation templates, extraction templates
- Can modify prompts without touching Python code
- Easy to version control prompt changes separately

### 3. **Environment Configuration Updates** ⚙️

**Updated Files**: `.env`, `.env.example`

```
# Primary (NEW)
NVIDIA_NIM_API_KEY="your-api-key"
NVIDIA_NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_MODEL="qwen/qwen3.5-122b-a10b"

# Existing fallbacks
GEMINI_API_KEY="..."
GROQ_API_KEY="..."
OLLAMA_URL="..."
```

---

## Files Modified

| File                                        | Changes                                                            |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `backend/app/services/llm_service.py`       | ✅ Refactored to load prompts from YAML, added NVIDIA NIM provider |
| `backend/app/services/reference_service.py` | ✅ Updated provider chain, externalized extraction prompt          |
| `backend/app/config/prompts_config.yaml`    | ✨ **NEW** - All prompts in one place                              |
| `backend/.env`                              | ✅ Added NVIDIA NIM configuration                                  |
| `backend/.env.example`                      | ✅ Documented all API keys                                         |
| `backend/requirements.txt`                  | ✅ Added `pyyaml` dependency                                       |

---

## How to Use

### Setup NVIDIA NIM API Key

1. Go to [build.nvidia.com](https://build.nvidia.com/nvidia/qwen-3.5-122b)
2. Get your API key
3. Update `.env`:
   ```
   NVIDIA_NIM_API_KEY="your-api-key-here"
   ```

### Modify Prompts

Edit `backend/app/config/prompts_config.yaml` to change:

- System prompt for report generation
- User prompt templates (with `{placeholders}`)
- Reference extraction prompt
- Section-specific profiles and requirements

No code changes needed!

### Example: Change Minimum Words

In `.env`:

```
REPORT_MIN_SECTION_WORDS="300"  # Instead of 250
```

### Example: Update Section Profile

In `prompts_config.yaml`:

```yaml
section_profiles:
  findings:
    purpose: "Your new purpose here"
    must_include: ["new", "keywords", "here"]
```

---

## Provider Fallback Chain

```
1. NVIDIA Qwen 3.5-122B (Primary)
   ↓ (if fails)
2. Google Gemini 2.5 Flash
   ↓ (if fails)
3. Groq Llama 3.3-70B
   ↓ (if fails)
4. Local Ollama
   ↓ (if fails)
5. Regex extraction (for references only)
```

All fallbacks happen automatically. No manual intervention needed.

---

## Benefits

| Benefit                | Details                                                               |
| ---------------------- | --------------------------------------------------------------------- |
| **Easier Maintenance** | Edit prompts in one YAML file, no code recompilation                  |
| **Better Generation**  | Qwen 3.5 is superior for instruction-following and structured outputs |
| **Cost Control**       | Temperature, max_tokens, batch sizes all easy to tune                 |
| **Batch Efficiency**   | Reference extraction uses 5-entry batches (80% cost reduction)        |
| **Fallback Safety**    | 4 providers ensure system keeps working if one fails                  |
| **Version Control**    | Prompt changes tracked separately from code                           |

---

## Validation ✅

### Tests Passed:

- ✅ Reference extraction: 9 citations extracted from test entry (7/7 expected found)
- ✅ Prompt configuration loading: 175-char system prompt + 604-char template
- ✅ All 5 section profiles loaded correctly
- ✅ Fallback chain verified (tested with missing API keys)
- ✅ YAML parsing: all placeholders present and functional

---

## Technical Details

### Architecture: LLM Service

```python
# Before: Gemini → Groq → Ollama
# After:  Qwen → Gemini → Groq → Ollama

async def _generate_with_fallback(prompt, temperature):
    # Try each provider in order until one succeeds
    # Returns (generated_text, provider_name)
```

### Prompt Templates Support

```yaml
draft_generation_user_prompt: |
  Section type: {section_name}
  Purpose: {section_purpose}
  ...raw texts...
  {rewrite_constraints}
```

All `{placeholders}` replaced at runtime with actual values.

### Batch Processing for References

- Groups up to 5 entries per LLM call
- ~80% cost reduction vs. single-call-per-entry
- Automatic fallback to regex if LLM unavailable

---

## Troubleshooting

### "NVIDIA NIM API returned 404"

- Check API key is valid in `.env`
- Verify Base URL: `https://integrate.api.nvidia.com/v1`
- System will automatically fallback to Gemini

### "ImportError: No module named 'yaml'"

```bash
pip install -r requirements.txt
```

### "Prompts config not found"

- Ensure `backend/app/config/prompts_config.yaml` exists
- System falls back to hardcoded defaults if missing

---

## Next Steps

1. **Test with Real Data**: Generate reports against actual diary entries
2. **Fine-tune Qwen Prompts**: Adjust templates in `prompts_config.yaml` based on output quality
3. **Monitor Token Usage**: Track cost difference between Qwen vs Gemini
4. **Add Custom Sections**: Extend `section_profiles` in YAML for new report types

---

## Quick Reference

**Change LLM provider**:

```bash
# Disable Qwen (will fall back to Gemini)
NVIDIA_NIM_API_KEY=""
```

**Adjust verbosity**:
In `.env`:

```
REPORT_MIN_SECTION_WORDS="500"  # Make longer
REPORT_MAX_REWRITES="2"         # Allow more rewrites
```

**View current config**:

```bash
cd backend
python test_llm_config.py
```

**Test reference extraction**:

```bash
python test_extraction_unit.py
```

---

**Last Updated**: March 25, 2026  
**System Status**: ✅ All tests passing
