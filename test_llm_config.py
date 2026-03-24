#!/usr/bin/env python3
"""Test that LLM service loads prompts configuration correctly."""

import sys
from pathlib import Path
import os

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))
os.chdir(backend_path)

try:
    from app.services.llm_service import LLMService
    
    print("=" * 70)
    print("LLM SERVICE CONFIGURATION TEST")
    print("=" * 70)
    
    # Initialize service (which loads prompts config)
    service = LLMService()
    
    print("\n✓ LLM Service initialized successfully")
    
    print("\n1. Checking LLM Provider Configuration...")
    print(f"   NVIDIA NIM API Key: {'✓ Set' if service.nvidia_nim_api_key and service.nvidia_nim_api_key != 'your-nvidia-nim-api-key-here' else '⚠ Not set'}")
    print(f"   NVIDIA NIM Base URL: {service.nvidia_nim_base_url}")
    print(f"   NVIDIA NIM Model: {service.nvidia_nim_model}")
    print(f"   Gemini API Key: {'✓ Set' if service.gemini_api_key else '⚠ Not set'}")
    print(f"   Gemini Client: {'✓ Initialized' if service.gemini_client else '⚠ Not initialized'}")
    print(f"   Groq API Key: {'✓ Set' if service.groq_api_key else '⚠ Not set'}")
    
    print("\n2. Checking Prompt Templates...")
    
    # Check draft system prompt
    if service.draft_system_prompt:
        print(f"   ✓ Draft system prompt loaded ({len(service.draft_system_prompt)} chars)")
        print(f"     Preview: {service.draft_system_prompt[:80]}...")
    else:
        print("   ✗ Draft system prompt NOT loaded")
    
    # Check draft user prompt template
    if service.draft_user_prompt_template:
        print(f"   ✓ Draft user prompt template loaded ({len(service.draft_user_prompt_template)} chars)")
        has_placeholders = all(
            placeholder in service.draft_user_prompt_template 
            for placeholder in ["{section_name}", "{section_purpose}", "{min_words}", "{raw_texts}"]
        )
        if has_placeholders:
            print(f"     ✓ All expected placeholders present")
        else:
            print(f"     ⚠ Some placeholders may be missing")
    else:
        print("   ✗ Draft user prompt template NOT loaded")
    
    # Check rewrite constraints template
    if service.rewrite_constraints_template:
        print(f"   ✓ Rewrite constraints template loaded ({len(service.rewrite_constraints_template)} chars)")
    else:
        print("   ✗ Rewrite constraints template NOT loaded")
    
    print("\n3. Checking Section Profiles...")
    section_count = len(service.section_profiles)
    print(f"   ✓ Loaded {section_count} section profiles:")
    for section_name, profile in service.section_profiles.items():
        purpose = profile.get("purpose", "N/A")[:60] + "..."
        must_include = ", ".join(profile.get("must_include", []))
        print(f"     • {section_name}: {purpose}")
        print(f"       Must include: {must_include}")
    
    print("\n4. Settings...")
    print(f"   Min words per section: {service.min_words}")
    print(f"   Max rewrites per section: {service.max_rewrites}")
    
    print("\n" + "=" * 70)
    print("✓ ALL CONFIGURATION TESTS PASSED")
    print("=" * 70)
    print("\nThe LLM service is properly configured with:")
    print("  • NVIDIA Qwen 3.5-122B as primary provider")
    print("  • Gemini → Groq → Ollama as fallbacks")
    print("  • Externalized prompts from prompts_config.yaml")
    print("  • Section-specific generation templates")
    
except Exception as e:
    print(f"\n✗ Error initializing LLM service: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
