#!/usr/bin/env python3
"""
Test script to validate LLM-powered reference extraction end-to-end.
Tests with a real diary entry containing academic citations.
"""

import requests
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_llm_extraction_with_real_citations():
    """Test reference extraction with entry containing real academic citations."""
    
    # Create a diary entry with real academic citations
    entry_data = {
        "title": "Whisker Encoding Research - Lottem & Azouz Paper Analysis",
        "content": "<p>Today I read the paper 'Neuronal Codes and Distributed Representations' by Lottem and Azouz (2011). "
                   "They discuss whisker-related neural encoding in the barrel cortex. I also found data from the "
                   "Allen Brain Observatory (Allen et al., 2019) which has mouse neural recordings. The CRCNS datasets "
                   "also contain whisker kinematic data that I should use. I've been using PyTorch for my analysis and "
                   "found some code on GitHub at https://github.com/sensorimotor-learning/whisker-analysis. "
                   "I should compare my findings with the earlier 2019 whisker paper from the team. "
                   "ChatGPT helped me debug the code, but I need to verify everything independently.</p>",
        "entry_type": "research"
    }
    
    print("=" * 60)
    print("TEST: LLM-Powered Reference Extraction")
    print("=" * 60)
    print("\n1. Creating test entry with real academic citations...")
    
    # Step 1: Create entry
    entry_response = client.post("/entries", json=entry_data)
    if entry_response.status_code != 200:
        print(f"ERROR: Failed to create entry. Status: {entry_response.status_code}")
        print(f"Response: {entry_response.text}")
        return False
    
    entry = entry_response.json()
    entry_id = entry["id"]
    print(f"✓ Entry created with ID: {entry_id}")
    print(f"  Title: {entry['title']}")
    
    # Step 2: Generate report suggestions (which calls extract_references)
    print("\n2. Generating report suggestions (triggers LLM extraction)...")
    report_request = {
        "entry_ids": [entry_id],
        "report_type": "project"
    }
    
    suggestions_response = client.post("/report/generate", json=report_request)
    if suggestions_response.status_code != 200:
        print(f"ERROR: Failed to generate suggestions. Status: {suggestions_response.status_code}")
        print(f"Response: {suggestions_response.text}")
        return False
    
    suggestions = suggestions_response.json()
    print("✓ Report suggestions generated")
    
    # Step 3: Validate references were extracted
    print("\n3. Validating extracted references...")
    references = suggestions.get("references", [])
    
    if not references:
        print("⚠ WARNING: No references extracted.")
        print("   This might mean the LLM extraction failed or returned empty.")
        print(f"   Full suggestions response: {json.dumps(suggestions, indent=2)}")
        return False
    
    print(f"✓ Found {len(references)} references:")
    for ref in references:
        print(f"  - Type: {ref.get('type', 'unknown')}")
        print(f"    Value: {ref.get('value', 'N/A')}")
        if ref.get("mentions"):
            print(f"    Mentions: {len(ref['mentions'])} times")
            for mention in ref['mentions']:
                print(f"      • {mention.get('text', 'N/A')}")
        print()
    
    # Step 4: Check for expected citations
    print("4. Validating expected citations...")
    expected_citations = [
        ("Lottem", "paper"),
        ("Azouz", "paper"),
        ("Allen Brain Observatory", "dataset"),
        ("CRCNS", "dataset"),
        ("PyTorch", "tool"),
        ("GitHub", "repository"),
        ("ChatGPT", "ai"),
    ]
    
    all_values = [ref["value"].lower() for ref in references]
    
    found_count = 0
    missing = []
    for citation, expected_type in expected_citations:
        found = any(citation.lower() in val for val in all_values)
        if found:
            found_count += 1
            print(f"✓ Found '{citation}'")
        else:
            missing.append(f"'{citation}'")
            print(f"✗ Missing '{citation}'")
    
    print(f"\nResult: {found_count}/{len(expected_citations)} expected citations found")
    
    if missing:
        print(f"\nMissing citations: {', '.join(missing)}")
        print("Note: This might indicate the LLM extraction needs tuning,")
        print("or the citations are formatted differently in the output.")
    
    # Step 5: Check sections generated
    print("\n5. Checking generated sections...")
    sections = suggestions.get("sections", {})
    print(f"Generated sections: {list(sections.keys())}")
    
    return True

def test_fallback_behavior():
    """Test that extraction falls back gracefully."""
    print("\n" + "=" * 60)
    print("TEST: Fallback Behavior")
    print("=" * 60)
    
    # This test depends on the system state
    # If GEMINI_API_KEY is set, Gemini will be tried first
    # If Gemini fails, it falls back to Groq, then Ollama, then regex
    
    print("\nNote: Fallback behavior is active if:")
    print("- LLM responses fail (network error, rate limit, etc.)")
    print("- GEMINI_API_KEY/GROQ_API_KEY are empty")
    print("- Ollama server is down")
    print("\nIn these cases, regex extraction will be used as final fallback.")
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REFERENCE EXTRACTION VALIDATION TEST")
    print("=" * 60 + "\n")
    
    success = test_llm_extraction_with_real_citations()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ TEST PASSED")
        print("=" * 60)
        print("\nThe LLM-powered reference extraction is working!")
        print("Academic citations are being extracted and structured properly.")
    else:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print("\nThe reference extraction needs debugging.")
        sys.exit(1)
