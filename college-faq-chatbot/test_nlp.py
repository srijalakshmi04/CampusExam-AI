"""
Comprehensive Verification Script for CampusExam AI NLP Engine
Tests 8 mandatory demo questions + edge cases + confidence scores
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, r"C:\Users\Lenovo\.gemini\antigravity-ide\scratch\college-faq-chatbot")

from app import nlp_engine, initialize_faq_model

# Ensure model is initialized
initialize_faq_model()

test_queries = [
    # 8 Required College Demo Questions
    ("When are the semester exams?", "exam_schedule"),
    ("Where can I get my hall ticket?", "hall_ticket"),
    ("What is the attendance requirement?", "attendance"),
    ("How can I apply for revaluation?", "revaluation"),
    ("When will the results be published?", "results"),
    ("What should I bring to the exam hall?", "required_stationery"),
    ("Can I use a calculator?", "calculator_rules"),
    ("What happens if I miss an exam?", "absence"),
    
    # Natural language variations & Slang
    ("when is my sem exam?", "exam_schedule"),
    ("how to get admit card online", "hall_ticket"),
    ("is 75 percent attendance compulsory for exams?", "attendance"),
    ("paper recheck fee and process", "revaluation"),
    ("are mobile phones allowed inside the exam hall?", "mobile_restrictions"),
    ("what is the pass mark for end sem?", "passing_criteria"),
    ("how to clear backlogs", "arrears"),
    
    # Edge cases
    ("what is the weather today in Paris?", "fallback_unknown"),
    ("", "empty_query")
]

print("\n" + "="*80)
print(" CAMPUS EXAM AI - NLP TEST SUITE")
print("="*80)

passed = 0
total = len(test_queries)

for query, expected_intent in test_queries:
    result = nlp_engine.predict_intent(query)
    pred_intent = result.get("intent")
    confidence = result.get("confidence", 0.0)
    conf_pct = result.get("confidence_percentage", "0%")
    status = result.get("status")

    is_match = (pred_intent == expected_intent)
    if is_match:
        passed += 1
        status_icon = "[PASS]"
    else:
        status_icon = "[FAIL]"

    print(f"\nQuery: '{query}'")
    print(f"  {status_icon} | Expected: {expected_intent} | Predicted: {pred_intent} | Conf: {conf_pct} ({confidence})")
    print(f"  Answer: {result.get('answer')[:75]}...")

print("\n" + "="*80)
print(f" TOTAL PASSED: {passed}/{total} ({round(passed/total*100, 1)}%)")
print("="*80)
