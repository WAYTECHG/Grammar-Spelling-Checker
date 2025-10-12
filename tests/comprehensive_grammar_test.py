import time
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from layer1.grammar_checker import EnhancedGrammarChecker

def test_grammar_checker():
    print("Initializing grammar checker...")
    start_time = time.time()
    checker = EnhancedGrammarChecker()
    initialization_time = time.time() - start_time
    print(f"Initialization completed in {initialization_time:.2f} seconds")
    
    test_cases = [
        # New cases from user feedback
        {"text": "She have went to the libary.", "expected": "She has went to the libary."},
        {"text": "My mom cooked a healthy dinner.", "expected": "My mom cooked a healthy dinner."},
        {"text": "My brother don't likes vegetables.", "expected": "My brother doesn't likes vegetables."},
        
        # Subject-verb agreement
        {"text": "He go to the store.", "expected": "He goes to the store."},
        {"text": "She run every morning.", "expected": "She runs every morning."},
        {"text": "It work well.", "expected": "It works well."},
        {"text": "The students was late.", "expected": "The students were late."},
        {"text": "The ground are wet after the rain.", "expected": "The ground is wet after the rain."},
        
        # Article usage
        {"text": "I need a apple.", "expected": "I need an apple."},
        {"text": "She bought a umbrella.", "expected": "She bought an umbrella."},
        {"text": "This is a honest mistake.", "expected": "This is an honest mistake."},
        
        # Modal verb errors
        {"text": "I could of gone to the party.", "expected": "I could have gone to the party."},
        {"text": "She would of called if she knew.", "expected": "She would have called if she knew."},
        {"text": "They should of been more careful.", "expected": "They should have been more careful."},
        
        # Gerund usage
        {"text": "I enjoy to read books.", "expected": "I enjoy reading books."},
        {"text": "She enjoys to read books.", "expected": "She enjoys reading books."},
        {"text": "I appreciate to helping others.", "expected": "I appreciate helping others."},
        {"text": "They suggest to waiting until tomorrow.", "expected": "They suggest waiting until tomorrow."},
        
        # Homophone confusion
        {"text": "Their is a problem here.", "expected": "There is a problem here."},
        {"text": "They're books are on the shelf.", "expected": "Their books are on the shelf."},
        
        # Multiple errors in one sentence
        {"text": "He go to the store and she enjoy to read books.", "expected": "He goes to the store and she enjoys reading books."},
        {"text": "Their was a apple on the table and she could of taken it.", "expected": "There was an apple on the table and she could have taken it."},
        
        # Correct sentences (should remain unchanged)
        {"text": "The students were on time for the exam.", "expected": "The students were on time for the exam."},
        {"text": "I need an apple for my lunch.", "expected": "I need an apple for my lunch."},
        {"text": "He goes to the park every day.", "expected": "He goes to the park every day."},
        {"text": "I enjoy reading books in the evening.", "expected": "I enjoy reading books in the evening."},
        {"text": "They could have arrived earlier.", "expected": "They could have arrived earlier."},
    ]
    
    print("\n" + "="*50)
    print("GRAMMAR CHECKER TEST RESULTS")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest #{i}: \"{test['text']}\"")
        
        start_time = time.time()
        issues = checker.check_grammar(test['text'])
        check_time = time.time() - start_time
        
        if issues:
            print(f"Found {len(issues)} issues:")
            for issue in issues:
                print(f"- {issue['description']}: \"{issue['original_phrase']}\" -> \"{issue['correction']}\" (Confidence: {issue['confidence']})")
            
            start_time = time.time()
            corrected = checker.get_final_correction(test['text'])
            correction_time = time.time() - start_time
            print(f"Corrected: \"{corrected}\"")
            print(f"Processing time: Check={check_time:.4f}s, Correction={correction_time:.4f}s")
            
            if corrected == test['expected']:
                print("✅ PASS")
                passed += 1
            else:
                print(f"❌ FAIL - Expected: \"{test['expected']}\"")
                failed += 1
        else:
            if test['text'] == test['expected']:
                print("No issues found (as expected).")
                print("✅ PASS")
                passed += 1
            else:
                print("No issues found, but corrections were expected.")
                print(f"❌ FAIL - Expected: \"{test['expected']}\"")
                failed += 1
    
    # Test readability calculation
    print("\n" + "="*50)
    print("READABILITY SCORE TESTS")
    print("="*50)
    
    readability_tests = [
        {"name": "Complex academic text", 
         "text": "The analysis of the quantum mechanical model requires understanding of advanced mathematical concepts including differential equations and linear algebra, which many undergraduate students find challenging to comprehend initially."},
        {"name": "Simple text", 
         "text": "The cat sat on the mat. It was happy. The sun was shining."},
        {"name": "Medium complexity", 
         "text": "The company's quarterly report indicated a 15% increase in revenue compared to the previous year, which analysts attribute to the successful launch of their new product line."}
    ]
    
    for test in readability_tests:
        score = checker.calculate_readability(test["text"])
        print(f"\n{test['name']} readability score: {score:.2f}")
        print(f"Text: \"{test['text']}\"")
    
    print("\nNote: Higher scores (closer to 100) indicate easier readability")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    test_grammar_checker() 