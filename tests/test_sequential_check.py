import os # run on python
import sys

# Add project root to Python path to allow for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the necessary checkers
from layer1.spelling_checker import EnhancedSpellingChecker
from layer1.grammar_checker import EnhancedGrammarChecker
from layer2.ai_corrector import T5GrammarCorrector, TRANSFORMERS_AVAILABLE

def print_local_results(title, original_text, corrected_text, corrections_list):
    """Helper function to print results for a local check."""
    print("\n" + "="*40)
    print(f"        {title.upper()}")
    print("="*40)
    
    print(f"\nInput Text:\n  '{original_text}'")
    print(f"\nCorrected Text:\n  '{corrected_text}'")
    
    print("\nCorrections Made:")
    if corrections_list:
        for i, correction in enumerate(corrections_list, 1):
            original = correction.get('original_phrase') or correction.get('original')
            corrected = correction.get('correction') or correction.get('corrected')
            source = correction.get('source', 'Unknown')
            
            print(f"  {i}. Original:  '{original}'")
            print(f"     Corrected: '{corrected}'")
            print(f"     Source:    {source}")
            print("-" * 25)
    else:
        print("  No corrections were made in this layer.")
    print("="*40)

def run_local_sequential_check(text):
    """
    Runs the full sequential check locally by importing and using the checkers directly.
    """
    try:
        # --- Step 1: Run Layer 1 (Rule-Based) ---
        print("--- Running Layer 1 (Rule-Based Check) Locally ---")
        spelling_checker = EnhancedSpellingChecker()
        grammar_checker = EnhancedGrammarChecker()

        # Spelling correction returns a corrected string and a list of changes
        l1_corrected_text, spelling_corrections = spelling_checker.correct_spelling(text)
        
        # Grammar correction runs on the spelling-corrected text and returns a list of changes
        grammar_corrections = grammar_checker.check_grammar(l1_corrected_text)

        # Combine all corrections from Layer 1
        all_l1_corrections = spelling_corrections + grammar_corrections
        for c in all_l1_corrections:
            c['source'] = 'Layer 1'

        # The final text from Layer 1 is the result of the spelling check
        final_l1_text = l1_corrected_text
        print_local_results("LAYER 1 RESULTS", text, final_l1_text, all_l1_corrections)

        # --- Step 2: Run Layer 2 (AI-Based) ---
        print("\n\n--- Running Layer 2 (AI Check) Locally ---")
        if not TRANSFORMERS_AVAILABLE: 
            print("AI models are not available. Skipping Layer 2.")
            return

        t5_corrector = T5GrammarCorrector()
        
        # Use Layer 1's output as input for Layer 2
        l2_input_text = final_l1_text
        final_l2_text, _, l2_corrections = t5_corrector.apply_t5_correction(l2_input_text)
        
        print_local_results("LAYER 2 RESULTS", l2_input_text, final_l2_text, l2_corrections)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please ensure all dependencies are installed and model files are in place.")

if __name__ == "__main__":
    paragraph = "she dont like go to school becaus her freinds is mean to her. they always make fun of she's clothes and call her names. sometime she cry at home and dont wanna go back. her mom telled the teacher but nothin change yet."
    run_local_sequential_check(paragraph)