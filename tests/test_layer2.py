import os # run on python
import sys
import json

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from layer2.ai_corrector import T5GrammarCorrector, PYTORCH_AVAILABLE, TRANSFORMERS_AVAILABLE

def predict_paragraph_with_t5(text):
    """
    Loads the T5 model, runs a prediction on the provided text,
    and prints the results to the console.
    """
    print("--- T5 Model Direct Prediction ---")

    if not (PYTORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        print("\n[ERROR] PyTorch or Transformers library not installed.")
        print("Please install the required dependencies to run the T5 model.")
        return

    try:
        # Initialize the T5 corrector
        # This will load the model and tokenizer from 'models/t5_grammar_spelling_model'
        print("\nLoading T5 model...")
        t5_corrector = T5GrammarCorrector()
        print("Model loaded successfully.")

        # Run the correction
        print("\nRunning prediction...")
        corrected_text, confidence, corrections = t5_corrector.apply_t5_correction(text)
        print("Prediction complete.")

        # --- Display Results ---
        print("\n" + "="*40)
        print("           PREDICTION RESULTS")
        print("="*40)

        print("\nOriginal Text:")
        print(f"  '{text}'")

        print("\nCorrected Text:")
        print(f"  '{corrected_text}'")

        print(f"\nOverall Confidence: {confidence:.2%}")

        print("\nCorrections Found:")
        if corrections:
            for i, corr in enumerate(corrections, 1):
                print(f"  {i}. Original:  '{corr['original']}'")
                print(f"     Corrected: '{corr['corrected']}'")
                print(f"     Confidence: {corr['confidence']:.2%}")
                print("-" * 30)
        else:
            print("  No corrections were made.")

        print("="*40)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Model file not found: {e}")
        print("Please ensure the T5 model exists in the 'models/t5_grammar_spelling_model' directory.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    paragraph = "she dont like go to school becaus her freinds is mean to her. they always make fun of she's clothes and call her names. sometime she cry at home and dont wanna go back. her mom telled the teacher but nothin change yet."
    predict_paragraph_with_t5(paragraph)