#!/usr/bin/env python3
import logging
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger("test_ai_corrector")

def test_t5_corrector():
    """Test the T5GrammarCorrector class."""
    from layer2.ai_corrector import T5GrammarCorrector
    
    logger.info("Testing T5GrammarCorrector...")
    
    corrector = T5GrammarCorrector()
    
    # Test model loading
    try:
        corrector.load_model()
        logger.info("✅ T5 model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load T5 model: {str(e)}")
        return False
    
    # Test text correction
    test_text = "i have goed to the store yesterdai and buyed milk"
    logger.info(f"Input text: '{test_text}'")
    
    try:
        corrected_text, confidence, corrections = corrector.apply_t5_correction(test_text)
        logger.info(f"Corrected text: '{corrected_text}'")
        logger.info(f"Confidence score: {confidence:.2f}")
        logger.info(f"Number of corrections: {len(corrections)}")
        
        for i, correction in enumerate(corrections):
            logger.info(f"Correction {i+1}: '{correction['original']}' → '{correction['corrected']}' (confidence: {correction['confidence']:.2f})")
        
        logger.info("✅ T5 correction completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to run T5 correction: {str(e)}")
        return False

def test_seq2seq_corrector():
    """Test the Seq2SeqCorrector class."""
    from layer2.ai_corrector import Seq2SeqCorrector
    
    logger.info("Testing Seq2SeqCorrector...")
    
    corrector = Seq2SeqCorrector()
    
    # Test model loading
    try:
        corrector.load_model()
        logger.info("✅ Seq2Seq model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Seq2Seq model: {str(e)}")
        return False
    
    # Test text correction
    test_text = "i have goed to the store yesterdai and buyed milk"
    logger.info(f"Input text: '{test_text}'")
    
    try:
        corrected_text, confidence, corrections = corrector.correct_text(test_text)
        logger.info(f"Corrected text: '{corrected_text}'")
        logger.info(f"Confidence score: {confidence:.2f}")
        logger.info(f"Number of corrections: {len(corrections)}")
        
        logger.info("✅ Seq2Seq correction completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to run Seq2Seq correction: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting AI corrector tests...")
    
    t5_success = test_t5_corrector()
    logger.info("\n" + "-"*50 + "\n")
    seq2seq_success = test_seq2seq_corrector()
    
    if t5_success and seq2seq_success:
        logger.info("\n✅ All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed!") 