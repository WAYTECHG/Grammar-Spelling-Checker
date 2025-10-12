from flask import Flask, render_template, request, jsonify
from layer1.spelling_checker import EnhancedSpellingChecker
from layer1.grammar_checker import EnhancedGrammarChecker
import re
import logging
import os
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For macOS, use the metal plugin
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import AI correctors in a try-except block to handle import errors gracefully
try:
    from layer2.ai_corrector import T5GrammarCorrector, PYTORCH_AVAILABLE, TRANSFORMERS_AVAILABLE
    T5_AVAILABLE = PYTORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
    if T5_AVAILABLE:
        logger.info("Successfully imported T5 model with PyTorch")
    else:
        logger.warning("PyTorch or Transformers not available. T5 will be disabled.")
except ImportError as e:
    T5_AVAILABLE = False
    logger.warning(f"Could not import T5 model: {str(e)}. T5 will be disabled.")

try:
    from layer2.ai_corrector import Seq2SeqCorrector, TENSORFLOW_AVAILABLE
    SEQ2SEQ_AVAILABLE = TENSORFLOW_AVAILABLE
    if SEQ2SEQ_AVAILABLE:
        logger.info("Successfully imported Seq2Seq model with TensorFlow")
    else:
        logger.warning("TensorFlow not available. Seq2Seq will run in fallback mode.")
except ImportError as e:
    SEQ2SEQ_AVAILABLE = False
    logger.warning(f"Could not import Seq2Seq model: {str(e)}. Seq2Seq will be disabled.")

# Set overall AI availability
AI_MODELS_AVAILABLE = T5_AVAILABLE #Can Set SEQ2SEQ_AVAILABLE to True to use Seq2Seq model


app = Flask(__name__)

# Dictionary to store checker instances (lazy loading)
checkers = {}

def get_checker(checker_type):
    """
    Implements lazy-loading for our checker instances.
    The first time a checker is requested, it is initialized and stored.
    Subsequent calls return the existing instance.
    """
    if checker_type not in checkers:
        if checker_type == 'spelling':
            checkers['spelling'] = EnhancedSpellingChecker()
        elif checker_type == 'grammar':
            # This will also try to connect to CoreNLP, but only on the first grammar check
            checkers['grammar'] = EnhancedGrammarChecker()
        elif checker_type == 't5' and T5_AVAILABLE:
            try:
                checkers['t5'] = T5GrammarCorrector()
                logger.info("T5 model loaded successfully")
            except Exception as e:
                logger.error(f"Error initializing T5GrammarCorrector: {str(e)}")
                return None
        elif checker_type == 'seq2seq' and SEQ2SEQ_AVAILABLE:
            try:
                checkers['seq2seq'] = Seq2SeqCorrector()
                logger.info("Seq2Seq model loaded successfully")
            except Exception as e:
                logger.error(f"Error initializing Seq2SeqCorrector: {str(e)}")
                return None
    
    return checkers.get(checker_type)

def should_trigger_layer_2(mode, corrections, text):
    """
    Determines whether to trigger the Layer 2 AI check based on specific criteria.

    This function evaluates three conditions as specified in the PRD:
    1. User explicitly selected "Deep Check" mode
    2. Any Layer 1 suggestion has a confidence score below the threshold
    3. A complex sentence structure is detected

    Args:
        mode (str): The check mode selected by the user ('quick' or 'deep').
        corrections (list): A list of correction dictionaries from Layer 1.
        text (str): The original text being checked.

    Returns:
        bool: True if Layer 2 should be triggered, False otherwise.
    """
    # 1. Trigger if the user explicitly selected "Deep Check"
    if mode == 'deep':
        logger.info("Layer 2 triggered: User selected Deep Check mode")
        return True
    
    # 2. Trigger if any Layer 1 suggestion has a confidence score below the threshold
    confidence_threshold = 0.6
    for correction in corrections:
        if correction.get('confidence', 1.0) < confidence_threshold:
            logger.info(f"Layer 2 triggered: Low confidence correction found ({correction.get('confidence', 0)})")
            return True
    
    # 3. Trigger if a complex sentence structure is detected
    # Simple heuristic: sentences with multiple commas or semicolons
    sentences = sent_tokenize(text)
    for sentence in sentences:
        comma_count = sentence.count(',')
        semicolon_count = sentence.count(';')
        word_count = len(sentence.split())
        
        # Define complexity criteria
        if ((comma_count + semicolon_count >= 2) and word_count > 15) or word_count > 25:
            logger.info(f"Layer 2 triggered: Complex sentence detected (commas: {comma_count}, semicolons: {semicolon_count}, words: {word_count})")
            return True
    
    logger.info("Layer 2 not triggered: No triggering conditions met")
    return False

def merge_corrections(l1_corrections, l2_corrections, original_text):
    """
    Merges corrections from Layer 1 and Layer 2, prioritizing Layer 2.

    This function:
    1. Adds all Layer 2 corrections to the final list first (they have priority)
    2. Identifies text spans covered by Layer 2 corrections
    3. Adds Layer 1 corrections only if they don't overlap with Layer 2 corrections
    4. Sorts all corrections by their position in the text
    5. Removes any duplicate corrections

    Args:
        l1_corrections (list): A list of correction dictionaries from Layer 1.
        l2_corrections (list): A list of correction dictionaries from Layer 2.
        original_text (str): The original user-submitted text.

    Returns:
        list: A merged and sorted list of corrections.
    """
    # Start with all Layer 2 corrections (they have priority)
    merged_corrections = l2_corrections.copy()
    
    # Track spans covered by Layer 2 corrections
    l2_spans = []
    for corr in l2_corrections:
        start = corr.get('start_index', 0)
        end = start + len(corr.get('original', ''))
        l2_spans.append((start, end))
    
    # Add Layer 1 corrections only if they don't overlap with Layer 2
    for l1_corr in l1_corrections:
        l1_start = l1_corr.get('start_index', 0)
        l1_end = l1_start + len(l1_corr.get('original', ''))
        
        # Check for overlap with any Layer 2 correction
        overlaps = False
        for l2_start, l2_end in l2_spans:
            # Check if the spans overlap
            if (l1_start <= l2_end and l1_end >= l2_start):
                overlaps = True
                break
        
        # Add the Layer 1 correction if it doesn't overlap
        if not overlaps:
            merged_corrections.append(l1_corr)
    
    # Sort corrections by their position in the text
    merged_corrections.sort(key=lambda x: x.get('start_index', 0))
    
    # Remove any duplicate corrections (same start position)
    unique_corrections = []
    seen_positions = set()
    
    for corr in merged_corrections:
        pos = corr.get('start_index', 0)
        if pos not in seen_positions:
            unique_corrections.append(corr)
            seen_positions.add(pos)
    
    return unique_corrections

@app.route('/')
def index():
    """
    Renders the main application page.
    """
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    """
    Receives text and processes it using only the Layer 1 engine.
    
    This endpoint:
    1. Processes text with the spelling checker
    2. Processes the spelling-corrected text with the grammar checker
    3. Combines and sorts all corrections
    4. Returns the corrected text and consolidated corrections
    
    Returns:
        JSON response with original_text, final_corrected_text, and corrections
    """
    text = request.form.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    print(f"Quick check requested for text: {text[:50]}{'...' if len(text) > 50 else ''}")

    spelling_checker = get_checker('spelling')
    grammar_checker = get_checker('grammar')

    # 1. Get spelling corrections and the first corrected text
    spelling_corrected_text, spelling_corrections = spelling_checker.correct_spelling(text)
    
    # 2. Get grammar corrections on the spelling-corrected text
    grammar_corrections = grammar_checker.check_grammar(spelling_corrected_text)
    
    # 3. Combine all corrections
    all_corrections = []
    
    # Format spelling corrections for frontend hover functionality
    for corr in spelling_corrections:
        all_corrections.append({
            "type": "spelling",
            "original": corr.get("original_phrase", ""),
            "corrected": corr.get("correction", ""),
            "start_index": corr.get("start", 0),
            "confidence": corr.get("confidence", 0.5),
            "description": corr.get("description", "Spelling correction"),
            "source": "Layer 1"
        })
    
    # Format grammar corrections for frontend hover functionality
    for corr in grammar_corrections:
        all_corrections.append({
            "type": "grammar",
            "original": corr.get("original_phrase", ""),
            "corrected": corr.get("correction", ""),
            "start_index": corr.get("start", 0),
            "confidence": corr.get("confidence", 0.5),
            "description": corr.get("description", "Grammar correction"),
            "source": "Layer 1"
        })
    
    # 4. Sort corrections by their position in the text
    all_corrections.sort(key=lambda x: x.get('start_index', 0))
    
    # 5. Apply corrections to get the final text
    final_corrected_text = spelling_corrected_text
    
    return jsonify({
        "original_text": text,
        "final_corrected_text": final_corrected_text,
        "corrections": all_corrections
    })

@app.route('/deepcheck', methods=['POST'])
def deep_check():
    text = request.form.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    logger.info(f"Deep check requested: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    if not AI_MODELS_AVAILABLE:
        logger.warning("AI models unavailable for Layer 2")
        return jsonify({
            "original_text": text,
            "final_corrected_text": text,
            "corrections": [],
            "layer2_triggered": True,
            "layer2_available": False
        })
    
    corrected_text = text
    corrections = []
    confidence = 1.0

    try:
        if T5_AVAILABLE:
            t5_corrector = get_checker('t5')
            if t5_corrector:
                corrected_text, confidence, corrections = t5_corrector.apply_t5_correction(text)
                logger.info(f"T5 correction done with confidence {confidence:.2f}")
        elif SEQ2SEQ_AVAILABLE:
            seq2seq_corrector = get_checker('seq2seq')
            if seq2seq_corrector:
                corrected_text, confidence, corrections = seq2seq_corrector.correct_text(text)
                logger.info(f"Seq2Seq correction done with confidence {confidence:.2f}")
    except Exception as e:
        logger.error(f"Deep check error: {e}")
        return jsonify({"error": "AI model failed"}), 500

    return jsonify({
        "original_text": text,
        "final_corrected_text": corrected_text,
        "corrections": corrections,
        "layer2_triggered": True,
        "layer2_available": True
    })


@app.route('/status')
def status():
    """
    Returns the status of various components of the application.
    """
    status_info = {
        "server": "running",
        "layer1": {
            "spelling_checker": "spelling" in checkers,
            "grammar_checker": "grammar" in checkers,
        },
        "layer2": {
            "t5_available": T5_AVAILABLE,
            "t5_loaded": "t5" in checkers,
            "seq2seq_available": SEQ2SEQ_AVAILABLE,
            "seq2seq_loaded": "seq2seq" in checkers,
        }
    }
    
    return jsonify(status_info)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1') 