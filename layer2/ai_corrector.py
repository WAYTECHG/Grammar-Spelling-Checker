"""
AI-powered grammar and spelling correction module (Layer 2).

This module implements the T5 and Seq2Seq models for advanced grammar and spelling correction.
The models are loaded on-demand (lazy loading) to conserve resources.

Implementation Notes:
- The T5 model processes text sentence-by-sentence using the HuggingFace transformers library.
- The Seq2Seq model has a fallback mode if TensorFlow is not available.
- Both models implement confidence scoring based on Levenshtein distance.
- The module handles model loading errors gracefully.

Models:
- T5 model: Pre-trained T5 model fine-tuned for grammar and spelling correction.
- Seq2Seq model: Custom-trained sequence-to-sequence model with attention mechanism.
"""

import os
import re
import nltk
import numpy as np
import logging
from nltk.tokenize import sent_tokenize
from Levenshtein import distance as levenshtein_distance
from pathlib import Path

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flag to track if PyTorch and Transformers are available
PYTORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch successfully imported")
except ImportError as e:
    logger.warning(f"PyTorch import error: {str(e)}")
    logger.warning("T5GrammarCorrector will not be available")

# Try to import transformers
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library successfully imported")
except ImportError as e:
    logger.warning(f"Transformers import error: {str(e)}")
    logger.warning("T5GrammarCorrector will not be available")

# Try to import TensorFlow
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    # For macOS, use the metal plugin
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback for PyTorch
    
    import tensorflow as tf
    # Configure TensorFlow for macOS
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow successfully imported")
except ImportError as e:
    logger.warning(f"TensorFlow import error: {str(e)}")
    logger.warning("Seq2SeqCorrector will run in fallback mode")

# Import tokenizer utils if available
try:
    from layer2.tokenizer_utils import convert_tokens_to_string
    logger.info("Tokenizer utilities successfully imported")
except ImportError as e:
    logger.warning(f"Tokenizer utilities import error: {str(e)}")

class T5GrammarCorrector:
    """
    Grammar and spelling correction using a pre-trained T5 Transformer model.
    
    The T5 model is a state-of-the-art sequence-to-sequence model that has been
    fine-tuned specifically for grammar and spelling correction tasks.
    
    This implementation:
    1. Loads the model and tokenizer on-demand
    2. Processes text sentence-by-sentence for better accuracy
    3. Provides confidence scoring for corrections
    4. Returns a detailed list of corrections with their positions
    
    Attributes:
        model: The T5 model for text generation
        tokenizer: The tokenizer for the T5 model
        model_path: Path to the model directory
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the T5 grammar corrector.
        
        Args:
            model_path (str, optional): Path to the model directory. Defaults to None,
                                        which will use the default path.
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot initialize T5GrammarCorrector.")
            
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is not available. Cannot initialize T5GrammarCorrector.")
            
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    "models", "t5_grammar_spelling_model")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        try:
            logger.info(f"Loading T5 model from {self.model_path}")
            # Use PyTorch for model loading - works well on macOS Silicon
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Configure device - use MPS (Metal Performance Shaders) if available on macOS
            if torch.backends.mps.is_available():
                logger.info("Using MPS (Metal) device for T5 model")
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                logger.info("Using CUDA device for T5 model")
                self.device = torch.device("cuda")
            else:
                logger.info("Using CPU device for T5 model")
                self.device = torch.device("cpu")
                
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info("T5 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading T5 model: {str(e)}")
            raise

    def calculate_confidence(self, original, corrected):
        """
        Calculate confidence score based on Levenshtein distance.
        
        Returns a score between 0 and 1, where 1 means high confidence.
        The confidence is inversely proportional to the normalized edit distance.
        
        Args:
            original (str): The original text
            corrected (str): The corrected text
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if original == corrected:
            return 1.0  # No changes, full confidence
            
        max_len = max(len(original), len(corrected))
        if max_len == 0:
            return 1.0  # Edge case: both strings empty
            
        # Calculate normalized Levenshtein distance (0 to 1, where 0 means identical)
        norm_distance = levenshtein_distance(original, corrected) / max_len
        
        # Convert to confidence score (1 - normalized distance)
        # Apply a sigmoid-like transformation to keep scores in a more useful range
        confidence = 1.0 - (norm_distance * 0.7)  # Scale factor can be adjusted
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def apply_t5_correction(self, text):
        """
        Apply T5 correction to the input text.
        
        Args:
            text (str): The text to correct.
            
        Returns:
            tuple: (corrected_text, confidence_score, corrections)
                - corrected_text (str): The corrected text.
                - confidence_score (float): The confidence score of the correction.
                - corrections (list): A list of correction dictionaries.
        """
        # Split text into sentences for better processing
        sentences = sent_tokenize(text)
        
        corrected_sentences = []
        all_corrections = []
        total_confidence = 0.0
        
        for sentence in sentences:
            if not sentence.strip():
                corrected_sentences.append(sentence)
                continue
                
            # Prepare input for T5
            input_text = f"grammar: {sentence}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # Generate correction
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode the generated text
            corrected_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove any "grammar:" prefix that might be in the output
            corrected_sentence = re.sub(r'^grammar:\s*', '', corrected_sentence, flags=re.IGNORECASE)
            
            # Fix duplicate letter issues (like "everythng" -> "everythngg")
            corrected_sentence = self._fix_duplicate_letters(sentence, corrected_sentence)
            
            # Calculate confidence based on Levenshtein distance
            confidence = self.calculate_confidence(sentence, corrected_sentence)
            
            # Only add corrections if there's an actual change
            if sentence != corrected_sentence:
                start_index = text.find(sentence)
                correction = {
                    "original": sentence,
                    "corrected": corrected_sentence,
                    "confidence": confidence,
                    "type": "grammar",
                    "source": "T5 AI Model",
                    "start_index": start_index
                }
                all_corrections.append(correction)
            
            corrected_sentences.append(corrected_sentence)
            total_confidence += confidence
        
        # Combine corrected sentences
        corrected_text = " ".join(corrected_sentences)
        avg_confidence = total_confidence / len(sentences) if sentences else 0.0
        
        return corrected_text, avg_confidence, all_corrections
        
    def _fix_duplicate_letters(self, original, corrected):
        """
        Fix issues with duplicate letters in the corrected text.
        
        Args:
            original (str): The original text
            corrected (str): The corrected text with potential duplicate letters
            
        Returns:
            str: The corrected text with duplicate letter issues fixed
        """
        # Split into words
        original_words = original.split()
        corrected_words = corrected.split()
        
        # If word counts don't match, try more advanced fixes
        if len(original_words) != len(corrected_words):
            # Try to fix common issues with word boundaries
            corrected = re.sub(r'(\w)(\s+)(\1)', r'\1\2', corrected)
            # Try again after fixing
            corrected_words = corrected.split()
            
        # If still don't match, return with basic duplicate removal
        if len(original_words) != len(corrected_words):
            # Basic duplicate letter removal at word boundaries
            corrected = re.sub(r'(\w)\1{2,}', r'\1\1', corrected)
            return corrected
            
        fixed_words = []
        for i, (orig_word, corr_word) in enumerate(zip(original_words, corrected_words)):
            # Check for duplicate letter pattern at the end of the word
            if len(corr_word) > len(orig_word):
                # Check for duplicate last letter
                if len(corr_word) >= 2 and corr_word[-1] == corr_word[-2]:
                    corr_word = corr_word[:-1]
                
                # Check for other duplicate letters within the word
                for j in range(1, len(corr_word)-1):
                    # If we have 3 of the same letter in a row, remove one
                    if (corr_word[j-1] == corr_word[j] == corr_word[j+1]):
                        corr_word = corr_word[:j] + corr_word[j+1:]
                        break
                        
                # Check if a letter was duplicated that doesn't exist in the original
                for j in range(len(corr_word)-1):
                    if (j < len(orig_word) and 
                        corr_word[j] != orig_word[j] and 
                        j+1 < len(corr_word) and 
                        corr_word[j] == corr_word[j+1]):
                        corr_word = corr_word[:j+1] + corr_word[j+2:]
                        break
            
            # Check for common misspellings that add 's' or 'es'
            if corr_word.endswith('ss') and not orig_word.endswith('ss'):
                if corr_word.endswith('ess') and not orig_word.endswith('ess'):
                    corr_word = corr_word[:-2]
                else:
                    corr_word = corr_word[:-1]
                    
            fixed_words.append(corr_word)
            
        return " ".join(fixed_words)

class Seq2SeqCorrector:
    """
    Grammar and spelling correction using a custom-trained Seq2Seq model.
    
    This implementation:
    1. Loads the model and tokenizer on-demand
    2. Provides a fallback mode if TensorFlow is not available
    3. Returns a detailed list of corrections with their positions
    
    Attributes:
        model: The Seq2Seq model for text generation
        model_path: Path to the model file
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Seq2Seq grammar corrector.
        
        Args:
            model_path (str, optional): Path to the model file. Defaults to None,
                                       which will use the default path.
        """
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    "models", "seq2seq_model.keras")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = None
        self.tokenizer = None
        self.fallback_mode = not TENSORFLOW_AVAILABLE
        
        if self.fallback_mode:
            logger.warning("Seq2SeqCorrector initialized in fallback mode due to missing TensorFlow.")
        else:
            try:
                logger.info(f"Loading Seq2Seq model from {self.model_path}")
                # Import here to avoid TensorFlow import if not needed
                from layer2.keras_models import CustomGRUDecoder, CustomAttention
                import tensorflow as tf
                
                # Custom objects needed for loading the model
                custom_objects = {
                    'CustomGRUDecoder': CustomGRUDecoder,
                    'CustomAttention': CustomAttention
                }
                
                # Load the model
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects
                )
                
                # Try to load tokenizer
                try:
                    from layer2.tokenizer_utils import get_tokenizer
                    self.tokenizer = get_tokenizer()
                    logger.info("Seq2Seq tokenizer loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer: {str(e)}. Using fallback tokenization.")
                    self.tokenizer = None
                
                logger.info("Seq2Seq model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Seq2Seq model: {str(e)}")
                logger.warning("Seq2SeqCorrector will run in fallback mode")
                self.fallback_mode = True

    def correct_text(self, text):
        """
        Apply Seq2Seq correction to the input text.
        
        Args:
            text (str): The text to correct.
            
        Returns:
            tuple: (corrected_text, confidence_score, corrections)
                - corrected_text (str): The corrected text.
                - confidence_score (float): The confidence score of the correction.
                - corrections (list): A list of correction dictionaries.
        """
        if self.fallback_mode:
            # In fallback mode, just return the original text with a low confidence score
            return text, 0.1, []
        
        # Split text into sentences for better processing
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        corrected_sentences = []
        all_corrections = []
        total_confidence = 0.0
        
        for sentence in sentences:
            if not sentence.strip():
                corrected_sentences.append(sentence)
                continue
            
            try:
                # Process the sentence with the Seq2Seq model
                if self.tokenizer:
                    # Use the proper tokenizer if available
                    from layer2.tokenizer_utils import preprocess_sentence, convert_tokens_to_string
                    
                    # Preprocess the sentence
                    input_seq = preprocess_sentence(sentence, self.tokenizer)
                    
                    # Generate prediction
                    prediction = self.model.predict(input_seq, verbose=0)
                    
                    # Convert prediction to string
                    corrected_sentence = convert_tokens_to_string(prediction, self.tokenizer)
                else:
                    # Simple word-by-word processing as fallback
                    words = sentence.split()
                    corrected_words = []
                    
                    for word in words:
                        # Simple word-level correction logic
                        if len(word) > 3:  # Only process words longer than 3 characters
                            # Add simple corrections for common errors
                            if word.lower() == "teh":
                                corrected_words.append("the")
                            elif word.lower() == "tha":
                                corrected_words.append("that")
                            elif word.lower() == "thier":
                                corrected_words.append("their")
                            else:
                                corrected_words.append(word)
                        else:
                            corrected_words.append(word)
                    
                    corrected_sentence = " ".join(corrected_words)
                
                # Calculate confidence based on Levenshtein distance
                confidence = self.calculate_confidence(sentence, corrected_sentence)
                
                # Only add corrections if there's an actual change
                if sentence != corrected_sentence:
                    correction = {
                        "original": sentence,
                        "corrected": corrected_sentence,
                        "confidence": confidence,
                        "type": "grammar",
                        "source": "Seq2Seq AI Model",
                        "start_index": text.find(sentence)
                    }
                    all_corrections.append(correction)
                
                corrected_sentences.append(corrected_sentence)
                total_confidence += confidence
            
            except Exception as e:
                logger.error(f"Error processing sentence with Seq2Seq model: {str(e)}")
                # In case of error, keep the original sentence
                corrected_sentences.append(sentence)
                total_confidence += 0.1  # Low confidence for error cases
        
        # Combine corrected sentences
        corrected_text = " ".join(corrected_sentences)
        avg_confidence = total_confidence / len(sentences) if sentences else 0.0
        
        return corrected_text, avg_confidence, all_corrections
    
    def calculate_confidence(self, original, corrected):
        """
        Calculate confidence score based on Levenshtein distance.
        
        Returns a score between 0 and 1, where 1 means high confidence.
        
        Args:
            original (str): The original text
            corrected (str): The corrected text
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if original == corrected:
            return 1.0  # No changes, full confidence
            
        max_len = max(len(original), len(corrected))
        if max_len == 0:
            return 1.0  # Edge case: both strings empty
            
        # Calculate normalized Levenshtein distance (0 to 1, where 0 means identical)
        norm_distance = levenshtein_distance(original, corrected) / max_len
        
        # Convert to confidence score (1 - normalized distance)
        confidence = 1.0 - (norm_distance * 0.7)  # Scale factor can be adjusted
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence)) 