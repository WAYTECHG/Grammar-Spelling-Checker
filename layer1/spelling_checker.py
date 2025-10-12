# This file will contain the EnhancedSpellingChecker class

import os
import re
import logging
import math
import jellyfish
from symspellpy.symspellpy import SymSpell
from symspellpy.verbosity import Verbosity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSpellingChecker:
    """
    A spelling checker that utilizes the SymSpell algorithm for candidate generation
    and a custom scoring model for ranking suggestions.
    """
    def __init__(self, ngram_model=None):
        logger.info("Initializing EnhancedSpellingChecker...")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.ngram_model = ngram_model
        self._initialize_contraction_map()
        self._load_dictionary()

    def _initialize_contraction_map(self):
        """Initializes a map of common contraction misspellings."""
        self.contraction_map = {
            "cant": "can't",
            "wont": "won't",
            "isnt": "isn't",
            "arent": "aren't",
            "wasnt": "wasn't",
            "werent": "weren't",
            "havent": "haven't",
            "hasnt": "hasn't",
            "hadnt": "hadn't",
            "wouldnt": "wouldn't",
            "shouldnt": "shouldn't",
            "couldnt": "couldn't",
            "im": "I'm",
            "youre": "you're",
            "hes": "he's",
            "shes": "she's",
            "its": "it's", # Note: This can be ambiguous with the possessive 'its'
            "theyre": "they're",
            "whatre": "what're",
            "theres": "there's"
        }

    def _load_dictionary(self):
        """
        Loads the clean frequency dictionary into the SymSpell object.
        """
        logger.info("Loading spelling dictionary...")
        dictionary_path = "frequency_dictionary_clean.txt"
        
        if not os.path.exists(dictionary_path):
            logger.error(f"CRITICAL: Spelling dictionary not found at '{dictionary_path}'. The checker will not work.")
            return False

        if self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            logger.info(f"Successfully loaded dictionary: {dictionary_path}")
            logger.info(f"Total words in dictionary: {self.sym_spell.word_count}")
            return True
        else:
            logger.error(f"Failed to load dictionary: {dictionary_path}")
            return False

    def get_candidates(self, word):
        """
        Generates potential correction candidates for a given word.
        """
        # Verbosity.ALL gets all possible suggestions
        return self.sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)

    def _calculate_score(self, original_word, suggestion_item, previous_word=None):
        """Calculates a score for a suggestion based on multiple factors."""
        original_len = len(original_word)
        suggestion_len = len(suggestion_item.term)

        # Higher frequency is better (normalize to prevent overpowering other scores)
        freq_score = suggestion_item.count / self.sym_spell.word_count if self.sym_spell.word_count > 0 else 0
        
        # Lower edit distance is better
        dist_score = 1 - (suggestion_item.distance / original_len)
        
        # Similar length is better
        len_score = 1 - (abs(original_len - suggestion_len) / original_len)

        # Phonetic similarity
        original_meta = jellyfish.metaphone(original_word)
        suggestion_meta = jellyfish.metaphone(suggestion_item.term)
        phonetic_score = 1 if original_meta == suggestion_meta else 0

        # Weighted combination
        score = (
            (freq_score * 0.4) +
            (dist_score * 0.3) +
            (len_score * 0.2) +
            (phonetic_score * 0.1)
        )

        # Contextual scoring using the N-gram model
        if self.ngram_model and previous_word:
            # N-gram score is a log probability (e.g., -5, -20). Higher is better.
            # We can add it to the score, but it needs to be weighted appropriately.
            # A simple approach is to use its exp() value to get a probability between 0 and 1.
            try:
                log_prob = self.ngram_model.score(suggestion_item.term, [previous_word.lower()])
                # Add a contextual bonus. We use exp to get a probability-like score.
                # The weight (e.g., 0.2) determines how much context influences the final decision.
                context_bonus = math.exp(log_prob) * 0.2
                score += context_bonus
            except Exception:
                # Could fail if previous_word is not in vocabulary, just skip the bonus.
                pass

        return score

    def _get_best_suggestion(self, original_word, suggestions, previous_word=None):
        """Finds the best suggestion from a list based on the custom score."""
        if not suggestions:
            return None
        return max(suggestions, key=lambda s: self._calculate_score(original_word, s, previous_word))

    def correct_spelling(self, text):
        """
        Corrects spelling in the text and returns the corrected text along with a list of changes.
        """
        if not self.sym_spell or not self.sym_spell.word_count > 0:
            logger.warning("SymSpell dictionary not loaded. Skipping spelling correction.")
            return text, []

        corrections = []
        final_text = list(text)
        # Use finditer to get match objects with start and end positions
        word_matches = re.finditer(r"\b\w+(?:['']\w+)?\b", text)
        
        offset = 0 # To track changes in text length

        for match in word_matches:
            word = match.group(0)
            start, end = match.start(), match.end()

            # Skip numbers
            if word.isdigit():
                continue

            # 1. Check our custom contraction map first
            lower_word = word.lower()
            if lower_word in self.contraction_map:
                corrected_word = self.contraction_map[lower_word]
                # Preserve casing
                if word.istitle():
                    corrected_word = "I'm" if corrected_word == "i'm" else corrected_word.title()
                
                current_start = start + offset
                final_text[current_start : current_start + len(word)] = list(corrected_word)
                
                corrections.append({
                    "type": "spelling",
                    "original_phrase": word,
                    "correction": corrected_word,
                    "start": current_start,
                    "end": current_start + len(corrected_word),
                    "confidence": 0.99, # High confidence for direct replacements
                    "description": f"Common contraction error: '{word}' corrected to '{corrected_word}'."
                })
                offset += len(corrected_word) - len(word)
                continue

            # 2. Check if the word is misspelled using SymSpell
            if self.sym_spell.lookup(lower_word, Verbosity.TOP, max_edit_distance=0, include_unknown=True)[0].distance > 0:
                suggestions = self.sym_spell.lookup(lower_word, Verbosity.CLOSEST, max_edit_distance=2)
                
                if suggestions:
                    best_suggestion = self._get_best_suggestion(word, suggestions)
                    
                    if best_suggestion and best_suggestion.term != word.lower():
                        corrected_word = best_suggestion.term
                        
                        # Preserve original casing
                        if word.istitle():
                            corrected_word = corrected_word.title()
                        elif word.isupper():
                            corrected_word = corrected_word.upper()
                            
                        # Check for duplicate letters in corrected word
                        # corrected_word = self._fix_duplicate_letters(word, corrected_word)

                        # Apply the correction to the list representation of the text
                        current_start = start + offset
                        current_end = end + offset
                        final_text[current_start:current_end] = list(corrected_word)
                        
                        # Add to our list of corrections
                        corrections.append({
                            "type": "spelling",
                            "original_phrase": word,
                            "correction": corrected_word,
                            "start": current_start,
                            "end": current_start + len(corrected_word),
                            "confidence": self._calculate_score(word, best_suggestion),
                            "description": f"Spelling error: '{word}' corrected to '{corrected_word}'."
                        })
                        
                        # Update offset for subsequent corrections
                        offset += len(corrected_word) - len(word)

        return "".join(final_text), corrections
        
    # def _fix_duplicate_letters(self, original_word, corrected_word):
    #     """
    #     Check for and fix duplicate letters that might have been introduced in the correction.
        
    #     Args:
    #         original_word (str): The original misspelled word
    #         corrected_word (str): The suggested correction that might have duplicate letters
            
    #     Returns:
    #         str: The corrected word with duplicate letters fixed
    #     """
    #     # If the corrected word is longer than the original
    #     if len(corrected_word) > len(original_word):
    #         # Check for duplicate letter at the end of the word
    #         if len(corrected_word) >= 2 and corrected_word[-1] == corrected_word[-2]:
    #             corrected_word = corrected_word[:-1]
            
    #         # Check for other duplicate letters within the word
    #         for i in range(1, len(corrected_word)-1):
    #             # If we have 3 of the same letter in a row, remove one
    #             if (corrected_word[i-1] == corrected_word[i] == corrected_word[i+1]):
    #                 corrected_word = corrected_word[:i] + corrected_word[i+1:]
    #                 break
                    
    #         # Check if a letter was duplicated that doesn't exist in the original
    #         for i in range(min(len(corrected_word)-1, len(original_word))):
    #             if (corrected_word[i] != original_word[i] and 
    #                 i+1 < len(corrected_word) and 
    #                 corrected_word[i] == corrected_word[i+1]):
    #                 corrected_word = corrected_word[:i+1] + corrected_word[i+2:]
    #                 break
        
    #     # Check for common misspellings that add 's' or 'es'
    #     if corrected_word.endswith('ss') and not original_word.endswith('ss'):
    #         if corrected_word.endswith('ess') and not original_word.endswith('ess'):
    #             corrected_word = corrected_word[:-2]
    #         else:
    #             corrected_word = corrected_word[:-1]
                
    #     return corrected_word 