import re
import logging
from dataclasses import dataclass, field
from nltk.parse.corenlp import CoreNLPParser
import json
import pandas as pd
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GrammarRule:
    """A dataclass to hold information about a single grammar rule."""
    id: str
    pattern: re.Pattern
    correction: str
    description: str

class EnhancedGrammarChecker:
    """
    A grammar checker that uses a combination of rule-based methods,
    N-gram models, and dependency parsing via Stanford CoreNLP.
    """
    def __init__(self, corenlp_url='http://127.0.0.1:9002'):
        """
        Initializes the grammar checker, setting up rules and the CoreNLP parser.
        """
        self.rules = self._initialize_grammar_rules()
        self.parser = None
        try:
            # Lazy-loading will be handled at the app level, but we check for connection here.
            self.parser = CoreNLPParser(url=corenlp_url)
            # A quick check to see if the server is responsive
            next(self.parser.raw_parse("test sentence"))
            logger.info("Successfully connected to Stanford CoreNLP server.")
        except Exception as e:
            logger.error(f"Failed to connect to Stanford CoreNLP server at {corenlp_url}. "
                         f"Please ensure it is running. Dependency parsing will be disabled. Error: {e}")
        self._initialize_homophones()
        
        # Initialize N-gram model attributes that will be trained later
        self.trigram_model = None
        self.ngram_counts = {}
        self.context_counts = {}

        # Load data and train the model upon initialization
        self.load_grammar_datasets()
        self.train_trigram_model()

    def _correct_verb_for_pronoun(self, match):
        """Helper function to apply correct verb endings for he/she/it."""
        pronoun = match.group(1)
        verb = match.group(2).lower()

        if verb == 'have':
            return f"{pronoun} has"
        if verb == 'do':
            return f"{pronoun} does"
        
        # Verbs ending in 'o', 's', 'x', 'z', 'sh', 'ch' take 'es'
        if verb.endswith(('o', 's', 'x', 'z')) or verb.endswith('sh') or verb.endswith('ch'):
            return f"{pronoun} {verb}es"
        
        return f"{pronoun} {verb}s"

    def _initialize_grammar_rules(self) -> list[GrammarRule]:
        """
        Initializes an extensible set of grammar rules.
        """
        logger.info("Initializing grammar rules...")
        rules = [
            GrammarRule(
                id="PLURAL_NOUN_WAS",
                pattern=re.compile(r'\b(\w+s)\s+(was)\b', re.IGNORECASE),
                correction=r'\1 were',
                description="Subject-verb agreement: Plural noun should be followed by 'were', not 'was'."
            ),
            GrammarRule(
                id="PRONOUN_VERB_S",
                pattern=re.compile(r'\b(he|she|it)\s+(start|begin|run|go|do|have|say|work)\b', re.IGNORECASE),
                correction=self._correct_verb_for_pronoun,
                description="Subject-verb agreement: 'he/she/it' should be followed by a verb ending in 's'."
            ),
            GrammarRule(
                id="GROUND_ARE_WET",
                pattern=re.compile(r'\b(ground)\s+(are)\b', re.IGNORECASE),
                correction=r'\1 is',
                description="Subject-verb agreement: 'ground' is singular and should be followed by 'is'."
            ),
            GrammarRule(
                id="MODAL_VERB_ERROR",
                pattern=re.compile(r'\b(could|would|should)\s+of\b', re.IGNORECASE),
                correction=r'\1 have',
                description="Modal verb error ('could of' -> 'could have')."
            ),
            GrammarRule(
                id="A_vs_AN",
                pattern=re.compile(r'\b(a)\s+([aeiouAEIOU]\w+)\b', re.IGNORECASE),
                correction=r'an \2',
                description="Incorrect article usage (e.g., 'a apple' -> 'an apple')."
            ),
            GrammarRule(
                id="A_vs_AN_HONORIFIC",
                pattern=re.compile(r'\b(a)\s+(honest|hour|heir|honor)\b', re.IGNORECASE),
                correction=r'an \2',
                description="Incorrect article usage for silent 'h'."
            ),
            GrammarRule(
                id="PRONOUN_DONT",
                pattern=re.compile(r'\b(he|she|it)\s+don\'t\b', re.IGNORECASE),
                correction=r"\1 doesn't",
                description="Incorrect use of 'don\\'t' with 'he/she/it'."
            ),
            GrammarRule(
                id="PRONOUN_DONT_SPELLING",
                pattern=re.compile(r'\b(he|she|it)\s+dont\b', re.IGNORECASE),
                correction=r"\1 doesn't",
                description="Incorrect spelling of 'dont'; should be 'doesn\\'t' with 'he/she/it'."
            ),
            GrammarRule(
                id="VERB_GERUND",
                pattern=re.compile(r'\b(enjoy|enjoys|appreciate|appreciates|suggest|suggests)\s+to\s+([a-zA-Z]+(?:ing)?)\b', re.IGNORECASE),
                correction=lambda m: f"{m.group(1)} {m.group(2) if m.group(2).endswith('ing') else m.group(2) + 'ing'}",
                description="Verb should be followed by a gerund (-ing form), not an infinitive."
            )
        ]
        logger.info(f"{len(rules)} grammar rules initialized.")
        return rules

    def _initialize_homophones(self):
        """Initializes a list of common homophone sets."""
        self.homophone_sets = [
            {'there', 'their', 'they\'re'},
            {'to', 'too', 'two'},
            {'your', 'you\'re'},
            {'its', 'it\'s'},
            {'affect', 'effect'},
            {'then', 'than'},
            {'which', 'witch'},
            {'weather', 'whether'},
        ]

    def load_grammar_datasets(self):
        """
        Loads and parses all grammar datasets specified in the PRD.
        Aggregates correct sentences for N-gram model training.
        """
        logger.info("Loading all grammar datasets...")
        self.training_sentences = []
        
        # Paths to datasets
        fce_train_path = 'data/grammar/fce/json/fce.train.json'
        conll14_path = 'data/grammar/conll14.json'
        jfleg_val_path = 'data/grammar/jfleg_validation.parquet'
        lang8_path = 'data/grammar/lang-8-en-1.0/entries.train'
        
        # Load datasets
        self._parse_fce(fce_train_path)
        self._parse_conll14(conll14_path)
        self._parse_jfleg(jfleg_val_path)
        self._parse_lang8(lang8_path)
        
        logger.info(f"Total correct training sentences loaded: {len(self.training_sentences)}")

    def _parse_fce(self, file_path):
        """Parses the FCE JSON-Lines dataset."""
        try:
            count = 0
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Extract corrected text from the edits field
                    if 'text' in item:
                        # The corrected text is the original with edits applied
                        # For simplicity, we'll just use the original text as training data
                        self.training_sentences.append(item['text'])
                        count += 1
            logger.info(f"Loaded {count} entries from FCE dataset: {file_path}")
        except FileNotFoundError:
            logger.warning(f"FCE dataset not found at {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error parsing FCE dataset {file_path}: {e}")

    def _parse_conll14(self, file_path):
        """Parses the CoNLL-14 JSON-Lines dataset."""
        try:
            count = 0
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Extract the text field as the training sentence
                    if 'text' in item:
                        self.training_sentences.append(item['text'])
                        count += 1
            logger.info(f"Loaded {count} entries from CoNLL-14 dataset: {file_path}")
        except FileNotFoundError:
            logger.warning(f"CoNLL-14 dataset not found at {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error parsing CoNLL-14 dataset {file_path}: {e}")

    def _parse_jfleg(self, file_path):
        """Parses the JFLEG Parquet dataset."""
        try:
            df = pd.read_parquet(file_path)
            for corrections in df['corrections']:
                # JFLEG also has multiple correct versions
                self.training_sentences.extend(corrections)
            logger.info(f"Loaded {len(df)} entries from JFLEG dataset: {file_path}")
        except FileNotFoundError:
            logger.warning(f"JFLEG dataset not found at {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error parsing JFLEG dataset {file_path}: {e}")

    def _parse_lang8(self, file_path):
        """Parses the Lang-8 tab-separated dataset."""
        try:
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        # The correct sentence is typically in the 4th column
                        # or subsequent columns if there are multiple corrections.
                        # We'll take the first one for simplicity.
                        self.training_sentences.append(parts[3])
                        count += 1
            logger.info(f"Loaded {count} entries from Lang-8 dataset: {file_path}")
        except FileNotFoundError:
            logger.warning(f"Lang-8 dataset not found at {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Error parsing Lang-8 dataset {file_path}: {e}")

    def train_trigram_model(self):
        """
        Trains a trigram language model with Laplace smoothing.
        If a cached model exists, it loads it instead of re-training.
        """
        model_path = "models/ngram_model.pkl"
        
        # 1. Check for a cached model first
        if os.path.exists(model_path):
            logger.info(f"Found cached N-gram model. Loading from {model_path}...")
            try:
                with open(model_path, 'rb') as f:
                    self.trigram_model, self.vocabulary = pickle.load(f)
                logger.info("Successfully loaded N-gram model from cache.")
                return
            except Exception as e:
                logger.error(f"Could not load cached model due to error: {e}. Re-training.")

        # 2. If no cached model, train a new one
        if not hasattr(self, 'training_sentences') or not self.training_sentences:
            logger.warning("No training sentences loaded. Skipping N-gram model training.")
            self.trigram_model = None
            return

        logger.info("Starting N-gram model training (this may take a while)...")
        
        # Tokenize the sentences
        tokenized_text = [sentence.lower().split() for sentence in self.training_sentences]
        
        # Create a vocabulary
        self.vocabulary = set(word for sent in tokenized_text for word in sent)
        
        # Prepare data for NLTK's language model
        n = 3
        train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
        
        # Initialize and train the Laplace (add-1) smoothed model
        self.trigram_model = Laplace(n)
        self.trigram_model.fit(train_data, padded_vocab)
        
        logger.info(f"N-gram model training complete. Vocabulary size: {len(self.trigram_model.vocab)}")

        # 3. Save the newly trained model to the cache
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump((self.trigram_model, self.vocabulary), f)
            logger.info(f"N-gram model saved to cache: {model_path}")
        except Exception as e:
            logger.error(f"Could not save N-gram model to cache: {e}")
        
    def check_grammar(self, text):
        """
        Checks the grammar of the input text using rule-based methods and
        dependency parsing.
        
        Args:
            text (str): The text to check.
            
        Returns:
            list: A list of grammar issues found in the text.
        """
        if not text.strip():
            return []
            
        issues = []
        
        # 1. Apply rule-based grammar checking
        for rule in self.rules:
            for match in re.finditer(rule.pattern, text):
                original_phrase = match.group(0)
                start_index = match.start()
                
                # Handle callable correction functions
                if callable(rule.correction):
                    corrected_phrase = rule.correction(match)
                else:
                    corrected_phrase = re.sub(rule.pattern, rule.correction, original_phrase)
                
                # Fix any duplicate letters in the correction
                corrected_phrase = self._fix_duplicate_letters(original_phrase, corrected_phrase)
                
                # Calculate confidence based on rule type and context
                confidence = self._calculate_confidence(original_phrase, corrected_phrase, rule.id)
                
                issues.append({
                    "id": rule.id,
                    "type": "grammar",
                    "original_phrase": original_phrase,
                    "correction": corrected_phrase,
                    "start": start_index,
                    "end": start_index + len(original_phrase),
                    "confidence": confidence,
                    "description": rule.description
                })
        
        # 2. Check for homophone errors
        homophone_issues = self._check_homophones(text)
        issues.extend(homophone_issues)
        
        # 3. Apply dependency parsing if CoreNLP is available
        if self.parser:
            try:
                # Split text into sentences for parsing
                sentences = sent_tokenize(text)
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    parse_issues = self._check_dependency_parse(sentence, text)
                    issues.extend(parse_issues)
            except Exception as e:
                logger.error(f"Error during dependency parsing: {e}")
                
        # Sort issues by their position in the text
        issues.sort(key=lambda x: x.get('start', 0))
        
        return issues

    def _fix_duplicate_letters(self, original_phrase, corrected_phrase):
        """
        Check for and fix duplicate letters that might have been introduced in the correction.
        
        Args:
            original_phrase (str): The original text
            corrected_phrase (str): The corrected text that might have duplicate letters
            
        Returns:
            str: The corrected text with duplicate letters fixed
        """
        # Split into words
        original_words = original_phrase.split()
        corrected_words = corrected_phrase.split()
        
        # If word counts don't match, return the corrected text as is
        if len(original_words) != len(corrected_words):
            return corrected_phrase
            
        fixed_words = []
        for orig_word, corr_word in zip(original_words, corrected_words):
            # Check for duplicate letter pattern at the end of the word
            if len(corr_word) > len(orig_word) and len(corr_word) >= 2:
                # Check for duplicate letter at the end
                if corr_word[-1] == corr_word[-2]:
                    corr_word = corr_word[:-1]
                    
                # Check for other common duplicate letter patterns
                for i in range(1, len(corr_word)-1):
                    # If we have 3 of the same letter in a row, remove one
                    if (corr_word[i-1] == corr_word[i] == corr_word[i+1] and 
                        (i-1 >= len(orig_word) or orig_word[i-1] != corr_word[i])):
                        corr_word = corr_word[:i] + corr_word[i+1:]
                        break
                    
            fixed_words.append(corr_word)
            
        return " ".join(fixed_words)

    def _check_homophones(self, text):
        """Checks for potential homophone confusion using the N-gram model."""
        if not self.trigram_model:
            return []

        issues = []
        tokens = word_tokenize(text)
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            for h_set in self.homophone_sets:
                if token_lower in h_set:
                    # Found a potential homophone, now score it against alternatives
                    context = (tokens[i-1].lower(),) if i > 0 else ()
                    
                    best_word = token_lower
                    # Get score of current word in context
                    best_score = self.trigram_model.score(token_lower, context)
                    
                    for alternative in h_set:
                        if alternative != token_lower:
                            alt_score = self.trigram_model.score(alternative, context)
                            if alt_score > best_score:
                                best_score = alt_score
                                best_word = alternative
                    
                    if best_word != token_lower:
                        # A better alternative was found
                        issue = {
                            'id': 'HOMOPHONE',
                            'description': f"Potential homophone error: '{token}' might be better as '{best_word}'.",
                            'correction': best_word,
                            'original_phrase': token,
                            'start': -1, # Position finding needs to be more robust
                            'end': -1,
                            'confidence': round(0.7 + (best_score / (best_score + self.trigram_model.score(token_lower, context) + 1e-9)), 2) # Normalize diff
                        }
                        issues.append(issue)
                        logger.info(f"Found homophone issue: '{token}' -> '{best_word}'")
                        break # Move to next token once a homophone set is processed
        return issues

    def get_final_correction(self, text):
        """
        Applies all identified grammar corrections sequentially to the text.
        Returns the fully corrected text.
        """
        issues = self.check_grammar(text)
        # Sort issues by start position, in reverse, to avoid index shifting issues
        sorted_issues = sorted(issues, key=lambda x: x['start'], reverse=True)
        
        corrected_text = text
        for issue in sorted_issues:
            # Skip applying 'passive voice' or other informational suggestions
            if issue['correction'].startswith("("):
                continue
            
            # Skip homophone issues for now as their start/end indices are not reliable yet
            if issue.get('id') == 'HOMOPHONE':
                continue
                
            start, end = issue['start'], issue['end']
            corrected_text = corrected_text[:start] + issue['correction'] + corrected_text[end:]
            
        return corrected_text

    def calculate_readability(self, text):
        """Calculates the Flesch-Kincaid Reading Ease score."""
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            syllables = self._count_syllables(words)
            
            if len(sentences) > 0 and len(words) > 0:
                score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
                return score
        except Exception as e:
            logger.error(f"Could not calculate readability: {e}")
            return None
        
        return None

    def _count_syllables(self, words):
        """A simple heuristic to count syllables in a list of words."""
        syllable_count = 0
        for word in words:
            word = word.lower()
            # Count vowels
            count = len(re.findall('[aeiouy]+', word))
            # Adjustments for common cases
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count = 1
            syllable_count += count
        return syllable_count

    def _calculate_confidence(self, original_phrase, corrected_phrase, rule_id):
        """
        Calculate confidence score for a grammar correction.
        
        Args:
            original_phrase (str): The original text
            corrected_phrase (str): The corrected text
            rule_id (str): The ID of the rule that generated this correction
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Base confidence level
        confidence = 0.85
        
        # Adjust confidence based on rule type
        if 'MODAL_VERB_ERROR' in rule_id:
            confidence = 0.95  # Very high confidence for modal verb errors
        elif 'A_vs_AN' in rule_id:
            confidence = 0.90  # High confidence for article errors
        elif 'PLURAL_NOUN_WAS' in rule_id:
            confidence = 0.88  # Good confidence for subject-verb agreement
            
        # Adjust confidence based on length of phrase
        if len(original_phrase.split()) > 5:
            confidence -= 0.05  # Longer phrases are more complex, slightly lower confidence
            
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def _check_dependency_parse(self, sentence, full_text):
        """
        Check for grammar issues using dependency parsing.
        
        Args:
            sentence (str): The sentence to check
            full_text (str): The full text (for finding the position of the sentence)
            
        Returns:
            list: A list of grammar issues found using dependency parsing
        """
        if not self.parser or not sentence.strip():
            return []
            
        issues = []
        
        try:
            # Get parse tree from CoreNLP
            parse_tree = next(self.parser.raw_parse(sentence))
            
            # Check for subject-verb agreement issues
            for subtree in parse_tree.subtrees():
                # Look for noun phrases followed by incorrect verb forms
                if subtree.label() == 'NP' and len(list(subtree)) > 0:
                    # Get the head noun
                    head_noun = None
                    for leaf in subtree.leaves():
                        if leaf.endswith('NN') or leaf.endswith('NNS'):
                            head_noun = leaf
                            break
                    
                    if head_noun:
                        # Check if it's plural (ends with 's')
                        is_plural = head_noun.endswith('NNS')
                        
                        # Look for verb phrases that follow
                        parent = subtree.parent()
                        if parent and any(child.label() == 'VP' for child in parent):
                            for sibling in parent:
                                if sibling.label() == 'VP':
                                    # Check verb form
                                    verb = sibling.leaves()[0] if sibling.leaves() else None
                                    if verb:
                                        # Simple check: plural nouns should not have singular verbs
                                        if is_plural and verb.endswith('VBZ'):  # 3rd person singular
                                            start_idx = full_text.find(sentence)
                                            if start_idx >= 0:
                                                verb_text = verb.split('/')[0]  # Get just the word
                                                verb_idx = sentence.find(verb_text)
                                                if verb_idx >= 0:
                                                    issues.append({
                                                        "id": "SVA_DEP_PARSE",
                                                        "type": "grammar",
                                                        "original_phrase": f"{head_noun.split('/')[0]} {verb_text}",
                                                        "correction": f"{head_noun.split('/')[0]} are" if verb_text == "is" else f"{head_noun.split('/')[0]} {verb_text}",
                                                        "start": start_idx + verb_idx - len(head_noun.split('/')[0]) - 1,
                                                        "end": start_idx + verb_idx + len(verb_text),
                                                        "confidence": 0.75,
                                                        "description": "Subject-verb agreement issue detected by parser."
                                                    })
        except Exception as e:
            logger.error(f"Error in dependency parsing: {e}")
            
        return issues

if __name__ == '__main__':
    # Example usage:
    # Make sure to run the Stanford CoreNLP server first.
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 15000
    
    checker = EnhancedGrammarChecker()
    
    # Example of N-gram model usage (if trained)
    checker.load_grammar_datasets()
    checker.train_trigram_model()
    if checker.trigram_model:
        perplexity = checker.trigram_model.perplexity("this is a test".split())
        print(f"\nPerplexity of 'this is a test': {perplexity}") 
    
    test_text = "He go to the store. their is too many options. I enjoy to reading."
    print(f"\nOriginal text: {test_text}")
    
    issues = checker.check_grammar(test_text)
    print("\nIdentified Issues:")
    for issue in issues:
        print(f"- {issue['description']}: '{issue['original_phrase']}' -> '{issue['correction']}' (Confidence: {issue['confidence']})")
        
    final_text = checker.get_final_correction(test_text)
    print(f"\nFinal corrected text (placeholder): {final_text}")
    
    readability_score = checker.calculate_readability(test_text)
    if readability_score is not None:
        print(f"\nReadability Score (Flesch-Kincaid): {readability_score:.2f}") 