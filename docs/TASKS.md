# Project Task Plan: Enhanced Spelling and Grammar Checker

This document outlines the detailed, step-by-step development plan for the project.

---

## Phase 1: Core Backend & Layer 1 Implementation

- [x] **Task 1.1: Environment & Setup**
  - [x] **1.1.1: Finalize `requirements.txt`**
    - [x] 1.1.1.1: Verify and pin versions for key libraries (`transformers`, `torch`, `nltk`, `symspellpy`, `flask`).
    - [x] 1.1.1.2: Add utility libraries like `pandas` for data handling and `requests` for server checks.
  - [x] **1.1.2: Verify and Integrate Datasets**
    - [x] **1.1.2.1: Verify Spelling Datasets**
        - [x] 1.1.2.1.1: Confirm the presence of Birkbeck, SCOWL, Google 10k, Wikipedia, and Aspell datasets in the `data/spelling/` directory.
    - [x] **1.1.2.2: Consolidate Spelling Datasets**
        - [x] 1.1.2.2.1: Re-create the `scripts/prepare_spelling_dict.py` script to process local datasets, excluding the Google Ngram data.
        - [x] 1.1.2.2.2: Run the script to generate the final `frequency_dictionary_clean.txt`.
    - [x] **1.1.2.3: Verify Grammar Datasets**
        - [x] 1.1.2.3.1: Confirm the presence of JFLEG, CoNLL-14, FCE, and Lang-8 datasets within the `data/grammar/` directory.
  - [x] **1.1.3: Configure Stanford CoreNLP**
    - [x] 1.1.3.1: Verify the `stanford-corenlp-4.5.10` directory exists in the project root.
    - [x] 1.1.3.2: Copy the `start_corenlp.sh` script from that directory to the project root.
    - [x] 1.1.3.3: Ensure the script has correct paths and memory allocation for our project structure.

- [x] **Task 1.2: Spelling Checker (Layer 1)**
  - [x] **1.2.1: Implement `EnhancedSpellingChecker` Class Initialization**
    - [x] 1.2.1.1: In `__init__`, initialize the `SymSpell` object with appropriate parameters (`max_dictionary_edit_distance`, `prefix_length`).
    - [x] 1.2.1.2: Implement the `load_dictionary` method to load the pre-processed `frequency_dictionary_clean.txt`.
    - [x] 1.2.1.3: Add logging to confirm successful loading and report the dictionary's word count.
  - [x] **1.2.2: Implement Candidate Generation**
    - [x] 1.2.2.1: Create a method that takes a single word and uses `symspell.lookup()` to generate a list of potential correction candidates.
  - [x] **1.2.3: Implement Custom Candidate Scoring**
    - [x] 1.2.3.1: Create a helper function to calculate the Levenshtein distance between two strings.
    - [x] 1.2.3.2: Create helper functions to generate Soundex and Metaphone keys for phonetic comparison.
    - [x] 1.2.3.3: Implement a `_calculate_score(word, candidate)` method that computes a final score based on the weighted formula specified in the PRD (Frequency, Edit Distance, Length Similarity, Phonetic Similarity).
  - [x] **1.2.4: Implement Main Correction Logic**
    - [x] 1.2.4.1: Implement the main `correct_spelling(text)` method.
    - [x] 1.2.4.2: This method should iterate through words in the input text.
    - [x] 1.2.4.3: Implement logic to handle and preserve common contractions (e.g., "don't", "can't").
    - [x] 1.2.4.4: For each non-contraction word, call the candidate generation method (1.2.2).
    - [x] 1.2.4.5: Score each generated candidate using the custom scoring method (1.2.3).
    - [x] 1.2.4.6: Select the highest-scoring candidate as the final correction for that word.
    - [x] 1.2.4.7: Reconstruct and return the fully corrected text.

- [x] **Task 1.3: Grammar Checker (Layer 1)**
  - [x] **1.3.1: Implement `EnhancedGrammarChecker` Class**
    - [x] 1.3.1.1: Create a `GrammarRule` data class to structure individual grammar rules (pattern, correction, description).
    - [x] 1.3.1.2: Research and document an initial, extensible set of grammar rules.
    - [x] 1.3.1.3: Implement a `_initialize_grammar_rules` method to define the list of `GrammarRule` objects.
    - [x] 1.3.1.4: In `__init__`, initialize the Stanford CoreNLP parser, with error handling in case the server is not available.
  - [x] **1.3.2: Implement Grammar Dataset Loading**
    - [x] 1.3.2.1: Create private helper methods to parse each grammar dataset format from their paths in `data/grammar/`.
    - [x] 1.3.2.2: Aggregate all correct sentences into a single `self.training_sentences` list.
  - [x] **1.3.3: Implement N-gram Model**
    - [x] 1.3.3.1: Implement `train_ngram_model` to process `self.training_sentences`.
    - [x] 1.3.3.2: Tokenize sentences, build a vocabulary, and use `nltk.util.ngrams` to generate trigrams.
    - [x] 1.3.3.3: Calculate and store `ngram_counts` and `context_counts`.
  - [x] **1.3.4: Implement `check_grammar` Method**
    - [x] 1.3.4.1: Iterate through the defined grammar rules and use `re.finditer` to find matches in the text.
    - [x] 1.3.4.2: Call the CoreNLP parser to get a dependency parse tree for the sentence.
    - [x] 1.3.4.3: Analyze the parse tree for structural issues (e.g., missing verb phrases).
    - [x] 1.3.4.4: Combine findings from rule-based and parsing checks into a list of identified issues.
  - [x] **1.3.5: Design and Implement Confidence Scoring**
    - [x] 1.3.5.1: Design a confidence scoring algorithm for grammar corrections.
    - [x] 1.3.5.2: Implement the scoring system within the `check_grammar` method.

- [x] **Task 1.4: Enhance Quick Check Algorithm (Layer 1)**
  - [x] **1.4.1: Enhance Spelling Checker**
    - [x] 1.4.1.1: Refine contraction handling to ensure they are preserved and correctly tokenized before checking.
    - [x] 1.4.1.2: Implement context-aware spelling correction. Use the N-gram model to re-rank spelling candidates based on the probability of the preceding word(s). This helps resolve common homophone errors (e.g., "their" vs. "there," "to" vs. "too") based on context.
    - [x] 1.4.1.3: Integrate a dedicated homophone checker that uses the contextual N-gram scores to flag potential mix-ups.
  - [x] **1.4.2: Enhance Grammar Checker**
    - [x] 1.4.2.1: Expand the grammar rule set. Based on common errors flagged by popular tools, add new categories for:
      - **Punctuation:** Comma splices, missing commas after introductory clauses, and incorrect semi-colon usage.
      - **Style:** Detection of passive voice with suggestions for an active-voice alternative.
      - **Clarity:** Flagging overly complex sentences or redundant phrases.
    - [x] 1.4.2.2: Implement readability analysis. Add a function to calculate and return a standard readability score, such as the Flesch-Kincaid Reading Ease.
    - [x] 1.4.2.3: Refine the confidence scoring for grammar rules. Make scores dynamic based on rule specificity and statistical data from the training corpora.

- [x] **Task 1.5: Unit Testing (Phase 1)**
  - [x] **1.5.1: Unit Tests for `EnhancedSpellingChecker`**
    - [x] 1.5.1.1: Test dictionary loading: successful load and word count verification.
    - [x] 1.5.1.2: Test dictionary loading failure with a non-existent file path.
    - [x] 1.5.1.3: Test candidate generation (`get_candidates`) with a known misspelled word (e.g., "speling").
    - [x] 1.5.1.4: Test custom scoring (`_calculate_score`) by comparing scores for two candidates to ensure the better one scores higher.
    - [x] 1.5.1.5: Test main correction logic (`correct_spelling`) with a sentence containing multiple errors, contractions, and punctuation.
    - [x] 1.5.1.6: Test `correct_spelling` with an already correct sentence to ensure no changes are made.
    - [x] 1.5.1.7: Test `correct_spelling` with an empty or whitespace-only string input.
    - [x] 1.5.1.8: Add a traceability test that shows word-by-word changes for a corrected sentence.
  - [x] **1.5.2: Unit Tests for `EnhancedGrammarChecker`**
    - [x] 1.5.2.1: Test `_initialize_grammar_rules` to ensure it returns a non-empty list of `GrammarRule` objects.
    - [x] 1.5.2.2: Test dataset parsers (`_parse_fce`, `_parse_conll14`, `_parse_jfleg`) individually by mocking file reads and verifying the output.
    - [x] 1.5.2.3: Add and test a `_parse_lang8` method to handle the tab-separated Lang-8 corpus.
    - [x] 1.5.2.4: Test `load_grammar_datasets` to confirm it calls all individual parsers and aggregates the results.
    - [x] 1.5.2.5: Test `train_ngram_model` on a small, controlled dataset to verify the model is created.
    - [x] 1.5.2.6: Test `check_grammar` for each individual rule with a matching incorrect sentence and verify the output structure.
    - [x] 1.5.2.7: Test `check_grammar` with a correct sentence to ensure no issues are returned.
    - [x] 1.5.2.8: (Advanced) Mock the `CoreNLPParser` to test both successful and failed connection scenarios in the `__init__` method.
    - [x] 1.5.2.9: Add a traceability test that shows word-by-word changes for a corrected sentence.
  - [x] **1.5.3: Layer 1 Integration Tests**
    - [x] 1.5.3.1: Create a test fixture that initializes both the `EnhancedSpellingChecker` and `EnhancedGrammarChecker`.
    - [x] 1.5.3.2: Write a test that runs text with both spelling and grammar errors through the spelling checker first, then feeds its output to the grammar checker, verifying that issues are correctly identified in sequence.

---

## Phase 2: API, Frontend, and Initial Integration

- [x] **Task 2.1: Flask Backend Integration**
  - [x] 2.1.1: Implement lazy-loading for all checker and model classes in `app.py` to ensure fast startup.
  - [x] 2.1.2: In the `/check` endpoint, orchestrate the Layer 1 flow: `correct_spelling` -> `check_grammar`.
  - [x] 2.1.3: Define and implement the final JSON response format, including `original_text`, `corrected_text`, and a list of `corrections`.
  - [x] 2.1.4: Implement the post-processing logic to apply corrections sequentially by position and generate the final `corrected_text`.

- [x] **Task 2.2: Frontend Development**
  - [x] 2.2.1: Implement side-by-side comparison view for original vs. corrected text.
  - [x] 2.2.2: In `script.js`, create a function to parse the `corrections` from the JSON response.
  - [x] 2.2.3: Dynamically reconstruct the result text, wrapping errors in `<span>` tags with appropriate classes (`.spelling`, `.grammar`).
  - [x] 2.2.4: Implement tooltip functionality on `mouseover` for error spans, displaying suggestions and confidence scores.
  - [x] 2.2.5: Add 'Accept'/'Reject' buttons to tooltips and implement the corresponding logic to modify the text.
  - [x] 2.2.6: Implement the 'Export to .txt' functionality.

- [x] **Task 2.3: Integration Testing (Phase 2)**
  - [x] **2.3.1: Manual Test Run of `/check` Endpoint**
    - [x] 2.3.1.1: Start the Stanford CoreNLP server using the `start_corenlp.sh` script.
    - [x] 2.3.1.2: Start the Flask application (`python app.py`).
    - [x] 2.3.1.3: Open the web interface in a browser.
    - [x] 2.3.1.4: Enter text with a mix of spelling and grammar errors into the text area.
    - [x] 2.3.1.5: Submit the text and visually confirm that the "Corrected Text" and "Identified Issues" sections populate correctly.
    - [x] 2.3.1.6: Check the browser's developer console and the Flask server logs for any errors.
  - [x] **2.3.2: Automated Integration Tests**
    - [x] 2.3.2.1: Write integration tests using a Flask test client for the `/check` endpoint.
    - [x] 2.3.2.2: **Test Response Structure:** Validate that the JSON response consistently contains the required keys: `original_text`, `corrected_text`, and `corrections`.
    - [x] 2.3.2.3: **Test Scenario - Spelling Errors Only:** Send text with only spelling mistakes and verify that `corrected_text` is accurate and the `corrections` list contains only spelling-related issues.
    - [x] 2.3.2.4: **Test Scenario - Grammar Errors Only:** Send text with only grammar mistakes and verify that `corrected_text` is accurate and the `corrections` list contains only grammar-related issues.
    - [x] 2.3.2.5: **Test Scenario - Mixed Errors:** Send text with both spelling and grammar mistakes and verify that all issues are caught and corrected appropriately.
    - [x] 2.3.2.6: **Test Scenario - Correct Text:** Send text that is already correct and verify that `corrected_text` matches `original_text` and the `corrections` list is empty.
    - [x] 2.3.2.7: **Test Scenario - Edge Cases:** Send an empty string, a string with only whitespace, and very long text to ensure the endpoint handles them gracefully without errors.

---

## Phase 3: Layer 2 AI Integration & Finalization

- [x] **Task 3.1: Integrate Pre-Trained AI Models**
  - [x] 3.1.1: Implement `T5GrammarCorrector` class with logic to load the pre-trained model from `t5_grammar_spelling_model/`.
  - [x] 3.1.2: Implement its `correct_text` method, ensuring it processes text sentence-by-sentence.
  - [x] 3.1.3: Implement `Seq2SeqCorrector` class with logic to load the pre-trained Keras model from `seq2seq_model.keras`.
  - [x] 3.1.4: Implement its `correct_text` method.
  - [x] 3.1.5: Implement a confidence scoring method for AI corrections based on Levenshtein distance.
  - [x] 3.1.6: Test that both models successfully load, receive input text, make predictions, and return properly formatted corrected output.
  - [x] 3.1.7: Implement proper token-to-word conversion for the Seq2Seq model using a tokenizer that matches the training implementation.

- [x] **Task 3.2: Full Workflow Integration**
  - [x] **3.2.1: Implement Layer 2 Trigger Logic**
    - [x] 3.2.1.1: Create a function skeleton for the trigger logic.
    - [x] 3.2.1.2: Implement logic to evaluate Layer 1 confidence scores against the threshold.
    - [x] 3.2.1.3: Add complex sentence structure detection using the dependency parse tree.
    - [x] 3.2.1.4: Integrate the user's "Deep Check" mode selection into the trigger.
  - [x] **3.2.2: Implement Post-Processing and Merging**
    - [x] 3.2.2.1: Add a mechanism in the backend to select which AI model to use (T5 or Seq2Seq).
    - [x] 3.2.2.2: Implement the `merge_corrections` logic to handle outputs from Layer 1 and the selected Layer 2 model.
    - [x] 3.2.2.3: Ensure conflict resolution in the merge logic correctly prioritizes Layer 2 suggestions.
    - [x] 3.2.2.4: Implement ranking of final suggestions based on confidence scores.
    - [x] 3.2.2.5: Add logic to maintain a history of corrections made.

- [x] **Task 3.3: Final Testing**
  - [x] 3.3.1: Write unit tests for the `T5GrammarCorrector` and `Seq2SeqCorrector`.
  - [x] 3.3.2: Write end-to-end tests for the `/deepcheck` workflow.
  - [x] 3.3.3: Conduct benchmark tests comparing the performance and accuracy of T5 vs. Seq2Seq.

---

## Phase 4: Documentation & Deployment

- [ ] **Task 4.1: Documentation**
  - [ ] 4.1.1: Create a final `README.md` with full setup, configuration, and usage instructions.
  - [ ] 4.1.2: Write a User Manual explaining the features and how to interpret results.
  - [ ] 4.1.3: Divide documentation tasks among six team members as outlined in `docs/DOCUMENTATION_PLAN.md`.

- [ ] **Task 4.2: Deployment**
  - [ ] 4.2.1: Create a multi-stage Dockerfile that handles dependency installation, dataset preparation, and running the application.
  - [ ] 4.2.2: Write clear, step-by-step deployment instructions.